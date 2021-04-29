import cython
import numpy as np
from numpy.random import PCG64
from scipy.integrate import dblquad, quad

from numpy.random cimport bitgen_t
from numpy.random.c_distributions cimport random_beta

from libc.float cimport DBL_MIN
from libc.math cimport fabs, exp, log, log1p, sqrt, isnan, HUGE_VAL

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

from scipy.optimize.cython_optimize cimport brentq
from scipy.special.cython_special cimport betainc, betaln, xlog1py, xlogy


cdef extern from "stdbool.h":
    ctypedef bint bool


cdef double log_beta_pdf(double theta, double p, double q):
    """Returns log of pdf of beta distribution.

    theta should be a double in the range [0, 1]. p and q are
    the shape parameters of the beta distribution.
    """
    cdef double output
    output = xlog1py(q - 1.0, -theta) + xlogy(p - 1.0, theta)
    output -= betaln(p, q)
    return output


cdef double beta_pdf(double theta, double p, double q):
    """Returns pdf of beta distribution

    theta should be a double in the range [0, 1]. p and q are
    the shape parameters of the beta distribution.
    """
    return exp(log_beta_pdf(theta, p, q))


@cython.cdivision(True)
cdef inline double coefficient(int n, double p, double q, double x):
    """Return nth term of continued fraction expansion used in betainc calc
    
    Continued fraction expansion is for hyp2f1(p + q, 1, p + 1, x)
    """
    cdef int m
    m = n // 2
    if n % 2 == 0:
        return m*(q-m)/((p+2*m-1)*(p+2*m)) * x
    else:
        return -(p+m)*(p+q+m)/((p+2*m)*(p+2*m+1)) * x


@cython.cdivision(True)
cdef double K(double p, double q, double x, double tol):
    """Returns hyp2f1(p + q, 1, p + 1, x)"""
    cdef int n
    cdef double delC, C, D
    delC = coefficient(1, p, q, x)
    C, D = 1 + delC, 1
    n = 2
    while fabs(delC) > tol:
        D = 1/(D*coefficient(n, p, q, x) + 1)
        delC *= (D - 1)
        C += delC
        n += 1
    return 1/C


cdef double _log_betainc(double p, double q, double x):
    """Returns log of incomplete beta function."""
    cdef double output
    if x <= p/(p + q):
        output = xlog1py(q, -x) + xlogy(p, x) - log(p)
        output -= betaln(p, q)
        output += log(K(p, q, x, 1e-20))
    else:
        output = log_diff(0, log_betainc(q, p, 1-x))
    return output


def log_betainc(p: float, q: float, x: float) -> float:
    """Returns log of incomplete beta function.

    Parameters
    ----------
    p : float
        First shape parameter for beta distribution
    q : float
        Second shape parameter for beta distribution
    x : float
        Argument of incomplete beta function. (Evaluate integral of beta pdf
        from 0 to x)

    Returns
    -------
    float
    """
    return _log_betainc(p, q, x)


cdef double log_diff(double log_p, double log_q):
    """Returns log(p - q) given log(p) = log_p and log(q) = log_q."""
    return log_p + log1p(-exp(log_q - log_p))


cdef double log_prevalence_cdf_fixed(double theta, int n, int t,
                                     double sensitivity,
                                     double specificity):
    """Returns log of prevalence cdf for fixed sensitivity and specificity."""
    cdef double c1, c2, logY
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    logY = log_betainc(t + 1, n - t + 1, c1)
    if c2 == 0:
        output = log(theta)
    elif c1 + c2 >= c1:
        output = log_diff(log_betainc(t + 1, n - t + 1, c1 + c2*theta),
                          logY)
        output -= log_diff(log_betainc(t + 1, n - t + 1, c1 + c2),
                           logY)
    else:
        output = log_diff(logY,
                          log_betainc(t + 1, n - t + 1, c1 + c2*theta))
        output -= log_diff(logY,
                           log_betainc(t + 1, n - t + 1, c1 + c2))
    if isnan(output):
        if t/n < c1:
            output = -HUGE_VAL if theta == 0.0 else 0.0
        elif c1 + c2 < t/n:
            output = 0.0 if theta == 1.0 else -HUGE_VAL
        else:
            output = log(theta)
    return output


cdef double prevalence_cdf_fixed(double theta, int n, int t,
                                 double sensitivity,
                                 double specificity):
    """Returns prevalence_cdf for fixed sensitivity and specificity."""
    return exp(log_prevalence_cdf_fixed(theta, n, t, sensitivity, specificity))


cdef class Params:
    """Parameters to pass to scipy.integrate dblquad

    Integration is used to marginalize prevalence cdf over priors for
    sensitivity and specificity.
    """
    cdef public int n, t
    cdef public double theta, sens_a, sens_b, spec_a, spec_b
    
    def __init__(self, theta, n, t, sens_a, sens_b, spec_a, spec_b):
        self.theta = theta
        self.n = n
        self.t = t
        self.sens_a = sens_a
        self.sens_b = sens_b
        self.spec_a = spec_a
        self.spec_b = spec_b


def integrand_cdf(double sens, double spec, Params p):
    """Integrand for marginalization."""
    return prevalence_cdf_fixed(p.theta, p.n, p.t, sens, spec) * \
        beta_pdf(sens, p.sens_a, p.sens_b) * \
        beta_pdf(spec, p.spec_a, p.spec_b)


@cython.cdivision(True)
cdef double _prevalence_cdf(double theta, int n, int t,
                            double sens_a, double sens_b,
                            double spec_a, double spec_b):
    """Compute marginalization integral with quadrature."""
    cdef double output, error
    p = Params(theta, n, t, sens_a, sens_b, spec_a, spec_b)
    output, error = dblquad(integrand_cdf, 0, 1, 0, 1, args=(p,),
                            epsabs=1e-3, epsrel=1e-3)
    if output < 0.0:
        return 0.0
    elif output > 1.0:
        return 1.0
    else:
        return output


cdef double _prevalence_cdf_mc_est(double theta, int n, int t,
                                   double sens_a, double sens_b,
                                   double spec_a, double spec_b,
                                   int num_samples):
    """Computes marginalization integral with importance sampling."""
    cdef int i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double result
    cdef double *sens_array
    cdef double *spec_array

    sens_array = <double *> PyMem_Malloc(num_samples * sizeof(double))
    spec_array = <double *> PyMem_Malloc(num_samples * sizeof(double))
    
    x = PCG64()
    capsule = x.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    i = 0
    with x.lock, nogil:
        for i in range(num_samples):
            sens_array[i] = random_beta(rng, sens_a, sens_b)
            spec_array[i] = random_beta(rng, spec_a, spec_b)
    result = 0
    i = 0
    for i in range(num_samples):
        result += prevalence_cdf_fixed(theta, n, t, sens_array[i],
                                       spec_array[i])
    PyMem_Free(sens_array)
    PyMem_Free(spec_array)
    return result/num_samples


def prevalence_cdf(float theta, int n, int t, float sens_a, float sens_b,
                   float spec_a, float spec_b, mc_est: bool=True,
                   num_mc_samples: int=5000) -> tuple[float, float]:
    """Returns prevalence_cdf as derived in Diggle, 2011.

    Parameters
    ----------
    theta : float
        Value of prevalence at which to calculate cdf.
    n : int
        Number of samples on which diagnostic test has been run.
    t : int
        Number of positives out of all samples.
    sens_a : float
        First shape parameter of beta prior for sensitivity
    sens_b : float
        Second shape parameter of beta prior for sensitivity
    spec_a : float
        First shape parameter of beta prior for specificity
    spec_b : float
        Second shape parameter of beta prior for specificity
    mc_est : Optional[bool]
       If True, calculate integral for marginalization over priors using
       importance sampling Monte-carlo. Otherwise use scipy's dblquad to
       calculate the integral. Using dblquad is more accurate but can behave
       poorly and take an excessive amount of time due to convergence issues
       for some values of the parameters.
    num_mc_samples : Optional[int]
       Number of samples to take when importance sampling if mc_est = True.

    Returns
    -------
    float
        Value of cdf at theta for given parameters.

    References
    ----------
    [1] Peter J. Diggle, "Estimating Prevalence Using an Imperfect Test",
        Epidemiology Research International, vol. 2011, Article ID 608719,
        5 pages, 2011. https://doi.org/10.1155/2011/608719
    """
    if mc_est:
        return _prevalence_cdf_mc_est(theta, n, t, sens_a, sens_b, spec_a, spec_b,
                                      num_mc_samples)
    else:
        return _prevalence_cdf(theta, n, t, sens_a, sens_b, spec_a, spec_b)


ctypedef struct inverse_cdf_params:
    int n
    int t
    int num_samples
    double sens_a
    double sens_b
    double spec_a
    double spec_b
    double val
    bint mc_est


@cython.cdivision(True)
cdef double f1(double theta, void *args):
    cdef inverse_cdf_params *params = <inverse_cdf_params *> args
    return _prevalence_cdf(theta, params.n, params.t,
                           params.sens_a, params.sens_b,
                           params.spec_a, params.spec_b) - params.val


@cython.cdivision(True)
cdef double f2(double theta, void *args):
    cdef inverse_cdf_params *params = <inverse_cdf_params *> args
    return _prevalence_cdf_mc_est(theta, params.n, params.t,
                                  params.sens_a, params.sens_b,
                                  params.spec_a, params.spec_b,
                                  params.num_samples) - params.val


ctypedef double (*function_1d)(double, void*)


cdef double _inverse_cdf(double x, int n, int t, double sens_a, double sens_b,
                         double spec_a, double spec_b, int num_samples,
                         function_1d func):
    cdef inverse_cdf_params args
    args.n, args.t = n, t
    args.num_samples = num_samples
    args.sens_a, args.sens_b = sens_a, sens_b
    args.spec_a, args.spec_b = spec_a, spec_b
    args.val = x
    return brentq(func, 0, 1, &args, 1e-3, 1e-3, 100, NULL)


def inverse_cdf(x, n, t, sens_a, sens_b, spec_a, spec_b, mc_est=True,
                num_mc_samples=5000):
    """Returns inverse of prevalence cdf evaluated at x for param values

    Uses scipy's brentq root finding algorithm to calculate inverse cdf
    by solving prevalence_cdf(theta, ...) = x for theta.

    Parameters
    ----------
    x : float
        Probability value at which to calculate inverse cdf.
    n : int
        Number of samples on which diagnostic test has been run.
    t : int
        Number of positives out of all samples.
    sens_a : float
        First shape parameter of beta prior for sensitivity
    sens_b : float
        Second shape parameter of beta prior for sensitivity
    spec_a : float
        First shape parameter of beta prior for specificity
    spec_b : float
        Second shape parameter of beta prior for specificity
    mc_est : Optional[bool]
       If True, calculate integral for marginalization over priors using
       importance sampling Monte-carlo. Otherwise use scipy's dblquad to
       calculate the integral. Using dblquad is more accurate but can behave
       poorly and take an excessive amount of time due to convergence issues
       for some values of the parameters.
    num_mc_samples : Optional[int]
       Number of samples to take when importance sampling if mc_est = True.

    Returns
    -------
    float
        Value of inverse cdf at x for given parameters.

    References
    ----------
    [1] Peter J. Diggle, "Estimating Prevalence Using an Imperfect Test",
        Epidemiology Research International, vol. 2011, Article ID 608719,
        5 pages, 2011. https://doi.org/10.1155/2011/608719
    """
    if mc_est:
        return _inverse_cdf(x, n, t, sens_a, sens_b, spec_a, spec_b,
                            num_mc_samples, f2)
    else:
        return _inverse_cdf(x, n, t, sens_a, sens_b, spec_a, spec_b,
                            num_mc_samples, f1)
        

cdef double interval_width(double x, void *args):
    cdef inverse_cdf_params *params = <inverse_cdf_params *> args
    cdef double left, right
    func = f2 if params.mc_est else f1
    left = _inverse_cdf(x, params.n, params.t,
                        params.sens_a, params.sens_b,
                        params.spec_a, params.spec_b,
                        params.num_samples, func)
    right = _inverse_cdf(x + params.val, params.n, params.t,
                         params.sens_a, params.sens_b,
                         params.spec_a, params.spec_b,
                         params.num_samples, func)
    return right - left


cdef (double, double) golden_section_search(function_1d func, double left,
                                            double right, double x_tol,
                                            double y_tol,
                                            void *args):
    """Returns a minimum obtained by func over the interval (left, right)

    Returns the minimum value of f(x), contrary to the typical practice
    of returning the value x at which the minimum is obtained. Guaranteed
    to be a global minimum if f is unimodal.
    """
    cdef double x1, x2, x3, x4
    cdef double func_at_x2, func_at_x3
    cdef double inv_phi = (sqrt(5) - 1) / 2

    x1, x4 = left, right
    x2, x3 = x4 - (x4 - x1) * inv_phi, x1 + (x4 - x1) * inv_phi
    func_at_x2, func_at_x3 = func(x2, args), func(x3, args)
    while True:
        if func_at_x2 < func_at_x3:
            x3, x4 = x2, x3
            if fabs(x4 - x1) < x_tol and fabs(func_at_x2 - func_at_x3) < y_tol:
                return x2, func_at_x2
            func_at_x3 = func_at_x2
            x2 = x4 - (x4 - x1) * inv_phi
            func_at_x2 = func(x2, args)
        else:
            x1, x2 = x2, x3
            if fabs(x4 - x1) < x_tol and fabs(func_at_x2 - func_at_x3) < y_tol:
                return x3, func_at_x3
            func_at_x2 = func_at_x3
            x3 = x1 + (x4 - x1) * inv_phi
            func_at_x3 = func(x3, args)


def highest_density_interval(n, t, sens_a, sens_b, spec_a, spec_b, alpha=0.1,
                             mc_est=True,
                             num_samples=5000):
    cdef double left, right
    cdef double argmin, min_
    cdef inverse_cdf_params args
    args.n, args.t = n, t
    args.sens_a, args.sens_b = sens_a, sens_b
    args.spec_a, args.spec_b = spec_a, spec_b
    args.val = 1 - alpha
    args.mc_est = mc_est
    func = f2 if mc_est else f1
    args.num_samples = num_samples
    argmin_width, min_width = golden_section_search(interval_width, 0, alpha,
                                                    1e-3, 1e-3, &args)
    left = _inverse_cdf(argmin_width, n, t, sens_a, sens_b, spec_a, spec_b,
                        num_samples, func)
    right = left + min_width
    return (max(0.0, left), min(right, 1.0))


def equal_tailed_interval(n, t, sens_a, sens_b, spec_a, spec_b, alpha=0.1,
                          mc_est=True, num_samples=5000):
    func = f2 if mc_est else f1
    return (_inverse_cdf(alpha/2, n, t, sens_a, sens_b, spec_a, spec_b,
                         num_samples, func),
            _inverse_cdf(1 - alpha/2, n, t, sens_a, sens_b, spec_a, spec_b,
                         num_samples, func))
