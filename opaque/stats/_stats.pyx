import cython
import numpy as np
from libc.math cimport fabs, exp, log, sqrt

from scipy.integrate import dblquad, quad
from scipy.optimize.cython_optimize cimport brentq
from scipy.special.cython_special cimport betainc, betaln, xlog1py, xlogy


cdef double log_beta_pdf(double theta, double p, double q):
    cdef double output
    output = xlog1py(q - 1.0, -theta) + xlogy(p - 1.0, theta)
    output -= betaln(p, q)
    return output


cdef double beta_pdf(double theta, double p, double q):
    return exp(log_beta_pdf(theta, p, q))


@cython.cdivision(True)
cdef inline double coefficient(int n, double p, double q, double x):
    cdef int m
    m = n // 2
    if n % 2 == 0:
        return m*(q-m)/((p+2*m-1)*(p+2*m)) * x
    else:
        return -(p+m)*(p+q+m)/((p+2*m)*(p+2*m+1)) * x


@cython.cdivision(True)
cdef double K(double p, double q, double x, double tol=1e-20):
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


cdef double log_betainc(double p, double q, double theta):
    cdef double output
    output = xlog1py(q, -theta) + xlogy(p, theta) - log(p)
    output -= betaln(p, q)
    output += log(K(p, q, theta))
    return output


cdef double log_diff(double log_p, double log_q):
    return log_p + exp(log_q - log_p)


cdef double prevalence_cdf_exact(double theta, int n, int t,
                                 double sensitivity,
                                 double specificity):
    cdef double c1, c2, numerator, denominator
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    if c2 == 0:
        return theta
    numerator = (betainc(t + 1, n - t + 1, c1 + c2*theta) -
                 betainc(t + 1, n - t + 1, c1))
    denominator = (betainc(t + 1, n - t + 1, c1 + c2) -
                   betainc(t + 1, n - t + 1, c1))
    if denominator == 0:
        if t/n < c1:
            return 0.0 if theta == 0.0 else 1.0
        elif t/n > c1 + c2:
            return 1.0 if theta == 1.0 else 0.0
        return theta
    return numerator/denominator

def py_cdf(theta, n, t, sens, spec):
    return prevalence_cdf_exact(theta, n, t, sens, spec)


cdef class Params:
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
    return prevalence_cdf_exact(p.theta, p.n, p.t, sens, spec) * \
        beta_pdf(sens, p.sens_a, p.sens_b) * \
        beta_pdf(spec, p.spec_a, p.spec_b)


@cython.cdivision(True)
cdef double prevalence_cdf(double theta, int n, int t,
                                     double sens_a, double sens_b,
                                     double spec_a, double spec_b):
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


def py_prevalence_cdf(theta, n, t, sens_a, sens_b, spec_a, spec_b):
    return prevalence_cdf(theta, n, t, sens_a, sens_b, spec_a, spec_b)


ctypedef struct inverse_cdf_params:
    int n
    int t
    double sens_a
    double sens_b
    double spec_a
    double spec_b
    double val


@cython.cdivision(True)
cdef double f(double theta, void *args):
    cdef inverse_cdf_params *params = <inverse_cdf_params *> args
    return prevalence_cdf(theta, params.n, params.t,
                          params.sens_a, params.sens_b,
                          params.spec_a, params.spec_b) - params.val


cdef double inverse_cdf(double x, int n, int t, double sens_a, double sens_b,
                        double spec_a, double spec_b):
    cdef inverse_cdf_params args
    args.n, args.t = n, t
    args.sens_a, args.sens_b = sens_a, sens_b
    args.spec_a, args.spec_b = spec_a, spec_b
    args.val = x
    return brentq(f, 0, 1, &args, 1e-3, 1e-3, 100, NULL)


cdef double interval_width(double x, void *args):
    cdef inverse_cdf_params *params = <inverse_cdf_params *> args
    cdef double left, right
    left = inverse_cdf(x, params.n, params.t,
                       params.sens_a, params.sens_b,
                       params.spec_a, params.spec_b)
    right = inverse_cdf(x + params.val, params.n, params.t,
                        params.sens_a, params.sens_b,
                        params.spec_a, params.spec_b)
    return right - left


ctypedef double (*function_1d)(double, void*)


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


def highest_density_interval(n, t, sens_a, sens_b, spec_a, spec_b, alpha):
    cdef double left, right
    cdef double argmin, min_
    cdef inverse_cdf_params args
    args.n, args.t = n, t
    args.sens_a, args.sens_b = sens_a, sens_b
    args.spec_a, args.spec_b = spec_a, spec_b
    args.val = 1 - alpha

    argmin_width, min_width = golden_section_search(interval_width, 0, alpha,
                                                    1e-3, 1e-3, &args)
    left = inverse_cdf(argmin_width, n, t, sens_a, sens_b, spec_a, spec_b)
    right = left + min_width
    return (max(0.0, left), min(right, 1.0))


def equal_tailed_interval(n, t, sens_a, sens_b, spec_a, spec_b, alpha):
    return (inverse_cdf(alpha/2, n, t, sens_a, sens_b, spec_a, spec_b),
            inverse_cdf(1 - alpha/2, n, t, sens_a, sens_b, spec_a, spec_b))
