import cython
import numpy as np
from libc.math cimport pow as cpow
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport fabs, exp, log, log1p, pi, sqrt


from scipy.integrate import dblquad
from scipy.optimize.cython_optimize cimport brentq
from scipy.special.cython_special cimport betainc, betaln, xlog1py, xlogy


cdef double beta_pdf(double theta, double p, double q):
    cdef double exponent
    exponent = xlog1py(q - 1.0, -theta) + xlogy(p - 1.0, theta)
    exponent -= betaln(p, q)
    return exp(exponent)


cdef double prevalence_cdf_exact(double theta, int n, int t,
                                 double sensitivity,
                                 double specificity):
    cdef double c1, c2, numerator, denominator
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    numerator = (betainc(t+1, n-t+1, c1 + c2*theta) - betainc(t+1, n-t+1, c1))
    denominator = betainc(t+1, n-t+1, c1 + c2) - betainc(t+1, n-t+1, c1)
    if denominator == 0:
        return theta
    return numerator/denominator


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


def integrand(double sens, double spec, Params p):
    return prevalence_cdf_exact(p.theta, p.n, p.t, sens, spec) * \
        beta_pdf(sens, p.sens_a, p.sens_b) * \
        beta_pdf(spec, p.spec_a, p.spec_b)
            
        
cdef double prevalence_cdf(double theta, int n, int t,
                           double sens_a, double sens_b,
                           double spec_a, double spec_b):
    p = Params(theta, n, t, sens_a, sens_b, spec_a, spec_b)
    return dblquad(integrand, 0, 1, 0, 1, args=(p,))[0]


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
