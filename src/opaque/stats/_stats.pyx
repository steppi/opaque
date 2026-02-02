# cython: cdivision=True
# cython: cpow=True

cimport cython
cimport numpy as np

from libc.float cimport DBL_EPSILON
from libc.math cimport exp, log, log1p, isinf, isnan, HUGE_VAL
from libc.stdint cimport int64_t
from numpy.math cimport INFINITY, NAN
from scipy.optimize.cython_optimize cimport brentq
from scipy.special.cython_special cimport betainc, betaln, xlog1py, xlogy


np.import_array()
np.import_ufunc()

cdef extern from "stdbool.h":
    ctypedef bint bool


cdef inline double coefficient(int64_t n, double p, double q, double x) noexcept nogil:
    """Return nth coefficient of required continued fraction expansion.

    Continued fraction expansion is for hyp2f1(p + q, 1, p + 1, x).
    """
    cdef int64_t m
    m = n // 2
    if n % 2 == 0:
        return m*(q-m)/((p+2*m-1)*(p+2*m)) * x
    else:
        return -(p+m)*(p+q+m)/((p+2*m)*(p+2*m+1)) * x


cdef inline double K(double p, double q, double x, double tol) noexcept nogil:
    """Returns hyp2f1(p + q, 1, p + 1, x)

    Evaluates continued fraction in top down fashion using Lentz's
    algorithm.
    """
    cdef int64_t n
    cdef double delC, C, D, upper, lower

    delC = coefficient(1, p, q, x)
    C, D = 1 + delC, 1
    upper, lower = HUGE_VAL, -HUGE_VAL
    n = 2
    while upper - lower > tol:
        D = 1/(D*coefficient(n, p, q, x) + 1)
        delC *= (D - 1)
        C += delC
        n += 1
        if n % 4 == 0 or n % 4 == 1:
            # nth convergent < true value if n % 4 is 0 or 1
            lower = 1/C
        else:
            # nth convergent > true value if n % 4 is 1 or 2
            upper = 1/C
    return 1/C


cdef inline double log_betainc(double p, double q, double x) noexcept nogil:
    """Returns log of incomplete beta function."""
    cdef double output
    cdef double eps = 2.220446049250313e-16
    if isnan(x):
        return NAN
    if x <= 0:
        return -INFINITY
    if x >= 1:
        return 0.0
    if x <= p/(p + q):
        if (p <= 20 and q <= 20):
            output = log(betainc(p, q, x))
            if not (isnan(output) or output == 0 or isinf(output)):
                return output
        output = log(K(p, q, x, eps))
        output += xlog1py(q, -x) + xlogy(p, x) - log(p)
        output -= betaln(p, q)
    else:
        if (p <= 20 and q <= 20):
            output = log1p(-betainc(q, p, 1 - x))
            if not (isnan(output) or output == 0 or isinf(output)):
                return output
        output = log_diff(0, log_betainc(q, p, 1-x))
    return output


cdef inline double log_diff(double log_p, double log_q) noexcept nogil:
    """Returns log(p - q) given log(p) = log_p and log(q) = log_q."""
    return log_p + log1p(-exp(log_q - log_p))


cdef inline double log_prevalence_cdf_fixed(
        double theta, int64_t n, int64_t t, double sensitivity, double specificity
) noexcept nogil:
    """Returns log of prevalence cdf for fixed sensitivity and specificity."""
    cdef bool anti_test
    cdef double c1, c2, logX, logY, logZ, log_delta, num, den
    if theta >= 1:
        return 1.
    if theta <= 0:
        return -INFINITY
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    if c2 == 0:
        # If c2 == 0, the test is uninformative. Uniform distribution
        return log(theta)
    anti_test = False
    # If c2 < 0, then the diagnostic test is an anti-test. That is, the test
    # will produce useful results if the returned labels are flipped. When it
    # says positive, then it is likely the result is actually negative.
    if c2 < 0:
        c1, c2 = 1 - c1, -c2
        theta = 1 - theta
        anti_test = True
    logY = log_betainc(t + 1, n - t + 1, c1)
    logX = log_betainc(t + 1, n - t + 1, c1 + c2*theta)
    if logX <= logY:
        # logX < logY can happen for very small theta due to numerical issues.
        num = -INFINITY
    else:
        num = log_diff(logX, logY)
    logZ = log_betainc(t + 1, n - t + 1, c1 + c2)
    if logZ <= logY:
        den = -INFINITY
    else:
        den = log_diff(logZ, logY)
    if isinf(num) and isinf(den):
        return log(theta if not anti_test else 1 - theta)
    return num - den


cdef inline double prevalence_cdf_fixed(
        double theta, int64_t n, int64_t t, double sensitivity, double specificity
) noexcept nogil:
    """Returns prevalence_cdf for fixed sensitivity and specificity."""
    cdef double c1, c2, result
    # There can be numerical difficulties for small t. Side step this.
    if t < n / 3:
        return 1 - prevalence_cdf_fixed(
            1 - theta, n, n - t, specificity, sensitivity
        )
    if theta >= 1:
        return 1.0
    if theta <= 0:
        return 0.0
    return exp(
        log_prevalence_cdf_fixed(theta, n, t, sensitivity, specificity)
    )


ctypedef double (*prev_func_ptr)(double, int64_t, int64_t, double, double) noexcept nogil


ctypedef struct inv_cdf_args:
    int64_t n
    int64_t t
    double sens
    double spec
    double p
    prev_func_ptr prev_func


cdef inline double func(double theta, void* args) noexcept nogil:
    cdef:
        inv_cdf_args *myargs = <inv_cdf_args *> args
        int64_t n = myargs.n
        int64_t t = myargs.t
        double sens = myargs.sens
        double spec = myargs.spec
        double p = myargs.p
        prev_func_ptr prev_func = myargs.prev_func
    return prev_func(theta, n, t, sens, spec) - p


cdef inline double inverse_prevalence_cdf_fixed(
    double p, int64_t n, int64_t t, double sensitivity, double specificity,
    prev_func_ptr prev_func,
) noexcept nogil:
    """Inverse of prevalence cdf for fixed sensivivity and specificity."""
    cdef inv_cdf_args args
    args.n = n
    args.t = t
    args.sens = sensitivity
    args.spec = specificity
    args.p = p
    args.prev_func = prev_func
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return brentq(
        func, 0.0, 1.0, <inv_cdf_args *> &args, 1e-100, DBL_EPSILON, 100, NULL
    )


cdef inline double prevalence_cdf_positive_fixed(
        double psi, int64_t n, int64_t t, double sensitivity, double specificity
) noexcept nogil:
    cdef:
        double c1, c2, theta
    if psi >= 1:
        return 1.0
    if psi <= 0:
        return 0.0
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    theta = c1*psi / (c1 + c2 - c2*psi)
    return prevalence_cdf_fixed(theta, n, t, sensitivity, specificity)


cdef inline double prevalence_cdf_negative_fixed(
        double psi, int64_t n, int64_t t, double sensitivity, double specificity
) noexcept nogil:
    cdef:
        double c1, c2, theta
    if psi >= 1:
        return 1.0
    if psi <= 0:
        return 0.0
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    theta = (1 - c1)*psi / (1 - c1 - c2 + c2*psi)
    if isnan(theta):
        theta = 1.0
    return prevalence_cdf_fixed(theta, n, t, sensitivity, specificity)


@cython.ufunc
cdef double log_betainc_ufunc(double p, double q, double x) nogil:
    return log_betainc(p, q, x)


@cython.ufunc
cdef double prevalence_cdf_fixed_ufunc(
        double theta, int64_t n, int64_t t, double sensitivity, double specificity,
        int64_t cond
) nogil:
    if cond == 0:
        return prevalence_cdf_fixed(theta, n, t, sensitivity, specificity)
    if cond == 1:
        return prevalence_cdf_positive_fixed(theta, n, t, sensitivity, specificity)
    if cond == -1:
        return prevalence_cdf_negative_fixed(theta, n, t, sensitivity, specificity)
    return NAN



@cython.ufunc
cdef double inverse_prevalence_cdf_fixed_ufunc(
    double p, int64_t n, int64_t t, double sensitivity, double specificity,
    int64_t cond
) nogil:
    if cond == 0:
        return inverse_prevalence_cdf_fixed(
            p, n, t, sensitivity, specificity, prevalence_cdf_fixed
        )
    if cond == 1:
        return inverse_prevalence_cdf_fixed(
            p, n, t, sensitivity, specificity, prevalence_cdf_positive_fixed
        )
    if cond == -1:
        return inverse_prevalence_cdf_fixed(
            p, n, t, sensitivity, specificity, prevalence_cdf_negative_fixed
        )
    return NAN
