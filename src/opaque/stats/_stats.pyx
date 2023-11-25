# distutils: language=c++
# cython: cdivision=True
# cython: cpow=True

cimport cython
cimport numpy as np

from libc.math cimport exp, log, log1p, isinf, isnan, HUGE_VAL
from numpy.math cimport INFINITY, NAN
from scipy.special.cython_special cimport betainc, betaln, xlog1py, xlogy


cdef extern from "stdbool.h":
    ctypedef bint bool


@cython.cdivision(True)
cdef inline double coefficient(int n, double p, double q, double x) noexcept nogil:
    """Return nth coefficient of required continued fraction expansion.

    Continued fraction expansion is for hyp2f1(p + q, 1, p + 1, x).
    """
    cdef int m
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
    cdef int n
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


cdef api double log_betainc(double p, double q, double x) noexcept nogil:
    """Returns log of incomplete beta function."""
    cdef double output
    cdef double eps = 2.220446049250313e-16
    if isnan(x):
        return NAN
    if x <= 0:
        return -INFINITY
    if x >= 1:
        return 0.0
    output = log(betainc(p, q, x))
    if not (isnan(output) or output == 0 or isinf(output)):
        return output
    if x <= p/(p + q):
        output = log(K(p, q, x, eps))
        output += xlog1py(q, -x) + xlogy(p, x) - log(p)
        output -= betaln(p, q)
    else:
        output = log_diff(0, log_betainc(q, p, 1-x))
    return output


cdef inline double log_diff(double log_p, double log_q) noexcept nogil:
    """Returns log(p - q) given log(p) = log_p and log(q) = log_q."""
    return log_p + log1p(-exp(log_q - log_p))


cdef inline double log_prevalence_cdf_fixed(
        double theta, int n, int t, double sensitivity, double specificity
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


cdef api double prevalence_cdf_fixed(
        double theta, int n, int t, double sensitivity, double specificity
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


@cython.cdivision(True)
cdef api double prevalence_cdf_positive_fixed(
        double psi, int n, int t, double sensitivity, double specificity
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


@cython.cdivision(True)
cdef api double prevalence_cdf_negative_fixed(
        double psi, int n, int t, double sensitivity, double specificity
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
