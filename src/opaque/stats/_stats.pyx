import cython
cimport numpy as np

from libc.math cimport exp, log, log1p, isnan, HUGE_VAL
from numpy.math cimport INFINITY
from scipy.special.cython_special cimport betaln, xlog1py, xlogy


@cython.cdivision(True)
cdef inline double coefficient(int n, double p, double q, double x) nogil:
    """Return nth coefficient of required continued fraction expansion.

    Continued fraction expansion is for hyp2f1(p + q, 1, p + 1, x).
    """
    cdef int m
    m = n // 2
    if n % 2 == 0:
        return m*(q-m)/((p+2*m-1)*(p+2*m)) * x
    else:
        return -(p+m)*(p+q+m)/((p+2*m)*(p+2*m+1)) * x


@cython.cdivision(True)
cdef double K(double p, double q, double x, double tol) nogil:
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


@cython.cdivision(True)
cpdef double log_betainc(double p, double q, double x) nogil:
    """Returns log of incomplete beta function."""
    cdef double output
    if x <= p/(p + q):
        output = log(K(p, q, x, 1e-16))
        output += xlog1py(q, -x) + xlogy(p, x) - log(p)
        output -= betaln(p, q)
    else:
        output = log_diff(0, log_betainc(q, p, 1-x))
    return output


cdef inline double log_diff(double log_p, double log_q) nogil:
    """Returns log(p - q) given log(p) = log_p and log(q) = log_q."""
    return log_p + log1p(-exp(log_q - log_p))


cdef double log_prevalence_cdf_fixed(
        double theta, int n, int t, double sensitivity, double specificity
) nogil:
    """Returns log of prevalence cdf for fixed sensitivity and specificity."""
    cdef double c1, c2, logY, log_delta
    if theta >= 1:
        return 1.
    if theta <= 0:
        return -INFINITY
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    if c2 <= 0:
        return log(theta)
    logY = log_betainc(t + 1, n - t + 1, c1)
    output = log_diff(log_betainc(t + 1, n - t + 1, c1 + c2*theta), logY)
    output -= log_diff(log_betainc(t + 1, n - t + 1, c1 + c2), logY)
    if isnan(output):
        output = log(theta)
    return output


cpdef api double prevalence_cdf_fixed(
        double theta, int n, int t, double sensitivity, double specificity
) nogil:
    """Returns prevalence_cdf for fixed sensitivity and specificity."""
    if theta >= 1:
        return 1.0
    if theta <= 0:
        return 0.0
    return exp(log_prevalence_cdf_fixed(theta, n, t, sensitivity, specificity))


@cython.cdivision(True)
cpdef api double prevalence_cdf_positive_fixed(
        double psi, int n, int t, double sensitivity, double specificity
) nogil:
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
cpdef api double prevalence_cdf_negative_fixed(
        double psi, int n, int t, double sensitivity, double specificity
) nogil:
    cdef:
        double c1, c2, theta
    if psi >= 1:
        return 1.0
    if psi <= 0:
        return 0.0
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    theta = (1 - c1)*psi / (1 - c1 - c2 + c2*psi)
    return prevalence_cdf_fixed(theta, n, t, sensitivity, specificity)
