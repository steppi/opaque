import pytest
import numpy as np
import mpmath as mp
from itertools import product

from opaque.stats import inverse_prevalence_cdf, prevalence_cdf, log_betainc


def mpmath_log_betainc(p, q, x):
    return mp.log(mp.betainc(p, q, 0, x, regularized=True))


@pytest.mark.parametrize('test_input',
                          [(p, q, x)
                           for p, q in product([10, 50, 100, 200, 1000],
                                               repeat=2)
                           for x in [0.1, 0.2, 0.4, 0.8]])
def test_log_betainc(test_input):
    tol=1e-12
    p, q, x = test_input
    expected = float(mpmath_log_betainc(p, q, x))
    observed = log_betainc(p, q, x)
    assert abs(expected - observed) < tol


@pytest.mark.parametrize('test_input',
                         [(n, t, sens_a, sens_b, spec_a, spec_b)
                          for n in [100, 1000]
                          for t in [n//3, 2*n // 3]
                          for sens_a, sens_b in [(60, 40), (80, 20)]
                          for spec_a, spec_b in [(60, 40), (80, 20)]])
def test_prevalence_cdf_mc(test_input):
    """Test importance sample estimate of prevalence cdf

    Test against exact integration through quadrature. Ensure that
    max absolute error and mean absolute error are within acceptable
    tolerance.
    """
    num_steps = 20
    n, t, sens_a, sens_b, spec_a, spec_b = test_input
    line = np.linspace(0, 1, num_steps)
    y_quad = np.fromiter((prevalence_cdf(theta, n, t, sens_a, sens_b,
                                         spec_a, spec_b, mc_est=False)
                          for theta in line), dtype=float)
    y_mc = np.fromiter((prevalence_cdf(theta, n, t, sens_a, sens_b,
                                       spec_a, spec_b, mc_est=True)
                        for theta in line), dtype=float)
    mean_absolute_error = np.sum(np.abs(y_quad - y_mc))/num_steps
    max_absolute_error = np.max(np.abs(y_quad - y_mc))
    assert mean_absolute_error < 0.002
    assert max_absolute_error < 0.02


@pytest.mark.parametrize('test_input',
                         [(n, t, sens_a, sens_b, spec_a, spec_b)
                          for n in [20, 100]
                          for t in [n//3, 2*n // 3]
                          for sens_a, sens_b in [(60, 40), (80, 20)]
                          for spec_a, spec_b in [(60, 40), (80, 20)]])
def test_inverse_prevalence_cdf_mc(test_input):
    """Test inverse prevalence cdf when using importance sample estimate of cdf

    Test against exact integration through quadrature. Ensure that
    max absolute error and mean absolute error are within acceptable
    tolerance.
    """
    num_steps = 20
    n, t, sens_a, sens_b, spec_a, spec_b = test_input
    line = np.linspace(0, 1, num_steps)
    y_quad = np.fromiter((inverse_prevalence_cdf(x, n, t, sens_a, sens_b,
                                                 spec_a, spec_b, mc_est=False)
                          for x in line), dtype=float)
    y_mc = np.fromiter((inverse_prevalence_cdf(x, n, t, sens_a, sens_b,
                                               spec_a, spec_b, mc_est=True)
                        for x in line), dtype=float)
    mean_absolute_error = np.sum(np.abs(y_quad - y_mc))/num_steps
    max_absolute_error = np.max(np.abs(y_quad - y_mc))
    assert mean_absolute_error < 0.001
    assert max_absolute_error < 0.01
