import pytest
import numpy as np

from opaque.stats import inverse_prevalence_cdf, prevalence_cdf


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
