import pytest
import numpy as np
from scipy.stats import beta
from opaque.simulations.prevalence import run_trial_for_theta
from opaque.stats import equal_tailed_interval, highest_density_interval


@pytest.mark.parametrize(
    "test_input",
    [
        (0.4, 12, 4, 70, 30, 100),
        (0.8, 20, 5, 60, 40, 20),
        (0.2, 10, 5, 80, 20, 1000)
    ],
)
def test_equal_tailed_interval(test_input):
    n_trials = 100
    theta, sens_a, sens_b, spec_a, spec_b, sample_size = test_input
    hits = 0
    for _ in range(n_trials):
        sensitivity = beta.rvs(sens_a, sens_b)
        specificity = beta.rvs(spec_a, spec_b)
        n, t, _, _, _ = run_trial_for_theta(
            theta,
            sensitivity,
            specificity,
            sample_size,
            np.random.RandomState()
        )
        interval = equal_tailed_interval(
            n, t, sens_a, sens_b, spec_a, spec_b, alpha=0.1
        )
        if interval[0] <= theta <= interval[1]:
            hits += 1
    coverage_rate = hits / n_trials
    assert coverage_rate > 0.8


@pytest.mark.parametrize(
    "test_input", [(0.4, 12, 4, 70, 30, 100), (0.8, 20, 5, 60, 40, 20)]
)
def test_highest_density_interval(test_input):
    n_trials = 100
    theta, sens_a, sens_b, spec_a, spec_b, sample_size = test_input
    hits = 0
    for _ in range(n_trials):
        sensitivity = beta.rvs(sens_a, sens_b)
        specificity = beta.rvs(spec_a, spec_b)
        n, t, _, _, _ = run_trial_for_theta(
            theta,
            sensitivity,
            specificity,
            sample_size,
            np.random.RandomState()
        )
        interval = highest_density_interval(
            n, t, sens_a, sens_b, spec_a, spec_b, alpha=0.1
        )
        print(interval, n, t)
        if interval[0] <= theta <= interval[1]:
            hits += 1
    coverage_rate = hits / n_trials
    print(coverage_rate)
    assert coverage_rate > 0.8
