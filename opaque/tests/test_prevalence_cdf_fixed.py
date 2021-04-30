"""This tests compare the closed form expression for the Prevalence
CDF when the diagnostic test has fixed sensitivity and specificity
with empirical distributions based on simulations. Ensures mean absolute
error between calculated CDF and empirical CDF remains within tolerance
tol = 0.05.
"""
import pytest
import numpy as np
from scipy.stats import binom
from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF

from opaque.stats import prevalence_cdf_fixed


def simulation(sensitivity, specificity, n_trials=1000, samples_per_trial=100,
               num_grid_points=100, seed=None):
    results = defaultdict(list)
    random_state = np.random.RandomState(seed)
    binom_pos = binom(1, sensitivity)
    binom_neg = binom(1, 1 - specificity)
    binom_pos.random_state = random_state
    binom_neg.random_state = random_state
    for theta in np.linspace(0, 1, num_grid_points):
        binom_ground = binom(1, theta)
        binom_ground.random_state = random_state
        for i in range(n_trials):
            ground = binom_ground.rvs(size=samples_per_trial)
            pos = binom_pos.rvs(size=samples_per_trial)
            neg = binom_neg.rvs(size=samples_per_trial)
            observed = np.where(ground == 1, pos, neg)
            n, t = samples_per_trial, sum(observed)
            results[(n, t)].append(theta)
    results = {(n, t): ECDF(theta_list)
               for (n, t), theta_list in results.items()}
    return results


@pytest.fixture(scope='session')
def simulation1():
    sens, spec = 0.8, 0.7
    return sens, spec, simulation(sens, spec, seed=561)


@pytest.fixture(scope='session')
def simulation2():
    sens, spec = 0.7, 0.8
    return sens, spec, simulation(sens, spec, seed=1105)


@pytest.fixture(scope='session')
def simulation3():
    sens, spec = 0.6, 0.9
    return (sens, spec, simulation(sens, spec, seed=1729))


def get_mae_for_test(n, t, sens, spec, results):
    ecdf = results[(n, t)]
    x = np.linspace(0, 1, 100)
    y1 = ecdf(x)
    y2 = np.fromiter((prevalence_cdf_fixed(theta, n, t,
                                           sens,
                                           spec)
                      for theta in x), dtype=float)
    mean_absolute_error = np.sum(np.abs(y1 - y2))/len(x)
    return mean_absolute_error


@pytest.mark.parametrize('test_input',
                         [(100, t) for t in range(20, 81)])
def test_prevalence_cdf_fixed_sim1(test_input, simulation1):
    n, t = test_input
    sens, spec, results = simulation1
    if (n, t) in results:
        mean_absolute_error = get_mae_for_test(n, t, sens, spec, results)
        assert mean_absolute_error < 0.05


@pytest.mark.parametrize('test_input',
                         [(100, t) for t in range(20, 81)])
def test_prevalence_cdf_fixed_sim2(test_input, simulation2):
    n, t = test_input
    sens, spec, results = simulation2
    if (n, t) in results:
        mean_absolute_error = get_mae_for_test(n, t, sens, spec, results)
        assert mean_absolute_error < 0.05


@pytest.mark.parametrize('test_input',
                         [(100, t) for t in range(20, 81)])
def test_prevalence_cdf_fixed_sim3(test_input, simulation3):
    n, t = test_input
    sens, spec, results = simulation3
    if (n, t) in results:
        mean_absolute_error = get_mae_for_test(n, t, sens, spec, results)
        assert mean_absolute_error < 0.05
