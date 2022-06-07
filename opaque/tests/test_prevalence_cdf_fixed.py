"""This tests compare the closed form expression for the Prevalence
CDF when the diagnostic test has fixed sensitivity and specificity
with empirical distributions based on simulations. Ensures mean absolute
error between calculated CDF and empirical CDF remains within tolerance
tol = 0.05.
"""
import os
import json
import pytest
import numpy as np

from opaque.stats import prevalence_cdf_fixed
from opaque.locations import TEST_DATA_LOCATION


def get_simulation_results(test_data_filename):
    with open(os.path.join(TEST_DATA_LOCATION, test_data_filename)) as f:
        sim_results = json.load(f)
    results, info = sim_results["results"], sim_results["info"]
    return results, info


@pytest.fixture
def simulation0():
    return get_simulation_results("prevalence_cdf_simulation_fixed0.json")


@pytest.fixture
def simulation1():
    return get_simulation_results("prevalence_cdf_simulation_fixed1.json")


@pytest.fixture
def simulation2():
    return get_simulation_results("prevalence_cdf_simulation_fixed2.json")


class TestPrevalenceCdfFixed:
    def calculate_cdf(self, n, t, sens, spec, num):
        return np.fromiter(
            (
                prevalence_cdf_fixed(theta, n, t, sens, spec)
                for theta in np.linspace(0, 1, num)
            ),
            dtype=float,
        )

    def get_mae_for_testcase(self, n, t, sens, spec, num_grid_points, results):
        simulated_cdf = np.array(results[f"{n}:{t}"])
        calculated_cdf = self.calculate_cdf(n, t, sens, spec, num_grid_points)
        residuals = np.abs(simulated_cdf - calculated_cdf)
        mean_absolute_error = np.sum(residuals) / num_grid_points
        max_absolute_error = np.max(residuals)
        return mean_absolute_error, max_absolute_error

    @pytest.mark.parametrize("test_input", [(20, t) for t in range(5, 16)])
    def test_prevalence_cdf_fixed_sim0(self, test_input, simulation0):
        results, info = simulation0
        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens, spec = info["sensitivity"], info["specificity"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n, t, sens, spec, num_grid_points, results
        )
        assert mae < 0.01
        assert mxae < 0.1

    @pytest.mark.parametrize("test_input", [(100, t) for t in range(20, 81)])
    def test_prevalence_cdf_fixed_sim1(self, test_input, simulation1):
        results, info = simulation1
        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens, spec = info["sensitivity"], info["specificity"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n, t, sens, spec, num_grid_points, results
        )
        assert mae < 0.01
        assert mxae < 0.1

    @pytest.mark.parametrize("test_input", [(100, t) for t in range(200, 801)])
    def test_prevalence_cdf_fixed_sim2(self, test_input, simulation2):
        results, info = simulation2
        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens, spec = info["sensitivity"], info["specificity"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n, t, sens, spec, num_grid_points, results
        )
        assert mae < 0.01
        assert mxae < 0.1
