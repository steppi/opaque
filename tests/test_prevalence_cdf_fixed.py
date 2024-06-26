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
from opaque.stats import prevalence_cdf_negative_fixed
from opaque.stats import prevalence_cdf_positive_fixed

here = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_LOCATION = os.path.join(here, 'data')


def get_simulation_results(test_data_filename):
    with open(os.path.join(TEST_DATA_LOCATION, test_data_filename)) as f:
        sim_results = json.load(f)
    return sim_results


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
    def get_mae_for_testcase(
            self, n, t, sens, spec, num_grid_points, results, mode
    ):
        if mode == "unconditional":
            pfunc = prevalence_cdf_fixed
        elif mode == "negative":
            pfunc = prevalence_cdf_negative_fixed
        elif mode == "positive":
            pfunc = prevalence_cdf_positive_fixed
        else:
            raise ValueError(f"Invalid mode {mode}")
        simulated_cdf = np.array(results[f"{n}:{t}"])
        theta = np.linspace(0, 1, num_grid_points)
        calculated_cdf = pfunc(theta, n, t, sens, spec)
        residuals = np.abs(simulated_cdf - calculated_cdf)
        mean_absolute_error = np.sum(residuals) / num_grid_points
        max_absolute_error = np.max(residuals)
        return mean_absolute_error, max_absolute_error

    @pytest.mark.parametrize("test_input", [(20, t) for t in range(5, 16)])
    @pytest.mark.parametrize(
        "mode", [
            "unconditional",
            pytest.param(
                "positive", marks=pytest.mark.xfail(
                    reason="Simulation results inaccurate."
                ),
            ),
            pytest.param(
                "negative", marks=pytest.mark.xfail(
                    reason="Simulation results inaccurate."
                ),
            ),
        ]
    )
    def test_prevalence_cdf_fixed_sim0(self, test_input, mode, simulation0):
        info = simulation0["info"]
        if mode == "unconditional":
            results = simulation0["results"]
        elif mode == "positive":
            results = simulation0["results_pos"]
        elif mode == "negative":
            results = simulation0["results_neg"]

        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens, spec = info["sensitivity"], info["specificity"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n, t, sens, spec, num_grid_points, results, mode
        )
        if mode == "unconditional":
            assert mae < 0.01
            assert mxae < 0.1
        else:
            # Looser conditions for conditional prevalence reflect inaccuracies
            # in the simulation, not the method.
            assert mae < 0.025

    @pytest.mark.parametrize("test_input", [(100, t) for t in range(20, 81)])
    @pytest.mark.parametrize("mode", ["unconditional", "positive", "negative"])
    def test_prevalence_cdf_fixed_sim1(self, test_input, mode, simulation1):
        info = simulation1["info"]
        if mode == "unconditional":
            results = simulation1["results"]
        elif mode == "positive":
            results = simulation1["results_pos"]
        elif mode == "negative":
            results = simulation1["results_neg"]

        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens, spec = info["sensitivity"], info["specificity"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n, t, sens, spec, num_grid_points, results, mode
        )
        if mode == "unconditional":
            assert mae < 0.01
            assert mxae < 0.1
        else:
            # Looser conditions for conditional prevalence reflect inaccuracies
            # in the simulation, not the method.
            assert mae < 0.03

    @pytest.mark.parametrize("test_input", [(100, t) for t in range(200, 801)])
    @pytest.mark.parametrize("mode", ["unconditional", "positive", "negative"])
    def test_prevalence_cdf_fixed_sim2(self, test_input, mode, simulation2):
        info = simulation2["info"]
        if mode == "unconditional":
            results = simulation2["results"]
        elif mode == "positive":
            results = simulation2["results_pos"]
        elif mode == "negative":
            results = simulation2["results_neg"]

        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens, spec = info["sensitivity"], info["specificity"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n, t, sens, spec, num_grid_points, results, mode
        )
        if mode == "unconditional":
            assert mae < 0.01
            assert mxae < 0.1
        else:
            # Looser conditions for conditional prevalence reflect inaccuracies
            # in the simulation, not the method.
            assert mae < 0.03
