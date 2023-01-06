"""These tests compare the closed form expression for the Prevalence
CDF with and empirical CDF computed through simulation.
"""
import os
import json
import pytest
import numpy as np

from opaque.stats import prevalence_cdf

here = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_LOCATION = os.path.join(here, 'data')


def get_simulation_results(test_data_filename):
    with open(os.path.join(TEST_DATA_LOCATION, test_data_filename)) as f:
        sim_results = json.load(f)
    return sim_results


@pytest.fixture
def simulation0():
    return get_simulation_results("prevalence_cdf_simulation0.json")


@pytest.fixture
def simulation1():
    return get_simulation_results("prevalence_cdf_simulation1.json")


@pytest.fixture
def simulation2():
    return get_simulation_results("prevalence_cdf_simulation2.json")


class TestPrevalenceCdf(object):
    def calculate_cdf(
            self, n, t, sens_a, sens_b, spec_a, spec_b, num, mode
    ):
        return np.fromiter(
            (
                prevalence_cdf(
                    theta, n, t, sens_a, sens_b, spec_a, spec_b, mode=mode
                )
                for theta in np.linspace(0, 1, num)
            ),
            dtype=float,
        )

    def get_mae_for_testcase(
            self,
            n,
            t,
            sens_a,
            sens_b,
            spec_a,
            spec_b,
            num_grid_points,
            results,
            mode,
    ):
        simulated_cdf = np.array(results[f"{n}:{t}"])
        calculated_cdf = self.calculate_cdf(
            n, t, sens_a, sens_b, spec_a, spec_b, num_grid_points, mode
        )
        residuals = np.abs(simulated_cdf - calculated_cdf)
        mean_absolute_error = np.sum(residuals) / num_grid_points
        max_absolute_error = np.max(residuals)
        return mean_absolute_error, max_absolute_error

    @pytest.mark.parametrize("test_input", [(100, t) for t in range(20, 81)])
    @pytest.mark.parametrize("mode", ["unconditional", "positive", "negative"])
    def test_prevalence_cdf_sim0(self, test_input, mode, simulation0):
        info = simulation0["info"]
        if mode == "unconditional":
            results = simulation0["results"]
        elif mode == "positive":
            results = simulation0["results_pos"]
        elif mode == "negative":
            results = simulation0["results_neg"]

        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens_a, sens_b = info["sens_prior"]
        spec_a, spec_b = info["spec_prior"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n,
            t,
            sens_a,
            sens_b,
            spec_a,
            spec_b,
            num_grid_points,
            results,
            mode,
        )
        assert mae < 0.06

    @pytest.mark.parametrize(
        "test_input", [(1000, t) for t in range(300, 701)]
    )
    @pytest.mark.parametrize("mode", ["unconditional", "positive", "negative"])
    def test_prevalence_cdf_sim1(self, test_input, mode, simulation1):
        info = simulation1["info"]
        if mode == "unconditional":
            results = simulation1["results"]
        elif mode == "positive":
            results = simulation1["results_pos"]
        elif mode == "negative":
            results = simulation1["results_neg"]

        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens_a, sens_b = info["sens_prior"]
        spec_a, spec_b = info["spec_prior"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n,
            t,
            sens_a,
            sens_b,
            spec_a,
            spec_b,
            num_grid_points,
            results,
            mode,
        )
        if mode == "unconditional":
            assert mae < 0.06
        else:
            # Looser conditions for conditional prevalence reflect inaccuracies
            # in the simulation, not the method.
            assert mae < 0.09

    @pytest.mark.parametrize("test_input", [(50, t) for t in range(10, 41)])
    def test_prevalence_cdf_sim2(self, test_input, simulation2):
        info = simulation2["info"]
        results = simulation2["results"]

        n, t = test_input
        num_grid_points = info["num_grid_points"]
        sens_a, sens_b = info["sens_prior"]
        spec_a, spec_b = info["spec_prior"]
        if f"{n}:{t}" not in results:
            return
        mae, mxae = self.get_mae_for_testcase(
            n,
            t,
            sens_a,
            sens_b,
            spec_a,
            spec_b,
            num_grid_points,
            results,
            "unconditional",
        )
        assert mae < 0.06
