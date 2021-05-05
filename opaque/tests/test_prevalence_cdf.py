"""These tests compare the closed form expression for the Prevalence
CDF with and empirical CDF computed through simulation.
"""
import os
import json
import pytest
import numpy as np

from opaque.stats import prevalence_cdf
from opaque.locations import TEST_DATA_LOCATION


def get_simulation_results(test_data_filename):
    with open(os.path.join(TEST_DATA_LOCATION,
                           test_data_filename)) as f:
        sim_results = json.load(f)
    results, info = sim_results['results'], sim_results['info']
    return results, info


@pytest.fixture
def simulation0():
    return get_simulation_results('prevalence_cdf_simulation0.json')


@pytest.fixture
def simulation1():
    return get_simulation_results('prevalence_cdf_simulation1.json')


@pytest.fixture
def simulation2():
    return get_simulation_results('prevalence_cdf_simulation2.json')


class TestPrevalenceCdfFixed(object):
    def get_simulation_results(self, test_data_filename):
        with open(os.path.join(TEST_DATA_LOCATION,
                               'prevalence_cdf_simulation_fixed0.json')) as f:
            sim_results = json.load(f)
        results, info = sim_results['results'], sim_results['info']
        return results, info

    def calculate_cdf(self, n, t, sens_a, sens_b,
                      spec_a, spec_b, num):
        return np.fromiter((prevalence_cdf(theta, n, t, sens_a,
                                           sens_b, spec_a, spec_b)
                            for theta in np.linspace(0, 1, num)), dtype=float)

    def get_mae_for_testcase(self, n, t, sens_a, sens_b,
                             spec_a, spec_b,
                             num_grid_points, results):
        simulated_cdf = np.array(results[f'{n}:{t}'])
        calculated_cdf = self.\
            calculate_cdf(n, t, sens_a, sens_b, spec_a, spec_b,
                          num_grid_points)
        residuals = np.abs(simulated_cdf - calculated_cdf)
        mean_absolute_error = np.sum(residuals)/num_grid_points
        max_absolute_error = np.max(residuals)
        return mean_absolute_error, max_absolute_error

    @pytest.mark.parametrize('test_input',
                             [(100, t) for t in range(20, 81)])
    def test_prevalence_cdf_fixed_sim0(self, test_input, simulation0):
        results, info = simulation0
        n, t = test_input
        num_grid_points = info['num_grid_points']
        sens_a, sens_b = info['sens_prior']
        spec_a, spec_b = info['spec_prior']
        if f'{n}:{t}' not in results:
            return
        mae, mxae = self.get_mae_for_testcase(n, t, sens_a, sens_b,
                                              spec_a, spec_b,
                                              num_grid_points, results)
        assert mae < 0.05

    @pytest.mark.parametrize('test_input',
                             [(1000, t) for t in range(300, 701)])
    def test_prevalence_cdf_fixed_sim1(self, test_input, simulation1):
        results, info = simulation1
        n, t = test_input
        num_grid_points = info['num_grid_points']
        sens_a, sens_b = info['sens_prior']
        spec_a, spec_b = info['spec_prior']
        if f'{n}:{t}' not in results:
            return
        mae, mxae = self.get_mae_for_testcase(n, t, sens_a, sens_b,
                                              spec_a, spec_b,
                                              num_grid_points, results)
        assert mae < 0.05
        
    @pytest.mark.parametrize('test_input',
                             [(50, t) for t in range(10, 41)])
    def test_prevalence_cdf_fixed_sim2(self, test_input, simulation2):
        results, info = simulation2
        n, t = test_input
        num_grid_points = info['num_grid_points']
        sens_a, sens_b = info['sens_prior']
        spec_a, spec_b = info['spec_prior']
        if f'{n}:{t}' not in results:
            return
        mae, mxae = self.get_mae_for_testcase(n, t, sens_a, sens_b,
                                              spec_a, spec_b,
                                              num_grid_points, results)
        assert mae < 0.05
