import pytest
from opaque.simulations.end_to_end import EndtoEndSimulator


@pytest.mark.parametrize('test_input',
                         [([0.8, 0.2, 0.5, 0.2],
                           [5, 0.3, 0.1, 0.0],
                           [1.2, 0.3, 0.2, 0.1],
                           [3, 0.6, 0.2, 0.1],
                           0.0, 0.0, 0.1, 0.1),
                          ([0.8, 0.2, 0.5, 0.2],
                           [5, 0.3, 0.1, 0.0],
                           [1.2, 0.3, 0.2, 0.1],
                           [3, 0.6, 0.2, 0.1],
                           0.2, 0.2, 0.5, 0.5)])
def test_end_to_end(test_input):
    sim = EndtoEndSimulator(*test_input[0:4],
                            sens_noise_mean=test_input[4],
                            spec_noise_mean=test_input[5],
                            sens_noise_disp=test_input[6],
                            spec_noise_disp=test_input[7])
    results = sim.run()
    hits = len(results[(results.left <= results.theta) &
                       (results.theta <= results.right)])
    assert hits/len(results) > 0.8


