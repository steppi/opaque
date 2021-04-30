import pytest
import numpy as np
import mpmath as mp
from itertools import product

from opaque.stats import log_betainc


def mpmath_log_betainc(p, q, x):
    """mpmath referenec implementation for log_betainc."""
    return mp.log(mp.betainc(p, q, 0, x, regularized=True))


@pytest.mark.parametrize('test_input',
                          [(p, q, x)
                           for p, q in product([10, 50, 100, 200, 500, 1000],
                                               repeat=2)
                           for x in [0.1, 0.2, 0.4, 0.8]])
def test_log_betainc(test_input):
    tol=1e-12
    p, q, x = test_input
    expected = float(mpmath_log_betainc(p, q, x))
    observed = log_betainc(p, q, x)
    assert abs(expected - observed) < tol
