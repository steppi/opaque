import mpmath as mp
import numpy as np
import pytest

from itertools import product

from opaque.stats import log_betainc


def mpmath_log_betainc(p, q, x):
    """mpmath reference implementation for log_betainc."""
    if x < p / (p + q):
        return mp.log(mp.betainc(p, q, 0, x, regularized=True))
    else:
        return mp.log1p(-mp.betainc(q, p, 0, 1 - x, regularized=True))


@pytest.mark.parametrize("p", [10, 50, 100, 200, 500, 1000])
@pytest.mark.parametrize("q", [10, 50, 100, 200, 500, 1000])
def test_log_betainc(p, q):
    X = np.linspace(0, 1, 100)
    expected = np.fromiter(
        (mpmath_log_betainc(p, q, x) for x in X),
        dtype=float
    )
    observed = log_betainc(p, q, X)
    M = max(p, q)
    if M <= 100:
        rtol = 1e-13
    else:
        rtol = 5e-11
    np.testing.assert_allclose(observed, expected, rtol=rtol, atol=1e-20)

