import ctypes
import logging
import numpy as np
import scipy.special as sc


from numba import int64
from numba import njit
from numba import float64
from numba import vectorize
from numba.extending import get_cython_function_address
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
from scipy.stats import qmc
from sklearn.utils.validation import column_or_1d
from statsmodels.stats.proportion import proportion_confint
from typing import Any


logger = logging.getLogger(__file__)


def _round_interval(left, right, digits=6):
    scale = 10**digits
    return (np.floor(left * scale) / scale, np.ceil(right * scale) / scale)


def true_positives(
        y_true: ArrayLike, y_pred: ArrayLike, pos_label: Any = 1
) -> float:
    y_true, y_pred = column_or_1d(y_true), column_or_1d(y_pred)
    return np.sum((y_true == pos_label) & (y_pred == pos_label))


def true_negatives(
        y_true: ArrayLike, y_pred: ArrayLike, pos_label: Any = 1
) -> float:
    y_true, y_pred = column_or_1d(y_true), column_or_1d(y_pred)
    return np.sum((y_true != pos_label) & (y_pred != pos_label))


def false_positives(
        y_true: ArrayLike, y_pred: ArrayLike, pos_label: Any = 1
) -> float:
    y_true, y_pred = column_or_1d(y_true), column_or_1d(y_pred)
    return np.sum((y_true != pos_label) & (y_pred == pos_label))


def false_negatives(
        y_true: ArrayLike, y_pred: ArrayLike, pos_label: Any = 1
) -> float:
    y_true, y_pred = column_or_1d(y_true), column_or_1d(y_pred)
    return np.sum((y_true == pos_label) & (y_pred != pos_label))


def sensitivity_score(
    y_true: ArrayLike, y_pred: ArrayLike, pos_label: Any = 1
) -> float:
    tp = true_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)
    try:
        result = tp / (tp + fn)
    except ZeroDivisionError:
        logger.warning("No positive examples in sample. Returning 0.0")
        result = 0.0
    return result


def specificity_score(
    y_true: ArrayLike, y_pred: ArrayLike, pos_label: Any = 1
) -> float:
    tn = true_negatives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    try:
        result = tn / (tn + fp)
    except ZeroDivisionError:
        logger.warning("No negative examples in sample. Returning 0.0")
        result = 0.0
    return result


def youdens_j_score(
        y_true: ArrayLike, y_pred: ArrayLike, pos_label: Any = 1
) -> float:
    sens = sensitivity_score(y_true, y_pred, pos_label)
    spec = specificity_score(y_true, y_pred, pos_label)
    return sens + spec - 1


def KL_beta(a1, b1, a2, b2):
    """Compute KL Divergence between two beta distributions."""
    output = sc.betaln(a1, b1) - sc.betaln(a2, b2)
    output -= (a2 - a1) * sc.digamma(a1) + (b2 - b1) * sc.digamma(b1)
    output += (a2 - a1 + b2 - b1) * sc.digamma(a1 + b1)
    return output


def NKLD(p_true, p_pred):
    """Compute Normalized Kullback-Leibler Divergence Metric."""
    KL = (
        p_true * np.log(p_true / p_pred)
        + (1 - p_true) * np.log((1 - p_true) / (1 - p_pred))
    )
    return np.mean(2 * sc.expit(KL) - 1)


def simple_prevalence_interval(
        n: int,
        t: int,
        sens: float,
        spec: float,
        alpha: float = 0.1,
        method: str = "beta",
) -> tuple[float,float]:
    """Compute simple prevalence interval based on linear transform.

    phi = sens * theta + (1 - spec) * (1 - theta)

    Where phi is proportion of positive tests and theta is unknown
    prevalence value.
    """
    a, b = proportion_confint(t, n, alpha=alpha, method=method)
    J = sens + spec - 1
    fnr = 1 - spec
    c = min(max(0, (a - fnr) / J), 1)
    d = max(min(1, (b - fnr) / J), 0)
    return c, d


def _make_log_betainc_ufunc():
    addr = get_cython_function_address("opaque.stats._stats", "log_betainc")
    functype = ctypes.CFUNCTYPE(
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    )
    func_for_numba = functype(addr)
    @vectorize([float64(float64, float64, float64)])
    def ufunc(p, q, x):
        return func_for_numba(p, q, x)
    return ufunc

log_betainc = _make_log_betainc_ufunc()

def _make_prevalence_ufunc(function_name):
    addr = get_cython_function_address("opaque.stats._stats", function_name)
    functype = ctypes.CFUNCTYPE(
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    )
    func_for_numba = functype(addr)
    @vectorize([float64(float64, int64, int64, float64, float64)])
    def ufunc(theta, n, t, sensitivity, specificity):
        return func_for_numba(theta, n, t, sensitivity, specificity)
    return ufunc


prevalence_cdf_fixed = _make_prevalence_ufunc("prevalence_cdf_fixed")
prevalence_cdf_positive_fixed = _make_prevalence_ufunc("prevalence_cdf_positive_fixed")
prevalence_cdf_negative_fixed = _make_prevalence_ufunc("prevalence_cdf_negative_fixed")


def prevalence_cdf(
        theta: ArrayLike,
        n: int,
        t: int,
        sens_a: float,
        sens_b: float,
        spec_a: float,
        spec_b: float,
        log2_num_qmc_points: int = 10,
        mode: str = "unconditional",
) -> NDArray:
    """Returns prevalence_cdf as derived in Diggle, 2011 [0].

    Parameters
    ----------
    theta : Arraylike of float
        Value of prevalence at which to calculate cdf.
    n : int
        Number of samples on which diagnostic test has been run.
    t : int
        Number of positives out of all samples.
    sens_a : float
        First shape parameter of beta prior for sensitivity.
    sens_b : float
        Second shape parameter of beta prior for sensitivity.
    spec_a : float
        First shape parameter of beta prior for specificity.
    spec_b : float
        Second shape parameter of beta prior for specificity.
    log2_num_qmc_points : Optional[int]
       Use 2**log2_num_qmc_points sample points in Sobol sequence.
       Sobol sequences require the number of sample points to be a
       power of 2. Controls accuracy at expense of compute time.
       Default = 10
    mode : Optional[str]
        If "unconditional" standard prevalence cdf. If "positive",
        prevalence cdf conditioned on positive diagnostic test result.
        If "negative", prevalence cdf conditioned on negative test
        result. Default "unconditional".

    Returns
    -------
    float
        Value of cdf at theta for given parameters.

    References
    ----------
    [0] Peter J. Diggle, "Estimating Prevalence Using an Imperfect Test",
        Epidemiology Research International, vol. 2011, Article ID 608719,
        5 pages, 2011. https://doi.org/10.1155/2011/608719
    """
    if mode == "unconditional":
        pfunc = prevalence_cdf_fixed
    elif mode == "positive":
        pfunc = prevalence_cdf_positive_fixed
    elif mode == "negative":
        pfunc = prevalence_cdf_negative_fixed
    else:
        raise ValueError(
            'mode should be one of "unconditional", "positive", "negative", '
            f'got "{mode}"'
        )
    theta = np.asarray(theta)
    sampler = qmc.Sobol(d=2, scramble=False)
    sample= sampler.random_base2(m=log2_num_qmc_points)
    sens_sample = sc.betaincinv(sens_a, sens_b, sample[:, 0])
    spec_sample = sc.betaincinv(spec_a, spec_b, sample[:, 1])
    return pfunc(
        theta[..., np.newaxis],
        n,
        t,
        sens_sample[np.newaxis, :],
        spec_sample[np.newaxis, :],
    ).mean(axis=-1)


def inverse_prevalence_cdf(
        x: float,
        n: int,
        t: int,
        sens_a: float,
        sens_b: float,
        spec_a: float,
        spec_b: float,
        log2_num_qmc_points: int = 10,
        mode: str = "unconditional",
) -> float:
    """Returns inverse of prevalence cdf evaluated at x for param values

    As derived in Diggle 2011 [0].
    Uses root finding algorithm to calculate inverse cdf by solving
    prevalence_cdf(theta, ...) = x for theta.

    Parameters
    ----------
    x : float
        Probability value at which to calculate inverse cdf.
    n : int
        Number of samples on which diagnostic test has been run.
    t : int
        Number of positives out of all samples.
    sens_a : float
        First shape parameter of beta prior for sensitivity.
    sens_b : float
        Second shape parameter of beta prior for sensitivity.
    spec_a : float
        First shape parameter of beta prior for specificity.
    spec_b : float
        Second shape parameter of beta prior for specificity.
    log2_num_qmc_points : Optional[int]
       Use 2**log2_num_qmc_points sample points in Sobol sequence.
       Sobol sequences require the number of sample points to be a
       power of 2. Controls accuracy at expense of compute time.
       Default = 10
    mode : Optional[str]
        If "unconditional" standard prevalence cdf. If "positive",
        prevalence cdf conditioned on positive diagnostic test result.
        If "negative", prevalence cdf conditioned on negative test
        result. Default "unconditional".

    Returns
    -------
    float
        Value of inverse cdf at x for given parameters.

    References
    ----------
    [0] Peter J. Diggle, "Estimating Prevalence Using an Imperfect Test",
        Epidemiology Research International, vol. 2011, Article ID 608719,
        5 pages, 2011. https://doi.org/10.1155/2011/608719
    """
    def f(theta):
        return prevalence_cdf(
            theta,
            n,
            t,
            sens_a,
            sens_b,
            spec_a,
            spec_b,
            log2_num_qmc_points=log2_num_qmc_points,
            mode=mode
        ) - x

    if x == 0 or x == 1:
        return x

    return root_scalar(
        f,
        method="brentq",
        bracket = [np.nextafter(0, -1), np.nextafter(1, 2)],
        xtol=np.nextafter(0, 1),
    ).root 


def equal_tailed_interval(
        n: int,
        t: int,
        sens_a: float,
        sens_b: float,
        spec_a: float,
        spec_b: float,
        alpha: float = 0.1,
        log2_num_qmc_points: int = 10,
        mode: str = "unconditional",
) -> tuple[float, float]:
    """Returns equal tailed prevalence credible interval [1].

    Interval of posterior distribution (left, right) such that
    the left and right tails [0, left] and [right, 1] each capture
    probability 1 - alpha/2.

    Parameters
    ----------
    n : int
        Number of samples on which diagnostic test has been run.
    t : int
        Number of positives out of all samples.
    sens_a : float
        First shape parameter of beta prior for sensitivity.
    sens_b : float
        Second shape parameter of beta prior for sensitivity.
    spec_a : float
        First shape parameter of beta prior for specificity.
    spec_b : float
        Second shape parameter of beta prior for specificity.
    alpha : float
        Significance level. Interval of posterior accounts for
        probability 1 - alpha.
    log2_num_qmc_points : Optional[int]
       Use 2**log2_num_qmc_points sample points in Sobol sequence.
       Sobol sequences require the number of sample points to be a
       power of 2. Controls accuracy at expense of compute time.
       Default = 10
    mode : Optional[str]
        If "unconditional" standard prevalence cdf. If "positive",
        prevalence cdf conditioned on positive diagnostic test result.
        If "negative", prevalence cdf conditioned on negative test
        result. Default "unconditional".

    Returns
    -------
    tuple[float, float]
        tuple(left, right) where left, and right are the endpoints of the
        prevalence credible interval.

    References
    ----------
    [0] Peter J. Diggle, "Estimating Prevalence Using an Imperfect Test",
        Epidemiology Research International, vol. 2011, Article ID 608719,
        5 pages, 2011. https://doi.org/10.1155/2011/608719
    [1] https://en.wikipedia.org/wiki/Credible_interval
    """
    left, right = (
        inverse_prevalence_cdf(
            alpha/2,
            n,
            t,
            sens_a,
            sens_b,
            spec_a,
            spec_b,
            log2_num_qmc_points=log2_num_qmc_points,
            mode=mode
        ),
        inverse_prevalence_cdf(
            1 - alpha/2,
            n,
            t,
            sens_a,
            sens_b,
            spec_a,
            spec_b,
            log2_num_qmc_points=log2_num_qmc_points,
            mode=mode
        ),
    )
    return _round_interval(left, right)


def highest_density_interval(
        n: int,
        t: int,
        sens_a: float,
        sens_b: float,
        spec_a: float,
        spec_b: float,
        alpha: float = 0.1,
        log2_num_qmc_points: int = 10,
        mode: str = "unconditional",
) -> tuple[float, float]:
    """Returns highest density prevalence credible interval [1].

    Interval of posterior distribution of minimal width that captures
    probability 1 - alpha.

    Parameters
    ----------
    n : int
        Number of samples on which diagnostic test has been run.
    t : int
        Number of positives out of all samples.
    sens_a : float
        First shape parameter of beta prior for sensitivity.
    sens_b : float
        Second shape parameter of beta prior for sensitivity.
    spec_a : float
        First shape parameter of beta prior for specificity.
    spec_b : float
        Second shape parameter of beta prior for specificity.
    alpha : float
        Significance level. Interval of posterior accounts for
        probability 1 - alpha.
    log2_num_qmc_points : Optional[int]
       Use 2**log2_num_qmc_points sample points in Sobol sequence.
       Sobol sequences require the number of sample points to be a
       power of 2. Controls accuracy at expense of compute time.
       Default = 10
    mode : Optional[str]
        If "unconditional" standard prevalence cdf. If "positive",
        prevalence cdf conditioned on positive diagnostic test result.
        If "negative", prevalence cdf conditioned on negative test
        result. Default "unconditional".

    Returns
    -------
    tuple[float, float]
        tuple(left, right) where left, and right are the endpoints of the
        prevalence credible interval.

    References
    ----------
    [0] Peter J. Diggle, "Estimating Prevalence Using an Imperfect Test",
        Epidemiology Research International, vol. 2011, Article ID 608719,
        5 pages, 2011. https://doi.org/10.1155/2011/608719
    [1] https://en.wikipedia.org/wiki/Credible_interval
    """
    def f(x):
        A = inverse_prevalence_cdf(
                x + 1 - alpha, n, t, sens_a, sens_b, spec_a, spec_b,
                log2_num_qmc_points=log2_num_qmc_points, mode=mode
        )
        B = inverse_prevalence_cdf(
            x, n, t, sens_a, sens_b, spec_a, spec_b,
            log2_num_qmc_points=log2_num_qmc_points, mode=mode
        )
        return A - B

    res = minimize_scalar(f, method="bounded", bounds=[0, alpha])
    right = inverse_prevalence_cdf(
        res.x + 1 - alpha, n, t, sens_a, sens_b, spec_a, spec_b,
        log2_num_qmc_points=log2_num_qmc_points, mode=mode
    )
    left = right - res.fun
    return _round_interval(left, right)
