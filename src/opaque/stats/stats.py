import logging
import numpy as np
import scipy.special as sc

from typing import Any
from numpy.typing import ArrayLike
from scipy.special import betaln, digamma
from sklearn.utils.validation import column_or_1d
from statsmodels.stats.proportion import proportion_confint


logger = logging.getLogger(__file__)


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
        n, t, sens, spec, alpha=0.1, method="beta"
):
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
