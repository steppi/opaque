import logging
import numpy as np
from typing import Any
from numpy.typing import ArrayLike
from scipy.special import betaln, digamma
from sklearn.utils.validation import column_or_1d


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
    output = betaln(a1, b1) - betaln(a2, b2)
    output -= (a2 - a1) * digamma(a1) + (b2 - b1) * digamma(b1)
    output += (a2 - a1 + b2 - b1) * digamma(a1 + b1)
    return output
