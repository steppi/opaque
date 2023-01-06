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


def binomial_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    assert y_true.shape[1] == 2
    # Require all samples to have at least one trial
    assert np.all((y_true[:, 0] > 0) & (y_pred[:, 0] > 0))
    N_true, K_true = y_true[:, 0], y_true[:, 1]
    N_pred, K_pred = y_pred[:, 0], y_pred[:, 1]
    # Renormalize predictions to have same number of trials as true.
    # This makes it convenient to use in GridSearchCV, where it's not easy
    # to pass the number of trials to predict.
    if not np.all(N_true == N_pred):
        K_pred = K_pred * N_true / N_pred
    p_hat_true = (K_true + 1) / (N_true + 2)
    var = N_true * p_hat_true * (1 - p_hat_true)
    residuals = (K_true - K_pred)**2 / var
    return np.sum(residuals) / len(residuals)
