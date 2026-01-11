import logging
import numpy as np
import scipy.special as sc
import scipy.stats as stats

from typing import NamedTuple

from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from ._stats import log_betainc_ufunc as log_betainc
from ._stats import prevalence_cdf_fixed_ufunc as prevalence_cdf_fixed
from ._stats import prevalence_cdf_positive_fixed_ufunc as prevalence_cdf_positive_fixed
from ._stats import prevalence_cdf_negative_fixed_ufunc as prevalence_cdf_negative_fixed
from ._stats import inverse_prevalence_cdf_fixed_ufunc as inverse_prevalence_cdf_fixed
from ._stats import (
    inverse_prevalence_cdf_positive_fixed_ufunc as inverse_prevalence_cdf_positive_fixed
)
from ._stats import (
    inverse_prevalence_cdf_positive_fixed_ufunc as inverse_prevalence_cdf_negative_fixed
)


logger = logging.getLogger(__file__)


def _round_interval(left, right, *, digits=6):
    scale = 10**digits
    left, right = np.floor(left * scale) / scale, np.ceil(right * scale) / scale
    return np.clip(left, 0.0, 1.0), np.clip(right, 0.0, 1.0)


def KL_beta(a1, b1, a2, b2):
    """Compute KL Divergence between two beta distributions."""
    output = sc.betaln(a1, b1) - sc.betaln(a2, b2)
    output -= (a2 - a1) * sc.digamma(a1) + (b2 - b1) * sc.digamma(b1)
    output += (a2 - a1 + b2 - b1) * sc.digamma(a1 + b1)
    return output


def log_score_betabinom(k_true, n_true, a_pred, b_pred):
    """Compute log score from betabinomial likelihood."""
    vals = stats.betabinom.logpmf(k_true, n_true, a_pred, b_pred)
    return np.mean(np.where(vals > 0, -np.inf, vals))


def sample_prevalence_posterior(
        n: ArrayLike,
        t: ArrayLike,
        sens_a: float,
        sens_b: float,
        spec_a: float,
        spec_b: float,
        *,
        n_samples: int = 1,
        rng=None,
        condition=None,
):
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    if condition == "positive":
        inv_func = inverse_prevalence_cdf_positive_fixed
    elif condition == "negative":
        inv_func = inverse_prevalence_cdf_negative_fixed
    elif condition is None:
        inv_func = inverse_prevalence_cdf_fixed
    else:
        raise ValueError(
            "condition must be one of None, 'positive', or 'negative'."
            f" received {condition}"
        )

    n, t = np.asarray(n), np.asarray(t)
    data_shape = np.broadcast_shapes(n.shape, t.shape)

    sens = rng.beta(sens_a, sens_b, size=n_samples)
    spec = rng.beta(spec_a, spec_b, size=n_samples)

    U = rng.uniform(0.0, 1.0, size=(n_samples,) + data_shape)

    theta = inv_func(
        U,
        n[np.newaxis, ...],
        t[np.newaxis, ...],
        sens.reshape(sens.shape + (1,) * n.ndim),
        spec.reshape(spec.shape + (1,) * n.ndim),
    )
    return theta[()]


def _hdi_from_sample(theta_samples, *, alpha=0.1):
    theta = np.sort(theta_samples)
    n = len(theta)
    interval_idx_inc = int(np.floor((1 - alpha) * n))
    n_intervals = n - interval_idx_inc

    interval_width = theta[interval_idx_inc:] - theta[:n_intervals]

    min_idx = np.argmin(interval_width)
    hdi_min = theta[min_idx]
    hdi_max = theta[min_idx + interval_idx_inc]
    
    return hdi_min, hdi_max
        

def prevalence_cdf(
        theta: ArrayLike,
        n: int,
        t: int,
        sens_a: float,
        sens_b: float,
        spec_a: float,
        spec_b: float,
        *,
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


def equal_tailed_interval(
        n: int,
        t: int,
        sens_a: float,
        sens_b: float,
        spec_a: float,
        spec_b: float,
        *,
        alpha: float = 0.1,
        n_samples: int = 10000,
        condition: str | None = None,
        rng=None,
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
    n_samples : Optional[int]
        Number of Monte-carlo samples used to estimate distribution.
        Default: 10000
    condition : Optional[str]
        If ``None`` computes eti for prevalence among all cases. If "positive",
        compute eti for prevalence among cases with positive diagnostic
        test result. If "negative", compute eti for prevalence among cases
        with negative diagnostic test result.

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
    sample = sample_prevalence_posterior(
        n, t, sens_a, sens_b, spec_a, spec_b, n_samples=1000, rng=rng,
        condition=condition
    )
    left, right = np.quantile(sample, [alpha/2, 1.0 - alpha/2])
    return _round_interval(left, right)


def highest_density_interval(
        n: int,
        t: int,
        sens_a: float,
        sens_b: float,
        spec_a: float,
        spec_b: float,
        *,
        alpha: float = 0.1,
        n_samples: int =10000,
        condition: str | None = None,
        rng=None,
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
    n_samples : Optional[int]
        Number of Monte-carlo samples used to estimate distribution.
        Default: 10000
    condition : Optional[str]
        If ``None`` computes hdi for prevalence among all cases. If "positive",
        compute hdi for prevalence among cases with positive diagnostic
        test result. If "negative", compute hdi for prevalence among cases
        with negative diagnostic test result.

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
    sample = sample_prevalence_posterior(
        n, t, sens_a, sens_b, spec_a, spec_b, n_samples=1000, rng=rng,
        condition=condition
    )
    return _round_interval(*_hdi_from_sample(sample, alpha=alpha))


class Metrics(NamedTuple):
    precision: float
    recall: float


def sample_estimated_metrics(
        n_pred_pos,
        t_pred_pos_diag_pos,
        n_pred_neg,
        t_pred_neg_diag_pos,
        sens_a,
        sens_b,
        spec_a,
        spec_b,
        *,
        n_samples=1,
        rng=None,
):
    samples = sample_prevalence_posterior(
        [n_pred_pos, n_pred_neg],
        [t_pred_pos_diag_pos, t_pred_neg_diag_pos],
        sens_a,
        sens_b,
        spec_a,
        spec_b,
        n_samples=n_samples,
        rng=rng,
    )
    precision = samples[:, 0]
    false_omission_rate = samples[:, 1]
    recall = precision / (
        precision + false_omission_rate * n_pred_neg / n_pred_pos
    )
    return Metrics(precision, recall)


class HighestDensityRegion2d:
    def __init__(self, samples, x_metric, y_metric, **kde_kwargs):
        self.samples = samples
        self.x_metric = x_metric
        self.y_metric = y_metric
        self.kde = stats.gaussian_kde(samples.T, **kde_kwargs)

        densities = self.kde(self.samples.T)
        self.sorted_densities = np.sort(densities)[::-1]
        self.cumsum = np.cumsum(self.sorted_densities)
        self.cumsum /= self.cumsum[-1]

    def threshold(self, alpha):
        idx = np.searchsorted(self.cumsum, alpha)
        return self.sorted_densities[idx]

    def contains(self, points, *, alpha=0.9):
        points = np.atleast_2d(points)
        densities = self.kde(points.T)
        return densities >= self.threshold(alpha)

    def plot(self, xlims=(0, 1), ylims=(0, 1), *, grid_size=200, alpha=0.9):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        x = np.linspace(xlims[0], xlims[1], grid_size)
        y = np.linspace(ylims[0], ylims[1], grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        Z = self.kde(grid_points.T).reshape(X.shape)

        thresh = self.threshold(alpha)

        fig, ax = plt.subplots(figsize=(6, 5))

        heatmap = ax.contourf(X, Y, Z, levels=100, cmap='viridis')
        fig.colorbar(heatmap, ax=ax, label='Density')

        ax.contour(X, Y, Z, levels=[thresh], linestyles="--", colors='red', linewidths=2)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel(self.x_metric)
        ax.set_ylabel(self.y_metric)
        ax.set_title(f'2D Density with {int(alpha*100)}% HDR')

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

        ax.set_axisbelow(False)

        return fig, ax
