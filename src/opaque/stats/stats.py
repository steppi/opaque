import logging
import numpy as np
import scipy.special as sc
import scipy.stats as stats

from typing import NamedTuple

from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from opaque.utils import load_array, serialize_array

from ._stats import log_betainc_ufunc as log_betainc
from ._stats import prevalence_cdf_fixed_ufunc as prevalence_cdf_fixed
from ._stats import inverse_prevalence_cdf_fixed_ufunc as inverse_prevalence_cdf_fixed



logger = logging.getLogger(__file__)


def _round_interval(left, right, *, digits=6):
    scale = 10**digits
    left, right = np.floor(left * scale) / scale, np.ceil(right * scale) / scale
    return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))


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

    if condition is None:
        condition = 0
    elif condition == "positive":
        condition = 1
    elif condition == "negative":
        condition = -1

    n, t = np.asarray(n), np.asarray(t)
    data_shape = np.broadcast_shapes(n.shape, t.shape)

    sens = rng.beta(sens_a, sens_b, size=n_samples)
    spec = rng.beta(spec_a, spec_b, size=n_samples)

    U = rng.uniform(0.0, 1.0, size=(n_samples,) + data_shape)

    theta = inverse_prevalence_cdf_fixed(
        U,
        n[np.newaxis, ...],
        t[np.newaxis, ...],
        sens.reshape(sens.shape + (1,) * n.ndim),
        spec.reshape(spec.shape + (1,) * n.ndim),
        condition,
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
        cond = 0
    elif mode == "positive":
        cond = 1
    elif mode == "negative":
        cond = -1
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
    return prevalence_cdf_fixed(
        theta[..., np.newaxis],
        n,
        t,
        sens_sample[np.newaxis, :],
        spec_sample[np.newaxis, :],
        cond,
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
    return _round_interval(left, right, digits=2)


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
    return _round_interval(*_hdi_from_sample(sample, alpha=alpha), digits=2)


class Metrics(NamedTuple):
    precision: float
    recall: float


def sample_estimated_metrics(
        n,
        t,
        m,
        u,
        sens_a,
        sens_b,
        spec_a,
        spec_b,
        *,
        n_samples=1,
        rng=None,
):
    samples = sample_prevalence_posterior(
        [n, n, n, m],
        [t, t, t, u],
        sens_a,
        sens_b,
        spec_a,
        spec_b,
        n_samples=n_samples,
        condition=[0, -1, 1, 0],
        rng=rng,
    )
    precision = 1.0 - samples[:, 0]
    false_omission_rate = 1.0 - samples[:, 3]
    recall = precision / (
        precision + false_omission_rate * m / n
    )
    recall = np.where(np.isnan(recall), 0.0, recall)

    precision_requiring_consensus = 1.0 - samples[:, 1]
    s = m + n - t
    false_omission_rate_requiring_consensus = (
        s - samples[:, 3] * m - samples[:, 2] * (n - t)
    ) / s
    recall_requiring_consensus = precision_requiring_consensus / (
        precision_requiring_consensus + false_omission_rate_requiring_consensus * s / n
    )
    return {
        "standard": Metrics(precision, recall),
        "consensus": Metrics(
            precision_requiring_consensus, recall_requiring_consensus
        )
    }


class HighestDensityRegion2d:
    def __init__(self, x_metric, y_metric, *, grid_size=200, **kde_kwargs):
        self.x_metric = x_metric
        self.y_metric = y_metric
        self.grid_size = grid_size
        self.kde_kwargs = kde_kwargs
        self._fitted = False
        self.X = None
        self.Y = None
        self.Z = None
        self.kde = None
        self.sorted_densities = None
        self.sorted_mass_cumsum = None

    def fit(self, samples):
        self.kde = stats.gaussian_kde(samples.T, **self.kde_kwargs)
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        densities = self.kde(grid_points.T)
        cell_area = 1.0 / (self.grid_size - 1)**2
        sorted_indices = np.argsort(densities)[::-1]

        self.X = X
        self.Y = Y
        self.Z = densities.reshape(X.shape)
        self.sorted_densities = densities[sorted_indices]
        sorted_mass_cumsum = np.cumsum(self.sorted_densities * cell_area)
        sorted_mass_cumsum /= sorted_mass_cumsum[-1]
        self.sorted_mass_cumsum = sorted_mass_cumsum
        self._fitted = True

    def threshold(self, alpha):
        if not self._fitted:
            raise RuntimeError("kde has not been fit.")
        idx = np.searchsorted(self.sorted_mass_cumsum, alpha, side="right")
        idx = min(idx, len(self.sorted_densities) - 1)
        return self.sorted_densities[idx]

    def contains(self, points, *, alpha=0.9):
        points = np.atleast_2d(points)
        densities = self.kde(points.T)
        return densities >= self.threshold(alpha)

    def plot(self, xlims=(0, 1), ylims=(0, 1), *, alpha=0.9):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        thresh = self.threshold(alpha)

        fig, ax = plt.subplots(figsize=(6, 5))

        heatmap = ax.contourf(self.X, self.Y, self.Z, levels=100, cmap='viridis')
        fig.colorbar(heatmap, ax=ax, label='Density')

        ax.contour(
            self.X, self.Y, self.Z, levels=[thresh], linestyles="--",
            colors='red', linewidths=2
        )

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

        return fig, ax

    def to_json(self):
        if not self._fitted:
            raise RuntimeError("kde has not been fitted.")
        return {
            "x_metric": self.x_metric,
            "y_metric": self.y_metric,
            "grid_size": self.grid_size,
            "X": serialize_array(self.X, compress=True),
            "Y": serialize_array(self.Y, compress=True),
            "Z": serialize_array(self.Z, compress=True),
            "kde": {
                "dataset": serialize_array(self.kde.dataset, compress=True),
                "weights": serialize_array(self.kde._weights, compress=True),
                "factor": float(self.kde.factor)
            },
            "sorted_densities": serialize_array(self.sorted_densities, compress=True),
            "sorted_mass_cumsum": serialize_array(self.sorted_mass_cumsum, compress=True),
        }

    @classmethod
    def from_json(cls, hdr_info):
        x_metric, y_metric = hdr_info["x_metric"], hdr_info["y_metric"]
        grid_size = hdr_info["grid_size"]
        result = cls(x_metric, y_metric, grid_size=grid_size)
        dataset = load_array(hdr_info["kde"]["dataset"], compressed=True)
        weights = load_array(hdr_info["kde"]["weights"], compressed=True)
        factor = float(hdr_info["kde"]["factor"])
        result.kde = stats.gaussian_kde(dataset, weights=weights, bw_method=factor)
        result.X = load_array(hdr_info["X"], compressed=True)
        result.Y = load_array(hdr_info["Y"], compressed=True)
        result.Z = load_array(hdr_info["Z"], compressed=True)
        result.sorted_densities = load_array(
            hdr_info["sorted_densities"], compressed=True
        )
        result.sorted_mass_cumsum = load_array(
            hdr_info["sorted_mass_cumsum"], compressed=True
        )
        result._fitted = True
        return result
