import numpy as np
from multiprocessing import Pool
from scipy.stats import beta, binom
from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF


class PrevalenceSimulation:
    """Manage simulations of prevalence distribution

    Parameters
    ----------
    fixed : bool
        If fixed is True, sensitivity and specificity should be
        set to fixed values.
    """

    def __init__(
        self,
        sensitivity,
        specificity,
        samples_per_trial=100,
        num_grid_points=100,
        seed=None,
    ):
        self.num_grid_points = num_grid_points
        self.samples_per_trial = samples_per_trial
        self.random_state = np.random.RandomState(seed)
        info_dict = {}
        info_dict["num_grid_points"] = num_grid_points
        if isinstance(sensitivity, float):
            self.sample_sens = lambda: sensitivity
            info_dict["sensitivity"] = sensitivity
        else:
            sens_a, sens_b = sensitivity
            sens_prior = beta(sens_a, sens_b)
            sens_prior.random_state = self.random_state
            info_dict["sens_prior"] = [sens_a, sens_b]

            def sample_sens():
                return sens_prior.rvs()

            self.sample_sens = sample_sens

        if isinstance(specificity, float):
            self.sample_spec = lambda: specificity
            info_dict["specificity"] = specificity
        else:
            spec_a, spec_b = specificity
            spec_prior = beta(spec_a, spec_b)
            spec_prior.random_state = self.random_state
            info_dict["spec_prior"] = [spec_a, spec_b]

            def sample_spec():
                return spec_prior.rvs()

            self.sample_spec = sample_spec
        self.info_dict = info_dict

    def sample_sens_spec(self):
        return self.sample_sens(), self.sample_spec()

    def run(self, n_trials=1000, n_jobs=1):
        points = (
            (
                theta,
                *self.sample_sens_spec(),
                self.samples_per_trial,
                np.random.RandomState(self.random_state.randint(10 ** 6)),
            )
            for _ in range(n_trials)
            for theta in np.linspace(0, 1, self.num_grid_points)
        )
        with Pool(n_jobs) as pool:
            results = pool.starmap(run_trial_for_theta, points)
        aggregate_results = defaultdict(list)
        for n, t, theta in results:
            aggregate_results[(n, t)].append(theta)
        aggregate_results = {
            (n, t): ECDF(theta_list)
            for (n, t), theta_list in aggregate_results.items()
        }
        self.aggregate_results = aggregate_results
        self.info_dict["n_trials"] = n_trials

    def get_results_dict(self, num_grid_points=100):
        x = np.linspace(0, 1, num_grid_points)
        return {
            "results": {
                f"{n}:{t}": ecdf(x).tolist()
                for (n, t), ecdf in self.aggregate_results.items()
            },
            "info": self.info_dict,
        }


def run_trial_for_theta(
    theta, sensitivity, specificity, samples_per_trial, random_state
):
    binom_ground = binom(1, theta)
    binom_ground.random_state = random_state
    binom_pos = binom(1, sensitivity)
    binom_pos.random_state = random_state
    binom_neg = binom(1, 1 - specificity)
    binom_neg.random_state = random_state
    ground = binom_ground.rvs(size=samples_per_trial)
    pos = binom_pos.rvs(size=samples_per_trial)
    neg = binom_neg.rvs(size=samples_per_trial)
    observed = np.where(ground == 1, pos, neg)
    n, t = samples_per_trial, sum(observed)
    return n, t, theta
