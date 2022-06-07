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
        aggregate_results = defaultdict(list)
        aggregate_results_pos = defaultdict(list)
        aggregate_results_neg = defaultdict(list)
        with Pool(n_jobs) as pool:
            results = pool.imap(run_trial_for_theta, points, chunksize=100)
            for n, t, theta, theta_pos, theta_neg in results:
                aggregate_results[(n, t)].append(theta)
                aggregate_results_pos[(n, t)].append(theta_pos)
                aggregate_results_neg[(n, t)].append(theta_neg)
        aggregate_results = {
            (n, t): ECDF(theta_list)
            for (n, t), theta_list in aggregate_results.items()
        }
        aggregate_results_pos = {
            (n, t): ECDF(theta_list)
            for (n, t), theta_list in aggregate_results_pos.items()
        }
        aggregate_results_neg = {
            (n, t): ECDF(theta_list)
            for (n, t), theta_list in aggregate_results_neg.items()
        }
        self.aggregate_results = aggregate_results
        self.aggregate_results_pos = aggregate_results_pos
        self.aggregate_results_neg = aggregate_results_neg
        self.info_dict["n_trials"] = n_trials

    def get_results_dict(self, num_grid_points=100):
        x = np.linspace(0, 1, num_grid_points)
        results_dict = self.aggregate_results
        results_dict_pos = self.aggregate_results_pos
        results_dict_neg = self.aggregate_results_neg
        return {
            "results": {
                f"{n}:{t}": ecdf(x).tolist()
                for (n, t), ecdf in results_dict.items()
            },
            "results_pos": {
                f"{n}:{t}": ecdf(x).tolist()
                for (n, t), ecdf in results_dict_pos.items()
            },
            "results_neg": {
                f"{n}:{t}": ecdf(x).tolist()
                for (n, t), ecdf in results_dict_neg.items()
            },
            "info": self.info_dict,
        }


def run_trial_for_theta(args):
    theta, sensitivity, specificity, samples_per_trial, random_state = args
    # Set up distributions
    ground_dist = binom(1, theta)
    ground_dist.random_state = random_state
    # Distribution for test results given example is positive.
    pos_dist = binom(1, sensitivity)
    pos_dist.random_state = random_state
    # Distribution for test results given example is negative.
    neg_dist = binom(1, 1 - specificity)
    neg_dist.random_state = random_state

    ground_truth = ground_dist.rvs(size=samples_per_trial)
    # Hypothetical diagnostic test results for positive examples.
    pos = pos_dist.rvs(size=samples_per_trial)
    # Hypothetical results for negative samples.
    neg = neg_dist.rvs(size=samples_per_trial)
    # Observed test results.
    test_results = np.where(ground_truth == 1, pos, neg)

    # Conditional prevalence.
    theta_pos = np.sum(ground_truth[test_results == 1])
    theta_pos /= np.sum(test_results == 1)
    theta_neg = np.sum(ground_truth[test_results == 0])
    theta_neg /= np.sum(test_results == 0)

    n, t = samples_per_trial, np.sum(test_results)
    return n, t, theta, theta_pos, theta_neg
