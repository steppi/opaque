import math
import numpy as np
from multiprocessing import Pool
from scipy.special import expit
from scipy.stats import beta, binom
from sklearn.model_selection import KFold
from opaque.beta_regression import BetaRegressor
from opaque.simulations.prevalence import run_trial_for_theta


class BetaSimulation(object):
    def __init__(self, sens_coefs_mean, sens_coefs_disp,
                 spec_coefs_mean, spec_coefs_disp,
                 sens_noise_mean=0.0, sens_noise_disp=0.0,
                 spec_noise_mean=0.0, spec_noise_disp=0.0,
                 cov=None, seed=None):
        if cov is None:
            cov = np.diag(np.full(len(sens_coefs_mean) - 1, 1.0))
        else:
            cov = np.array(cov)
        self.random_state = np.random.RandomState(seed)
        assert len(sens_coefs_mean) == len(sens_coefs_disp) == cov.shape[0] + 1
        assert len(spec_coefs_mean) == len(spec_coefs_disp) == cov.shape[0] + 1
        self.sens_coefs_mean = np.array(sens_coefs_mean)
        self.sens_coefs_disp = np.array(sens_coefs_disp)
        self.spec_coefs_mean = np.array(spec_coefs_mean)
        self.spec_coefs_disp = np.array(spec_coefs_disp)
        self.sens_noise_mean = sens_noise_mean
        self.sens_noise_disp = sens_noise_disp
        self.spec_noise_mean = spec_noise_mean
        self.spec_noise_disp = spec_noise_disp
        self.cov = cov
        
    def generate_data(self, size):
        X =  self.random_state.multivariate_normal(np.zeros(self.cov.shape[0]),
                                                   self.cov, size=size)
        X = np.hstack([np.full((X.shape[0], 1), 1), X])
        sens_mu = expit(X.dot(self.sens_coefs_mean) +
                   np.random.normal(0, self.sens_noise_mean, size=size))
        sens_nu = np.exp(X.dot(self.sens_coefs_disp) +
                    self.random_state.normal(0, self.sens_noise_disp,
                                             size=size))
        sens_prior = beta(sens_mu * sens_nu, (1 - sens_mu) * sens_nu)
        sens_prior.random_state = self.random_state
        sens = sens_prior.rvs()
        spec_mu = expit(X.dot(self.spec_coefs_mean) +
                        np.random.normal(0, self.spec_noise_mean, size=size))
        spec_nu = np.exp(X.dot(self.spec_coefs_disp) +
                         np.random.normal(0, self.spec_noise_disp, size=size))
        spec_prior = beta(spec_mu * spec_nu, (1 - spec_mu) * spec_nu)
        spec_prior.random_state = self.random_state
        spec = spec_prior.rvs()
        sens.shape = sens_mu.shape = sens_nu.shape = (size, 1)
        spec.shape = spec_mu.shape = spec_nu.shape = (size, 1)
        data = np.hstack([X, sens, spec, sens_mu, sens_nu, spec_mu, spec_nu,
                          sens_mu * sens_nu, (1 - sens_mu) * sens_nu,
                          spec_mu * spec_nu, (1 - spec_mu) * spec_mu])
        return data


def simulate_anomaly_detection(sens, spec, theta_a=1.0, theta_b=1.0,
                               n_mean=6.0, n_sigma=1.0, random_state=None,
                               n_jobs=1):
    if random_state is None:
        random_state = np.random.RandomState()
    points = ((random_state.random_sample(),
               sens, spec, math.floor(random_state.lognormal(mean=n_mean,
                                                             sigma=n_sigma)),
               np.random.RandomState(random_state.randint(10**6)))
              for sens, spec in zip(sens, spec))
    with Pool(n_jobs) as pool:
        results = pool.starmap(run_trial_for_theta, points)
    return results


def prevalidate_beta_regression(X, y, beta_regressor, n_splits=5,
                                shuffle=True, random_state=None):
    output = np.empty((X.shape[0], 2))
    if random_state is None:
        random_state = np.random.RandomState()
    splits = KFold(n_splits=n_splits, shuffle=shuffle,
                   random_state=random_state).split(X)
    for train, test in splits:
        beta_regressor.fit(X[train], y[train])
        shape_params = beta_regressor.predict_shape_params(X[test])
        output[test] = shape_params
    return output
