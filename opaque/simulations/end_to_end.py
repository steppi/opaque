import math
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.special import expit
from scipy.stats import beta, binom
from sklearn.model_selection import KFold
from opaque.beta_regression import BetaRegressor
from opaque.stats import equal_tailed_interval, KL_beta
from opaque.simulations.prevalence import run_trial_for_theta


class EndtoEndSimulator(object):
    def __init__(self, sens_coefs_mean, sens_coefs_disp,
                 spec_coefs_mean, spec_coefs_disp,
                 sens_noise_mean=0.0, sens_noise_disp=0.0,
                 spec_noise_mean=0.0, spec_noise_disp=0.0,
                 cov=None,
                 n_mean=6.0, n_sigma=1.0, random_state=None, n_jobs=1):
        if cov is None:
            cov = np.diag(np.full(len(sens_coefs_mean) - 1, 1.0))
        else:
            cov = np.array(cov)
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
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
        self.num_covariates = cov.shape[0]
        self.n_mean = n_mean
        self.n_sigma = n_sigma
        self.n_jobs = n_jobs
        
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
        data = np.hstack([X[:, 1:], sens, spec,
                          sens_mu, sens_nu, spec_mu, spec_nu,
                          sens_mu * sens_nu, (1 - sens_mu) * sens_nu,
                          spec_mu * spec_nu, (1 - spec_mu) * spec_nu])
        data = pd.DataFrame(data,
                            columns=[f'X{i}'
                                     for i in range(self.num_covariates)] +
                            ['sens', 'spec', 'sens_mu', 'sens_nu',
                             'spec_mu', 'spec_nu', 'sens_a', 'sens_b',
                             'spec_a', 'spec_b'])
        return data

    def simulate_anomaly_detection(self, sens_list, spec_list):
        points = ((self.random_state.random_sample(),
                   sens, spec,
                   math.floor(self.random_state.lognormal(mean=self.n_mean,
                                                          sigma=self.n_sigma)),
                   np.random.RandomState(self.random_state.randint(10**6)))
                   for sens, spec in zip(sens_list, spec_list))
        with Pool(self.n_jobs) as pool:
            results = pool.starmap(run_trial_for_theta, points)
        return results

    def run(self, size_train=1000, size_test=200):
        data_train = self.generate_data(size=size_train)
        data_test = self.generate_data(size=size_test)
        X_train = data_train.iloc[:, :self.num_covariates].values
        X_test = data_test.iloc[:, :self.num_covariates].values
        sens_train = data_train['sens'].values
        spec_train = data_train['spec'].values
        sens_test = data_test['sens'].values
        spec_test = data_test['spec'].values
        br = BetaRegressor()
        br.fit(X_train, sens_train)
        sens_shape = br.predict_shape_params(X_test)
        br.fit(X_train, spec_train)
        spec_shape = br.predict_shape_params(X_test)
        ad = self.simulate_anomaly_detection(sens_test, spec_test)
        points = []
        rows = []
        for i in range(len(ad)):
            n, t, theta = ad[i]
            sens_a_est, sens_b_est = sens_shape[i, :]
            spec_a_est, spec_b_est = spec_shape[i, :]
            sens_a, sens_b = data_test.iloc[i, -4], data_test.iloc[i, -3]
            spec_a, spec_b = data_test.iloc[i, -2], data_test.iloc[i, -1]
            point = [n, t, sens_a_est, sens_b_est, spec_a_est, spec_b_est]
            points.append(point)
            rows.append(point + [sens_a, sens_b, spec_a, spec_b,
                                 KL_beta(sens_a, sens_b,
                                         sens_a_est, sens_b_est),
                                 KL_beta(spec_a, spec_b,
                                         spec_a_est, spec_b_est), theta])
        with Pool(self.n_jobs) as pool:
            intervals = pool.starmap(equal_tailed_interval, points)
        data = np.array(rows)
        intervals = np.array(intervals)
        data = np.hstack([data, intervals])
        data = pd.DataFrame(data, columns=['n', 't', 'sens_a_est',
                                           'sens_b_est', 'spec_a_est',
                                           'spec_b_est', 'sens_a', 'sens_b',
                                           'spec_a', 'spec_b',
                                           'KL_sens', 'KL_spec',
                                           'theta', 'left', 'right'])
        return data
