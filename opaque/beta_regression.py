import pymc3 as pm
from sklearn.base import BaseEstimator


class BetaRegressor(BaseEstimator):
    def __init__(self, coefficient_prior_type='normal',
                 coefficient_prior_scale=100.0,
                 intercept_prior_type='normal',
                 intercept_prior_scale=1000.0,
                 **sampler_args):
        self.coefficient_prior_type = coefficient_prior_type
        self.coefficient_prior_scale = coefficient_prior_scale
        self.intercept_prior_type = intercept_prior_type
        self.intercept_prior_scale = intercept_prior_scale
        self.sampler_args = sampler_args
        self.model_ = None
        self.trace_ = None

    def fit(self, X, y, mean_use_cols=None, disp_use_cols=None):
        with pm.Model() as model:
            if self.intercept_prior_type == 'normal':
                int_mean = pm.Normal('int_mean', 0.0,
                                     self.intercept_prior_scale)
                int_disp = pm.Normal('int_disp', 0.0,
                                     self.intercept_prior_scale)
            elif self.intercept_prior_type == 'laplace':
                int_mean = pm.Laplace('int_mean',
                                      self.intercept_prior_scale)
                int_disp = pm.Laplace('int_disp',
                                      self.intercept_prior_scale)
            else:
                raise ValueError('intercept_prior_type must be one of "normal" '
                                 'or "laplace"')
            X_mean = X[:] if mean_use_cols is None else X[:, mean_use_cols]
            X_disp = X[:] if disp_use_cols is None else X[:, disp_use_cols]
            ncols_mean = X_mean.shape[1]
            ncols_disp = X_disp.shape[1]
            if self.coefficient_prior_type == 'normal':
                beta_mean = pm.Normal('beta_mean', 0.0,
                                      self.coefficient_prior_scale,
                                       shape=ncols_mean)
                beta_disp = pm.Normal('beta_disp', 0.0,
                                      self.coefficient_prior_scale,
                                       shape=ncols_disp)
            elif self.coefficient_prior_type == 'laplace':
                beta_mean = pm.Laplace('beta_mean',
                                       self.coefficient_prior_scale,
                                        shape=ncols_mean)
                beta_disp = pm.Laplace('beta_disp',
                                       self.coefficient_prior_scale,
                                        shape=ncols_disp)
            else:
                ValueError('coefficient_prior_type must be one of "normal" '
                           'or "laplace"')
            mu_est = pm.math.invlogit(int_mean +
                                      pm.math.dot(X_mean, beta_mean))
            nu_est = pm.math.exp(int_disp + pm.math.dot(X_disp, beta_disp))
            y_est = pm.Beta('y_est', alpha=mu_est * nu_est,
                            beta = (1 - mu_est) * nu_est, observed=y)
            trace = pm.sample(**self.sampler_args)
            self.trace_ = trace
            self.model_ = model
