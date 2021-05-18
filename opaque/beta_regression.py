import numpy as np
import pymc3 as pm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BetaRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        coefficient_prior_type="normal",
        coefficient_prior_scale=100.0,
        intercept_prior_type="normal",
        intercept_prior_scale=1000.0,
        rescale_response=True,
        **sampler_args
    ):
        self.coefficient_prior_type = coefficient_prior_type
        self.coefficient_prior_scale = coefficient_prior_scale
        self.intercept_prior_type = intercept_prior_type
        self.intercept_prior_scale = intercept_prior_scale

        self.rescale_response = rescale_response
        self.mean_use_cols = None
        self.disp_use_cols = None
        self.model_ = None
        self.trace_ = None
        if "return_inferencedata" not in sampler_args:
            sampler_args["return_inferencedata"] = True
        if "progressbar" not in sampler_args:
            sampler_args["progressbar"] = False
        self.sampler_args = sampler_args

    def fit(self, X, y, mean_use_cols=None, disp_use_cols=None):
        self.mean_use_cols = mean_use_cols
        self.disp_use_cols = disp_use_cols
        X, y = check_X_y(X, y)
        if self.rescale_response:
            y = (y * (len(y) - 1) + 0.5) / len(y)
        with pm.Model() as model:
            if self.intercept_prior_type == "normal":
                int_mean = pm.Normal(
                    "int_mean",
                    0.0,
                    self.intercept_prior_scale
                )
                int_disp = pm.Normal(
                    "int_disp",
                    0.0,
                    self.intercept_prior_scale
                    )
            elif self.intercept_prior_type == "laplace":
                int_mean = pm.Laplace("int_mean", self.intercept_prior_scale)
                int_disp = pm.Laplace("int_disp", self.intercept_prior_scale)
            else:
                raise ValueError(
                    'intercept_prior_type must be one of "normal" '
                    'or "laplace"'
                )
            X_mean = X[:] if mean_use_cols is None else X[:, mean_use_cols]
            X_disp = X[:] if disp_use_cols is None else X[:, disp_use_cols]
            ncols_mean = X_mean.shape[1]
            ncols_disp = X_disp.shape[1]
            X_mean_obs = pm.Data("X_mean_obs", X_mean)
            X_disp_obs = pm.Data("X_disp_obs", X_disp)
            y_obs = pm.Data("y_obs", y)
            if self.coefficient_prior_type == "normal":
                beta_mean = pm.Normal(
                    "beta_mean",
                    0.0,
                    self.coefficient_prior_scale,
                    shape=ncols_mean
                )
                beta_disp = pm.Normal(
                    "beta_disp",
                    0.0,
                    self.coefficient_prior_scale,
                    shape=ncols_disp
                )
            elif self.coefficient_prior_type == "laplace":
                beta_mean = pm.Laplace(
                    "beta_mean", self.coefficient_prior_scale, shape=ncols_mean
                )
                beta_disp = pm.Laplace(
                    "beta_disp", self.coefficient_prior_scale, shape=ncols_disp
                )
            else:
                ValueError(
                    'coefficient_prior_type must be one of "normal" '
                    'or "laplace"'
                )
            mu_est = pm.Deterministic(
                "mu_est",
                pm.math.invlogit(
                    int_mean + pm.math.dot(X_mean_obs, beta_mean),
                    )
            )
            nu_est = pm.Deterministic(
                "nu_est",
                pm.math.exp(int_disp + pm.math.dot(X_disp_obs, beta_disp))
            )
            y_est = pm.Beta(
                "y_est",
                alpha=mu_est * nu_est,
                beta=(1 - mu_est) * nu_est,
                observed=y_obs,
            )
            trace = pm.sample(**self.sampler_args)
            self.trace_ = trace
            self.model_ = model

    def get_posterior_predictions(self, X, **pymc3_args):
        check_is_fitted(self)
        if "var_names" not in pymc3_args:
            pymc3_args["var_names"] = ["y_est", "mu_est", "nu_est"]
        mean_use_cols, disp_use_cols = self.mean_use_cols, self.disp_use_cols
        X = check_array(X)
        X_mean = X[:] if mean_use_cols is None else X[:, mean_use_cols]
        X_disp = X[:] if disp_use_cols is None else X[:, disp_use_cols]
        with self.model_:
            pm.set_data(
                {
                    "X_mean_obs": X_mean,
                    "X_disp_obs": X_disp,
                    "y_obs": np.empty(len(X_mean)),
                }
            )
            post_pred = pm.fast_sample_posterior_predictive(
                self.trace_,
                **pymc3_args
                )

        return post_pred

    def predict(self, X):
        post_pred = self.get_posterior_predictions(X, var_names=["y_est"])
        return np.mean(post_pred["y_est"], axis=0)

    def predict_shape_params(self, X):
        post_pred = self.get_posterior_predictions(X)
        mu = np.mean(post_pred["mu_est"], axis=0)
        nu = np.mean(post_pred["nu_est"], axis=0)
        alpha = mu * nu
        beta = (1 - mu) * nu
        return np.vstack([alpha, beta]).T
