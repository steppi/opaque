import numpy as np
import pymc3 as pm
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BetaBinomialRegressor:
    def __init__(
        self,
        coefficient_prior_type="normal",
        coefficient_prior_scale=100.0,
        intercept_prior_scale=1000.0,
        **sampler_args
    ):
        self.coefficient_prior_type = coefficient_prior_type
        self.coefficient_prior_scale = coefficient_prior_scale
        self.intercept_prior_scale = intercept_prior_scale

        self.mean_use_cols = None
        self.disp_use_cols = None
        self.model_ = None
        self.trace_ = None
        if "return_inferencedata" not in sampler_args:
            sampler_args["return_inferencedata"] = True
        if "progressbar" not in sampler_args:
            sampler_args["progressbar"] = False
        self.sampler_args = sampler_args

    def fit(
            self,
            X,
            y,
            mean_use_cols=None,
            disp_use_cols=None,
    ):
        self.mean_use_cols = mean_use_cols
        self.disp_use_cols = disp_use_cols
        X, y = check_array(X), check_array(y)
        assert X.shape[0] == y.shape[0]
        assert y.shape[1] == 2
        N, K = y[:, 0], y[:, 1]

        with pm.Model() as model:
            intercept_mean = pm.Normal(
                "intercept_mean",
                0.0,
                self.intercept_prior_scale
            )
            intercept_disp = pm.Normal(
                "intercept_disp",
                0.0,
                self.intercept_prior_scale
            )

            X_mean = X[:] if mean_use_cols is None else X[:, mean_use_cols]
            X_disp = X[:] if disp_use_cols is None else X[:, disp_use_cols]
            ncols_mean = X_mean.shape[1]
            ncols_disp = X_disp.shape[1]
            X_mean_obs = pm.Data("X_mean_obs", X_mean)
            X_disp_obs = pm.Data("X_disp_obs", X_disp)
            N = pm.Data("N", N)
            K_obs = pm.Data("K_obs", K)
            if self.coefficient_prior_type == "normal":
                coef_mean = pm.Normal(
                    "coef_mean",
                    0.0,
                    self.coefficient_prior_scale,
                    shape=ncols_mean
                )
                coef_disp = pm.Normal(
                    "coef_disp",
                    0.0,
                    self.coefficient_prior_scale,
                    shape=ncols_disp
                )
            elif self.coefficient_prior_type == "laplace":
                coef_mean = pm.Laplace(
                    "coef_mean", self.coefficient_prior_scale, shape=ncols_mean
                )
                coef_disp = pm.Laplace(
                    "coef_disp", self.coefficient_prior_scale, shape=ncols_disp
                )
            else:
                ValueError(
                    'coefficient_prior_type must be one of "normal" '
                    'or "laplace"'
                )
            mu = pm.Deterministic(
                "mu",
                pm.math.invlogit(
                    intercept_mean + pm.math.dot(X_mean_obs, coef_mean),
                    )
            )
            nu = pm.Deterministic(
                "nu",
                pm.math.exp(
                    intercept_disp + pm.math.dot(X_disp_obs, coef_disp)
                )
            )
            K_est = pm.BetaBinomial(
                "K_est",
                n=N,
                alpha=mu * nu,
                beta=(1 - mu) * nu,
                observed=K_obs,
            )
            trace = pm.sample(**self.sampler_args)
            self.trace_ = trace
            self.model_ = model

    def get_posterior_predictions(self, X, N, **pymc3_args):
        check_is_fitted(self)
        if "var_names" not in pymc3_args:
            pymc3_args["var_names"] = ["K_est", "mu", "nu"]
        mean_use_cols, disp_use_cols = self.mean_use_cols, self.disp_use_cols
        X, N = check_X_y(X, N)
        X_mean = X[:] if mean_use_cols is None else X[:, mean_use_cols]
        X_disp = X[:] if disp_use_cols is None else X[:, disp_use_cols]
        with self.model_:
            pm.set_data(
                {
                    "X_mean_obs": X_mean,
                    "X_disp_obs": X_disp,
                    "N": N,
                    "K_obs": np.empty(len(X_mean)),
                }
            )
            post_pred = pm.fast_sample_posterior_predictive(
                self.trace_,
                **pymc3_args
                )
        return post_pred

    def predict(self, X, N):
        post_pred = self.get_posterior_predictions(X, N, var_names=["K_est"])
        return np.mean(post_pred["K_est"], axis=0)

    def predict_shape_params(self, X):
        X = check_array(X)
        N = np.full(X.shape[0], 1000)
        post_pred = self.get_posterior_predictions(X, N)
        mu = np.mean(post_pred["mu"], axis=0)
        nu = np.mean(post_pred["nu"], axis=0)
        alpha = mu * nu
        beta = (1 - mu) * nu
        return np.vstack([alpha, beta]).T
