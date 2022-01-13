import numpy as np
import pickle
import pymc3 as pm
from typing import NamedTuple

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from opaque.ood.utils import AnyMethodPipeline


class BetaBinomialRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            coefficient_prior_type="normal",
            coefficient_prior_scale=100.0,
            intercept_prior_scale=1000.0,
            **sampler_args,
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
                    "coef_mean",
                    mu=0,
                    b=self.coefficient_prior_scale,
                    shape=ncols_mean,
                )
                coef_disp = pm.Laplace(
                    "coef_disp",
                    mu=0,
                    b=self.coefficient_prior_scale,
                    shape=ncols_disp,
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

    def predict(self, X, N=None):
        if N is None:
            N = np.ones(size=(X.shape[0]))
        post_pred = self.get_posterior_predictions(X, N, var_names=["K_est"])
        preds = np.mean(post_pred["K_est"], axis=0)
        return np.vstack([N, preds]).T

    def predict_shape_params(self, X):
        X = check_array(X)
        N = np.full(X.shape[0], 1)
        post_pred = self.get_posterior_predictions(X, N)
        mu_sample = post_pred["mu"]
        nu_sample = post_pred["nu"]
        alpha_sample = mu_sample * nu_sample
        beta_sample = (1 - mu_sample) * nu_sample
        alpha = np.mean(alpha_sample, axis=0)
        beta = np.mean(beta_sample, axis=0)
        alpha_var = np.var(alpha_sample, axis=0)
        beta_var = np.var(beta_sample, axis=0)
        return (
            np.vstack([alpha, beta]).T,
            np.vstack([alpha_var, beta_var]).T,
        )

    def get_model_info(self):
        check_is_fitted(self)
        return {
            'model': self.model_,
            'trace': self.trace_,
            'params': self.get_params(),
            'sampler_args': self.sampler_args,
            'mean_use_cols': self.mean_use_cols,
            'disp_use_cols': self.disp_use_cols,
        }

    @classmethod
    def load(cls, model_info):
        params = model_info['params']
        sampler_args = model_info['sampler_args']
        instance = cls(**params, **sampler_args)
        instance.model_ = model_info['model']
        instance.trace_ = model_info['trace']
        instance.mean_use_cols = model_info['mean_use_cols']
        instance.disp_use_cols = model_info['disp_use_cols']
        return instance


class ShapeParamResults(NamedTuple):
    sens_alpha: float
    sens_beta: float
    spec_alpha: float
    spec_beta: float
    sens_alpha_var: float
    sens_beta_var: float
    spec_alpha_var: float
    spec_beta_var: float


class DiagnosticTestPriorModel:
    def __init__(
            self,
            sens_pipeline,
            spec_pipeline,
            validation=None,
    ):
        # Some basic validation. Each pipeline should have only two steps,
        # a transformer and an estimator. Estimator should be a
        # BetaBinomialRegressor model.
        if validation is None or not isinstance(validation, dict):
            validation = {}
        self.__validation = validation

        for pipeline in sens_pipeline, spec_pipeline:
            assert len(pipeline.steps) == 2
            transformer = pipeline.steps[0][1]
            estimator = pipeline.steps[1][1]
            check_is_fitted(transformer)
            check_is_fitted(estimator)
            assert isinstance(estimator, BetaBinomialRegressor)
        self.sens_pipeline = sens_pipeline
        self.spec_pipeline = spec_pipeline

    @property
    def validation(self):
        return self.__validation

    @property
    def params(self):
        sens_estimator = self.sens_pipeline.steps[1][1]
        spec_estimator = self.spec_pipeline.steps[1][1]
        sens_params = sens_estimator.get_params()
        spec_params = spec_estimator.get_params()
        sens_sampler_args = sens_estimator.sampler_args
        spec_sampler_args = spec_estimator.sampler_args
        params = {}
        for key, value in sens_params:
            params[f"sens__{key}"] = value
        for key, value in spec_params:
            params[f"spec__{key}"] = value
        for key, value in sens_sampler_args:
            params[f"sens__sampler__{key}"] = value
        for key, value in spec_sampler_args:
            params[f"spec__sampler__{key}"] = value
        return params

    def predict_shape_params(
            self,
            nu,
            max_features,
            log_num_entrez,
            log_num_mesh,
            sens_neg_set,
            mean_spec,
            std_spec,
    ):
        X = np.array(
            [
                [
                    nu,
                    max_features,
                    log_num_entrez,
                    log_num_mesh,
                    sens_neg_set,
                    mean_spec,
                    std_spec,
                ]
            ]
        )
        sens_shape, sens_shape_var = self.sens_pipeline.apply_method(
            'predict_shape_params', X
        )
        spec_shape, spec_shape_var = self.spec_pipeline.apply_method(
            'predict_shape_params', X
        )
        return ShapeParamResults(
            sens_alpha=sens_shape[0, 0],
            sens_beta=sens_shape[0, 1],
            spec_alpha=spec_shape[0, 0],
            spec_beta=spec_shape[0, 1],
            sens_alpha_var=sens_shape_var[0, 0],
            sens_beta_var=sens_shape_var[0, 1],
            spec_alpha_var=spec_shape_var[0, 0],
            spec_beta_var=spec_shape_var[0, 1],
        )

    def dump(self, filepath):
        sens_estimator = self.sens_pipeline.steps[1][1]
        sens_transformer = self.sens_pipeline.steps[0][1]
        spec_estimator = self.spec_pipeline.steps[1][1]
        spec_transformer = self.spec_pipeline.steps[0][1]
        sens_model_info = sens_estimator.get_model_info()
        spec_model_info = spec_estimator.get_model_info()
        with open(filepath, 'wb') as file_handle:
            pickle.dump(
                {
                    'sens_model_info': sens_model_info,
                    'sens_transformer': sens_transformer,
                    'spec_model_info': spec_model_info,
                    'spec_transformer': spec_transformer,
                },
                file_handle,
            )

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            info = pickle.load(f)
        sens_model_info = info['sens_model_info']
        sens_transformer = info['sens_transformer']
        spec_model_info = info['spec_model_info']
        spec_transformer = info['spec_transformer']
        sens_estimator = BetaBinomialRegressor.load(sens_model_info)
        spec_estimator = BetaBinomialRegressor.load(spec_model_info)
        sens_pipeline = AnyMethodPipeline(
            [
                ('transform', sens_transformer),
                ('estimator', sens_estimator),
            ]
        )
        spec_pipeline = AnyMethodPipeline(
            [
                ('transform', spec_transformer),
                ('estimator', spec_estimator),
            ]
        )
        return cls(sens_pipeline, spec_pipeline)
