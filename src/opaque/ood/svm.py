import numpy as np
from scipy.sparse import issparse
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from opaque.utils import load_array, serialize_array


class LinearOneClassSVM(BaseEstimator, OutlierMixin):
    def __init__(
            self,
            *,
            nu=0.5,
            tol=1e-3,
            verbose=False,
            solver="libsvm",
            **kwargs,
    ):
        self.nu = nu
        self.tol = tol
        self.verbose = False
        if solver not in {"libsvm", "sgd"}:
            raise ValueError(
                f"solver should be one of 'libsvm' or 'sgd', got {solver}"
            )
        self.solver = solver
        kwargs.pop("nu", None)
        kwargs.pop("tol", None)
        kwargs.pop("verbose", None)
        self.kwargs = kwargs

    def _fit_libsvm(self, X):
        model = OneClassSVM(
            kernel='linear',
            nu=self.nu,
            tol=self.tol,
            verbose=self.verbose,
            **self.kwargs,
        )
        model.fit(X)
        return model.coef_, model.intercept_

    def _fit_sgd(self, X):
        model = SGDOneClassSVM(
            nu=self.nu,
            tol=self.tol,
            verbose=self.verbose,
            **self.kwargs,
        )
        model.fit(X)
        return model.coef_, -model.offset_

    def fit(self, X, y=None):
        if self.solver == "libsvm":
            coef, intercept = self._fit_libsvm(X)
        else:
            coef, intercept = self._fit_sgd(X)
        self.coef_, self.intercept_ = coef, intercept
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        return safe_sparse_dot(
            X, self.coef_.T, dense_output=True
        ) + self.intercept_

    def predict(self, X):
        check_is_fitted(self)
        scores = self.decision_function(X)
        return np.where(scores > 0, 1.0, -1.0)

    def get_model_info(self):
        check_is_fitted(self)
        return {
            'coef': serialize_array(self.coef_),
            'intercept': self.intercept_,
            'params': self.get_params()
        }

    def feature_scores(self):
        check_is_fitted(self)
        if not issparse(self.coef_):
            return -self.coef_ / self.intercept_
        else:
            return -self.coef_.toarray() / self.intercept_

    @classmethod
    def load_model_info(cls, model_info):
        model = LinearOneClassSVM(**model_info["params"])
        model.intercept_ = model_info["intercept"]
        model.coef_ = load_array(model_info["coef"])
        return model


class SerializableOneClassSVM(OneClassSVM):
    def get_model_info(self):
        check_is_fitted(self)
        return {
            "gamma": self._gamma,
            "sparse": self._sparse,
            "params": self.get_params(),
            "shape_fit": list(self.shape_fit_),
            "support": serialize_array(self.support_),
            "intercept": serialize_array(self.intercept_),
            "dual_coef": serialize_array(self.dual_coef_),
            "n_support": serialize_array(self._n_support),
            "support_vectors": serialize_array(self.support_vectors_),
        }

    def feature_scores(self):
        check_is_fitted(self)
        if self.kernel == "linear":
            return self.coef_
        else:
            raise AttributeError(
                "Feature scores only available when using a linear kernel."
            )

    @classmethod
    def load_model_info(cls, model_info):
        model = SerializableOneClassSVM(**model_info["params"])
        model.fit_status_ = 0.0
        model._gamma = model_info["gamma"]
        model._sparse = model_info["sparse"]
        model.shape_fit_ = tuple(model_info["shape_fit"])
        model.support_ = load_array(model_info["support"])
        model.intercept_ = load_array(model_info["intercept"])
        model.dual_coef_ = load_array(model_info["dual_coef"])
        model._n_support = load_array(model_info["n_support"])
        model.support_vectors_ = load_array(model_info["support_vectors"])
        model.offset_ = -model.intercept_
        model.class_weight_ = np.array([])
        model._intercept_ = model.intercept_
        model._dual_coef_ = model.dual_coef_
        model._probA, model._probB = np.array([]), np.array([])
        return model
