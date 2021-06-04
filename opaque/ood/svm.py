import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted

from opaque.ood.utils import load_array, serialize_array


class LinearOneClassSVM(OneClassSVM):
    def __init__(
        self,
        nu=0.5,
        tol=0.001,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    ):
        super().__init__(
            kernel="linear",
            nu=nu,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=False,
            max_iter=-1,
        )

    def get_model_info(self):
        check_is_fitted(self)
        return {
            "sparse": self._sparse,
            "params": self.get_params(),
            "shape_fit": list(self.shape_fit_),
            "support": serialize_array(self.support_),
            "intercept": serialize_array(self.intercept_),
            "dual_coef": serialize_array(self.dual_coef_),
            "support_vectors": serialize_array(self.support_vectors_),
            "n_support": serialize_array(self._n_support)
        }

    def feature_scores(self):
        check_is_fitted(self)
        return self.coef_

    @classmethod
    def load_model_info(cls, model_info):
        model = LinearOneClassSVM(**model_info["params"])
        model._sparse = model_info["sparse"]
        model.shape_fit_ = tuple(model_info["shape_fit"])
        model.intercept_ = load_array(model_info["intercept"])
        model.dual_coef_ = load_array(model_info["dual_coef"])
        model.support_vectors_ = load_array(model_info["support_vectors"])
        model.support_ = load_array(model_info["support"])
        model._n_support = load_array(model_info["n_support"])
        model.fit_status_ = 0.0
        model._intercept_ = model.intercept_
        model._dual_coef_ = model.dual_coef_
        model._gamma = 0.0
        model.offset_ = -model.intercept_
        model.class_weight_ = np.array([])
        model._probA, model._probB = np.array([]), np.array([])
        return model
