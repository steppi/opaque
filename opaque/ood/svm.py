import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted

from opaque.ood.utils import load_array, serialize_array


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
            "support_vectors": serialize_array(self.support_vectors_),
            "n_support": serialize_array(self._n_support)
        }

    def feature_scores(self):
        check_is_fitted(self)
        try:
            return self.coef_
        except AttributeError:
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
