import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted


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
            "support": self.support_.tolist(),
            "intercept": self.intercept_.tolist(),
            "dual_coef": self.dual_coef_.tolist(),
            "support_vectors": self.support_vectors_.tolist(),
            "n_support": self._n_support.tolist()
        }

    @classmethod
    def load_model_info(cls, model_info):
        model = LinearOneClassSVM(**model_info["params"])
        model._sparse = model_info["sparse"]
        model.shape_fit_ = tuple(model_info["shape_fit"])
        model.intercept_ = np.array(model_info["intercept"])
        model.dual_coef_ = np.array(model_info["dual_coef"])
        model.support_vectors_ = np.array(model_info["support_vectors"])
        model.support_ = np.array(model_info["support"], dtype=np.int32)
        model._n_support = np.array(model_info["n_support"], dtype=np.int32)

        model._intercept_ = model.intercept_
        model._dual_coef_ = model.dual_coef_
        model._gamma = 0.0
        model._probA, model._probB = np.array([]), np.array([])
        return model
