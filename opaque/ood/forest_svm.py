import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method

from opaque.ood._tree_kernel import tree_kernel


class AnyMethodPipeline(Pipeline):
    """sklearn Pipeline that allows any method of final estimator to be called
    """
    @if_delegate_has_method(delegate='_final_estimator')
    def apply_method(self, method, X, **params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return getattr(self.steps[-1][-1], method)(X, **params)


def make_forest_kernel(trained_forest):
    trees = [est.tree_ for est in trained_forest.estimators_]

    def K(X, Y):
        n1, p1 = X.shape
        n2, p2 = Y.shape
        if p1 != p2:
            raise ValueError("Input matrices do not have compatible shape.")
        X_32, Y_32 = X.astype(np.float32), Y.astype(np.float32)
        return tree_kernel(X_32, Y_32, trees)

    return K


class ForestOneClassSVM(BaseEstimator):
    def __init__(
        self,
        forest,
        # OneClassSVM parameters
        tol=1e-3,
        nu=0.5,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    ):

        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.forest = forest
        self.estimator_ = None

    def fit(self, X, y, sample_weight=None, **params):
        num_classes = len(set(y))
        if num_classes > 1:
            forest_estimator = self.forest
            forest_estimator.fit(X, y, sample_weight=sample_weight)
            kernel = make_forest_kernel(forest_estimator)
        else:
            kernel = "linear"
        estimator = OneClassSVM(
            kernel=kernel,
            tol=self.tol,
            nu=self.nu,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
        )
        estimator.fit(X)
        self.estimator_ = estimator
        return self

    def ood_predict(self, X):
        return self.estimator_.predict(X)

    def ood_decision_function(self, X):
        return self.estimator_.decision_function(X)

    def forest_predict(self, X):
        return self.forest.predict(X)

    def forest_predict_proba(self, X):
        return self.forest.predict_proba(X)

    def forest_predict_log_proba(self, X):
        return self.forest.predict_log_proba(X)

    def predict(self, X):
        ood_predictions = self.ood_predict(X)
        forest_predictions = self.forest_predict(X)
        return np.where(ood_predictions == -1.0, None, forest_predictions)
