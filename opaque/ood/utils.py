from sklearn.pipeline import Pipeline


class AnyMethodPipeline(Pipeline):
    """sklearn Pipeline that allows any method of final estimator to be called
    """
    def apply_method(self, method, X, **params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return getattr(self.steps[-1][-1], method)(Xt, **params)
