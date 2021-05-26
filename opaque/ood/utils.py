import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


class AnyMethodPipeline(Pipeline):
    """sklearn Pipeline that allows any method of final estimator to be called
    """
    def apply_method(self, method, X, **params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return getattr(self.steps[-1][-1], method)(Xt, **params)


class NestedKFold:
    def __init__(self, outter_splitter=None, inner_splitter=None):
        if outter_splitter is None:
            self.outter_splitter = StratifiedKFold()
        if inner_splitter is None:
            self.inner_splitter = StratifiedKFold()

    def split(self, X, y):
        outter_splits = self.outter_splitter.split(X, y)
        for outter_train, outter_test in outter_splits:
            inner_splits = self.inner_splitter.split(X[outter_train],
                                                     y[outter_train])
            inner_splits = (
                (outter_train[inner_train], outter_train[inner_test])
                for inner_train, inner_test in inner_splits
            )
            yield inner_splits, outter_train, outter_test


class SubsetSampler:
    def __init__(self, iterable, random_state=None):
        self.arr = list(iterable)
        self.current_index = 0
        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        elif random_state is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = random_state
        self.rng.shuffle(self.arr)

    def _reset(self):
        self.rng.shuffle(self.arr)
        self.current_index = 0

    def sample(self, k):
        if self.current_index + k >= len(self.arr):
            self._reset()
        res = self.arr[self.current_index:self.current_index + k]
        self.current_index += k
        return sorted(res)

    def multiple_samples(self, n, k):
        """Draw n samples each of length k without replacement."""
        results = set()
        i = 0
        # This class should never be the bottleneck, so why not spend
        # cycles to contend with coupon collectors edge cases instead of
        # thinking of a more clever way.
        while i < n * np.log(n) and len(results) < n:
            results.add(tuple(self.sample(k)))
            i += 1
        return list(results)
