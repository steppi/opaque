import arviz
import io
import numpy as np
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import tempfile


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
        else:
            self.outter_splitter = outter_splitter
        if inner_splitter is None:
            self.inner_splitter = StratifiedKFold()
        else:
            self.inner_splitter = inner_splitter

    def split(self, X, y=None):
        outter_splits = self.outter_splitter.split(X, y)
        for outter_train, outter_test in outter_splits:
            inner_splits = self.inner_splitter.split(
                outter_train,
                None if y is None else y[outter_train],
            )
            inner_splits = (
                (outter_train[inner_train], outter_train[inner_test])
                for inner_train, inner_test in inner_splits
            )
            yield inner_splits, outter_train, outter_test


class SubsetSampler:
    def __init__(self, iterable, *, random_state=None):
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


def serialize_array(X):
    memfile = io.BytesIO()
    if sparse.issparse(X):
        sparse.save_npz(memfile, X)
    else:
        np.save(memfile, X)
    memfile.seek(0)
    return memfile.read().decode('latin-1')


def load_array(data):
    memfile = io.BytesIO()
    memfile.write(data.encode('latin-1'))
    memfile.seek(0)
    X = np.load(memfile)
    if isinstance(X, np.ndarray):
        return X
    memfile.seek(0)
    X = sparse.load_npz(memfile)
    if sparse.issparse(X):
        return X
    raise ValueError("Input is not valid npz data.")


def dump_trace(trace):
    """Dump an arviz trace to a string of bytes."""
    with tempfile.NamedTemporaryFile() as tf:
        trace.to_netcdf(tf.name)
        tf.seek(0)
        result = tf.read().decode('latin-1')
    return result


def load_trace(data):
    """Load an arviz trace from a string of bytes produced by dump_trace."""
    original_arviz_data_load = arviz.rcParams['data.load']
    arviz.rcParams['data.load'] = 'eager'
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(data.encode('latin-1'))
        tf.seek(0)
        trace = arviz.from_netcdf(tf.name)
    arviz.rcParams['data.load'] = original_arviz_data_load
    if isinstance(trace, arviz.data.inference_data.InferenceData):
        return trace
    raise ValueError("Input is not valid net_cdf data.")
