"""Real-fit fixtures and scoped observation hooks for parallel error-policy tests."""
import threading
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD
import ensemble_dwd.estimator as estimator_module
import ensemble_dwd._parallel_base as parallel_module


DEFAULT = dict(divide="warn", over="warn", invalid="warn", under="ignore")
STRICT = dict(DEFAULT, divide="raise")


@pytest.fixture(autouse=True)
def restore_runtime():
    policy, handler = np.geterr(), np.geterrcall()
    np.seterr(**DEFAULT)
    np.seterrcall(None)
    # Most probes need only two workers to expose the boundary/queue reuse.
    # The dedicated sixteen-worker case explicitly restores a native16 caller.
    try:
        with threadpool_limits(1):
            yield
    finally:
        np.seterr(**policy)
        np.seterrcall(handler)


def data(n=32, *, offset=0., far=False):
    # Fixed labels make every even-sized chunk of the unchanged seed7
    # permutation eligible; this is fixture construction, not a redraw.
    permutation = np.random.RandomState(7).permutation(n)
    y = np.empty(n, dtype=int)
    y[permutation] = np.arange(n) % 2
    if far:
        X = np.column_stack((40. * y, np.zeros(n)))
    else:
        X = np.random.RandomState(13).normal(size=(n, 3)) * .2
        X += y[:, None] * np.array([.5, -.3, .2])
    return X + offset, y


def model(**options):
    params = dict(M=2, theta=.25, sv_selection_ratio=1., lambd=.1,
                  gamma=.4, random_state=7, max_iter=3, stopping="fixed",
                  acceleration=None, backend="spectral", affine_computation="standard",
                  base_jobs=2, store_base_estimators=True)
    params.update(options)
    return EnsembleKernelDWD(**params)


class Sink:
    def __init__(self, fail=False):
        self.events, self.fail, self.lock = [], fail, threading.Lock()

    def __reduce__(self):
        raise TypeError("IR4 handler deliberately cannot be pickled")

    def record(self, event):
        with self.lock:
            self.events.append((threading.get_ident(), event))
        if self.fail:
            raise RuntimeError("IR4 callback sentinel")

    def __call__(self, description, flag):
        self.record((description, flag))

    def write(self, description):
        self.record(description)


def trigger(category):
    if category == "divide":
        return np.divide(np.float64(1), np.float64(0))
    if category == "over":
        return np.multiply(np.float64(np.finfo(float).max), np.float64(2))
    if category == "under":
        return np.multiply(np.float64(np.finfo(float).tiny), np.float64(np.finfo(float).tiny))
    return np.sqrt(np.float64(-1))


def install_fit_probe(monkeypatch, full_size, *, event=None, mutate_after=False):
    original, rows, lock = KernGDWD.fit, [], threading.Lock()
    replacement_handler = Sink()

    def observed(learner, X, y, *args, **kwargs):
        row = dict(thread=threading.get_ident(), size=len(y), policy=np.geterr(),
                   handler=np.geterrcall(), returned=False)
        with lock:
            rows.append(row)
        if event is not None and len(y) < full_size:
            trigger(event)
        result = original(learner, X, y, *args, **kwargs)
        row["returned"] = True
        if mutate_after and len(y) < full_size:
            # Deliberately outside any production scoring errstate, so only
            # the proposed task boundary can restore this worker mutation.
            np.seterr(divide="ignore", over="raise", invalid="raise", under="ignore")
            np.seterrcall(replacement_handler)
        return result

    monkeypatch.setattr(KernGDWD, "fit", observed)
    return rows


def install_score_probe(monkeypatch, *, event=None, mutate=False, fail_type=None):
    original, rows, lock = estimator_module._binary_scores, [], threading.Lock()
    replacement_handler = Sink()

    def observed(learner, X, **kwargs):
        with lock:
            rows.append(dict(thread=threading.get_ident(), policy=np.geterr(), handler=np.geterrcall()))
        if event is not None:
            trigger(event)
        result = original(learner, X, **kwargs)
        if mutate:
            np.seterr(divide="ignore", over="raise", invalid="raise", under="ignore")
            np.seterrcall(replacement_handler)
        if fail_type is not None:
            raise fail_type("IR4 task failure sentinel")
        return result

    monkeypatch.setattr(estimator_module, "_binary_scores", observed)
    return rows


class PoolObserver:
    """Observe actual executor boundaries without replacing its numerical work."""
    def __init__(self, monkeypatch, *, seed_handler=None):
        observer = self
        self.rows, self.initialized, self.lock = [], [], threading.Lock()
        original = parallel_module.ThreadPoolExecutor

        class ObservedExecutor(original):
            def __init__(self, *args, **kwargs):
                initialize = kwargs.get("initializer")

                def observed_initializer(*initargs):
                    if initialize is not None:
                        initialize(*initargs)
                    if seed_handler is not None:
                        np.seterrcall(seed_handler)
                    with observer.lock:
                        observer.initialized.append(threading.get_ident())

                kwargs["initializer"] = observed_initializer
                super().__init__(*args, **kwargs)

            def submit(self, function, *args, **kwargs):
                def outside_task():
                    row = dict(thread=threading.get_ident(), before=np.geterr(),
                               handler_before=np.geterrcall())
                    try:
                        return function(*args, **kwargs)
                    finally:
                        row.update(after=np.geterr(), handler_after=np.geterrcall())
                        with observer.lock:
                            observer.rows.append(row)
                return super().submit(outside_task)

        monkeypatch.setattr(parallel_module, "ThreadPoolExecutor", ObservedExecutor)

    def assert_restored_and_joined(self):
        assert self.rows
        for row in self.rows:
            assert row["before"] == row["after"]
            assert row["handler_before"] is row["handler_after"]
        assert not set(self.initialized).intersection(t.ident for t in threading.enumerate())


def caught(call):
    try:
        call()
    except BaseException as error:
        return error
    pytest.fail("Expected a caller-policy exception")


def assert_same_fit(left, right, X):
    for first, second in zip(left.subset_indices_, right.subset_indices_):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(left.selected_indices_, right.selected_indices_)
    np.testing.assert_array_equal(left.predict(X), right.predict(X))
    np.testing.assert_allclose(left.consensus_scores_, right.consensus_scores_, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(left.decision_function(X), right.decision_function(X), rtol=2e-8, atol=2e-10)
    assert all(base.kernel_computation_ == "guarded_sklearn_rel_v2" for base in right.estimators_)
