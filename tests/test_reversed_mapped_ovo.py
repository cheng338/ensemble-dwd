"""Check parent preparation and actual pair fits for reverse-sliced mappings."""
import os

import numpy as np
import pytest
from joblib import cpu_count, parallel_config
from sklearn.base import BaseEstimator, ClassifierMixin

from ensemble_dwd import OvOClassifier
import ensemble_dwd.multiclass as multiclass


def _check_snapshot(result, original):
    # Fail in the parent if this implementation would dispatch a mapped view.
    assert type(result) is np.ndarray and result.flags.owndata
    np.testing.assert_array_equal(result, original)


@pytest.mark.parametrize('mode', ['r', 'r+'])
def test_reversed_process_preparation(tmp_path, mode):
    data = np.arange(60.).reshape(12, 5)
    path = tmp_path / 'data.npy'
    np.save(path, data)
    backing = np.load(path, mmap_mode=mode)
    a = np.asarray(backing)
    labels = a[::-1, 0]
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        for view in (backing[::-1], a[:, ::-1], a[::-2, ::-1]):
            X, y, transpose = multiclass._pair_worker_inputs(view, labels, None)
            _check_snapshot(X, view)
            _check_snapshot(y, labels)
            assert not transpose
            X, y, transpose = multiclass._pair_worker_inputs(view, labels, 1)
            assert X is view and y is labels and not transpose
        ordinary_labels = data[::-1, 0]
        for ordinary in (backing, a[::2], a[::2, :1][:, ::-1], data[::-1]):
            X, y, transpose = multiclass._pair_worker_inputs(ordinary, ordinary_labels, None)
            assert X is ordinary and y is ordinary_labels and not transpose
    with parallel_config(backend='threading', n_jobs=2):
        view = a[::-1, ::-1]
        X, y, transpose = multiclass._pair_worker_inputs(view, labels, None)
        assert X is view and y is labels and not transpose


class _RecordingBinary(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.pid_ = os.getpid()
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.classes_ = np.unique(y)
        self.centers_ = [X[y == cls, 0].mean() for cls in self.classes_]
        return self

    def predict(self, X):
        return self.classes_[np.abs(X[:, :1] - self.centers_).argmin(axis=1)]


@pytest.mark.parametrize(
    ('mode', 'backend', 'configured_jobs', 'requested_jobs', 'process_workers'),
    [
        ('r', 'loky', 2, None, True),
        ('r+', 'loky', 2, None, True),
        pytest.param('r', 'loky', 1.9, None, False, id='inherited-fraction-serial'),
        ('r', 'loky', '1', None, False),
        ('r', 'loky', '2', None, True),
        ('r', 'loky', -cpu_count(), None, False),
        ('r', 'loky', '2', 1, False),
        ('r', 'loky', '1', 2, True),
        ('r', 'threading', '2', None, False),
    ],
)
def test_reversed_public_ovo(tmp_path, mode, backend, configured_jobs,
                             requested_jobs, process_workers, monkeypatch):
    rows = np.arange(18)
    labels = rows % 3
    features = np.column_stack([20 * labels + rows, rows, -rows]).astype(float)
    np.save(tmp_path / 'X.npy', features[::-1, ::-1])
    np.save(tmp_path / 'y.npy', labels[::-1])
    X = np.asarray(np.load(tmp_path / 'X.npy', mmap_mode=mode))[::-1, ::-1]
    y = np.load(tmp_path / 'y.npy', mmap_mode=mode)[::-1]
    reference = OvOClassifier(_RecordingBinary(), n_jobs=1).fit(features, labels)
    prepare = multiclass._pair_worker_inputs
    prepared = []
    dispatched = []

    def checked_prepare(X, y, n_jobs):
        worker_X, worker_y, transpose = prepare(X, y, n_jobs)
        if process_workers:
            _check_snapshot(worker_X, X)
            _check_snapshot(worker_y, y)
        else:
            assert worker_X is X and worker_y is y
        assert not transpose
        prepared.append((worker_X, worker_y))
        return worker_X, worker_y, transpose

    dispatch = multiclass.Parallel.__call__

    def checked_dispatch(parallel, tasks):
        # Check every actual task in the parent before joblib starts a worker.
        tasks = list(tasks)
        assert len(prepared) == 1 and len(tasks) == len(reference.pairs_)
        safe_X, safe_y = prepared[0]
        for function, args, _ in tasks:
            assert function is multiclass._fit_pair
            assert args[1] is safe_X and args[2] is safe_y
        dispatched.append(len(tasks))
        return dispatch(parallel, tasks)

    monkeypatch.setattr(multiclass, '_pair_worker_inputs', checked_prepare)
    monkeypatch.setattr(multiclass.Parallel, '__call__', checked_dispatch)
    with parallel_config(backend=backend, n_jobs=configured_jobs,
                         inner_max_num_threads=1 if backend == 'loky' else None):
        # Verify the gate even before public validation or job creation.
        checked_prepare(X, y, requested_jobs)
        prepared.clear()
        model = OvOClassifier(_RecordingBinary(), n_jobs=requested_jobs).fit(X, y)
    assert dispatched == [len(reference.pairs_)]
    assert model.pairs_ == reference.pairs_
    for child, expected in zip(model.estimators_, reference.estimators_):
        assert (child.pid_ != os.getpid()) == process_workers
        np.testing.assert_array_equal(child.X_, expected.X_)
        np.testing.assert_array_equal(child.y_, expected.y_)
    for actual, expected in zip(model.pair_indices_, reference.pair_indices_):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(model.vote_counts(features), reference.vote_counts(features))
    np.testing.assert_array_equal(model.predict(features), reference.predict(features))
    np.testing.assert_array_equal(X, features)
    np.testing.assert_array_equal(y, labels)
