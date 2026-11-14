"""The paired DWD's ordered reducer preserves serial ensemble fitting.

This is a small real-solver propagation check, not a performance benchmark.
No prototype module, workspace path or separately installed test copy is used.
"""
import importlib
import importlib.util
import platform
import struct
import sys
import sysconfig

import numpy as np
import pytest

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD


_TIMING_FIELDS = frozenset({
    'cholesky_setup_seconds', 'eigenvalue_check_seconds', 'elapsed_seconds',
    'initial_state_check_seconds', 'optimization_seconds', 'setup_seconds',
    'spectral_setup_seconds', 'total_seconds',
})


@pytest.fixture
def ordered(monkeypatch):
    name = 'dwd._ordered_gradient_reduction'
    if importlib.util.find_spec(name) is None:
        pytest.skip('This check requires the paired development DWD dependency.')
    # Production loads the facade from the feature helper, whose owner and
    # original callback tokens must already exist when the facade is imported.
    importlib.import_module('dwd._linear_feature_diagnostics')
    facade = importlib.import_module(name)
    monkeypatch.setattr(facade, '_LOAD_ATTEMPTED', False)
    monkeypatch.setattr(facade, '_ACCEL', None)
    monkeypatch.setattr(facade, '_ENTRY', None)
    return facade


def _data():
    rows, columns = np.arange(48)[:, None], np.arange(12)[None, :]
    X = ((rows*17+columns*31+rows*columns) % 257-128).astype(float)/256.0
    y = np.tile(np.array(['negative', 'positive']), 24)
    return X, y


def _fit(X, y):
    return EnsembleKernelDWD(
        M=1, theta=1., sv_selection_ratio=1., base_jobs=1,
        kernel='linear', backend='lbfgs', lambd=.25, q=1.,
        max_iter=4, stopping='fixed', random_state=17,
        store_base_estimators=True,
    ).fit(X, y)


def _assert_same(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, np.generic):
        _assert_same(left.item(), right.item())
    elif isinstance(left, dict):
        left_keys, right_keys = set(left)-_TIMING_FIELDS, set(right)-_TIMING_FIELDS
        assert left_keys == right_keys
        for key in left_keys:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    elif type(left) is float:
        assert type(right) is float
        assert struct.pack('=d', left) == struct.pack('=d', right)
    else:
        assert type(left) is type(right) and left == right


def _assert_fits_same(left, right, X):
    for name in ('classes_', 'selected_indices_', 'consensus_scores_',
                 'n_iter_', 'converged_', 'subset_indices_', 'base_diagnostics_'):
        _assert_same(getattr(left, name), getattr(right, name))
    _assert_same(left.decision_function(X), right.decision_function(X))
    _assert_same(left.predict(X), right.predict(X))
    learners_left = left.estimators_ + [left.final_estimator_]
    learners_right = right.estimators_ + [right.final_estimator_]
    assert len(learners_left) == len(learners_right) == 2
    for a, b in zip(learners_left, learners_right):
        assert a.diagnostics_['rkhs_geometry'] == 'linear_features'
        assert b.diagnostics_['rkhs_geometry'] == 'linear_features'
        for name in ('dual_coef_', 'intercept_', 'objective_history_',
                     'n_iter_', 'converged_', 'final_objective_', 'diagnostics_'):
            _assert_same(getattr(a, name), getattr(b, name))


def test_absent_ordered_binary_preserves_base_and_final_fits(ordered, monkeypatch):
    X, y = _data()
    # Simulate the documented permanent portable fallback, then an absent
    # optional module on first use. No imported dependency is modified on disk.
    with monkeypatch.context() as portable:
        portable.setattr(ordered, '_LOAD_ATTEMPTED', True)
        portable.setattr(ordered, '_ACCEL', None)
        expected = _fit(X, y)
    monkeypatch.setitem(sys.modules, 'dwd._ordered_reduction_accel', None)
    actual = _fit(X, y)
    assert ordered._LOAD_ATTEMPTED is True and ordered._ACCEL is None
    _assert_fits_same(actual, expected, X)


def test_actual_native_reduction_reaches_both_base_and_final_fit(ordered, monkeypatch):
    if (sys.implementation.name != 'cpython'
            or sys.platform not in ('win32', 'linux')
            or platform.machine().lower() not in ('amd64', 'x86_64')
            or sys.maxsize <= 2**32
            or sysconfig.get_config_var('Py_GIL_DISABLED') not in (None, 0)):
        pytest.skip('Native ordered-gradient platform contract does not apply.')
    name = 'dwd._ordered_reduction_accel'
    if importlib.util.find_spec(name) is None:
        pytest.skip('Portable DWD build has no ordered-gradient binary.')
    # A present binary on a declared native runtime must import and qualify;
    # a failing entry is not converted into a skip.
    native = importlib.import_module(name)
    entry = native.reduce_rows
    X, y = _data()
    with monkeypatch.context() as portable:
        portable.setattr(ordered, '_LOAD_ATTEMPTED', True)
        portable.setattr(ordered, '_ACCEL', None)
        expected = _fit(X, y)
    witnessed = set()
    old_profile = sys.getprofile()

    def profile(frame, event, arg):
        if event == 'c_call' and arg is entry:
            cursor = frame
            while cursor is not None:
                owner = cursor.f_locals.get('self')
                if type(owner) is KernGDWD:
                    witnessed.add(id(owner))
                    break
                cursor = cursor.f_back

    try:
        sys.setprofile(profile)
        actual = _fit(X, y)
    finally:
        sys.setprofile(old_profile)
    learners = actual.estimators_ + [actual.final_estimator_]
    assert witnessed == {id(model) for model in learners}
    assert ordered._ACCEL is native and ordered._ENTRY is entry
    _assert_fits_same(actual, expected, X)
