"""Checkpoint entry imports preserve estimator hooks and durable recovery.

Only dispatch and runtime records are controlled. Entry, folds, receipts and
leases remain real; task counts here do not establish distinct worker readiness.
The companion process tests exercise actual loky serialization and imports.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from threadpoolctl import threadpool_info

from test_execution_runtime_roles import cp
from runtime_test_support import synthetic_runtime


@dataclass
class DispatchState:
    runtime: dict = field(default_factory=synthetic_runtime)
    role: str = 'parent'
    events: list = field(default_factory=list)
    batches: list = field(default_factory=list)
    fingerprint_roles: list = field(default_factory=list)
    import_count: int = 0
    fail_import_at: int = 0
    work_count: int = 0
    fail_work_at: int = 0
    error: object = None
    before_work: object = None
    after_fit: object = None
    import_result_transform: object = None

    def hook(self, name):
        self.events.append((self.role, name))
        if self.role == 'import':
            raise AssertionError('Import-only work touched an estimator hook')
        if self.role == 'probe' and self.events.count(('probe', name)) > 1:
            raise AssertionError('Entry repeated an estimator-bearing hook')


STATE = None


class ObservableClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, offset=0., random_state=31):
        self.offset = offset
        self.random_state = random_state

    def get_params(self, deep=True):
        STATE.hook('get_params')
        return super().get_params(deep=deep)

    def __getstate__(self):
        STATE.hook('serialize')
        return dict(self.__dict__)

    def fit(self, X, y):
        STATE.hook('fit')
        self.classes_, self.n_features_in_ = np.unique(y), np.asarray(X).shape[1]
        self.converged_, self.criterion_reached_ = True, True
        self.stopping_reason_, self.n_iter_ = 'fixed', 1
        if STATE.after_fit is not None:
            STATE.after_fit()
        return self

    def predict(self, X):
        STATE.hook('predict')
        return (np.asarray(X)[:, 0] > self.offset).astype(int)


@pytest.fixture
def dispatch(monkeypatch):
    state = DispatchState()
    monkeypatch.setitem(globals(), 'STATE', state)

    def fingerprint(**kwargs):
        state.fingerprint_roles.append(state.role)
        return deepcopy(state.runtime)
    monkeypatch.setattr(cp, 'runtime_fingerprint', fingerprint)

    class Parallel:
        def __init__(self, **options):
            self.options = options
            self.active = False

        def __enter__(self):
            assert not self.active
            self.active = True
            return self

        def __exit__(self, *unused):
            assert self.active
            self.active = False

        def __call__(self, pending):
            tasks = list(pending)
            kinds = [('probe' if fn is cp._execution_runtime_probe else
                      'import' if fn is cp._execution_import_probe else 'work')
                     for fn, _, _ in tasks]
            state.batches.append({'kinds': kinds, 'options': dict(self.options)})

            def execute():
                result, old_role = [], state.role
                try:
                    for (fn, args, kwargs), kind in zip(tasks, kinds):
                        state.role = kind
                        if kind == 'probe':
                            # Deterministic hook observation, not a process claim.
                            args = (pickle.loads(pickle.dumps(args[0])), *args[1:])
                        elif kind == 'import':
                            assert args == () and kwargs == {}
                            state.import_count += 1
                            if state.import_count == state.fail_import_at:
                                raise state.error
                        else:
                            state.work_count += 1
                            if state.work_count == state.fail_work_at:
                                raise state.error
                            if state.before_work is not None:
                                state.before_work()
                        result.append(fn(*args, **kwargs))
                    if 'import' in kinds and state.import_result_transform is not None:
                        result = state.import_result_transform(result, kinds)
                    return result
                finally:
                    state.role = old_role

            if self.options.get('return_as') == 'generator':
                assert self.active
                assert kinds in (['import'] * (self.options['n_jobs'] - 1), ['probe'])
                def stream():
                    # Controlled lazy delivery, not evidence of async overlap.
                    assert self.active
                    yield from execute()
                return stream()
            assert not self.active and all(kind == 'work' for kind in kinds)
            return execute()
    monkeypatch.setattr(cp, 'Parallel', Parallel)
    return state


def make_store(tmp_path, *, resume=False, jobs=6):
    X = np.array([[-4.], [-3.], [-2.], [-1.], [1.], [2.], [3.], [4.]])
    y = (X[:, 0] > 0).astype(int)
    folds = [(np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])),
             (np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6]))]
    return cp.CheckpointedCV(tmp_path / 'run', ObservableClassifier(), X, y, folds,
                            jobs=jobs, resume=resume, reuse_equivalent_folds=False,
                            error_policy='raise')


def receipts(store):
    return {path.name: path.read_bytes() for path in (store.folder / 'folds').glob('*.json')}


def native_state():
    return (dict(os.environ), np.geterr(), sorted(
        (row['filepath'], row['user_api'], row['num_threads']) for row in threadpool_info()))


def assert_released(store):
    assert store.lease is None
    with cp._Lease(store.folder / 'session.lock'):
        pass


@pytest.mark.parametrize('work', ['unused', 'artifact-only', 2, 6, 8])
def test_entry_imports_do_not_multiply_estimator_hooks_or_repeat_on_search(
        tmp_path, dispatch, work):
    store = make_store(tmp_path)
    before = native_state()
    with store:
        assert dispatch.events.count(('probe', 'serialize')) == 1
        assert dispatch.events.count(('probe', 'get_params')) == 1
        assert [batch['kinds'] for batch in dispatch.batches] == [
            ['import'] * 5, ['probe']]
        if work == 'artifact-only':
            store.write_artifact('metadata', {'x': 1})
            assert store.read_artifact('metadata') == {'x': 1}
        elif isinstance(work, int):
            grid = {'offset': [0., .1, .2, .3][:work // 2]}
            result = store.search(grid)
            assert result.checkpoint_summary_['new_folds'] == work
            assert dispatch.import_count == 5
            assert dispatch.events.count(('probe', 'serialize')) == 1
            assert dispatch.events.count(('probe', 'get_params')) == 1
            assert all(batch['options'] == (
                {'n_jobs': 6, 'pre_dispatch': 6, 'batch_size': 1}
                | ({'return_as': 'generator'} if batch['kinds'] and batch['kinds'][0] in ('import', 'probe') else {}))
                for batch in dispatch.batches)
            frozen = receipts(store)
            dispatch.batches.clear()
            dispatch.events.clear()
            replay = store.search(grid)
            assert replay.checkpoint_summary_['new_folds'] == 0
            assert replay.checkpoint_summary_['reused_folds'] == work
            assert sum(len(batch['kinds']) for batch in dispatch.batches) == 0
            assert not any(name in ('fit', 'predict', 'serialize') for _, name in dispatch.events)
            assert receipts(store) == frozen
        assert 'import' not in dispatch.fingerprint_roles
        assert not any(role == 'import' for role, _ in dispatch.events)
        assert dispatch.import_count == 5
        assert all(batch['options'] == (
            {'n_jobs': 6, 'pre_dispatch': 6, 'batch_size': 1}
            | ({'return_as': 'generator'} if batch['kinds'] and batch['kinds'][0] in ('import', 'probe') else {}))
            for batch in dispatch.batches)
    assert native_state() == before
    assert_released(store)


def test_serial_search_has_no_probe_or_import_dispatch(tmp_path, dispatch):
    with make_store(tmp_path, jobs=1) as store:
        assert store.search({'offset': [0.]}).checkpoint_summary_['new_folds'] == 2
    assert dispatch.batches == []
    assert not any(role in ('probe', 'import') for role, _ in dispatch.events)
    assert_released(store)


@pytest.mark.parametrize('pending', [0, 1, 2])
def test_resume_preserves_saved_folds_without_probe_or_import_tasks(tmp_path, dispatch, pending):
    store = make_store(tmp_path)
    if pending:
        # Four tasks; commit the first two or three before interruption.
        dispatch.fail_work_at = 5 - pending
        dispatch.error = KeyboardInterrupt('controlled interruption after durable receipts')
        with pytest.raises(KeyboardInterrupt):
            with store:
                store.search({'offset': [0., .5]})
    else:
        with store:
            store.search({'offset': [0., .5]})
    frozen = receipts(store)
    assert len(frozen) == 4 - pending
    assert_released(store)
    dispatch.fail_work_at = 0
    dispatch.batches.clear()
    dispatch.events.clear()
    import_count = dispatch.import_count
    with make_store(tmp_path, resume=True) as resumed:
        result = resumed.search({'offset': [0., .5]})
        assert result.checkpoint_summary_['new_folds'] == pending
        assert result.checkpoint_summary_['reused_folds'] == 4 - pending
        assert all(kind == 'work' for batch in dispatch.batches for kind in batch['kinds'])
        assert dispatch.import_count == import_count
        assert sum(name == 'fit' for _, name in dispatch.events) == pending
        assert all(receipts(resumed)[name] == data for name, data in frozen.items())
    assert_released(resumed)


@pytest.mark.parametrize('failure', [
    'exception', 'KeyboardInterrupt', 'non-none', 'missing-result', 'extra-result'])
def test_import_failure_refuses_before_identity_and_releases_for_retry(tmp_path, dispatch, failure):
    dispatch.error = (KeyboardInterrupt('controlled interruption before any fold')
                      if failure == 'KeyboardInterrupt' else
                      ImportError('controlled import-only unpickle/import refusal'))
    if failure in ('exception', 'KeyboardInterrupt'):
        dispatch.fail_import_at = 1
    else:
        def malformed(rows, kinds):
            if failure == 'missing-result':
                return rows[:-1]
            if failure == 'extra-result':
                return rows + [None]
            rows[kinds.index('import')] = 'unexpected import result'
            return rows
        dispatch.import_result_transform = malformed
    store = make_store(tmp_path)
    before = native_state()
    expected_error = type(dispatch.error) if failure in ('exception', 'KeyboardInterrupt') else cp.CheckpointError
    with pytest.raises(expected_error) as caught:
        with store:
            pytest.fail('An import failure must prevent entry')
    if failure in ('exception', 'KeyboardInterrupt'):
        assert caught.value is dispatch.error
    else:
        assert str(caught.value) == 'Invalid entry import preparation result'
    assert receipts(store) == {}
    assert not any(name in ('fit', 'predict') for _, name in dispatch.events)
    assert native_state() == before
    assert_released(store)
    assert not (store.folder / 'identity.json').exists()
    assert not (store.folder / 'owner.json').exists()
    assert not list((store.folder / 'sessions').glob('*.json'))
    dispatch.fail_import_at = 0
    dispatch.import_result_transform = None
    dispatch.events.clear()
    dispatch.batches.clear()
    with make_store(tmp_path) as retry:
        assert retry.search({'offset': [0.]}).checkpoint_summary_['new_folds'] == 2
        assert sum(name == 'fit' for _, name in dispatch.events) == 2
    assert_released(retry)


@pytest.mark.parametrize('when,fits', [('before', 0), ('after', 1)])
def test_entry_imports_do_not_replace_actual_worker_runtime_guards(tmp_path, dispatch, when, fits):
    store = make_store(tmp_path)
    with store:
        def mutate():
            dispatch.runtime.update(files_sha256='f' * 64)
        if when == 'before':
            dispatch.before_work = mutate
        else:
            dispatch.after_fit = mutate
        with pytest.raises(cp.CheckpointError, match='Worker runtime'):
            store.search({'offset': [0.]})
        assert not receipts(store)
        attempts = [json.loads(path.read_text())['payload']
                    for path in (store.folder / 'attempts').glob('*.json')]
        assert len(attempts) == 1 and attempts[0]['state'] == 'failed'
    assert sum(name == 'fit' for _, name in dispatch.events) == fits
    assert_released(store)
