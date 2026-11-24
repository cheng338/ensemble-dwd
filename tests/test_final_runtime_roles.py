"""Final-role authority remains exact within each execution.

These are contract tests using the existing observed native-count fixture.
Actual native loading and cross-process CV/resume are covered separately.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

from test_final_recovery import parent, reset_phase_events, event_counts
from test_phase_threads import PhaseSurface, observed_counts, cv_checkpoint


def extra_observer(monkeypatch, *, enabled):
    original = cv_checkpoint.runtime_fingerprint
    state = {'enabled': enabled}

    def observed(**kwargs):
        runtime = deepcopy(original(**kwargs))
        if state['enabled']:
            pool = deepcopy(runtime['native_pools'][0])
            pool.update(filepath=str(Path(__file__).resolve().with_name('__synthetic_extra_native__')),
                        sha256='e' * 64,
                        prefix='runtime-extra', user_api='blas')
            runtime['native_pools'].append(pool)
            # Match the full observer's stable path order, including when this
            # synthetic path sorts before real pools; retain every row and field.
            runtime['native_pools'].sort(key=lambda row: row['filepath'])
            runtime['files_sha256'] = 'e' * 64
            runtime['file_count'] += 1
        return runtime

    monkeypatch.setattr(cv_checkpoint, 'runtime_fingerprint', observed)
    return state


def payload(path):
    return json.loads(path.read_text(encoding='utf-8'))['payload']


def test_final_role_extra_is_recorded_and_completed_clean_replay_is_zero_work(
        parent, monkeypatch):
    state = extra_observer(monkeypatch, enabled=True)
    with threadpool_limits(limits=3):
        before = observed_counts()
        result = parent.recover()
        assert observed_counts() == before
    assert result['model'] is not None
    assert event_counts(parent.w) == (6, 1, 1)
    identity_path = parent.w.root / 'attempt/identity.json'
    identity = payload(identity_path)['identity']
    source = payload(parent.source / 'cv-checkpoints/identity.json')['identity']
    assert identity['runtime'] != source['runtime']
    assert 'runtime-extra' in json.dumps(identity['runtime'])
    committed = {str(p): p.read_bytes() for p in
                 (parent.w.root / 'attempt').rglob('*.json')}
    barrier = parent.barrier.read_bytes()
    state['enabled'] = False
    replay = parent.recover(resume=True)
    assert replay['model'] is None and replay['resumed_completed'] is True
    assert replay['completed'] == result['completed']
    assert event_counts(parent.w) == (6, 1, 1)
    assert parent.barrier.read_bytes() == barrier
    for name, contents in committed.items():
        assert Path(name).read_bytes() == contents
    parent.unchanged()


def test_incomplete_attempt_freezes_final_role_and_new_attempt_is_explicit(
        parent, monkeypatch):
    state = extra_observer(monkeypatch, enabled=True)
    original_fit = PhaseSurface.fit
    error = RuntimeError('controlled final-role failure before scoring')

    def failed(self, X, y):
        original_fit(self, X, y)
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(PhaseSurface, 'fit', failed)
        with pytest.raises(RuntimeError) as caught:
            parent.recover()
        assert caught.value is error
    assert event_counts(parent.w) == (6, 1, 0)
    assert not parent.barrier.exists()
    identity_path = parent.w.root / 'attempt/identity.json'
    original_identity = identity_path.read_bytes()
    state['enabled'] = False
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match=r'^Checkpoint identity mismatch: .*identity\.json$'):
        parent.recover(resume=True)
    assert identity_path.read_bytes() == original_identity
    assert event_counts(parent.w) == (6, 1, 0)
    assert not parent.barrier.exists()
    result = parent.recover('explicit-clean-attempt')
    assert result['model'] is not None
    assert event_counts(parent.w) == (6, 2, 1)
    assert identity_path.read_bytes() == original_identity
    parent.unchanged()


def test_pool_added_during_final_fit_refuses_before_scoring(parent, monkeypatch):
    state = extra_observer(monkeypatch, enabled=False)
    original_fit = PhaseSurface.fit

    def changed(self, X, y):
        result = original_fit(self, X, y)
        state['enabled'] = True
        return result

    monkeypatch.setattr(PhaseSurface, 'fit', changed)
    with threadpool_limits(limits=3):
        before = observed_counts()
        with pytest.raises(cv_checkpoint.CheckpointError,
                           match='Final recovery runtime changed within this execution attempt'):
            parent.recover()
        assert observed_counts() == before
    assert event_counts(parent.w) == (6, 1, 0)
    assert not parent.barrier.exists()
    assert not (parent.w.root / 'attempt/completed.json').exists()
    parent.unchanged()


def test_extra_pool_does_not_allow_retry_of_ambiguous_held_out_score(
        parent, monkeypatch):
    state = extra_observer(monkeypatch, enabled=True)
    original_predict = PhaseSurface.predict
    error = RuntimeError('controlled ambiguous held-out scoring')

    def failed(self, X):
        result = original_predict(self, X)
        if len(X) == len(parent.w.test_X):
            raise error
        return result

    with monkeypatch.context() as patch:
        patch.setattr(PhaseSurface, 'predict', failed)
        with pytest.raises(RuntimeError) as caught:
            parent.recover()
        assert caught.value is error
    assert event_counts(parent.w) == (6, 1, 1)
    assert parent.barrier.exists()
    barrier = parent.barrier.read_bytes()
    for name, resume, extra in [('attempt', True, True),
                                ('new-clean-attempt', False, False)]:
        state['enabled'] = extra
        with pytest.raises(cv_checkpoint.CheckpointError,
                           match='^Held-out scoring has already been reserved;'):
            parent.recover(name, resume=resume)
        assert event_counts(parent.w) == (6, 1, 1)
        assert parent.barrier.read_bytes() == barrier
    parent.unchanged()
