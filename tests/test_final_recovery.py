"""Final-only recovery preserves accepted selection and held-out-use guards.

Parents are newly computed tiny CV fixtures. Recovery retains their exact folds
and accepted configured pipeline; only the explicit final thread budget changes.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from path_support import extended_path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

# Import helpers only. No sibling test functions or unittest TestCase classes
# are imported or collected; this module owns its fixture reset explicitly.
from test_final_thread_preflight import Workflow
from test_phase_threads import (
    PhaseSurface, PhaseTransform, observed_counts, observed_runtime, synthetic_native_pools,
    adaptive_search, cv_checkpoint, phase_threads, new_run_directory,
)
import final_recovery
import evaluation_reporting
import work_reporting


@pytest.fixture(autouse=True)
def reset_phase_events():
    previous = (PhaseSurface.fits, PhaseSurface.predictions,
                PhaseSurface.fail_final, PhaseTransform.fits)
    PhaseSurface.fits, PhaseSurface.predictions = [], []
    PhaseSurface.fail_final, PhaseTransform.fits = None, []
    try:
        yield
    finally:
        (PhaseSurface.fits, PhaseSurface.predictions,
         PhaseSurface.fail_final, PhaseTransform.fits) = previous


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_counts(workflow):
    return (workflow.cv_count, workflow.final_count, workflow.held_out_count)


def assert_completed_reports(result, training_labels, test_labels, expected_prediction):
    """Inspect retained output only; do not obtain another model prediction."""
    completed = result['completed']
    refit = completed['refit']
    diagnostics = cv_checkpoint.diagnostics(result['model'], training_labels)
    assert refit['state'] == 'completed'
    assert refit['diagnostics'] == diagnostics
    assert refit['base_cache'] == work_reporting.summarize_cache_diagnostics(diagnostics)
    assert refit['numerical_checks'] == evaluation_reporting.numerical_check_summary(diagnostics)
    assert refit['selection_composition'] == evaluation_reporting.selection_composition(
        diagnostics, training_labels)
    assert completed['held_out_classification'] == evaluation_reporting.classification_report(
        test_labels, expected_prediction)
    assert completed['held_out_classification']['accuracy'] == completed['held_out_accuracy']
    assert refit['native_runtime']['state'] == 'completed'
    assert refit['native_runtime']['restored'] is True
    for key in ('fit_seconds', 'pipeline_fit_seconds', 'fit_management_seconds',
                'scoring_seconds', 'held_out_prediction_seconds',
                'held_out_accuracy_scoring_seconds'):
        assert np.isfinite(completed[key]) and completed[key] >= 0.
    assert completed['fit_seconds'] == pytest.approx(
        completed['pipeline_fit_seconds'] + completed['fit_management_seconds'])
    assert completed['scoring_seconds'] == pytest.approx(
        completed['held_out_prediction_seconds'] + completed['held_out_accuracy_scoring_seconds'])
    path = Path(extended_path(result['attempt_dir'])) / 'refits' / (completed['invocation'] + '.json')
    record = json.loads(path.read_text())
    assert record['payload'] == refit
    assert record['sha256'] == completed['refit_receipt_sha256']


def protect_cv(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Final-only recovery attempted CV work')
    monkeypatch.setattr(cv_checkpoint.CheckpointedCV, 'search', forbidden)
    monkeypatch.setattr(cv_checkpoint, '_fit_fold', forbidden)


class RecoveryFixture:
    def __init__(self, workflow):
        self.w = workflow
        self.source = workflow.root / 'run'
        self.finalization = self.source / 'cv-checkpoints' / 'finalization'
        self.handoff = self.finalization / 'handoff.json'
        self.barrier = self.finalization / 'scoring-started.json'
        self.model = clone(workflow.pipeline).set_params(
            ensemble__lambd=4., ensemble__gamma=4.)
        self.original_cv = workflow.hashes(cv_only=True)
        self.original_handoff = self.handoff.read_bytes()

    def recover(self, name='attempt', *, budget=2, resume=False, **overrides):
        args = dict(final_pipeline=self.model, X_train=self.w.X, y_train=self.w.y,
                    X_test=self.w.test_X, y_test=self.w.test_y,
                    source_run=self.source, attempt_dir=self.w.root / name,
                    final_native_threads=budget, resume=resume)
        args.update(overrides)
        with redirect_stdout(io.StringIO()):
            return final_recovery.recover_final_fit(**args)

    def unchanged(self):
        assert self.w.hashes(cv_only=True) == self.original_cv
        assert self.handoff.read_bytes() == self.original_handoff


@pytest.fixture
def parent(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, monkeypatch)
    # A real accepted tiny counting-estimator CV is committed. The cap changes
    # only after its six folds, so early preflight succeeds but final guard fails.
    workflow.cap_after_cv = 2
    with pytest.raises(cv_checkpoint.CheckpointError, match='wrong active budget'):
        workflow.run(budget=4)
    assert event_counts(workflow) == (6, 0, 0)
    assert (workflow.root / 'run/cv-checkpoints/finalization/handoff.json').is_file()
    workflow.cap_after_cv = None
    result = RecoveryFixture(workflow)
    assert result.original_cv and any('folds/' in p for p in result.original_cv)
    protect_cv(monkeypatch)
    return result


def test_persisted_handoff_roundtrip_preserves_original_typed_heldout_identity(parent):
    folder = parent.source / 'cv-checkpoints'
    parent_record = json.loads((folder / 'identity.json').read_text())['payload']
    identity = parent_record['identity']
    provenance = dict(identity['provenance'][1])
    typed_heldout = dict(provenance['data'][1])['held_out']
    original = cv_checkpoint.data_fingerprint(parent.w.test_X, parent.w.test_y)
    assert typed_heldout == cv_checkpoint._stable(original)
    raw = json.loads(parent.handoff.read_text())
    assert raw['payload']['held_out_identity'] == typed_heldout
    assert raw['payload']['held_out'] == original
    # Exercise the actual persisted reader, after JSON has sorted ordinary
    # object keys. Typed identity order is retained separately in the parent.
    with cv_checkpoint._Lease(folder / 'session.lock'), threadpool_limits(limits=1):
        loaded, handoff_digest = final_recovery._handoff(
            folder, identity, parent_record['identity_sha256'])
    assert loaded == raw['payload'] and handoff_digest == raw['sha256']
    assert event_counts(parent.w) == (6, 0, 0)
    assert not parent.barrier.exists()
    parent.unchanged()


def test_rechecksummed_heldout_tamper_does_not_change_parent_authority(parent):
    record = json.loads(parent.handoff.read_text())
    record['payload']['held_out']['X']['sha256'] = '0' * 64
    canonical = json.dumps(record['payload'], sort_keys=True, separators=(',', ':'),
                           allow_nan=False).encode('utf-8')
    record['sha256'] = hashlib.sha256(canonical).hexdigest()
    parent.handoff.write_text(json.dumps(record, sort_keys=True), encoding='utf-8')
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match='^Handoff held-out identity differs from its parent$'):
        parent.recover()
    assert event_counts(parent.w) == (6, 0, 0)
    assert parent.w.hashes(cv_only=True) == parent.original_cv
    assert not parent.barrier.exists()


def test_corrected_budget_fits_once_retains_cv_and_restores_native_scope(parent, record_property):
    arrays = [x.copy() for x in (parent.w.X, parent.w.y, parent.w.test_X, parent.w.test_y)]
    with threadpool_limits(limits=3):
        before = observed_counts()
        result = parent.recover()
        assert observed_counts() == before
    assert event_counts(parent.w) == (6, 1, 1)
    assert result['model'] is not None and result['completed']['held_out_accuracy'] == 1.
    assert result['model'].get_params()['ensemble__lambd'] == 4.
    assert all(value == 2 for path, value in PhaseSurface.fits[-1][1])
    assert all(value == 1 for path, value in PhaseSurface.predictions[-1][1])
    assert parent.barrier.is_file()
    assert_completed_reports(result, parent.w.y, parent.w.test_y, parent.w.test_y)
    parent.unchanged()
    for old, current in zip(arrays, (parent.w.X, parent.w.y, parent.w.test_X, parent.w.test_y)):
        np.testing.assert_array_equal(current, old)
    record_property('final_recovery_work', json.dumps(dict(
        cv_fits=6, new_final_fits=1, held_out_predictions=1,
        requested_final_threads=2, original_cv_sha256=parent.original_cv,
        handoff_sha256=digest(parent.handoff), caller_restored=True), sort_keys=True))


def test_completed_replay_returns_reports_not_estimator_and_does_no_work(parent):
    first = parent.recover()
    completion = parent.w.root / 'attempt/completed.json'
    committed = completion.read_bytes()
    before = deepcopy((PhaseSurface.fits, PhaseSurface.predictions, parent.w.scopes))
    parent.w.cap = 1  # Completed replay must not perform a new budget probe.
    replay = parent.recover(resume=True)
    assert replay['resumed_completed'] is True and replay['model'] is None
    assert replay['completed'] == first['completed']
    assert completion.read_bytes() == committed
    assert (PhaseSurface.fits, PhaseSurface.predictions, parent.w.scopes) == before
    parent.unchanged()


@pytest.mark.parametrize('field', ['X_train', 'y_train', 'X_test', 'y_test'])
def test_changed_data_refuses_before_fit_or_prediction(parent, field):
    values = dict(X_train=parent.w.X, y_train=parent.w.y,
                  X_test=parent.w.test_X, y_test=parent.w.test_y)
    changed = values[field].copy()
    changed.flat[0] += 1
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match='^Final recovery data differs from the accepted handoff$'):
        parent.recover(**{field: changed})
    assert event_counts(parent.w) == before
    assert not parent.barrier.exists()
    parent.unchanged()


@pytest.mark.parametrize('change', ['parameter', 'typed-parameter', 'preprocessor'])
def test_changed_configured_pipeline_refuses(parent, change):
    model = clone(parent.model)
    if change == 'parameter':
        model.set_params(ensemble__lambd=1.)
    elif change == 'typed-parameter':
        model.set_params(ensemble__lambd=np.float64(4.))
    else:
        model.set_params(transform='passthrough')
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match='^Supplied configured pipeline differs from the accepted final model$'):
        parent.recover(final_pipeline=model)
    assert event_counts(parent.w) == before
    parent.unchanged()


@pytest.mark.parametrize('damage', ['truncated', 'checksum', 'duplicate', 'nonfinite'])
def test_corrupt_handoff_is_not_recovered_from_public_mirrors(parent, damage):
    envelope = json.loads(parent.handoff.read_text(encoding='utf-8'))
    if damage == 'truncated':
        bad = '{'
    elif damage == 'checksum':
        envelope['sha256'] = '0' * 64
        bad = json.dumps(envelope)
    elif damage == 'duplicate':
        key = next(iter(envelope))
        item = json.dumps(key) + ':' + json.dumps(envelope[key])
        bad = '{' + item + ',' + item + '}'
    else:
        bad = '{"payload":NaN,"sha256":"' + '0' * 64 + '"}'
    expected = {
        'truncated': r'^Unreadable checkpoint: .*handoff\.json$',
        'checksum': r'^Checkpoint checksum mismatch: .*handoff\.json$',
        'duplicate': '^Duplicate JSON key in checkpoint$',
        'nonfinite': '^Nonfinite JSON token$',
    }
    parent.handoff.write_text(bad, encoding='utf-8')
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError, match=expected[damage]):
        parent.recover()
    assert event_counts(parent.w) == before
    assert parent.w.hashes(cv_only=True) == parent.original_cv
    assert not parent.barrier.exists()


def test_no_historical_handoff_import_from_public_selection(parent):
    parent.handoff.unlink()
    assert (parent.source / 'selection.json').is_file()
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match=r'^Unreadable checkpoint: .*handoff\.json$'):
        parent.recover()
    assert event_counts(parent.w) == before
    assert not parent.handoff.exists() and not parent.barrier.exists()


def test_corrupt_parent_identity_refuses(parent):
    identity = parent.source / 'cv-checkpoints/identity.json'
    record = json.loads(identity.read_text())
    record['sha256'] = '0' * 64
    identity.write_text(json.dumps(record), encoding='utf-8')
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match=r'^Checkpoint checksum mismatch: .*identity\.json$'):
        parent.recover()
    assert event_counts(parent.w) == before
    assert not parent.barrier.exists()


def test_non_count_runtime_change_refuses(parent, monkeypatch):
    original = cv_checkpoint.runtime_fingerprint
    def changed(**kwargs):
        record = original(**kwargs)
        return {**record, 'python': 'different-numerical-runtime'}
    monkeypatch.setattr(cv_checkpoint, 'runtime_fingerprint', changed)
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match='^Execution and coordinator common runtime differs: python$'):
        parent.recover()
    assert event_counts(parent.w) == before
    parent.unchanged()


def test_helper_source_change_refuses_without_modifying_frozen_sources(parent, monkeypatch):
    original_path = Path(final_recovery.__file__).absolute()
    temporary = parent.w.root / 'changed-final-recovery.py'
    temporary.write_bytes(original_path.read_bytes() + b'\n# isolated source mutation\n')
    file_hash = cv_checkpoint._file_hash
    def redirected(path):
        target = temporary if Path(path).absolute() == original_path else path
        return file_hash(target)
    monkeypatch.setattr(cv_checkpoint, '_file_hash', redirected)
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError, match='source identity changed'):
        parent.recover()
    assert event_counts(parent.w) == before
    assert not parent.barrier.exists()
    parent.unchanged()


def test_late_non_count_runtime_drift_blocks_scoring(parent, monkeypatch):
    fingerprint, real_fit = cv_checkpoint.runtime_fingerprint, PhaseSurface.fit
    changed = [False]
    def observed(**kwargs):
        record = fingerprint(**kwargs)
        return {**record, 'python': 'changed-after-fit'} if changed[0] else record
    def fit_then_drift(self, X, y):
        result = real_fit(self, X, y)
        changed[0] = True
        return result
    monkeypatch.setattr(cv_checkpoint, 'runtime_fingerprint', observed)
    monkeypatch.setattr(PhaseSurface, 'fit', fit_then_drift)
    with threadpool_limits(limits=3):
        before = observed_counts()
        with pytest.raises(cv_checkpoint.CheckpointError,
                           match='^Final-fit runtime changed beyond its thread budget$'):
            parent.recover()
        assert observed_counts() == before
    assert event_counts(parent.w) == (6, 1, 0)
    assert not parent.barrier.exists()
    parent.unchanged()


def test_unsupported_budget_refuses_without_clamp_or_model_work(parent):
    parent.w.cap = 2
    with threadpool_limits(limits=3):
        before = observed_counts()
        with pytest.raises(cv_checkpoint.CheckpointError, match='wrong active budget'):
            parent.recover(budget=4)
        assert observed_counts() == before
    assert event_counts(parent.w) == (6, 0, 0)
    assert not parent.barrier.exists()
    parent.unchanged()


@pytest.mark.parametrize('error_type', [ValueError, KeyboardInterrupt, SystemExit])
def test_failed_final_fit_retains_original_error_and_same_attempt_can_resume(
        parent, monkeypatch, error_type):
    original = error_type('controlled unscored recovery failure')
    real_fit = PhaseSurface.fit
    def failed_fit(self, X, y):
        real_fit(self, X, y)
        raise original
    with monkeypatch.context() as patch:
        patch.setattr(PhaseSurface, 'fit', failed_fit)
        with threadpool_limits(limits=3):
            before = observed_counts()
            with pytest.raises(error_type) as caught:
                parent.recover()
            assert caught.value is original and observed_counts() == before
    assert event_counts(parent.w) == (6, 1, 0)
    assert not parent.barrier.exists()
    failures = list((parent.w.root / 'attempt/invocations').glob('*.json'))
    assert failures and any('controlled unscored recovery failure' in p.read_text() for p in failures)
    saved = {str(p): p.read_bytes() for p in failures}
    result = parent.recover(resume=True)
    assert result['model'] is not None and event_counts(parent.w) == (6, 2, 1)
    assert all(Path(p).read_bytes() == raw for p, raw in saved.items())
    parent.unchanged()


def test_failed_unscored_attempt_allows_new_budget_not_changed_same_identity(parent, monkeypatch):
    real_fit = PhaseSurface.fit
    failure = RuntimeError('first final attempt failed before scoring')
    def failed(self, X, y):
        real_fit(self, X, y)
        raise failure
    with monkeypatch.context() as patch:
        patch.setattr(PhaseSurface, 'fit', failed)
        with pytest.raises(RuntimeError) as caught:
            parent.recover(budget=2)
        assert caught.value is failure
    identity = (parent.w.root / 'attempt/identity.json').read_bytes()
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match=r'^Checkpoint identity mismatch: .*identity\.json$'):
        parent.recover(budget=4, resume=True)
    assert event_counts(parent.w) == before
    assert (parent.w.root / 'attempt/identity.json').read_bytes() == identity
    second = parent.recover('corrected-attempt', budget=4)
    assert second['completed']['held_out_accuracy'] == 1.
    assert all(n == 4 for path, n in PhaseSurface.fits[-1][1])
    assert event_counts(parent.w) == (6, 2, 1)
    parent.unchanged()


@pytest.mark.parametrize('boundary', ['diagnostics', 'refit-receipt'])
def test_failed_diagnostics_or_refit_publication_leaves_scoring_unclaimed(
        parent, monkeypatch, boundary):
    original = RuntimeError('controlled final-diagnostic boundary failure')
    reached = []
    if boundary == 'diagnostics':
        def failed_diagnostics(*args, **kwargs):
            reached.append(boundary)
            raise original
        monkeypatch.setattr(cv_checkpoint, 'diagnostics', failed_diagnostics)
    else:
        write_receipt = cv_checkpoint._write_receipt
        def failed_refit(path, value):
            if Path(path).parent.name == 'refits':
                reached.append(boundary)
                raise original
            return write_receipt(path, value)
        monkeypatch.setattr(cv_checkpoint, '_write_receipt', failed_refit)
    with pytest.raises(RuntimeError) as caught:
        parent.recover()
    assert caught.value is original and reached == [boundary]
    assert event_counts(parent.w) == (6, 1, 0)
    assert not parent.barrier.exists()
    assert not list((parent.w.root / 'attempt/refits').glob('*.json'))
    failures = list((parent.w.root / 'attempt/invocations').glob('*.json'))
    assert len(failures) == 1
    payload = json.loads(failures[0].read_text())['payload']
    assert payload['phase'] == 'final_diagnostics'
    assert payload['exception_type'] == type(original).__name__
    assert payload['message'] == str(original)
    parent.unchanged()


def test_scoring_failure_blocks_same_new_and_normal_attempts(parent, monkeypatch):
    real_predict = PhaseSurface.predict
    failure = KeyboardInterrupt('scoring was started but did not commit')
    def interrupted(self, X):
        assert parent.barrier.is_file(), 'Reserve must precede the first held-out use.'
        real_predict(self, X)
        raise failure
    with monkeypatch.context() as patch:
        patch.setattr(PhaseSurface, 'predict', interrupted)
        with pytest.raises(KeyboardInterrupt) as caught:
            parent.recover()
        assert caught.value is failure
    assert event_counts(parent.w) == (6, 1, 1)
    barrier = parent.barrier.read_bytes()
    refits = list((parent.w.root / 'attempt/refits').glob('*.json'))
    assert len(refits) == 1
    retained_refit = refits[0].read_bytes()
    refit = json.loads(retained_refit)['payload']
    assert refit['artifact'] == 'completed_final_refit' and refit['state'] == 'completed'
    assert 'diagnostics' in refit and refit['native_runtime']['state'] == 'completed'
    for name, resume in [('attempt', True), ('alternate', False)]:
        with pytest.raises(cv_checkpoint.CheckpointError,
                           match='^Held-out scoring has already been reserved;'):
            parent.recover(name, resume=resume)
        assert event_counts(parent.w) == (6, 1, 1)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match='^Held-out scoring has already been reserved;'):
        parent.w.run(resume=True, budget=4)
    assert event_counts(parent.w) == (6, 1, 1)
    assert parent.barrier.read_bytes() == barrier
    assert refits[0].read_bytes() == retained_refit
    parent.unchanged()


def test_completion_commit_failure_never_repeats_scored_holdout(parent, monkeypatch):
    original = OSError('controlled completion commit failure')
    write_receipt = cv_checkpoint._write_receipt
    completed_path = parent.w.root / 'attempt/completed.json'
    def fail_commit(path, payload):
        if cv_checkpoint._io_path(path) == cv_checkpoint._io_path(completed_path):
            raise original
        return write_receipt(path, payload)
    with monkeypatch.context() as patch:
        patch.setattr(cv_checkpoint, '_write_receipt', fail_commit)
        with pytest.raises(OSError) as caught:
            parent.recover()
        assert caught.value is original
    assert event_counts(parent.w) == (6, 1, 1)
    assert not completed_path.exists() and parent.barrier.is_file()
    claim = parent.barrier.read_bytes()
    for name, resume in [('attempt', True), ('new-after-commit-error', False)]:
        with pytest.raises(cv_checkpoint.CheckpointError,
                           match='^Held-out scoring has already been reserved;'):
            parent.recover(name, resume=resume)
        assert event_counts(parent.w) == (6, 1, 1)
    assert parent.barrier.read_bytes() == claim
    parent.unchanged()


def test_completed_child_cannot_be_bypassed_via_new_destination(parent):
    parent.recover()
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match='^Held-out scoring has already been reserved;'):
        parent.recover('new-path', budget=4)
    assert event_counts(parent.w) == before
    parent.unchanged()


def test_parent_lease_excludes_competing_final_job(parent):
    before = event_counts(parent.w)
    with cv_checkpoint._Lease(parent.source / 'cv-checkpoints/session.lock'):
        with pytest.raises(cv_checkpoint.CheckpointBusyError,
                           match='^Live or inaccessible checkpoint owner:'):
            parent.recover()
    assert event_counts(parent.w) == before
    parent.unchanged()


@pytest.mark.parametrize('location', ['same', 'nested', 'ancestor'])
@pytest.mark.parametrize('spelling', ['native', 'alternate'])
def test_attempt_path_cannot_alias_or_contain_original_run(
        parent, location, spelling, record_property):
    destination = {'same': parent.source, 'nested': parent.source / 'child',
                   'ancestor': parent.source.parent}[location]
    source = parent.source
    form = 'native-path'
    if spelling == 'alternate':
        if os.name == 'nt':
            # conftest may already supply extended temporary paths. Construct
            # both spellings independently of the producer's path resolver.
            prefix = '\\\\?\\'

            def public_spelling(value):
                value = str(value)
                if value[:8].upper() == prefix + 'UNC\\':
                    return '\\\\' + value[8:]
                return value[len(prefix):] if value.startswith(prefix) else value

            public_source = Path(public_spelling(parent.source))
            public_text = str(public_source)
            destination = {'same': public_source, 'nested': public_source / 'child',
                           'ancestor': public_source.parent}[location]
            source = (prefix + 'UNC\\' + public_text[2:] if public_text.startswith('\\\\')
                      else prefix + public_text)
            assert str(source) != public_text
            assert public_spelling(source) == public_text
            assert public_spelling(parent.source) == public_text
            form = 'windows-extended-source/public-destination'
        else:
            # A lexical alias exercises resolution without requiring symlinks.
            source = str(parent.source.parent) + '/./' + parent.source.name
            form = 'posix-dot-source/public-destination'
            assert str(source) != str(parent.source)
    record_property('ir16_path_spelling', form)
    before = {str(p.relative_to(parent.source)): digest(p)
              for p in parent.source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='must not overlap'):
        parent.recover(source_run=source, attempt_dir=destination)
    assert event_counts(parent.w) == (6, 0, 0)
    after = {str(p.relative_to(parent.source)): digest(p)
             for p in parent.source.rglob('*') if p.is_file()}
    assert after == before


def test_owned_data_survives_caller_mutation_during_final_fit(parent, monkeypatch):
    arrays = (parent.w.X, parent.w.y, parent.w.test_X, parent.w.test_y)
    originals = [x.copy() for x in arrays]
    real_fit, real_predict = PhaseSurface.fit, PhaseSurface.predict
    def mutate_caller(self, X, y):
        np.testing.assert_array_equal(X, originals[0])
        np.testing.assert_array_equal(y, originals[1])
        for original in arrays:
            original[...] = 0
        np.testing.assert_array_equal(X, originals[0])
        np.testing.assert_array_equal(y, originals[1])
        return real_fit(self, X, y)
    def checked_prediction(self, X):
        np.testing.assert_array_equal(X, originals[2])
        return real_predict(self, X)
    monkeypatch.setattr(PhaseSurface, 'fit', mutate_caller)
    monkeypatch.setattr(PhaseSurface, 'predict', checked_prediction)
    result = parent.recover()
    assert result['completed']['held_out_accuracy'] == 1.
    assert event_counts(parent.w) == (6, 1, 1)
    parent.unchanged()


def test_corrupt_completed_child_refuses_without_new_model_work(parent):
    parent.recover()
    completion = parent.w.root / 'attempt/completed.json'
    record = json.loads(completion.read_text())
    record['sha256'] = '0' * 64
    completion.write_text(json.dumps(record), encoding='utf-8')
    before = event_counts(parent.w)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match=r'^Checkpoint checksum mismatch: .*completed\.json$'):
        parent.recover(resume=True)
    assert event_counts(parent.w) == before
    parent.unchanged()


def test_needs_review_never_publishes_accepted_handoff(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, monkeypatch)
    original_stage = adaptive_search.next_search_stage
    def hold(results, grid, config, state):
        if len(results['params']) == 2:
            return dict(state=state, review=dict(state='needs_review', reasons=['test hold'],
                        stopping_reason='controlled finite test hold'))
        return original_stage(results, grid, config, state)
    monkeypatch.setattr(adaptive_search, 'next_search_stage', hold)
    result = workflow.run(budget=2)
    assert result['review']['state'] == 'needs_review'
    assert event_counts(workflow) == (6, 0, 0)
    assert not (workflow.root / 'run/cv-checkpoints/finalization/handoff.json').exists()


def test_normal_completed_run_cannot_be_forked_for_rescoring(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, monkeypatch)
    workflow.run(budget=2)
    assert event_counts(workflow) == (6, 1, 1)
    prepared = RecoveryFixture(workflow)
    protect_cv(monkeypatch)
    before = event_counts(workflow)
    with pytest.raises(cv_checkpoint.CheckpointError,
                       match='^The original workflow already completed;'):
        prepared.recover()
    assert event_counts(workflow) == before
    prepared.unchanged()


def test_real_ensemble_recovery_matches_same_configured_reference_without_cv_replay(
        tmp_path, monkeypatch, record_property):
    from ensemble_dwd import EnsembleKernelDWD
    from ensemble_dwd._cache import _initialize_numerical_runtime
    from sklearn.preprocessing import StandardScaler

    _initialize_numerical_runtime()
    # Keep the controlled selection-policy seam independent of the source
    # observer's callable inventory. Actual native pool counts are still read;
    # dedicated source-identity guards separately cover inventory changes.
    monkeypatch.setattr(cv_checkpoint, 'runtime_fingerprint', observed_runtime)
    monkeypatch.setattr(cv_checkpoint, '_native_pool_fingerprint', synthetic_native_pools)
    X = np.column_stack((np.linspace(-2., 2., 48), np.sin(np.arange(48)),
                         (np.arange(48) % 5) / 8.))
    y = np.where(X[:, 0] < 0., -3, 7)
    Xt = np.column_stack((np.linspace(-1.8, 1.8, 8), np.sin(np.arange(8) + .25),
                          ((np.arange(8) + .5) % 5) / 8.))
    yt = np.where(Xt[:, 0] < 0., -3, 7)
    indices = np.arange(len(y))
    folds = [(indices[indices % 2 != k], indices[indices % 2 == k]) for k in (0, 1)]
    binary = EnsembleKernelDWD(M=2, theta=.5, sv_selection_ratio=1., lambd=.05,
        gamma=.2, max_iter=3, stopping='fixed', random_state=17,
        cache_dir=None, store_base_estimators=True)
    pipeline = Pipeline([('scale', StandardScaler()), ('ensemble', binary)])
    grid = {'ensemble__lambd': [.05, .1], 'ensemble__gamma': [.2]}
    config = adaptive_search.AdaptiveSearchConfig(refine=False, max_candidates=2)
    original_stage = adaptive_search.next_search_stage
    def accept(results, grid, config, state):
        if len(results['params']) == 2:
            return dict(state=state, review=dict(state='accepted', reasons=[],
                        stopping_reason='controlled finite real-model acceptance'))
        return original_stage(results, grid, config, state)
    monkeypatch.setattr(adaptive_search, 'next_search_stage', accept)
    source = Path(extended_path(new_run_directory(tmp_path / 'real-source')))
    real_fit = EnsembleKernelDWD.fit
    fit_rows = []
    original = KeyboardInterrupt('stop accepted parent before real final fit')
    def stop_final(self, data, labels, *args, **kwargs):
        fit_rows.append(len(data))
        if len(data) == 48:
            raise original
        return real_fit(self, data, labels, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(EnsembleKernelDWD, 'fit', stop_final)
        with pytest.raises(KeyboardInterrupt) as caught, redirect_stdout(io.StringIO()):
            adaptive_search.run_adaptive_search(pipeline, grid, X, y, Xt, yt, folds, source,
                config=config, final_native_threads=2, reuse_equivalent_folds=False)
        assert caught.value is original
    assert fit_rows == [24, 24, 24, 24, 48]
    retained = {str(p.relative_to(source)): digest(p)
                for p in (source / 'cv-checkpoints/folds').glob('*.json')}
    assert len(retained) == 4
    selected = json.loads((source / 'selection.json').read_text())['parameters']
    configured = clone(pipeline).set_params(**selected)
    protect_cv(monkeypatch)
    fit_rows.clear()
    def count_final(self, data, labels, *args, **kwargs):
        fit_rows.append(len(data))
        return real_fit(self, data, labels, *args, **kwargs)
    monkeypatch.setattr(EnsembleKernelDWD, 'fit', count_final)
    result = final_recovery.recover_final_fit(configured, X, y, Xt, yt,
        source, tmp_path / 'real-attempt', final_native_threads=2)
    assert fit_rows == [48] and result['model'] is not None
    reference = clone(configured)
    with threadpool_limits(limits=2):
        reference.fit(X, y)
    assert fit_rows == [48, 48]
    reference_prediction = reference.predict(Xt)
    for actual, expected in ((result['model'].decision_function(Xt), reference.decision_function(Xt)),
                             (result['model'].predict(Xt), reference_prediction)):
        np.testing.assert_array_equal(actual, expected)
    assert_completed_reports(result, y, yt, reference_prediction)
    assert all(digest(source / path) == value for path, value in retained.items())
    before = len(fit_rows)
    replay = final_recovery.recover_final_fit(configured, X, y, Xt, yt,
        source, tmp_path / 'real-attempt', final_native_threads=2, resume=True)
    assert replay['model'] is None and replay['resumed_completed'] is True
    assert len(fit_rows) == before
    record_property('ir16_real_final_recovery', json.dumps(dict(
        cv_fits=4, parent_final_entry_interrupted_before_fit=True,
        recovered_final_fits=1, independent_same_config_reference_fits=1,
        fixed_base_updates=3, claimed_converged=False, replay_fits=0,
        scoring_scope='distinct synthetic rows; numerical parity, not predictive performance',
        retained_fold_sha256=retained), sort_keys=True))
