"""Exact-source recovery with synthetic receipts and no numerical fitting.

The equivalent-fold numerical parity tests remain a separate panel.
"""
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import cv_checkpoint as checkpoint
import test_equivalent_fold_reuse as existing
from runtime_test_support import synthetic_runtime


def _forbid_fit(*args, **kwargs):
    raise AssertionError('Receipt restoration must not fit or score')


class SavedRun:
    """Use real identities/proofs and alias publication with synthetic science."""

    def __init__(self, root, *, pairs=False):
        root = checkpoint._io_path(root)
        self.root = root
        self.X, self.y = existing.multiclass_data() if pairs else existing.binary_data()
        folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.settings = dict(run_dir=root, estimator=existing.pipeline('dwd' if pairs else 'svm', multiclass=pairs),
            X=self.X, y=self.y, folds=folds, jobs=1, reuse_equivalent_folds=True,
            provenance={'test': 'alias-source-exact-attempt'}, pair_recovery=pairs)
        self.source_paths, self.attempt_paths, self.source_records = [], [], []
        self.alias_paths = []
        self.ratios = [.875, .88, .89]
        with self.session() as store:
            self.folder, self.identity = store.folder, store.identity_sha256
            for fold, (train, validation) in enumerate(store.folds):
                params = {'model__sv_selection_ratio': self.ratios[0]}
                # OvO wraps the binary model; the grid names that actual path.
                if pairs:
                    params = {'model__estimator__sv_selection_ratio': self.ratios[0]}
                self.parameter = next(iter(params))
                candidate = clone(store.estimator).set_params(**params)
                proof = checkpoint._fold_equivalence(candidate, self.identity, fold, self.y[train])
                assert proof is not None
                task = checkpoint._digest({'identity': self.identity,
                    'estimator': checkpoint._stable(candidate), 'fold': fold})
                attempt = uuid.uuid4().hex
                diagnostics = {'converged_': True, 'n_iter_': 3, 'cache_hit_': False,
                               'cache_status_': 'miss'}
                record = dict(schema=checkpoint.SCHEMA, identity_sha256=self.identity,
                    task_sha256=task, attempt=attempt, owner_token=store.token, pid=os.getpid(),
                    fold=fold, parameters=checkpoint._stable(params), state='complete',
                    started_at='2026-11-14T00:00:00+00:00',
                    completed_at='2026-11-14T00:00:01+00:00', failure=None,
                    score=.75, fit_time=1.25 + fold, score_time=.125,
                    accuracy_counts={'correct': len(validation) * 3 // 4,
                                     'n_validation': len(validation)},
                    diagnostics=diagnostics, equivalent_fold=proof)
                if pairs:
                    import pair_recovery
                    pairs_list = [[0, 1], [0, 2], [1, 2]]
                    events = [dict(pair_index=i, pair=pair, status='fitted_saved',
                        work_origin='fitted', cache_committed=True, fit_seconds=.25,
                        load_seconds=0., save_seconds=.125, historical_fit_seconds=0.)
                        for i, pair in enumerate(pairs_list)]
                    record['pair_recovery'] = dict(schema=1, policy=pair_recovery.POLICY,
                        pairs=events, cleanup_status='pending', fitted_pairs=3,
                        recovered_pairs=0, fitted_unsaved_pairs=0,
                        current_pair_fit_seconds=.75, pair_load_seconds=0.,
                        pair_save_seconds=.375, historical_reused_pair_fit_seconds=0.)
                    diagnostics['pairwise_estimators'] = [dict(pair=pair, work_origin='fitted',
                        diagnostics={'converged_': True}) for pair in pairs_list]
                source_path = store.folder / 'folds' / (task + '.json')
                attempt_path = store.folder / 'attempts' / (attempt + '.json')
                checkpoint._write_receipt(source_path, record)
                checkpoint._write_receipt(attempt_path, record)
                self.source_paths.append(source_path)
                self.attempt_paths.append(attempt_path)
                self.source_records.append(record)
                for ratio in self.ratios[1:]:
                    alias_params = {self.parameter: ratio}
                    candidate = clone(store.estimator).set_params(**alias_params)
                    alias_proof = checkpoint._fold_equivalence(candidate, self.identity, fold, self.y[train])
                    alias_task = checkpoint._digest({'identity': self.identity,
                        'estimator': checkpoint._stable(candidate), 'fold': fold})
                    checkpoint._reuse_fold(store.folder, self.identity, store.execution_runtime,
                        store.token, alias_task, alias_params, fold, alias_proof, task,
                        validation_size=len(validation))
                    self.alias_paths.append(store.folder / 'folds' / (alias_task + '.json'))
        self.original = self.saved_bytes()

    def session(self, **kwargs):
        return checkpoint.CheckpointedCV(**self.settings, **kwargs)

    def grid(self, ratios=None):
        return {self.parameter: self.ratios if ratios is None else ratios}

    def resume(self, ratios=None):
        with self.session(resume=True) as store:
            return store.search(self.grid(ratios))

    def saved_bytes(self):
        return {path: path.read_bytes() for name in ('folds', 'attempts')
                for path in (self.root / 'cv-checkpoints' / name).glob('*.json')}

    def missing(self, *indices):
        for index in indices:
            self.source_paths[index].unlink()

    def aliases_for(self, source):
        task = self.source_records[source]['task_sha256']
        return [path for path in self.alias_paths
                if checkpoint._read_receipt(path, self.identity)['equivalent_reuse']['source_task_sha256'] == task]

    def reanchor(self, source, record):
        for path in self.aliases_for(source):
            alias = checkpoint._read_receipt(path, self.identity)
            alias['equivalent_reuse']['source_receipt_sha256'] = checkpoint._digest(record)
            checkpoint._write_receipt(path, alias)


@pytest.fixture
def run_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, 'runtime_fingerprint', lambda **kwargs: synthetic_runtime())
    monkeypatch.setattr(checkpoint, '_fit_fold', _forbid_fit)
    monkeypatch.setattr(checkpoint.CandidateEvaluator, 'fit', _forbid_fit)
    monkeypatch.setattr(checkpoint, 'eligible_accuracy', _forbid_fit)
    created = []

    def create(*, pairs=False):
        run = SavedRun(tmp_path / str(len(created)), pairs=pairs)
        created.append(run)
        return run

    return create


@pytest.fixture
def saved(run_factory):
    return run_factory()


@pytest.mark.parametrize('ratios', [[.88], [.875, .88, .89]])
def test_exact_source_restore_preserves_all_bytes_and_replay_accounting(saved, ratios):
    saved.missing(0)
    result = saved.resume(ratios)
    assert saved.saved_bytes() == saved.original
    summary = result.checkpoint_summary_
    assert summary['restored_source_receipts'] == 1
    assert summary['executed_folds'] == summary['new_folds'] == 0
    assert summary['reused_folds'] == 2 * len(ratios)
    work = result.work_summary_
    assert work['evaluations']['executed'] == 0
    assert work['evaluations']['checkpoint_replayed'] == 2 * len(ratios)
    assert work['current_worker_seconds'] == {'fit': 0., 'score': 0., 'equivalent_alias': 0.}
    expected_historical_fit = sum(row['fit_time'] for row in saved.source_records) if .875 in ratios else 0.
    assert work['historical_receipt_seconds']['replayed_fit'] == expected_historical_fit
    repeated = saved.resume(ratios)
    assert 'restored_source_receipts' not in repeated.checkpoint_summary_
    np.testing.assert_equal(result.cv_results_['mean_test_score'], repeated.cv_results_['mean_test_score'])
    assert saved.saved_bytes() == saved.original


def test_multiple_sources_and_later_stage_create_direct_aliases_only(saved):
    saved.missing(0, 1)
    result = saved.resume([.88])
    assert result.checkpoint_summary_['restored_source_receipts'] == 2
    next_stage = saved.resume([.90])
    assert next_stage.checkpoint_summary_['executed_folds'] == 0
    assert next_stage.work_summary_['evaluations']['equivalent_aliases_created'] == 2
    for facts in next_stage.checkpoint_summary_['equivalent_reuse'][0]:
        source = next(row for row in saved.source_records
                      if row['task_sha256'] == facts['source_task_sha256'])
        assert facts['source_receipt_sha256'] == checkpoint._digest(source)
    assert all(path.read_bytes() == content for path, content in saved.original.items())


def test_healthy_aliases_never_scan_attempts_or_lstat_sources(saved, monkeypatch):
    original_glob, original_lstat = Path.glob, Path.lstat
    attempts = saved.folder / 'attempts'

    def guarded_glob(path, pattern, *args, **kwargs):
        assert path != attempts, 'Healthy resume scanned attempts'
        return original_glob(path, pattern, *args, **kwargs)

    def guarded_lstat(path, *args, **kwargs):
        assert path not in saved.source_paths, 'Healthy source validation added an lstat'
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'glob', guarded_glob)
    monkeypatch.setattr(Path, 'lstat', guarded_lstat)
    monkeypatch.setattr(checkpoint, '_restore_alias_sources', _forbid_fit)
    assert saved.resume().checkpoint_summary_['executed_folds'] == 0


def test_validators_remain_read_only_when_source_missing(saved):
    saved.missing(0)
    before = saved.saved_bytes()
    alias = saved.aliases_for(0)[0]
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._valid_fold(alias, saved.identity, alias.stem, validation_size=16)
    assert saved.saved_bytes() == before


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'checksum', 'running', 'failed',
    'ineligible', 'identity', 'task', 'fold', 'denominator', 'alias-chain', 'filename'])
def test_missing_source_needs_exact_valid_complete_attempt(saved, damage):
    attempt = saved.attempt_paths[0]
    row = deepcopy(saved.source_records[0])
    saved.missing(0)
    if damage == 'missing':
        attempt.unlink()
    elif damage == 'truncated':
        attempt.write_bytes(b'{')
    elif damage == 'checksum':
        envelope = json.loads(attempt.read_text())
        envelope['payload']['fit_time'] += 1.
        attempt.write_text(json.dumps(envelope))
    elif damage == 'filename':
        attempt.rename(attempt.with_name('copy.json'))
    else:
        if damage in ('running', 'failed'):
            row['state'] = damage
        elif damage == 'ineligible':
            row.update(state='ineligible', score=None, accuracy_counts=None,
                       failure={'category': 'ineligible_selection'})
        elif damage == 'identity':
            row['identity_sha256'] = '0' * 64
        elif damage == 'task':
            row['task_sha256'] = '0' * 64
        elif damage == 'fold':
            row['fold'] = 1
        elif damage == 'denominator':
            row['accuracy_counts'] = {key: 2 * value for key, value in row['accuracy_counts'].items()}
        elif damage == 'alias-chain':
            row['equivalent_reuse'] = deepcopy(checkpoint._read_receipt(saved.aliases_for(0)[0],
                saved.identity)['equivalent_reuse'])
        checkpoint._write_receipt(attempt, row)
        # Exercise semantic validation even when the malformed duplicate's
        # checksum and the aliases' anchors have been recomputed consistently.
        saved.reanchor(0, row)
    before = saved.saved_bytes()
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume()
    assert saved.saved_bytes() == before
    assert not saved.source_paths[0].exists()


@pytest.mark.parametrize('damage', ['checksum', 'valid-changed-score', 'denominator', 'directory'])
def test_existing_bad_source_is_never_replaced_from_good_attempt(saved, damage, monkeypatch):
    source = saved.source_paths[0]
    row = deepcopy(saved.source_records[0])
    if damage == 'directory':
        source.unlink()
        source.mkdir()
    elif damage == 'checksum':
        envelope = json.loads(source.read_text())
        envelope['payload']['fit_time'] += 1.
        source.write_text(json.dumps(envelope))
    else:
        if damage == 'valid-changed-score':
            row['score'] = .5
            row['accuracy_counts']['correct'] = 8
        else:
            row['accuracy_counts'] = {key: 2 * value for key, value in row['accuracy_counts'].items()}
        checkpoint._write_receipt(source, row)
    before = None if source.is_dir() else source.read_bytes()
    monkeypatch.setattr(checkpoint, '_restore_alias_sources', _forbid_fit)
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume([.88])
    assert source.is_dir() if before is None else source.read_bytes() == before
    assert saved.attempt_paths[0].read_bytes() == saved.original[saved.attempt_paths[0]]


@pytest.mark.parametrize('field', ['policy', 'source_parameters', 'source_fit_time',
    'source_score_time', 'diagnostics_source_task_sha256', 'reuse_seconds'])
def test_alias_anchor_fields_are_checked_before_restoration(saved, field):
    path = saved.aliases_for(0)[0]
    alias = checkpoint._read_receipt(path, saved.identity)
    values = dict(policy='other', source_parameters={}, source_fit_time=99., source_score_time=99.,
                  diagnostics_source_task_sha256='0' * 64, reuse_seconds=-1.)
    alias['equivalent_reuse'][field] = values[field]
    checkpoint._write_receipt(path, alias)
    saved.missing(0)
    before = saved.saved_bytes()
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume([.88])
    assert saved.saved_bytes() == before


def test_conflicting_alias_digests_refuse_before_attempt_scan(saved, monkeypatch):
    path = saved.aliases_for(0)[0]
    alias = checkpoint._read_receipt(path, saved.identity)
    alias['equivalent_reuse']['source_receipt_sha256'] = '0' * 64
    checkpoint._write_receipt(path, alias)
    saved.missing(0)
    original_glob = Path.glob

    def guarded(path, *args, **kwargs):
        assert path != saved.folder / 'attempts'
        return original_glob(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'glob', guarded)
    with pytest.raises(checkpoint.CheckpointError, match='disagree'):
        saved.resume()
    assert not saved.source_paths[0].exists()


def test_previously_indexed_alias_is_included_in_all_source_anchors(saved):
    first, second = saved.aliases_for(0)
    task = saved.source_records[0]['task_sha256']
    with saved.session(resume=True) as store:
        checkpoint._valid_fold(first, saved.identity, first.stem, validation_size=16)
        saved.missing(0)
        with pytest.raises(checkpoint._MissingEquivalentSource) as missing:
            checkpoint._valid_fold(second, saved.identity, second.stem, validation_size=16)
        changed = checkpoint._read_receipt(first, saved.identity)
        changed['equivalent_reuse']['source_receipt_sha256'] = '0' * 64
        checkpoint._write_receipt(first, changed)
        with pytest.raises(checkpoint.CheckpointError, match='disagree'):
            checkpoint._restore_alias_sources(saved.folder, saved.identity, store.token,
                [missing.value], {task: [first]}, [16, 16])
    assert not saved.source_paths[0].exists()


@pytest.mark.parametrize('conflict', [False, True])
def test_source_disappears_between_aliases_in_coordinator_index_scan(saved, monkeypatch, conflict):
    first, second = saved.aliases_for(0)
    order = [*saved.source_paths, first, second,
             *(path for path in saved.alias_paths if path not in (first, second))]
    positions = {path: index for index, path in enumerate(order)}
    original_sorted, original_valid = sorted, checkpoint._valid_fold
    first_reads = []
    removed = False

    def index_order(iterable, *args, **kwargs):
        values = list(iterable)
        if values and all(isinstance(path, Path) and path.parent == saved.folder / 'folds'
                          for path in values):
            # Deterministic filesystem-order injection, retaining the actual
            # receipt names and all production validation/dispatch operations.
            return original_sorted(values, key=lambda path: positions[path])
        return original_sorted(values, *args, **kwargs)

    def validate_then_remove(path, *args, **kwargs):
        nonlocal removed
        if Path(path) == first:
            first_reads.append(saved.source_paths[0].exists())
        record = original_valid(path, *args, **kwargs)
        if Path(path) == first and not removed:
            removed = True
            saved.missing(0)
            if conflict:
                changed = deepcopy(record)
                changed['equivalent_reuse']['source_receipt_sha256'] = '0' * 64
                checkpoint._write_receipt(first, changed)
        return record

    monkeypatch.setattr(checkpoint, 'sorted', index_order, raising=False)
    monkeypatch.setattr(checkpoint, '_valid_fold', validate_then_remove)
    if conflict:
        with pytest.raises(checkpoint.CheckpointError, match='disagree'):
            saved.resume()
        assert not saved.source_paths[0].exists()
        assert saved.source_paths[1].read_bytes() == saved.original[saved.source_paths[1]]
    else:
        result = saved.resume()
        assert result.checkpoint_summary_['restored_source_receipts'] == 1
        assert result.checkpoint_summary_['executed_folds'] == 0
        assert saved.saved_bytes() == saved.original
    assert removed
    assert first_reads[:2] == [True, False]
    assert not list(saved.folder.glob('folds/.source-restore-*.tmp'))


def test_unrelated_auxiliary_failures_do_not_discard_exact_attempt(saved):
    unrelated = saved.folder / 'attempts' / ('f' * 32 + '.json')
    unrelated.write_bytes(b'not a complete record')
    alternative = deepcopy(saved.source_records[0])
    alternative['attempt'] = 'e' * 32
    alternative['fit_time'] += .5
    path = saved.folder / 'attempts' / (alternative['attempt'] + '.json')
    checkpoint._write_receipt(path, alternative)
    saved.missing(0)
    assert saved.resume().checkpoint_summary_['restored_source_receipts'] == 1
    assert unrelated.read_bytes() == b'not a complete record'
    assert saved.source_paths[0].read_bytes() == saved.original[saved.source_paths[0]]


def test_entire_plan_is_validated_before_any_source_is_published(saved):
    saved.missing(0, 1)
    saved.attempt_paths[1].unlink()
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume()
    assert all(not path.exists() for path in saved.source_paths)


@pytest.mark.parametrize('phase', ['candidate-read', 'before-publish'])
def test_stale_owner_cannot_publish_recovery(saved, monkeypatch, phase):
    saved.missing(0)
    original_read = checkpoint._read_receipt
    calls = 0

    def intercepted(path, identity, **options):
        nonlocal calls
        result = original_read(path, identity, **options)
        if Path(path) == saved.attempt_paths[0]:
            calls += 1
            if calls == (1 if phase == 'candidate-read' else 2):
                owner = checkpoint._read_json(saved.folder / 'owner.json')
                owner['token'] = 'replacement-owner'
                checkpoint.write_json_atomic(saved.folder / 'owner.json', owner, replace=True)
        return result

    monkeypatch.setattr(checkpoint, '_read_receipt', intercepted)
    with pytest.raises(checkpoint.CheckpointBusyError):
        saved.resume()
    assert not saved.source_paths[0].exists()


def test_source_task_lease_and_live_coordinator_block_recovery(saved):
    saved.missing(0)
    with saved.session(resume=True) as store:
        with pytest.raises(checkpoint.CheckpointBusyError):
            with saved.session(resume=True):
                pass
        lock = saved.folder / 'locks' / (saved.source_records[0]['task_sha256'] + '.lock')
        with checkpoint._Lease(lock), pytest.raises(checkpoint.CheckpointBusyError):
            store.search(saved.grid())
        assert not saved.source_paths[0].exists()
    assert saved.resume().checkpoint_summary_['restored_source_receipts'] == 1


def test_alias_change_during_attempt_read_aborts_publication(saved, monkeypatch):
    saved.missing(0)
    original_read = checkpoint._read_receipt
    changed = False
    alias_path = saved.aliases_for(0)[0]

    def intercepted(path, identity, **options):
        nonlocal changed
        result = original_read(path, identity, **options)
        if Path(path) == saved.attempt_paths[0] and not changed:
            changed = True
            alias = original_read(alias_path, identity)
            alias['equivalent_reuse']['reuse_seconds'] += .1
            checkpoint._write_receipt(alias_path, alias)
        return result

    monkeypatch.setattr(checkpoint, '_read_receipt', intercepted)
    with pytest.raises(checkpoint.CheckpointError, match='alias changed'):
        saved.resume()
    assert not saved.source_paths[0].exists()


def test_captured_attempt_bytes_must_match_validated_record(saved, monkeypatch):
    saved.missing(0)
    original_read = Path.read_bytes

    def intercepted(path):
        content = original_read(path)
        if path == saved.attempt_paths[0]:
            envelope = json.loads(content)
            envelope['payload']['fit_time'] += 1.
            return json.dumps(envelope).encode()
        return content

    monkeypatch.setattr(Path, 'read_bytes', intercepted)
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume()
    assert not saved.source_paths[0].exists()


@pytest.mark.parametrize('change', ['data', 'folds', 'estimator', 'provenance', 'reuse-mode'])
def test_caller_identity_change_refuses_before_recovery(saved, monkeypatch, change):
    saved.missing(0)
    settings = dict(saved.settings)
    if change == 'data':
        settings['X'] = saved.X.copy()
        settings['X'][0, 0] += .01
    elif change == 'folds':
        settings['folds'] = settings['folds'][::-1]
    elif change == 'estimator':
        settings['estimator'] = clone(settings['estimator']).set_params(model__gamma=.3)
    elif change == 'provenance':
        settings['provenance'] = {'test': 'changed-caller'}
    else:
        settings['reuse_equivalent_folds'] = 'auto'
    monkeypatch.setattr(checkpoint, '_restore_alias_sources', _forbid_fit)
    with pytest.raises(checkpoint.CheckpointError):
        with checkpoint.CheckpointedCV(**settings, resume=True) as store:
            store.search(saved.grid())
    assert not saved.source_paths[0].exists()


def test_unreadable_present_source_does_not_enter_recovery(saved, monkeypatch):
    original_read = Path.read_text

    def denied(path, *args, **kwargs):
        if path == saved.source_paths[0]:
            raise PermissionError('synthetic unreadable existing source')
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', denied)
    monkeypatch.setattr(checkpoint, '_restore_alias_sources', _forbid_fit)
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume()
    assert saved.source_paths[0].read_bytes() == saved.original[saved.source_paths[0]]


def test_target_collision_is_not_overwritten(saved, monkeypatch):
    saved.missing(0)
    original_link = checkpoint.os.link

    def collide(source, target, *args, **kwargs):
        Path(target).write_bytes(b'concurrent destination')
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, 'link', collide)
    with pytest.raises(checkpoint.CheckpointError, match='Cannot restore'):
        saved.resume()
    assert saved.source_paths[0].read_bytes() == b'concurrent destination'
    assert not list(saved.folder.glob('folds/.source-restore-*.tmp'))


def test_temporary_collision_is_not_deleted_or_replaced(saved, monkeypatch):
    saved.missing(0)
    identifier = 'a' * 32
    temporary = saved.folder / 'folds' / ('.source-restore-' + identifier + '.tmp')
    temporary.write_bytes(b'unowned temporary')
    # Enter the session before fixing the recovery temporary's UUID, so the
    # coordinator still receives an independently generated generation token.
    with saved.session(resume=True) as store:
        monkeypatch.setattr(checkpoint.uuid, 'uuid4', lambda: type('FixedUUID', (), {'hex': identifier})())
        with pytest.raises(checkpoint.CheckpointError):
            store.search(saved.grid())
    assert temporary.read_bytes() == b'unowned temporary'
    assert not saved.source_paths[0].exists()


def test_fsync_failure_keeps_missing_destination_and_removes_owned_temp(saved, monkeypatch):
    saved.missing(0)
    original_fsync = checkpoint.os.fsync
    original_publish = checkpoint._publish_restored_source

    def failed_publish(path, content, **kwargs):
        def fail_fsync(descriptor):
            raise OSError(errno.EIO, 'synthetic recovery fsync failure')
        with monkeypatch.context() as patcher:
            patcher.setattr(checkpoint.os, 'fsync', fail_fsync)
            return original_publish(path, content, **kwargs)

    monkeypatch.setattr(checkpoint, '_publish_restored_source', failed_publish)
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume()
    assert checkpoint.os.fsync is original_fsync
    assert not saved.source_paths[0].exists()
    assert not list(saved.folder.glob('folds/.source-restore-*.tmp'))


def test_owner_change_during_temporary_fsync_refuses_publication(saved, monkeypatch):
    saved.missing(0)
    original_publish = checkpoint._publish_restored_source
    original_fsync = checkpoint.os.fsync

    def changed_publish(path, content, **kwargs):
        def change_owner(descriptor):
            original_fsync(descriptor)
            owner_path = saved.folder / 'owner.json'
            owner = checkpoint._read_json(owner_path)
            owner['token'] = 'replacement-during-temporary-fsync'
            # Inject the external ownership change without recursively calling
            # the atomic writer's fsync hook.
            owner_path.write_text(json.dumps(owner), encoding='utf-8')
        with monkeypatch.context() as patcher:
            patcher.setattr(checkpoint.os, 'fsync', change_owner)
            return original_publish(path, content, **kwargs)

    monkeypatch.setattr(checkpoint, '_publish_restored_source', changed_publish)
    with pytest.raises(checkpoint.CheckpointBusyError):
        saved.resume()
    assert not saved.source_paths[0].exists()
    assert not list(saved.folder.glob('folds/.source-restore-*.tmp'))


@pytest.mark.parametrize('phase', ['before-link', 'after-link'])
def test_interrupted_publication_preserves_complete_or_absent_source(saved, monkeypatch, phase):
    saved.missing(0)
    original_link = checkpoint.os.link

    def interrupt(source, target, *args, **kwargs):
        if phase == 'after-link':
            original_link(source, target, *args, **kwargs)
        raise KeyboardInterrupt('injected recovery publication boundary')

    with monkeypatch.context() as patcher:
        patcher.setattr(checkpoint.os, 'link', interrupt)
        with pytest.raises(KeyboardInterrupt):
            saved.resume()
    if phase == 'before-link':
        assert not saved.source_paths[0].exists()
    else:
        assert saved.source_paths[0].read_bytes() == saved.original[saved.source_paths[0]]
    assert not list(saved.folder.glob('folds/.source-restore-*.tmp'))
    assert saved.resume().checkpoint_summary_['executed_folds'] == 0
    assert saved.saved_bytes() == saved.original


def test_interruption_between_two_restores_preserves_first_and_recovers_remaining(saved, monkeypatch):
    saved.missing(0, 1)
    original_link = checkpoint.os.link
    published = []

    def interrupted(source, target, *args, **kwargs):
        original_link(source, target, *args, **kwargs)
        published.append(Path(target))
        raise KeyboardInterrupt('one complete source was published')

    with monkeypatch.context() as patcher:
        patcher.setattr(checkpoint.os, 'link', interrupted)
        with pytest.raises(KeyboardInterrupt):
            saved.resume()
    assert len(published) == 1
    assert published[0].read_bytes() == saved.original[published[0]]
    assert sum(path.exists() for path in saved.source_paths) == 1
    result = saved.resume()
    assert result.checkpoint_summary_['restored_source_receipts'] == 1
    assert saved.saved_bytes() == saved.original


def test_unsupported_hard_link_refuses_without_replacing_fallback(saved, monkeypatch):
    saved.missing(0)

    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOTSUP, 'synthetic filesystem does not support hard links')

    monkeypatch.setattr(checkpoint.os, 'link', unsupported)
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume()
    assert not saved.source_paths[0].exists()
    assert not list(saved.folder.glob('folds/.source-restore-*.tmp'))


def test_dangling_source_symlink_is_not_absence(saved):
    saved.missing(0)
    path = saved.source_paths[0]
    try:
        path.symlink_to(path.with_name('nonexistent-source-target'))
    except (OSError, NotImplementedError) as error:
        pytest.skip('Actual symlink creation unavailable: ' + str(error))
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume()
    assert path.is_symlink()
    assert os.readlink(path).endswith('nonexistent-source-target')


@pytest.mark.skipif(os.name == 'nt' or not hasattr(os, 'mkfifo'), reason='POSIX FIFO unavailable')
def test_fifo_destination_collision_is_never_opened(saved, monkeypatch):
    saved.missing(0)
    original_link = checkpoint.os.link

    def collide(source, target, *args, **kwargs):
        os.mkfifo(target)
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, 'link', collide)
    with pytest.raises(checkpoint.CheckpointError):
        saved.resume()
    import stat
    assert stat.S_ISFIFO(saved.source_paths[0].lstat().st_mode)


def test_pair_recovery_receipt_restores_original_pending_payload(run_factory):
    saved = run_factory(pairs=True)
    saved.missing(0)
    result = saved.resume()
    assert result.checkpoint_summary_['restored_source_receipts'] == 1
    assert result.work_summary_['evaluations']['executed'] == 0
    assert all(path.read_bytes() == content for path, content in saved.original.items())
    record = checkpoint._read_receipt(saved.source_paths[0], saved.identity)
    assert record['pair_recovery']['cleanup_status'] == 'pending'
    assert record == saved.source_records[0]
