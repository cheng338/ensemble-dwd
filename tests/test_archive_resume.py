"""Archive location is provenance, while loaded data and policies remain identity."""

from contextlib import redirect_stdout
from copy import deepcopy
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

import ensemble_dwd

_REPOSITORY = Path(ensemble_dwd.__file__).resolve().parents[2]
for _directory in (_REPOSITORY / 'examples', _REPOSITORY / 'tests'):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import adaptive_search
import cv_checkpoint as checkpoint
from runtime_test_support import synthetic_runtime
from search_support import _io_path, data_fingerprint
import tune_mnist
import tune_mnist_multiclass
import tune_mnist_svm


_CLIS = {'binary-dwd': tune_mnist, 'binary-svm': tune_mnist_svm,
         'multiclass-dwd': tune_mnist_multiclass}


class _Surface(ClassifierMixin, BaseEstimator):
    attempts = []
    predictions = []
    fail_at = None

    def __init__(self, lambd=4.0, gamma=4.0, random_state=42):
        self.lambd = lambd
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y):
        rows = tuple(np.asarray(X)[:, 0].tolist())
        type(self).attempts.append((rows, self.lambd, self.gamma, self.random_state))
        if len(type(self).attempts) == type(self).fail_at:
            raise KeyboardInterrupt('bounded interruption after committed folds')
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        type(self).predictions.append(tuple(X[:, 0].tolist()))
        labels = X[:, 1].astype(np.int64)
        return labels if (self.lambd, self.gamma) == (4.0, 4.0) else np.roll(labels, 1)


class _WorkflowBoundary(Exception):
    def __init__(self, args, kwargs):
        self.invocation = SimpleNamespace(args=args, kwargs=kwargs)


@pytest.fixture(autouse=True)
def _isolated_runtime_and_counters(monkeypatch):
    # Real loaders, checkpoint identities/leases/receipts and adaptive policy run.
    # Library discovery and actual numerical fitting are separate test scopes.
    monkeypatch.setattr(checkpoint, 'runtime_fingerprint',
                        lambda **kwargs: synthetic_runtime(source='ir34-archive-resume'))
    _Surface.attempts = []
    _Surface.predictions = []
    _Surface.fail_at = None


@pytest.fixture
def archive(tmp_path):
    root = _io_path(tmp_path)
    path = root / 'source.npz'
    training_labels = np.tile(np.arange(10), 6)
    test_labels = np.tile(np.arange(10), 3)
    arrays = {'x_train': np.column_stack([np.arange(60), training_labels]),
              'y_train': training_labels,
              'x_test': np.column_stack([np.arange(100, 130), test_labels]),
              'y_test': test_labels}
    np.savez(path, **arrays)
    return SimpleNamespace(root=root, path=path, arrays=arrays)


def _invoke_cli(monkeypatch, kind, archive_path, run_dir, *, resume=False, extra=()):
    argv = ['--data', str(archive_path), '--run', '--no-cache',
            '--output-dir', str(run_dir)]
    if kind == 'multiclass-dwd':
        argv += ['--train-size', '40', '--test-size', '20', '--folds', '2', '--smoke-grid']
    else:
        argv += ['--digits', '7', '9', '--no-refine', '--max-expansions', '0']
    if resume:
        argv += ['--resume']
    argv += list(extra)

    def capture(*args, **kwargs):
        raise _WorkflowBoundary(args, kwargs)

    output = io.StringIO()
    with monkeypatch.context() as local:
        local.setattr(adaptive_search, 'run_adaptive_search', capture)
        with redirect_stdout(output), pytest.raises(_WorkflowBoundary) as caught:
            _CLIS[kind].main(argv)
    invocation = caught.value.invocation
    invocation.output = output.getvalue()
    return invocation


def _session(invocation, *, resume=False, settings=None, estimator=None):
    _, _, X, y, X_test, y_test, folds, run_dir = invocation.args
    provenance = {
        'settings': deepcopy(invocation.kwargs['settings'] if settings is None else settings),
        'held_out': data_fingerprint(X_test, y_test),
    }
    return checkpoint.CheckpointedCV(
        run_dir, _Surface() if estimator is None else estimator, X, y, folds,
        jobs=1, error_policy='raise', resume=resume, provenance=provenance,
        reuse_equivalent_folds=False)


def _scientific_bytes(run_dir):
    folder = _io_path(run_dir) / 'cv-checkpoints'
    files = [folder / 'identity.json']
    for child in ('folds', 'attempts', 'artifacts'):
        directory = folder / child
        if directory.exists():
            files.extend(directory.glob('*.json'))
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in files if path.is_file()}


def _origin_file(kind, run_dir):
    return _io_path(run_dir) / ('data-selection.json' if kind == 'multiclass-dwd'
                               else 'archive-location.json')


@pytest.mark.parametrize('kind', list(_CLIS))
def test_moved_identical_archive_reuses_real_checkpoint_without_refit(
        monkeypatch, archive, kind, record_property):
    """This same-source baseline reproduction needs no candidate-only API."""
    run_dir = archive.root / 'run'
    first = _invoke_cli(monkeypatch, kind, archive.path, run_dir)
    with _session(first) as store:
        original = store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    before = (len(_Surface.attempts), len(_Surface.predictions))
    relocated = archive.root / 'elsewhere' / 'renamed.npz'
    relocated.parent.mkdir()
    shutil.copyfile(archive.path, relocated)
    assert archive.path.read_bytes() == relocated.read_bytes()
    archive.path.unlink()
    resumed = _invoke_cli(monkeypatch, kind, relocated, run_dir, resume=True)
    for original_array, resumed_array in zip(first.args[2:6], resumed.args[2:6]):
        np.testing.assert_array_equal(original_array, resumed_array)
    with _session(resumed, resume=True) as store:
        replay = store.search({'lambd': [4.0]})
    assert before == (len(_Surface.attempts), len(_Surface.predictions))
    assert frozen == _scientific_bytes(run_dir)
    assert replay.checkpoint_summary_['new_folds'] == 0
    assert replay.checkpoint_summary_['reused_folds'] == len(first.args[6])
    np.testing.assert_array_equal(original.cv_results_['mean_test_score'],
                                  replay.cv_results_['mean_test_score'])
    record_property('archive_resume_evidence', json.dumps({
        'cli': kind, 'old_archive_missing': not archive.path.exists(),
        'archive_sha256': hashlib.sha256(relocated.read_bytes()).hexdigest(),
        'new_folds': replay.checkpoint_summary_['new_folds'],
        'reused_folds': replay.checkpoint_summary_['reused_folds'],
        'fit_calls': len(_Surface.attempts), 'prediction_calls': len(_Surface.predictions),
        'scientific_receipts_preserved': frozen == _scientific_bytes(run_dir),
        'scope': 'Actual CLI loading/settings and real CheckpointedCV; bounded synthetic estimator/runtime'
    }, sort_keys=True))


@pytest.mark.parametrize('kind', list(_CLIS))
def test_partial_checkpoint_after_relocation_runs_only_missing_folds(monkeypatch, archive, kind):
    run_dir = archive.root / 'partial'
    first = _invoke_cli(monkeypatch, kind, archive.path, run_dir)
    _Surface.fail_at = 2
    with pytest.raises(KeyboardInterrupt):
        with _session(first) as store:
            store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    committed_call = _Surface.attempts[0]
    committed_files = [name for name in frozen if name.startswith('folds/')]
    assert len(committed_files) == 1 and len(_Surface.attempts) == 2
    moved = archive.root / 'moved.npz'
    archive.path.rename(moved)
    resumed = _invoke_cli(monkeypatch, kind, moved, run_dir, resume=True)
    _Surface.fail_at = None
    with _session(resumed, resume=True) as store:
        result = store.search({'lambd': [4.0]})
    current = _scientific_bytes(run_dir)
    assert all(current[name] == content for name, content in frozen.items())
    assert _Surface.attempts.count(committed_call) == 1
    assert len(_Surface.attempts) == len(first.args[6]) + 1
    assert result.checkpoint_summary_['reused_folds'] == 1
    assert result.checkpoint_summary_['new_folds'] == len(first.args[6]) - 1


@pytest.mark.parametrize('kind', list(_CLIS))
def test_initial_archive_origin_is_kept_out_of_compatibility_settings(monkeypatch, archive, kind):
    run_dir = archive.root / 'origin'
    first = _invoke_cli(monkeypatch, kind, archive.path, run_dir)
    assert 'data_path' not in first.kwargs['settings']
    origin = _origin_file(kind, run_dir)
    saved = origin.read_bytes()
    record = json.loads(saved)
    assert record['data_path'] == str(archive.path.resolve())
    if kind != 'multiclass-dwd':
        assert record == {'schema_version': 1, 'data_path': str(archive.path.resolve())}
    else:
        assert {key: value for key, value in record.items() if key != 'data_path'} == \
               json.loads(json.dumps(first.kwargs['settings'], default=lambda value: value.tolist()))
    moved = archive.root / 'new-location.npz'
    archive.path.rename(moved)
    resumed = _invoke_cli(monkeypatch, kind, moved, run_dir, resume=True)
    assert 'Input archive: ' + str(moved.resolve()) in resumed.output
    assert 'data_path' not in resumed.kwargs['settings']
    assert origin.read_bytes() == saved
    assert checkpoint._stable(first.kwargs['settings']) == checkpoint._stable(resumed.kwargs['settings'])


@pytest.mark.parametrize('kind', list(_CLIS))
def test_repacked_identical_loaded_arrays_are_not_archive_byte_identity(monkeypatch, archive, kind):
    run_dir = archive.root / 'repacked'
    first = _invoke_cli(monkeypatch, kind, archive.path, run_dir)
    with _session(first) as store:
        store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    repacked = archive.root / 'compressed.npz'
    np.savez_compressed(repacked, **archive.arrays)
    assert repacked.read_bytes() != archive.path.read_bytes()
    resumed = _invoke_cli(monkeypatch, kind, repacked, run_dir, resume=True)
    with _session(resumed, resume=True) as store:
        result = store.search({'lambd': [4.0]})
    assert result.checkpoint_summary_['new_folds'] == 0
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))
    assert frozen == _scientific_bytes(run_dir)


@pytest.mark.parametrize('kind', list(_CLIS))
@pytest.mark.parametrize('mutation', ['train-X', 'train-y', 'test-X', 'test-y', 'train-order', 'test-order'])
def test_changed_loaded_data_refuses_before_reuse_or_refit(monkeypatch, archive, kind, mutation):
    run_dir = archive.root / 'changed'
    first = _invoke_cli(monkeypatch, kind, archive.path, run_dir)
    with _session(first) as store:
        store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    origin = _origin_file(kind, run_dir).read_bytes()
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    arrays = {key: value.copy() for key, value in archive.arrays.items()}
    split, field = mutation.split('-')
    feature_key, label_key = ('x_train', 'y_train') if split == 'train' else ('x_test', 'y_test')
    if field == 'X':
        arrays[feature_key][7, 0] += 1
    elif field == 'y':
        arrays[label_key][7] = 9
    else:
        for key in (feature_key, label_key):
            arrays[key][[7, 9]] = arrays[key][[9, 7]]
    changed = archive.root / 'changed-data.npz'
    np.savez(changed, **arrays)
    resumed = _invoke_cli(monkeypatch, kind, changed, run_dir, resume=True)
    with pytest.raises(checkpoint.CheckpointError):
        with _session(resumed, resume=True):
            pytest.fail('Changed data reached an admitted checkpoint session')
    assert frozen == _scientific_bytes(run_dir)
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))
    assert _origin_file(kind, run_dir).read_bytes() == origin


@pytest.mark.parametrize('mutation', ['preprocessing', 'digits', 'fold-order', 'seed', 'runtime'])
def test_nonlocation_compatibility_guards_remain_strict(monkeypatch, archive, mutation):
    run_dir = archive.root / 'policy'
    first = _invoke_cli(monkeypatch, 'binary-dwd', archive.path, run_dir)
    with _session(first) as store:
        store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    settings, args, estimator = deepcopy(first.kwargs['settings']), list(first.args), _Surface()
    if mutation == 'preprocessing':
        settings['preprocessing'] = {'mode': 'standard'}
    elif mutation == 'digits':
        settings['digits'] = [9, 7]
    elif mutation == 'fold-order':
        args[6] = list(reversed(args[6]))
    elif mutation == 'seed':
        estimator = _Surface(random_state=17)
    else:
        monkeypatch.setattr(checkpoint, 'runtime_fingerprint',
                            lambda **kwargs: synthetic_runtime(source='changed-runtime'))
    resumed = SimpleNamespace(args=tuple(args), kwargs=first.kwargs)
    with pytest.raises(checkpoint.CheckpointError):
        with _session(resumed, resume=True, settings=settings, estimator=estimator):
            pytest.fail('Changed compatibility settings reached an admitted session')
    assert frozen == _scientific_bytes(run_dir)
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))


def test_generic_provenance_path_and_old_path_bearing_identity_are_not_rewritten(monkeypatch, archive):
    run_dir = archive.root / 'generic'
    invocation = _invoke_cli(monkeypatch, 'binary-dwd', archive.path, run_dir)
    old_settings = {**invocation.kwargs['settings'], 'data_path': str(archive.path.resolve())}
    with _session(invocation, settings=old_settings) as store:
        store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    for changed in ({**old_settings, 'data_path': 'other.npz'}, invocation.kwargs['settings']):
        with pytest.raises(checkpoint.CheckpointError):
            with _session(invocation, resume=True, settings=changed):
                pytest.fail('Generic path-bearing identity was silently migrated')
        assert frozen == _scientific_bytes(run_dir)
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))


def test_multiclass_unselected_source_rows_still_belong_to_identity(monkeypatch, archive):
    run_dir = archive.root / 'full-source'
    first = _invoke_cli(monkeypatch, 'multiclass-dwd', archive.path, run_dir)
    with _session(first) as store:
        store.search({'lambd': [4.0]})
    settings = first.kwargs['settings']
    selected = set(settings['subsampling']['train_indices'].tolist())
    omitted = next(index for index in range(60) if index not in selected)
    arrays = {key: value.copy() for key, value in archive.arrays.items()}
    arrays['x_train'][omitted, 0] += 1
    altered = archive.root / 'unselected-row-change.npz'
    np.savez(altered, **arrays)
    frozen = _scientific_bytes(run_dir)
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    resumed = _invoke_cli(monkeypatch, 'multiclass-dwd', altered, run_dir, resume=True)
    for old, new in zip(first.args[2:6], resumed.args[2:6]):
        np.testing.assert_array_equal(old, new)
    with pytest.raises(checkpoint.CheckpointError):
        with _session(resumed, resume=True):
            pytest.fail('Full-source provenance was dropped with the location')
    assert frozen == _scientific_bytes(run_dir)
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))


@pytest.mark.parametrize('kind', list(_CLIS))
def test_relative_and_resolved_archive_paths_have_same_semantics(monkeypatch, archive, kind):
    run_dir = archive.root / 'resolved'
    first = _invoke_cli(monkeypatch, kind, archive.path, run_dir)
    with _session(first) as store:
        store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    monkeypatch.chdir(archive.root)
    resumed = _invoke_cli(monkeypatch, kind, Path('source.npz'), run_dir, resume=True)
    with _session(resumed, resume=True) as store:
        result = store.search({'lambd': [4.0]})
    assert result.checkpoint_summary_['new_folds'] == 0
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))
    assert frozen == _scientific_bytes(run_dir)


@pytest.mark.parametrize('kind', list(_CLIS))
def test_symlink_archive_origin_does_not_change_loaded_identity(monkeypatch, archive, kind):
    link = archive.root / 'archive-link.npz'
    try:
        link.symlink_to(archive.path)
    except (NotImplementedError, OSError) as error:
        unsupported = (isinstance(error, NotImplementedError)
                       or getattr(error, 'errno', None) in (errno.ENOSYS, errno.EOPNOTSUPP)
                       or (os.name == 'nt' and
                           (isinstance(error, PermissionError) or getattr(error, 'winerror', None) == 1314)))
        if unsupported:
            pytest.skip('Filesystem or Windows privileges do not permit this symlink control: ' + str(error))
        raise
    assert link.resolve() == archive.path.resolve()
    run_dir = archive.root / 'symlink'
    first = _invoke_cli(monkeypatch, kind, link, run_dir)
    with _session(first) as store:
        store.search({'lambd': [4.0]})
    origin = _origin_file(kind, run_dir).read_bytes()
    assert json.loads(origin)['data_path'] == str(archive.path.resolve())
    frozen = _scientific_bytes(run_dir)
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    resumed = _invoke_cli(monkeypatch, kind, archive.path, run_dir, resume=True)
    with _session(resumed, resume=True) as store:
        result = store.search({'lambd': [4.0]})
    assert result.checkpoint_summary_['new_folds'] == 0
    assert frozen == _scientific_bytes(run_dir)
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))
    assert _origin_file(kind, run_dir).read_bytes() == origin


def _adaptive(invocation, *, resume=False):
    _, _, X, y, X_test, y_test, folds, run_dir = invocation.args
    model = Pipeline([('ensemble', _Surface())])
    grid = {'ensemble__lambd': [1.0, 4.0, 16.0],
            'ensemble__gamma': [1.0, 4.0, 16.0]}
    config = adaptive_search.AdaptiveSearchConfig(refine=False, max_candidates=9)
    with redirect_stdout(io.StringIO()):
        return adaptive_search.run_adaptive_search(
            model, grid, X, y, X_test, y_test, folds, run_dir, config=config,
            settings=invocation.kwargs['settings'], resume=resume,
            reuse_equivalent_folds=False, error_policy='raise')


@pytest.mark.parametrize('interrupted', [False, True], ids=['completed', 'partial'])
@pytest.mark.parametrize('relocation', ['copy', 'rename'])
def test_adaptive_completed_and_partial_resume_preserve_work_and_aggregates(
        monkeypatch, archive, interrupted, relocation, record_property):
    run_dir = archive.root / 'adaptive'
    first = _invoke_cli(monkeypatch, 'binary-dwd', archive.path, run_dir)
    if interrupted:
        _Surface.fail_at = 3
        with pytest.raises(KeyboardInterrupt):
            _adaptive(first)
        _Surface.fail_at = None
        completed = None
        accepted_calls = list(_Surface.attempts[:2])
        assert len([name for name in _scientific_bytes(run_dir) if name.startswith('folds/')]) == 2
    else:
        initial = _adaptive(first)
        completed = initial['completed']
        accepted_calls = list(_Surface.attempts)
        assert completed['held_out_accuracy'] == 1.0
    frozen = _scientific_bytes(run_dir)
    origin = _origin_file('binary-dwd', run_dir).read_bytes()
    before = (len(_Surface.attempts), len(_Surface.predictions))
    moved = archive.root / 'adaptive-moved.npz'
    if relocation == 'copy':
        shutil.copyfile(archive.path, moved)
    else:
        archive.path.rename(moved)
    resumed = _invoke_cli(monkeypatch, 'binary-dwd', moved, run_dir, resume=True)
    result = _adaptive(resumed, resume=True)
    current = _scientific_bytes(run_dir)
    assert all(current[name] == content for name, content in frozen.items())
    assert all(_Surface.attempts.count(call) == 1 for call in accepted_calls)
    assert _origin_file('binary-dwd', run_dir).read_bytes() == origin
    if interrupted:
        assert result['completed']['held_out_accuracy'] == 1.0
        # Nine candidates, five fixed folds, one interrupted attempt, one refit.
        assert len(_Surface.attempts) == 47
        assert sum(min(rows) >= 100 for rows in _Surface.predictions) == 1
    else:
        assert result['resumed_completed'] and result['model'] is None
        assert result['completed'] == completed
        assert result['work_summary']['held_out_scoring']['state'] == 'not_performed_replay'
        assert before == (len(_Surface.attempts), len(_Surface.predictions))
        assert current == frozen
    after = (len(_Surface.attempts), len(_Surface.predictions))
    again = _adaptive(resumed, resume=True)
    assert again['resumed_completed'] and again['completed'] == result['completed']
    assert after == (len(_Surface.attempts), len(_Surface.predictions))
    record_property('archive_adaptive_resume_evidence', json.dumps({
        'initially_interrupted': interrupted, 'relocation': relocation,
        'initial_fit_calls': before[0], 'final_fit_calls': after[0],
        'final_prediction_calls': after[1], 'aggregate_replay_equal': True,
        'second_replay_added_fits_or_predictions': False,
        'original_committed_receipts_unchanged': True,
        'held_out_accuracy': result['completed']['held_out_accuracy']
    }, sort_keys=True))


@pytest.mark.parametrize('split', ['training', 'held-out'])
def test_completed_adaptive_changed_data_refuses_before_status_or_origin_write(
        monkeypatch, archive, split):
    run_dir = archive.root / 'completed-refusal'
    first = _invoke_cli(monkeypatch, 'binary-dwd', archive.path, run_dir)
    _adaptive(first)
    frozen = _scientific_bytes(run_dir)
    reports = {name: (_io_path(run_dir) / name).read_bytes()
               for name in ('run.json', 'status.json', 'completed.json', 'archive-location.json')}
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    arrays = {key: value.copy() for key, value in archive.arrays.items()}
    arrays['x_train' if split == 'training' else 'x_test'][7, 0] += 1
    changed = archive.root / 'changed-for-replay.npz'
    np.savez(changed, **arrays)
    resumed = _invoke_cli(monkeypatch, 'binary-dwd', changed, run_dir, resume=True)
    with pytest.raises(checkpoint.CheckpointError):
        _adaptive(resumed, resume=True)
    assert frozen == _scientific_bytes(run_dir)
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))
    assert all((_io_path(run_dir) / name).read_bytes() == content
               for name, content in reports.items())


@pytest.mark.parametrize('kind', list(_CLIS))
@pytest.mark.parametrize('representation', ['feature-dtype', 'label-dtype', 'fortran-layout', 'archive-keys'])
def test_archive_representation_obeys_existing_loaded_identity_scope(
        monkeypatch, archive, kind, representation):
    run_dir = archive.root / 'representation'
    first = _invoke_cli(monkeypatch, kind, archive.path, run_dir)
    with _session(first) as store:
        store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    arrays = {key: value.copy() for key, value in archive.arrays.items()}
    if representation == 'feature-dtype':
        for key in ('x_train', 'x_test'):
            arrays[key] = arrays[key].astype(np.float64)
    elif representation == 'label-dtype':
        for key in ('y_train', 'y_test'):
            arrays[key] = arrays[key].astype(np.float64)
    elif representation == 'fortran-layout':
        for key in ('x_train', 'x_test'):
            arrays[key] = np.asfortranarray(arrays[key])
    else:
        arrays['X_train'] = arrays.pop('x_train')
        arrays['X_test'] = arrays.pop('x_test')
    changed = archive.root / 'new-representation.npz'
    np.savez(changed, **arrays)
    resumed = _invoke_cli(monkeypatch, kind, changed, run_dir, resume=True)
    for old, new in zip(first.args[2:6], resumed.args[2:6]):
        np.testing.assert_array_equal(old, new)
        assert old.dtype == new.dtype
    # Binary loading normalizes these representations. Multiclass additionally
    # binds original full-source feature dtype and the archive key schema.
    should_refuse = kind == 'multiclass-dwd' and representation in ('feature-dtype', 'archive-keys')
    if should_refuse:
        with pytest.raises(checkpoint.CheckpointError):
            with _session(resumed, resume=True):
                pytest.fail('Full-source identity was relaxed beyond archive location')
    else:
        with _session(resumed, resume=True) as store:
            replay = store.search({'lambd': [4.0]})
        assert replay.checkpoint_summary_['new_folds'] == 0
    assert frozen == _scientific_bytes(run_dir)
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))


@pytest.mark.parametrize('kind', list(_CLIS))
def test_initial_origin_write_failure_precedes_checkpoint_and_fit(monkeypatch, archive, kind):
    run_dir = archive.root / 'origin-write-failure'
    origin = _origin_file(kind, run_dir)
    calls = []

    def fail_origin(path, payload, **kwargs):
        assert Path(path).name == origin.name
        calls.append((path, payload, kwargs))
        raise OSError('injected origin-write failure')

    with monkeypatch.context() as local:
        local.setattr(_CLIS[kind], 'write_json_atomic', fail_origin)
        with pytest.raises(OSError, match='injected origin-write failure'):
            _invoke_cli(local, kind, archive.path, run_dir)
    assert len(calls) == 1 and calls[0][1]['data_path'] == str(archive.path.resolve())
    assert run_dir.is_dir() and (run_dir / 'failures').is_dir()
    assert not origin.exists() and not _scientific_bytes(run_dir)
    assert not _Surface.attempts and not _Surface.predictions
    # A partial fresh directory is neither silently replaced nor a valid saved
    # checkpoint. Recovery of an interrupted origin write is not introduced.
    with pytest.raises(FileExistsError):
        _invoke_cli(monkeypatch, kind, archive.path, run_dir)
    resumed = _invoke_cli(monkeypatch, kind, archive.path, run_dir, resume=True)
    with pytest.raises(checkpoint.CheckpointError):
        with _session(resumed, resume=True):
            pytest.fail('An origin-write failure created a resumable checkpoint')
    assert not origin.exists() and not _scientific_bytes(run_dir)
    assert not _Surface.attempts and not _Surface.predictions


@pytest.mark.parametrize('origin_mutation', ['edit-path', 'remove-record'])
def test_observational_origin_cannot_authorize_changed_data(monkeypatch, archive, origin_mutation):
    run_dir = archive.root / 'observational-only'
    first = _invoke_cli(monkeypatch, 'binary-dwd', archive.path, run_dir)
    with _session(first) as store:
        store.search({'lambd': [4.0]})
    frozen = _scientific_bytes(run_dir)
    counts = (len(_Surface.attempts), len(_Surface.predictions))
    arrays = {key: value.copy() for key, value in archive.arrays.items()}
    arrays['x_train'][7, 0] += 1
    changed = archive.root / 'tampered-origin-data.npz'
    np.savez(changed, **arrays)
    origin = _origin_file('binary-dwd', run_dir)
    if origin_mutation == 'edit-path':
        origin.write_text(json.dumps({'schema_version': 1, 'data_path': str(changed.resolve())}),
                          encoding='utf-8')
        altered_origin = origin.read_bytes()
    else:
        origin.unlink()
        altered_origin = None
    resumed = _invoke_cli(monkeypatch, 'binary-dwd', changed, run_dir, resume=True)
    with pytest.raises(checkpoint.CheckpointError):
        with _session(resumed, resume=True):
            pytest.fail('Observational location overrode actual data identity')
    assert frozen == _scientific_bytes(run_dir)
    assert counts == (len(_Surface.attempts), len(_Surface.predictions))
    assert (origin.read_bytes() if origin.exists() else None) == altered_origin
