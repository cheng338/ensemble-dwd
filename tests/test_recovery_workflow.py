"""Checkpoint recovery regression checks for ordinary pytest execution.

These tests do not mock runtime/source identity or fit a replacement estimator.
Patches observe science, inject an interruption, or reject prohibited work.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import cv_checkpoint as cp
import pair_recovery as pr
import _pair_store as ps


RESULT_FIELDS = ('params', 'mean_test_score', 'std_test_score',
                 'rank_test_score', 'split0_test_score', 'split1_test_score')


def read(path):
    return json.loads(cp._io_path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(cp._io_path(path).read_bytes()).hexdigest()


def files(folder, pattern='*'):
    folder = cp._io_path(folder)
    return {str(p.relative_to(folder)): sha(p)
            for p in folder.rglob(pattern) if p.is_file()}


def save(path, value):
    with cp._io_path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


@pytest.fixture
def panel(tmp_path):
    root = cp._io_path(tmp_path / 'workflow')
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(912)
    X = np.vstack([rng.normal(0, .12, (16, 2)) + center
                   for center in [(-2, -1), (2, -1), (0, 2), (0, -3)]])
    y = np.repeat(np.arange(4), 16)
    folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(X, y))
    with threadpool_limits(limits=1):
        yield root, X, y, folds


def pipeline(mode):
    controls = dict(lambd=.1, max_iter=4, stopping='fixed',
                    random_state=42, prediction_batch_size=32)
    if mode == 'full':
        binary = KernGDWD(kernel='rbf', kernel_kws={'gamma': .2},
                          initialization='zero', **controls)
    else:
        binary = EnsembleKernelDWD(M=1, theta=1., gamma=.2,
                    sv_selection_ratio=.88,
                    store_base_estimators=(mode == 'retained'), **controls)
    return Pipeline([('scale', StandardScaler()),
                     ('model', OvOClassifier(binary, n_jobs=1))])


def session(panel, name, mode, *, pair, resume=False, reuse=False):
    root, X, y, folds = panel
    run = root / name
    (run / 'failures').mkdir(parents=True, exist_ok=True)
    return cp.CheckpointedCV(run, pipeline(mode), X, y, folds,
        jobs=1, error_policy='raise', resume=resume, pair_recovery=pair,
        reuse_equivalent_folds=reuse,
        provenance={'study': 'ir24-public-workflow', 'seed': 912,
                    'arm': 'candidate', 'scope': mode})


def execute(panel, name, mode, *, pair, resume=False, reuse=False, grid=None):
    with session(panel, name, mode, pair=pair, resume=resume, reuse=reuse) as cv:
        return cv.search({} if grid is None else grid)


def mathematical_model(model):
    if type(model) is KernGDWD:
        return pr.snapshot({name: getattr(model, name)
                            for name in ('classes_', 'n_features_in_', '_Xfit',
                                         'dual_coef_', 'intercept_')})
    assert type(model) is EnsembleKernelDWD
    return {'classes': pr.snapshot(model.classes_),
            'features': int(model.n_features_in_),
            'selected': pr.snapshot(model.selected_indices_),
            'subsets': pr.snapshot(model.subset_indices_),
            'consensus': pr.snapshot(model.consensus_scores_),
            'final': mathematical_model(model.final_estimator_),
            'bases': (None if model.estimators_ is None else
                      [mathematical_model(m) for m in model.estimators_])}


@contextmanager
def observe_science(panel, collection):
    _, X, _, _ = panel
    original = cp.eligible_accuracy

    def score(evaluator, validation_X, validation_y):
        result = original(evaluator, validation_X, validation_y)
        assert evaluator.eligible_
        fitted = evaluator.estimator_
        scaler, ovo = (step for _, step in fitted.steps)
        transformed = scaler.transform(X)
        collection.append({
            'scaler': pr.snapshot(scaler),
            'pairs': pr.snapshot(ovo.pairs_),
            'rows': pr.snapshot(ovo.pair_indices_),
            'models': [mathematical_model(m) for m in ovo.estimators_],
            'scores': [pr.snapshot(m.decision_function(transformed))
                       for m in ovo.estimators_],
            'votes': pr.snapshot(ovo.vote_counts(transformed)),
            'predictions': pr.snapshot(fitted.predict(X)),
            'accuracy_counts': dict(evaluator.accuracy_counts_),
        })
        return result

    with patch.object(cp, 'eligible_accuracy', score):
        yield


@contextmanager
def interrupt_after_two_pairs():
    original = pr.FoldContext._pair_finished

    def finished(context, event):
        original(context, event)
        if context.fold == 1 and len(context.events) == 2:
            raise KeyboardInterrupt('checkpoint test: two committed pairs in unfinished fold')

    with patch.object(pr.FoldContext, '_pair_finished', finished):
        yield


def source_proof(folder):
    manifest = read(folder / 'identity.json')['payload']
    identity = manifest['identity']
    expected_provenance = [cp._stable({'study': 'ir24-public-workflow',
        'seed': 912, 'arm': 'candidate', 'scope': mode})
        for mode in ('full', 'lean', 'retained')]
    assert identity['provenance'] in expected_provenance
    pins = identity.get('pair_recovery', {}).get('source_hashes', {})
    for path, expected in pins.items():
        assert sha(path) == expected
    return {'arm': 'candidate', 'identity_sha256': manifest['identity_sha256'],
            'identity_file_sha256': sha(folder / 'identity.json'),
            'encoded_provenance': identity['provenance'],
            'pair_source_pins': pins,
            'current_source': {Path(module.__file__).name:
                               {'path': str(Path(module.__file__).resolve()),
                                'sha256': sha(module.__file__)}
                               for module in (cp, pr, ps)},
            'cross_version_adoption': False}


def assert_scores_equal(left, right):
    for field in RESULT_FIELDS:
        np.testing.assert_equal(left.cv_results_[field], right.cv_results_[field])
    assert left.best_index_ == right.best_index_


@pytest.mark.parametrize('mode', ['full', 'lean', 'retained'])
def test_partial_pair_payload_recovery_matches_plain_science(panel, mode):
    root = panel[0]
    expected_science, resumed_science = [], []
    with observe_science(panel, expected_science):
        expected = execute(panel, 'plain', mode, pair=False)
    with observe_science(panel, resumed_science), interrupt_after_two_pairs(), \
            pytest.raises(KeyboardInterrupt, match='checkpoint test: two committed pairs'):
        execute(panel, 'partial', mode, pair=True)
    folder = cp._io_path(root / 'partial' / 'cv-checkpoints')
    completed = files(folder / 'folds', '*.json')
    assert len(completed) == 1 and len(resumed_science) == 1
    stores = [p for p in (folder / 'pair-stores').iterdir()
              if p.is_dir() and len(list(p.glob('*.pkl'))) == 2]
    assert len(stores) == 1
    store = stores[0]
    models = sorted(store.glob('*.pkl'))
    damaged, healthy = models
    original = damaged.read_bytes()
    assert original
    damaged.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    original_pairs = files(store)
    assert len(models) == 2
    # Preserve the injected evidence outside the live store for later review.
    (root / 'original-payload.bin').write_bytes(original)
    (root / 'damaged-payload.bin').write_bytes(damaged.read_bytes())
    proof = source_proof(folder)
    save(root / 'BEFORE.json', {'source': proof, 'completed': completed,
         'pair_store': str(store), 'pair_files': original_pairs,
         'damaged': damaged.name, 'healthy': healthy.name})

    cleanup_observations = []
    original_cleanup = pr._cleanup_store

    def cleanup(context, record):
        if context.fold == 1:
            # Existing records are not overwritten while recomputing this key.
            for name, expected_hash in original_pairs.items():
                assert sha(store / name) == expected_hash
            assert files(folder / 'folds', '*.json').items() >= completed.items()
            authoritative = cp._valid_fold(folder / 'folds' / (context.task + '.json'),
                context.identity, context.task, parameters=context.parameters,
                fold=context.fold, validation_size=len(context.validation))
            assert authoritative == record
            summary = record['pair_recovery']
            assert summary['fitted_pairs'] == 5
            assert summary['recovered_pairs'] == 1
            assert summary['fitted_unsaved_pairs'] == 1
            failed = [event for event in summary['pairs']
                      if event.get('failed_load_seconds', 0) > 0]
            assert len(failed) == 1
            assert failed[0]['status'] == 'fitted_unsaved'
            assert failed[0]['cache_committed'] is False
            assert failed[0]['load_seconds'] == 0
            assert failed[0]['historical_fit_seconds'] == 0
            assert summary['pair_failed_load_seconds'] == failed[0]['failed_load_seconds']
            cleanup_observations.append({'old_bytes_preserved': True,
                                         'authoritative_fold_verified': True,
                                         'summary': summary})
        return original_cleanup(context, record)

    with observe_science(panel, resumed_science), patch.object(pr, '_cleanup_store', cleanup):
        actual = execute(panel, 'partial', mode, pair=True, resume=True)
    assert len(cleanup_observations) == 1
    assert len(expected_science) == len(resumed_science) == 2
    assert resumed_science == expected_science
    assert_scores_equal(expected, actual)
    after = files(folder / 'folds', '*.json')
    assert after.items() >= completed.items()
    assert not list((folder / 'pair-stores').rglob('*.pkl'))
    assert source_proof(folder) == proof
    current = actual.work_summary_['pair_recovery']['current']
    assert current['fitted_pairs'] == 5 and current['recovered_pairs'] == 1
    failed_load = actual.work_summary_['pair_recovery']['current_worker_seconds'].get('pair_failed_load', 0)
    assert failed_load == cleanup_observations[0]['summary']['pair_failed_load_seconds']
    with patch.object(pr, 'fit_pipeline', side_effect=AssertionError('completed folds refitted')), \
            patch.object(cp, 'eligible_accuracy', side_effect=AssertionError('completed folds rescored')):
        replay = execute(panel, 'partial', mode, pair=True, resume=True)
    assert replay.checkpoint_summary_['reused_folds'] == 2
    assert replay.checkpoint_summary_['new_folds'] == 0
    assert replay.work_summary_['evaluations']['executed'] == 0
    assert files(folder / 'folds', '*.json') == after
    assert_scores_equal(actual, replay)
    save(root / 'RESULT.json', {'arm': 'candidate', 'outcome': 'recovered',
         'new_pair_fits_after_damage': 5, 'recovered_pairs': 1,
         'plain_science': expected_science, 'resumed_science': resumed_science,
         'cleanup': cleanup_observations,
         'work_summary': actual.work_summary_,
         'replay_summary': replay.checkpoint_summary_,
         'replay_work_summary': replay.work_summary_,
         'old_completed_folds_preserved': True,
         'source': proof})


@pytest.mark.parametrize('request_scope', ['alias-only', 'full-grid'])
def test_missing_alias_source_recovers_exact_attempt_without_fit_or_score(panel, request_scope):
    root = panel[0]
    grid = {'model__estimator__sv_selection_ratio': [.88, .93]}
    initial_science = []
    with observe_science(panel, initial_science):
        initial = execute(panel, 'aliases', 'retained', pair=False, reuse='auto', grid=grid)
    folder = cp._io_path(root / 'aliases' / 'cv-checkpoints')
    records = [(p, read(p)['payload']) for p in sorted((folder / 'folds').glob('*.json'))]
    sources = [(p, r) for p, r in records if 'equivalent_reuse' not in r]
    aliases = [(p, r) for p, r in records if 'equivalent_reuse' in r]
    assert len(sources) == len(aliases) == 2
    assert len(initial_science) == 2
    source_path, source = sources[0]
    dependents = [r for _, r in aliases
                  if r['equivalent_reuse']['source_task_sha256'] == source['task_sha256']]
    assert len(dependents) == 1
    assert dependents[0]['equivalent_reuse']['source_receipt_sha256'] == cp._digest(source)
    attempts = files(folder / 'attempts', '*.json')
    duplicate_paths = [p for p in (folder / 'attempts').glob('*.json')
                       if read(p)['payload'] == source]
    assert len(duplicate_paths) == 1
    assert source['state'] == 'complete'
    before = files(folder / 'folds', '*.json')
    proof = source_proof(folder)
    save(root / 'SOURCE-BACKUP.json', read(source_path))
    save(root / 'BEFORE.json', {'source': proof, 'fold_files': before,
         'attempt_files': attempts, 'missing_source': source_path.name,
         'exact_duplicate': duplicate_paths[0].name,
         'source_payload_sha256': cp._digest(source)})
    source_path.unlink()
    requested = ({'model__estimator__sv_selection_ratio': [.93]}
                 if request_scope == 'alias-only' else grid)

    with patch.object(cp, '_fit_fold', side_effect=AssertionError('completed source refitted')), \
            patch.object(cp, 'eligible_accuracy', side_effect=AssertionError('completed source rescored')):
        recovered = execute(panel, 'aliases', 'retained', pair=False, resume=True,
                            reuse='auto', grid=requested)
        assert read(source_path)['payload'] == source
        assert files(folder / 'folds', '*.json') == before
        assert files(folder / 'attempts', '*.json') == attempts
        assert source_proof(folder) == proof
        # Repeated full-grid resume also checks every preserved source and alias.
        replay = execute(panel, 'aliases', 'retained', pair=False, resume=True,
                         reuse='auto', grid=grid)
    assert recovered.checkpoint_summary_['executed_folds'] == 0
    assert replay.checkpoint_summary_['executed_folds'] == 0
    assert_scores_equal(initial, replay)
    expected_column = 1 if request_scope == 'alias-only' else None
    for field in ('mean_test_score', 'std_test_score', 'split0_test_score', 'split1_test_score'):
        expected_scores = initial.cv_results_[field]
        if expected_column is not None:
            expected_scores = expected_scores[expected_column:expected_column + 1]
        np.testing.assert_equal(recovered.cv_results_[field], expected_scores)
    assert files(folder / 'folds', '*.json') == before
    assert files(folder / 'attempts', '*.json') == attempts
    save(root / 'RESULT.json', {'arm': 'candidate', 'outcome': 'exact_source_restored',
         'scope': request_scope, 'new_fits': 0, 'new_scores': 0,
         'initial_science': initial_science,
         'source_payload_sha256': cp._digest(source),
         'source_restored_sha256': sha(source_path),
         'folds_preserved': True, 'attempts_preserved': True,
         'recovered_summary': recovered.checkpoint_summary_,
         'replay_summary': replay.checkpoint_summary_, 'source': proof})
