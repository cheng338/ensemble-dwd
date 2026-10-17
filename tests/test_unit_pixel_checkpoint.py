"""Pixel scaling keeps exact checkpoint, reuse and partial-pair semantics."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint
import mnist_preprocessing
from mnist_preprocessing import UnitPixelScaler, make_preprocessor
import pair_recovery as recovery
import test_equivalent_fold_reuse as reuse_tests
import test_pair_recovery as recovery_tests


class CustomPixelScaler(UnitPixelScaler):
    """Subclasses must not inherit exact-transformer reuse permission."""


class PixelCheckpointTests(unittest.TestCase):
    assert_scores_equal = reuse_tests.EquivalentFoldReuseTests.assert_scores_equal

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pix-cv-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        values, self.y = reuse_tests.binary_data()
        self.X = (values + 3.) * 40.
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        limits = threadpool_limits(1)
        limits.__enter__()
        self.addCleanup(limits.__exit__, None, None, None)

    def model(self, kind='dwd', input_range='0-255'):
        model = reuse_tests.pipeline(kind)
        return model.set_params(scale=UnitPixelScaler(input_range=input_range))

    def session(self, name, model, *, resume=False, reuse=False, jobs=1):
        folder = self.root / name
        (folder / 'failures').mkdir(parents=True, exist_ok=True)
        return checkpoint.CheckpointedCV(folder, model, self.X, self.y, self.folds,
                jobs=jobs, error_policy='raise', resume=resume,
                reuse_equivalent_folds=reuse, provenance={'test': 'unit-pixels'})

    def test_exact_identity_and_equivalence_do_not_cross_input_conventions(self):
        models = [reuse_tests.pipeline(), self.model(), self.model(input_range='0-1')]
        identities = [checkpoint._digest(checkpoint._stable(model)) for model in models]
        proofs = [checkpoint._fold_equivalence(model, 'a' * 64, 0, self.y) for model in models]
        self.assertEqual(len(set(identities)), 3)
        self.assertTrue(all(proof is not None for proof in proofs))
        self.assertEqual(len({proof['key'] for proof in proofs}), 3)
        default = reuse_tests.pipeline().set_params(scale=make_preprocessor())
        self.assertEqual(checkpoint._digest(checkpoint._stable(default)), identities[1])
        for model in models:
            a = checkpoint._fold_equivalence(model, 'a' * 64, 0, self.y)
            b = checkpoint._fold_equivalence(clone(model).set_params(model__sv_selection_ratio=.88),
                                             'a' * 64, 0, self.y)
            self.assertEqual(a['key'], b['key'])

    def test_invalid_custom_and_modified_transformers_cannot_reuse(self):
        for scaler in (UnitPixelScaler(input_range='auto'),
                       CustomPixelScaler(input_range='0-255')):
            self.assertIsNone(checkpoint._fold_equivalence(
                self.model().set_params(scale=scaler), 'a' * 64, 0, self.y))
        original = UnitPixelScaler.transform
        def changed(self, X):
            return original(self, X)
        with patch.object(UnitPixelScaler, 'transform', changed):
            self.assertIsNone(checkpoint._fold_equivalence(self.model(), 'a' * 64, 0, self.y))
            with self.assertRaisesRegex(ValueError, 'unchanged'):
                recovery.validate_supported(recovery_tests.pipeline().set_params(
                    scale=UnitPixelScaler(input_range='0-255')))

    def test_real_dwd_and_svm_ratio_reuse_matches_independent_fits(self):
        grid = {'model__sv_selection_ratio': [.875, .88]}
        for kind in ('dwd', 'svm'):
            with self.subTest(kind=kind):
                model = self.model(kind)
                with self.session(kind + '-direct', model) as session:
                    direct = session.search(grid)
                with self.session(kind + '-reuse', model, reuse=True) as session:
                    reused = session.search(grid)
                self.assert_scores_equal(reused, direct)
                self.assertEqual(reused.checkpoint_summary_['executed_folds'], 2)
                self.assertEqual(reused.checkpoint_summary_['equivalent_reused_folds'], 2)
                with patch.object(checkpoint, '_fit_fold', side_effect=AssertionError('duplicate fit')):
                    with self.session(kind + '-reuse', model, reuse=True, resume=True) as session:
                        replay = session.search(grid)
                self.assertEqual(replay.best_score_, reused.best_score_)

    def test_changed_preprocessing_refuses_resume_before_fitting(self):
        model = self.model('svm')
        with self.session('identity', model) as session:
            session.search({})
        before = recovery_tests.hashes(self.root / 'identity/cv-checkpoints', '*.json')
        for scaler in (StandardScaler(), UnitPixelScaler(input_range='0-1')):
            with patch.object(checkpoint, '_fit_fold', side_effect=AssertionError('mismatched resume fit')):
                with self.assertRaises(checkpoint.CheckpointError):
                    with self.session('identity', clone(model).set_params(scale=scaler), resume=True):
                        self.fail('A changed input convention cannot resume')
        self.assertEqual(before, recovery_tests.hashes(self.root / 'identity/cv-checkpoints', '*.json'))

    def test_new_default_refuses_a_standardized_run_before_fitting(self):
        model = self.model('svm').set_params(scale=StandardScaler())
        with self.session('old-standard', model) as session:
            session.search({})
        before = recovery_tests.hashes(self.root / 'old-standard/cv-checkpoints', '*.json')
        current = clone(model).set_params(scale=make_preprocessor())
        with patch.object(checkpoint, '_fit_fold', side_effect=AssertionError('mismatched resume fit')):
            with self.assertRaises(checkpoint.CheckpointError):
                with self.session('old-standard', current, resume=True):
                    self.fail('Unit-pixel default cannot adopt standardized results')
        self.assertEqual(before, recovery_tests.hashes(self.root / 'old-standard/cv-checkpoints', '*.json'))

    def test_two_worker_pipeline_matches_serial(self):
        model = self.model('svm')
        grid = {'model__C': [1., 2.]}
        with self.session('serial', model) as session:
            serial = session.search(grid)
        with self.session('parallel', model, jobs=2) as session:
            parallel = session.search(grid)
        self.assert_scores_equal(parallel, serial)

    def test_helper_source_is_guarded_in_shallow_runtime_and_pair_config(self):
        expected = (EXAMPLES / 'mnist_preprocessing.py').resolve()
        with patch.object(checkpoint, '_file_hash', wraps=checkpoint._file_hash) as hashed:
            checkpoint.runtime_fingerprint(check_inventory=False)
        self.assertIn(expected, {Path(call.args[0]).resolve() for call in hashed.call_args_list})
        config = recovery.normalize_config(True, None)
        self.assertIn(str(expected), config['source_hashes'])
        invalid = deepcopy(config)
        invalid['source_hashes'][str(expected)] = '0' * 64
        with self.assertRaisesRegex(checkpoint.CheckpointError, 'source changed'):
            recovery.check_source_hashes(invalid)
        with patch.object(mnist_preprocessing, '__file__', str(self.root / 'other.py')):
            with self.assertRaisesRegex(checkpoint.CheckpointError, 'preprocessing'):
                checkpoint.runtime_fingerprint(check_inventory=False)


class PixelPairRecoveryTests(unittest.TestCase):
    interrupt_after_two_pairs = recovery_tests.PairRecoveryIntegrationTests.interrupt_after_two_pairs
    observe_models = recovery_tests.PairRecoveryIntegrationTests.observe_models

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pix-pair-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        X, self.y = make_blobs(n_samples=96, n_features=6, centers=4,
                              cluster_std=2., random_state=20261007)
        self.raw = (X + 20.) * 5.
        assert self.raw.min() >= 0 and self.raw.max() <= 255
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(X, self.y))
        limits = threadpool_limits(1)
        limits.__enter__()
        self.addCleanup(limits.__exit__, None, None, None)

    def execute(self, name, mode, input_range, *, enabled=True, resume=False):
        folder = self.root / name
        (folder / 'failures').mkdir(parents=True, exist_ok=True)
        model = recovery_tests.pipeline(mode).set_params(scale=UnitPixelScaler(input_range=input_range))
        with checkpoint.CheckpointedCV(folder, model, self.X, self.y, self.folds,
                jobs=1, error_policy='raise', resume=resume, pair_recovery=enabled,
                provenance={'test': 'unit-pixel-pair-recovery'}) as session:
            return session.search({})

    def test_both_ranges_preserve_partial_pair_recovery_and_predictions(self):
        for mode in ('full', 'lean'):
            for input_range in ('0-255', '0-1'):
                with self.subTest(mode=mode, input_range=input_range):
                    self.X = self.raw if input_range == '0-255' else self.raw / 255.
                    name = mode + '-' + input_range
                    baseline, restored = [], []
                    with self.observe_models(baseline):
                        expected = self.execute(name + '-plain', mode, input_range, enabled=False)
                    with self.observe_models(restored), self.interrupt_after_two_pairs():
                        with self.assertRaisesRegex(KeyboardInterrupt, 'controlled pair boundary'):
                            self.execute(name, mode, input_range)
                    before = recovery_tests.hashes(self.root / name / 'cv-checkpoints/folds', '*.json')
                    self.assertEqual(len(before), 1)
                    with self.observe_models(restored):
                        actual = self.execute(name, mode, input_range, resume=True)
                    self.assertEqual(len(restored), 2)
                    for direct, resumed in zip(baseline, restored, strict=True):
                        for field in ('scaler', 'pairs', 'rows', 'models'):
                            self.assertEqual(direct[field], resumed[field])
                        for a, b in zip(direct['scores'], resumed['scores'], strict=True):
                            np.testing.assert_array_equal(a, b)
                        np.testing.assert_array_equal(direct['predictions'], resumed['predictions'])
                        np.testing.assert_array_equal(direct['votes'], resumed['votes'])
                    for field in ('params', 'mean_test_score', 'std_test_score', 'rank_test_score'):
                        np.testing.assert_equal(expected.cv_results_[field], actual.cv_results_[field])
                    after = recovery_tests.hashes(self.root / name / 'cv-checkpoints/folds', '*.json')
                    self.assertTrue(all(after[k] == v for k, v in before.items()))
                    self.assertEqual(actual.work_summary_['pair_recovery']['current']['recovered_pairs'], 2)
                    self.assertEqual(actual.work_summary_['pair_recovery']['current']['fitted_pairs'], 4)
                    with patch.object(recovery, 'fit_pipeline', side_effect=AssertionError('completed pair fit')):
                        replay = self.execute(name, mode, input_range, resume=True)
                    self.assertEqual(replay.checkpoint_summary_['reused_folds'], 2)


if __name__ == '__main__':
    unittest.main()
