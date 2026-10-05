"""Compact pairwise diagnostics survive CV, final refitting and completed resume."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ensemble_dwd import EnsembleKernelDWD, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import cv_checkpoint
from search_support import CandidateEvaluator, new_run_directory


class DiagnosticBinary(ClassifierMixin, BaseEstimator):
    """Deterministic nearest-centroid response surface; no numerical solver."""
    fits = 0
    predictions = 0

    def __init__(self, lambd=2., gamma=2.):
        self.lambd = lambd
        self.gamma = gamma

    def fit(self, X, y):
        type(self).fits += 1
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.centroids_ = np.asarray([X[y == label].mean(axis=0) for label in self.classes_])
        self.n_iter_ = 7
        self.converged_ = False
        self.criterion_reached_ = True
        self.objective_tolerance_met_ = True
        self.stopping_reason_ = 'objective'
        self.backend_used_ = 'synthetic'
        self.base_diagnostics_ = [{'converged_': False, 'stopping_reason_': 'max_iter'}]
        self.final_estimator_ = SimpleNamespace(
            n_iter_=11, converged_=False, criterion_reached_=False,
            objective_tolerance_met_=False, stopping_reason_='max_iter')
        self.subset_indices_ = [np.arange(2), np.arange(2, len(y))]
        self.selected_indices_ = np.arange(min(3, len(y)))
        return self

    def predict(self, X):
        type(self).predictions += 1
        squared = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        result = self.classes_[np.argmin(squared, axis=1)]
        return result if (self.lambd, self.gamma) == (2., 2.) else -result


class MulticlassDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        DiagnosticBinary.fits = DiagnosticBinary.predictions = 0
        self.X = np.repeat(np.array([[-3., 0.], [0., 1.], [3., 2.]]), 4, axis=0)
        self.y = np.repeat(np.array(['ant', 'moth', 'zebra']), 4)
        self.pairs = [['ant', 'moth'], ['ant', 'zebra'], ['moth', 'zebra']]

    def assert_pairwise(self, facts):
        self.assertEqual(set(facts), {'pairwise_estimators'})
        self.assertEqual([row['pair'] for row in facts['pairwise_estimators']], self.pairs)
        for row in facts['pairwise_estimators']:
            child = row['diagnostics']
            self.assertIs(child['converged_'], False)
            self.assertIs(child['criterion_reached_'], True)
            self.assertIs(child['objective_tolerance_met_'], True)
            self.assertEqual(child['stopping_reason_'], 'objective')
            self.assertEqual(child['n_iter_'], 7)
            self.assertEqual(child['base_diagnostics_'],
                             [{'converged_': False, 'stopping_reason_': 'max_iter'}])
            self.assertEqual(child['final_estimator'], {
                'n_iter_': 11, 'converged_': False, 'criterion_reached_': False,
                'objective_tolerance_met_': False, 'stopping_reason_': 'max_iter'})
            self.assertIn('subset_sizes', child)
            self.assertEqual(child['selected_size'], 3)
        text = json.dumps(facts, allow_nan=False)
        for forbidden in ('pair_indices_', 'subset_indices_', 'selected_indices_',
                          'centroids_', 'X_train_', 'classes_'):
            self.assertNotIn(forbidden, text)

    def test_actual_fitted_ovo_through_candidate_and_pipeline(self):
        evaluator = CandidateEvaluator(Pipeline([('ovo', OvOClassifier(DiagnosticBinary()))]))
        evaluator.fit(self.X, self.y)
        self.assertEqual(DiagnosticBinary.fits, 3)
        facts = cv_checkpoint.diagnostics(evaluator)
        self.assert_pairwise(facts)
        self.assertEqual(facts, cv_checkpoint.diagnostics(evaluator.estimator_))
        self.assertEqual(DiagnosticBinary.fits, 3)
        self.assertEqual(DiagnosticBinary.predictions, 0)

    def test_nonsequential_original_numeric_labels_are_not_binary_codes(self):
        model = OvOClassifier(DiagnosticBinary()).fit(
            self.X, np.repeat(np.array([-8, 4, 91], dtype=np.int64), 4))
        pairs = [row['pair'] for row in cv_checkpoint.diagnostics(model)['pairwise_estimators']]
        self.assertEqual(pairs, [[-8, 4], [-8, 91], [4, 91]])
        json.dumps(pairs, allow_nan=False)

    def test_binary_ensemble_diagnostics_remain_exact_and_do_not_recurse_bases(self):
        model = EnsembleKernelDWD()
        model.base_diagnostics_ = [{'converged_': False, 'stopping_reason_': 'max_iter'}]
        model.final_estimator_ = SimpleNamespace(
            n_iter_=100, converged_=False, criterion_reached_=True,
            stopping_reason_='objective')
        model.cache_hit_ = True
        model.gamma_ = .125
        model.subset_indices_ = [np.arange(4), np.arange(4, 9)]
        model.selected_indices_ = np.array([0, 2, 6])
        # Generic estimators_ belongs to binary base learners, not class pairs.
        model.estimators_ = [SimpleNamespace(n_iter_=999)]
        self.assertEqual(cv_checkpoint.diagnostics(model), {
            'cache_hit_': True, 'gamma_': .125,
            'base_diagnostics_': model.base_diagnostics_,
            'subset_sizes': [4, 5], 'selected_size': 3,
            'final_estimator': {'n_iter_': 100, 'converged_': False,
                                'criterion_reached_': True, 'stopping_reason_': 'objective'}})

    def test_unfitted_ovo_and_children_without_flags_do_not_invent_facts(self):
        model = OvOClassifier(DiagnosticBinary())
        self.assertEqual(cv_checkpoint.diagnostics(model), {})
        model.pairs_ = (('left', 'right'),)
        model.estimators_ = [SimpleNamespace()]
        self.assertEqual(cv_checkpoint.diagnostics(model), {
            'pairwise_estimators': [{'pair': ['left', 'right'], 'diagnostics': {}}]})

    def test_real_fold_and_final_receipts_preserve_pairwise_diagnostics_on_resume(self):
        pipeline = Pipeline([('ovo', OvOClassifier(DiagnosticBinary(), n_jobs=1))])
        prefix = 'ovo__estimator__'
        grid = {prefix + 'lambd': [1., 2., 4.], prefix + 'gamma': [1., 2., 4.]}
        folds = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(self.X, self.y))
        config = adaptive_search.AdaptiveSearchConfig(prefix=prefix, refine=False, max_candidates=9)
        with tempfile.TemporaryDirectory(prefix='ovo-diagnostics-') as temporary, \
                patch.object(cv_checkpoint, 'runtime_fingerprint',
                             return_value={'synthetic_runtime': 'ovo-diagnostics-v1'}), \
                contextlib.redirect_stdout(io.StringIO()):
            run_dir = new_run_directory(Path(temporary) / 'run')

            def execute(resume=False):
                return adaptive_search.run_adaptive_search(
                    pipeline, grid, self.X, self.y, self.X.copy(), self.y.copy(),
                    folds, run_dir, config=config, jobs=1, resume=resume)

            result = execute()
            self.assertEqual(result['review']['state'], 'accepted')
            self.assertEqual(result['completed']['held_out_accuracy'], 1.)
            self.assertEqual(DiagnosticBinary.fits, 9 * 2 * 3 + 3)
            self.assert_pairwise(result['refit']['diagnostics'])
            receipts = list((run_dir / 'cv-checkpoints' / 'folds').glob('*.json'))
            self.assertEqual(len(receipts), 18)
            for path in receipts:
                payload = json.loads(path.read_text(encoding='utf-8'))['payload']
                self.assertEqual(payload['state'], 'complete')
                self.assert_pairwise(payload['diagnostics'])
            final = json.loads((run_dir / 'refit.json').read_text(encoding='utf-8'))
            self.assert_pairwise(final['diagnostics'])
            completed = json.loads((run_dir / 'cv-checkpoints' / 'artifacts' / 'completed.json')
                                   .read_text(encoding='utf-8'))['payload']['value']
            self.assertEqual(completed['refit']['diagnostics'], final['diagnostics'])
            counts = (DiagnosticBinary.fits, DiagnosticBinary.predictions)
            # The authoritative completed receipt restores missing public mirrors.
            (run_dir / 'refit.json').unlink()
            resumed = execute(resume=True)
            self.assertTrue(resumed['resumed_completed'])
            self.assertIsNone(resumed['model'])
            self.assertEqual((DiagnosticBinary.fits, DiagnosticBinary.predictions), counts)
            self.assertEqual(resumed['refit']['diagnostics'], final['diagnostics'])
            self.assertEqual(json.loads((run_dir / 'refit.json').read_text(encoding='utf-8')),
                             final)


if __name__ == '__main__':
    unittest.main()
