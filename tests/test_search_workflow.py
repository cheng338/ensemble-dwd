"""Small classifier controls for durable external search and failure handling."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_dwd import IneligibleSelectionError, mnist_param_grid


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'tune_mnist.py'
SPEC = importlib.util.spec_from_file_location('ensemble_search_workflow_example', EXAMPLE_PATH)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)
from search_support import (CandidateEvaluator, data_fingerprint, eligible_accuracy,
                            json_safe, new_run_directory, write_json_atomic)
import search_support


class ControlledClassifier(ClassifierMixin, BaseEstimator):
    events = []

    def __init__(self, mode='good', checkpoint_dir=None, lambd=1., gamma=1.,
                 theta=0.1, sv_selection_ratio=0.3):
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        self.lambd = lambd
        self.gamma = gamma
        self.theta = theta
        self.sv_selection_ratio = sv_selection_ratio

    def fit(self, X, y):
        type(self).events.append(('fit', len(y)))
        if self.mode == 'ineligible':
            raise IneligibleSelectionError('The selected rows contain one class.')
        if self.mode == 'numerical':
            raise FloatingPointError('Numerical check failed.')
        if self.mode == 'memory':
            raise MemoryError('Allocation failed.')
        if len(y) == 10 and self.checkpoint_dir is not None:
            directory = Path(self.checkpoint_dir)
            assert (directory / 'coarse-cv.json').is_file()
            assert (directory / 'selection.json').is_file()
        if self.mode == 'final_failure' and len(y) == 10:
            raise IneligibleSelectionError('Full-training selection is ineligible.')
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        type(self).events.append(('predict', len(X)))
        if self.mode == 'prediction_failure':
            raise FloatingPointError('Prediction check failed.')
        if self.mode == 'bad_shape':
            return np.zeros(len(X) + 1, dtype=int)
        return (np.asarray(X)[:, 0] > 0).astype(int)


class SearchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = new_run_directory(Path(self.temporary.name) / 'run')
        self.X = np.tile([[-1., 2.], [1., -2.]], (5, 1))
        self.y = np.tile([0, 1], 5)
        self.X_test = self.X[:4].copy()
        self.y_test = self.y[:4].copy()
        self.folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(self.X, self.y))
        ControlledClassifier.events = []

    def read(self, name):
        return json.loads((self.run_dir / name).read_text(encoding='utf-8'))

    def failures(self):
        return [json.loads(path.read_text(encoding='utf-8'))
                for path in (self.run_dir / 'failures').glob('failure-*.json')]

    def run_workflow(self, grid, **kwargs):
        pipeline = Pipeline([('scale', StandardScaler()),
                             ('ensemble', ControlledClassifier(checkpoint_dir=str(self.run_dir)))])
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return example.run_search_workflow(
                pipeline, grid, self.X, self.y, self.X_test, self.y_test,
                self.folds, self.run_dir, verbose=0, **kwargs)

    def test_five_successful_folds_checkpoint_before_failed_final_fit(self):
        with self.assertRaisesRegex(IneligibleSelectionError, 'Full-training'):
            self.run_workflow({'ensemble__mode': ['final_failure']})
        self.assertEqual(ControlledClassifier.events.count(('fit', 8)), 5)
        self.assertEqual(ControlledClassifier.events.count(('fit', 10)), 1)
        self.assertNotIn(('predict', 4), ControlledClassifier.events)
        cv = self.read('coarse-cv.json')
        self.assertEqual(cv['fully_valid'], [True])
        for fold in range(5):
            self.assertEqual(cv['cv_results'][f'split{fold}_test_score'], [1.])
        self.assertEqual(self.read('selection.json')['parameters'],
                         {'ensemble__mode': 'final_failure'})
        self.assertEqual(self.read('refit.json')['state'], 'failed')
        self.assertEqual(self.read('status.json')['phase'], 'final_refit')
        self.assertFalse((self.run_dir / 'completed.json').exists())
        self.assertTrue(any(row['phase'] == 'final_refit' for row in self.failures()))

    def test_only_expected_ineligible_is_skipped_and_logged(self):
        result = self.run_workflow({'ensemble__mode': ['ineligible', 'good']})
        self.assertEqual(self.read('coarse-cv.json')['fully_valid'], [False, True])
        self.assertEqual(self.read('coarse-cv.json')['cv_results']['mean_test_score'], [None, 1.])
        self.assertEqual(len(self.failures()), 5)
        self.assertTrue(all(row['category'] == 'ineligible_selection' for row in self.failures()))
        self.assertEqual(result['completed']['held_out_accuracy'], 1.)
        self.assertEqual(ControlledClassifier.events.count(('predict', 4)), 1)
        self.assertEqual(self.read('refit.json')['state'], 'completed')
        self.assertEqual(self.read('status.json')['state'], 'completed')

    def test_debug_raise_propagates_expected_failure(self):
        with self.assertRaises(IneligibleSelectionError):
            self.run_workflow({'ensemble__mode': ['ineligible']}, error_policy='raise')
        self.assertEqual(self.read('status.json')['state'], 'failed')
        self.assertNotIn(('predict', 4), ControlledClassifier.events)

    def test_numerical_failure_aborts_and_records_cause(self):
        with self.assertRaises(FloatingPointError):
            self.run_workflow({'ensemble__mode': ['numerical', 'good']})
        self.assertTrue(any(row['category'] == 'numerical_error' and row['phase'] == 'cv_fit'
                            for row in self.failures()))
        self.assertFalse((self.run_dir / 'selection.json').exists())
        self.assertNotIn(('predict', 4), ControlledClassifier.events)

    def test_memory_failure_aborts_and_records_cause(self):
        with self.assertRaises(MemoryError):
            self.run_workflow({'ensemble__mode': ['memory', 'good']})
        self.assertTrue(any(row['category'] == 'memory_error' for row in self.failures()))
        self.assertEqual(self.read('status.json')['state'], 'failed')
        self.assertNotIn(('predict', 4), ControlledClassifier.events)

    def test_prediction_and_scorer_failures_are_logged(self):
        for mode, phase, error_type in [('prediction_failure', 'cv_predict', FloatingPointError),
                                        ('bad_shape', 'cv_scoring', ValueError)]:
            with self.subTest(mode=mode):
                adapter = CandidateEvaluator(ControlledClassifier(mode), failure_dir=str(self.run_dir / 'failures'))
                adapter.fit(self.X, self.y)
                with self.assertRaises(error_type):
                    eligible_accuracy(adapter, self.X_test, self.y_test)
                self.assertTrue(any(row['phase'] == phase for row in self.failures()))

    def test_parallel_expected_failures_have_unique_structured_records(self):
        self.run_workflow({'ensemble__mode': ['ineligible', 'good']}, jobs=2)
        failures = self.failures()
        self.assertEqual(len(failures), 5)
        self.assertEqual(len(list((self.run_dir / 'failures').glob('failure-*.json'))), 5)
        expected_hashes = {data_fingerprint(self.X[train], self.y[train])['X']['sha256']
                           for train, _ in self.folds}
        for row in failures:
            self.assertEqual(row['phase'], 'cv_fit')
            self.assertEqual(row['category'], 'ineligible_selection')
            self.assertIn(row['training_data']['X']['sha256'], expected_hashes)
            self.assertEqual(row['parameters']['ensemble__mode'], 'ineligible')
            self.assertIsInstance(row['pid'], int)
            self.assertNotIn('values', row['training_data']['X'])

    def test_all_invalid_search_preserves_complete_cv_results(self):
        with self.assertRaisesRegex(ValueError, 'No candidate'):
            self.run_workflow({'ensemble__mode': ['ineligible']},
                              selection_review=lambda _: self.fail('An invalid candidate reached review.'))
        self.assertEqual(self.read('coarse-cv.json')['fully_valid'], [False])
        self.assertIsNone(self.read('coarse-cv.json')['selected_parameters'])
        self.assertEqual(self.read('status.json')['state'], 'failed')
        self.assertFalse((self.run_dir / 'selection.json').exists())
        self.assertNotIn(('fit', 10), ControlledClassifier.events)

    def test_selection_review_holds_refit_and_test_after_durable_cv(self):
        called = []

        def review(parameters):
            self.assertTrue((self.run_dir / 'coarse-cv.json').is_file())
            self.assertEqual(self.read('selection.json')['parameters'], parameters)
            called.append(parameters.copy())
            parameters['ensemble__mode'] = 'numerical'  # A callback cannot alter the saved selection.
            return ['ensemble__C', 'ensemble__gamma']

        result = self.run_workflow({'ensemble__mode': ['good']}, selection_review=review)
        self.assertEqual(called, [{'ensemble__mode': 'good'}])
        self.assertEqual(ControlledClassifier.events.count(('fit', 8)), 5)
        self.assertNotIn(('fit', 10), ControlledClassifier.events)
        self.assertNotIn(('predict', 4), ControlledClassifier.events)
        self.assertIsNone(result['model'])
        self.assertIsNone(result['completed'])
        self.assertEqual(result['selection']['parameters'], {'ensemble__mode': 'good'})
        self.assertEqual(self.read('review.json'), result['review'])
        self.assertEqual(self.read('review.json')['reasons'], ['ensemble__C', 'ensemble__gamma'])
        self.assertEqual(self.read('status.json')['state'], 'needs_review')
        self.assertEqual(self.read('status.json')['phase'], 'parameter_review')
        self.assertFalse(self.read('status.json')['held_out_scored'])
        self.assertFalse(self.read('status.json')['final_refit_performed'])
        self.assertFalse((self.run_dir / 'refit.json').exists())
        self.assertFalse((self.run_dir / 'completed.json').exists())
        self.assertEqual(self.failures(), [])

    def test_empty_selection_review_continues_existing_final_workflow(self):
        for index, response in enumerate(([], None)):
            with self.subTest(response=response):
                self.run_dir = new_run_directory(Path(self.temporary.name) / f'accepted-{index}')
                ControlledClassifier.events = []
                result = self.run_workflow({'ensemble__mode': ['good']},
                                           selection_review=lambda _: response)
                self.assertEqual(self.read('review.json')['state'], 'accepted')
                self.assertEqual(ControlledClassifier.events.count(('fit', 10)), 1)
                self.assertEqual(ControlledClassifier.events.count(('predict', 4)), 1)
                self.assertEqual(result['completed']['held_out_accuracy'], 1.)
                self.assertEqual(self.read('status.json')['state'], 'completed')

    def test_review_exception_preserves_cv_and_records_correct_failure_phase(self):
        def failed_review(parameters):
            raise RuntimeError('Review callback failed.')

        with self.assertRaisesRegex(RuntimeError, 'Review callback failed'):
            self.run_workflow({'ensemble__mode': ['good']}, selection_review=failed_review)
        self.assertTrue((self.run_dir / 'coarse-cv.json').is_file())
        self.assertTrue((self.run_dir / 'selection.json').is_file())
        self.assertEqual(self.read('status.json')['phase'], 'parameter_review')
        self.assertEqual(self.read('status.json')['state'], 'failed')
        self.assertTrue(any(row['phase'] == 'parameter_review' for row in self.failures()))
        self.assertFalse((self.run_dir / 'refit.json').exists())
        self.assertFalse((self.run_dir / 'completed.json').exists())
        self.assertNotIn(('fit', 10), ControlledClassifier.events)
        self.assertNotIn(('predict', 4), ControlledClassifier.events)

    def test_malformed_review_response_cannot_silently_continue(self):
        with self.assertRaisesRegex(ValueError, 'list of nonempty strings'):
            self.run_workflow({'ensemble__mode': ['good']}, selection_review=lambda _: 'boundary')
        self.assertEqual(self.read('status.json')['phase'], 'parameter_review')
        self.assertNotIn(('fit', 10), ControlledClassifier.events)
        self.assertNotIn(('predict', 4), ControlledClassifier.events)

    def test_noncallable_review_rejected_before_cv(self):
        with self.assertRaisesRegex(TypeError, 'callable or None'):
            self.run_workflow({'ensemble__mode': ['good']}, selection_review=True)
        self.assertEqual(ControlledClassifier.events, [])
        self.assertFalse((self.run_dir / 'run.json').exists())

    def test_dwd_boundary_review_uses_lambd_gamma_including_refinement_outside_range(self):
        grid = mnist_param_grid('ensemble__')
        middle = {name: values[len(values) // 2] for name, values in grid.items()}
        for end in (0, -1):
            fractions = {**middle, 'ensemble__theta': grid['ensemble__theta'][end],
                         'ensemble__sv_selection_ratio': grid['ensemble__sv_selection_ratio'][end]}
            self.assertEqual(example.boundary_parameters(fractions, grid), [])
            endpoints = {**fractions, 'ensemble__lambd': grid['ensemble__lambd'][end],
                         'ensemble__gamma': grid['ensemble__gamma'][end]}
            self.assertEqual(example.boundary_parameters(endpoints, grid),
                             ['ensemble__lambd', 'ensemble__gamma'])
        outside = {**middle, 'ensemble__lambd': min(grid['ensemble__lambd']) / 4,
                   'ensemble__gamma': max(grid['ensemble__gamma']) * 10}
        self.assertEqual(example.boundary_parameters(outside, grid),
                         ['ensemble__lambd', 'ensemble__gamma'])

    def test_dwd_cli_wires_same_gate_before_final_fit_and_test(self):
        directory = Path(self.temporary.name) / 'dwd-cli'
        calls = []

        def workflow(pipeline, grid, *args, **kwargs):
            parameters = {name: values[len(values) // 2] for name, values in grid.items()}
            parameters['ensemble__lambd'] = min(grid['ensemble__lambd']) / 4
            reasons = kwargs['selection_review'](parameters)
            self.assertEqual(reasons, ['ensemble__lambd'])
            self.assertTrue(kwargs['refine'])
            calls.append(parameters)
            return {'model': None, 'completed': None, 'review': {'reasons': reasons}}

        output = io.StringIO()
        argv = [str(EXAMPLE_PATH), '--data', 'unused.npz', '--run', '--refine',
                '--output-dir', str(directory)]
        with (patch.object(sys, 'argv', argv), patch.object(example, 'load_local_data',
                return_value=(self.X, self.X_test, self.y, self.y_test)), patch.object(
                example, 'run_search_workflow', side_effect=workflow),
                contextlib.redirect_stdout(output)):
            example.main()
        self.assertEqual(len(calls), 1)
        self.assertEqual(ControlledClassifier.events, [])
        self.assertIn('lambda/gamma endpoints using training-only CV', output.getvalue())
        self.assertIn('test set was not scored', output.getvalue())
        self.assertFalse((directory / 'completed.json').exists())

    def test_local_stage_uses_identical_folds_and_no_intermediate_refit(self):
        grid = {'ensemble__lambd': [1.], 'ensemble__gamma': [1.],
                'ensemble__theta': [.1], 'ensemble__sv_selection_ratio': [.3]}
        self.run_workflow(grid, refine=True)
        coarse, local = self.read('coarse-cv.json'), self.read('local-cv.json')
        self.assertEqual(coarse['folds'], local['folds'])
        self.assertEqual(coarse['data'], local['data'])
        self.assertEqual(len(local['cv_results']['params']), 9)
        self.assertEqual(self.read('selection.json')['source_stage'], 'local')
        self.assertEqual(ControlledClassifier.events.count(('fit', 10)), 1)
        self.assertEqual(ControlledClassifier.events.count(('predict', 4)), 1)

    def test_adapter_clears_a_previous_fitted_model_after_failure(self):
        adapter = CandidateEvaluator(ControlledClassifier())
        adapter.fit(self.X, self.y)
        self.assertTrue(adapter.eligible_)
        adapter.set_params(estimator__mode='ineligible').fit(self.X, self.y)
        self.assertFalse(adapter.eligible_)
        self.assertFalse(hasattr(adapter, 'estimator_'))
        self.assertTrue(np.isnan(eligible_accuracy(adapter, self.X_test, self.y_test)))
        with self.assertRaises(IneligibleSelectionError):
            adapter.predict(self.X_test)
        adapter.set_params(estimator__mode='memory')
        with self.assertRaises(MemoryError):
            adapter.fit(self.X, self.y)
        self.assertFalse(adapter.eligible_)
        self.assertEqual(adapter.failure_['category'], 'memory_error')

    def test_strict_json_masks_nonfinite_and_preserves_exact_fold_order(self):
        masked = np.ma.array([1., 2., np.inf, np.nan], mask=[False, True, False, False])
        self.assertEqual(json_safe(masked), [1., None, None, None])
        self.run_workflow({'ensemble__mode': ['good']})
        manifest = self.read('run.json')
        self.assertEqual(manifest['data']['training'], data_fingerprint(self.X, self.y))
        self.assertEqual(manifest['folds'], [{'train': train.tolist(), 'validation': test.tolist()}
                                           for train, test in self.folds])

    def test_existing_run_and_existing_records_are_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            new_run_directory(self.run_dir)
        path = self.run_dir / 'sentinel.json'
        write_json_atomic(path, {'value': 1})
        with self.assertRaises(FileExistsError):
            write_json_atomic(path, {'value': 2})
        self.assertEqual(self.read('sentinel.json'), {'value': 1})

    def test_failed_worker_logging_cannot_replace_original_exception(self):
        for mode, error_type in [('memory', MemoryError), ('numerical', FloatingPointError)]:
            with self.subTest(mode=mode), warnings.catch_warnings():
                warnings.simplefilter('error')
                adapter = CandidateEvaluator(ControlledClassifier(mode),
                                             failure_dir=str(self.run_dir / 'failures'))
                with patch.object(search_support, 'write_json_atomic', side_effect=OSError('Disk full')):
                    with self.assertRaises(error_type):
                        adapter.fit(self.X, self.y)

    def test_failed_final_logging_cannot_replace_original_fit_exception(self):
        original_write = search_support.write_json_atomic

        def fail_failure_records(path, payload, **kwargs):
            if Path(path).parent.name == 'failures' or payload.get('state') == 'failed':
                raise OSError('Disk full')
            return original_write(path, payload, **kwargs)

        with warnings.catch_warnings(), patch.object(search_support, 'write_json_atomic', fail_failure_records):
            warnings.simplefilter('error', RuntimeWarning)
            with self.assertRaisesRegex(IneligibleSelectionError, 'Full-training'):
                self.run_workflow({'ensemble__mode': ['final_failure']})
        self.assertTrue((self.run_dir / 'coarse-cv.json').is_file())
        self.assertTrue((self.run_dir / 'selection.json').is_file())
        self.assertNotIn(('predict', 4), ControlledClassifier.events)


if __name__ == '__main__':
    unittest.main()
