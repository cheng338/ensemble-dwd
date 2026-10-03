"""Small response surfaces for legacy selection and archive-validation repairs."""
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ensemble_dwd import IneligibleSelectionError


PATH = Path(__file__).resolve().parents[1] / 'examples' / 'tune_mnist.py'
SPEC = importlib.util.spec_from_file_location('legacy_example_repairs', PATH)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class ParameterSensitiveClassifier(ClassifierMixin, BaseEstimator):
    """Changing M changes the known score; no numerical solver is involved."""
    fit_sizes = []

    def __init__(self, lambd=1., gamma=1., theta=.1, sv_selection_ratio=.3,
                 M=5, q=1., backend='auto', kernel_kws=None, fail_test=False):
        self.lambd = lambd
        self.gamma = gamma
        self.theta = theta
        self.sv_selection_ratio = sv_selection_ratio
        self.M = M
        self.q = q
        self.backend = backend
        self.kernel_kws = kernel_kws
        self.fail_test = fail_test

    def fit(self, X, y):
        type(self).fit_sizes.append(len(y))
        if self.M == 0:
            raise IneligibleSelectionError('Controlled ineligible local candidate.')
        if self.M == -1:
            raise FloatingPointError('Controlled unexpected numerical failure.')
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if self.fail_test and len(X) == 4:
            raise FloatingPointError('Controlled held-out prediction failure.')
        labels = (X[:, 0] > 0).astype(int)
        return labels if self.M == 3 else 1 - labels


class LegacyExampleRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = example.new_run_directory(Path(self.temporary.name) / 'run')
        self.X = np.tile([[-1., 2.], [1., -2.]], (5, 1))
        self.y = np.tile([0, 1], 5)
        self.folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(self.X, self.y))
        self.grid = {'ensemble__lambd': [1.], 'ensemble__gamma': [1.],
                     'ensemble__theta': [.1], 'ensemble__sv_selection_ratio': [.3],
                     'ensemble__M': [3], 'ensemble__q': [2.],
                     'ensemble__backend': ['spectral'],
                     'ensemble__kernel_kws': [{'coef0': 2.}]}
        ParameterSensitiveClassifier.fit_sizes = []

    def read(self, name):
        return json.loads((self.run_dir / (name + '.json')).read_text(encoding='utf-8'))

    def run_workflow(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return example.run_search_workflow(
                Pipeline([('ensemble', ParameterSensitiveClassifier())]), self.grid,
                self.X, self.y, self.X[:4], self.y[:4], self.folds, self.run_dir,
                jobs=1, verbose=0, **kwargs)

    def test_refinement_retains_full_parent_and_distinguishes_extra_parameters(self):
        first = {name: values[0] for name, values in self.grid.items()}
        second = {**deepcopy(first), 'ensemble__M': 4}
        third = {**deepcopy(first), 'ensemble__kernel_kws': {'coef0': 3.}}
        parents = [first, second, third]
        original = deepcopy(parents)
        means = np.array([.9, .8, .7])
        grid = example.local_refinement_grid({
            'params': parents, 'mean_test_score': means,
            'split0_test_score': means, 'split1_test_score': means})
        self.assertEqual(len(grid), 27)
        self.assertEqual(sum(row['ensemble__M'] == [4] for row in grid), 9)
        self.assertEqual(sum(row['ensemble__kernel_kws'] == [{'coef0': 3.}]
                             for row in grid), 9)
        self.assertTrue(all(row['ensemble__q'] == [2.] and
                            row['ensemble__backend'] == ['spectral'] for row in grid))
        grid[0]['ensemble__kernel_kws'][0]['coef0'] = 100.
        self.assertEqual(parents, original)
        self.assertEqual(grid[1]['ensemble__kernel_kws'], [{'coef0': 2.}])

    def test_real_workflow_preserves_tuned_M_and_other_parent_parameters(self):
        result = self.run_workflow(refine=True)
        model = result['model'].named_steps['ensemble']
        self.assertEqual((model.M, model.q, model.backend, model.kernel_kws),
                         (3, 2., 'spectral', {'coef0': 2.}))
        self.assertEqual(result['completed']['mean_cv_accuracy'], 1.)
        self.assertEqual(result['completed']['held_out_accuracy'], 1.)
        self.assertEqual(result['selection']['source_stage'], 'local')
        self.assertEqual(self.read('local-cv')['cv_results']['mean_test_score'], [1.] * 9)
        self.assertEqual(ParameterSensitiveClassifier.fit_sizes.count(10), 1)

    def test_refinement_does_not_require_pickleable_callable_kernel(self):
        kernel = lambda X, Y: X @ Y.T
        parent = {name: values[0] for name, values in self.grid.items()}
        parent['ensemble__kernel'] = kernel
        means = np.array([1.])
        grid = example.local_refinement_grid({
            'params': [parent], 'mean_test_score': means,
            'split0_test_score': means, 'split1_test_score': means})
        self.assertEqual(len(grid), 9)
        self.assertTrue(all(row['ensemble__kernel'][0] is kernel for row in grid))

    def test_worse_local_result_cannot_discard_coarse_incumbent(self):
        local = {**self.grid, 'ensemble__M': [5]}
        with patch.object(example, 'local_refinement_grid', return_value=[local]):
            result = self.run_workflow(refine=True)
        self.assertEqual(self.read('coarse-cv')['cv_results']['mean_test_score'], [1.])
        self.assertEqual(self.read('local-cv')['cv_results']['mean_test_score'], [0.])
        self.assertEqual(result['selection']['source_stage'], 'coarse')
        self.assertEqual(result['model'].named_steps['ensemble'].M, 3)
        self.assertEqual(result['completed']['held_out_accuracy'], 1.)

    def test_all_ineligible_local_results_preserve_coarse_and_failure_records(self):
        local = {**self.grid, 'ensemble__M': [0]}
        with patch.object(example, 'local_refinement_grid', return_value=[local]):
            result = self.run_workflow(refine=True)
        self.assertEqual(self.read('local-cv')['fully_valid'], [False])
        self.assertEqual(result['selection']['source_stage'], 'coarse')
        self.assertEqual(len(list((self.run_dir / 'failures').glob('failure-*.json'))), 5)
        self.assertEqual(ParameterSensitiveClassifier.fit_sizes.count(10), 1)

    def test_unexpected_local_failure_still_stops_without_refit(self):
        local = {**self.grid, 'ensemble__M': [-1]}
        with patch.object(example, 'local_refinement_grid', return_value=[local]):
            with self.assertRaisesRegex(FloatingPointError, 'unexpected numerical'):
                self.run_workflow(refine=True)
        self.assertEqual(self.read('status')['state'], 'failed')
        self.assertNotIn(10, ParameterSensitiveClassifier.fit_sizes)
        self.assertFalse((self.run_dir / 'completed.json').exists())

    def test_accepted_review_tracks_successful_refit_and_test(self):
        result = self.run_workflow(selection_review=lambda _: [])
        review = self.read('review')
        self.assertTrue(review['final_refit_performed'])
        self.assertTrue(review['held_out_scored'])
        self.assertEqual(review, result['review'])

    def test_review_keeps_refit_success_when_held_out_prediction_fails(self):
        self.grid['ensemble__fail_test'] = [True]
        with self.assertRaisesRegex(FloatingPointError, 'held-out prediction'):
            self.run_workflow(selection_review=lambda _: [])
        review = self.read('review')
        self.assertTrue(review['final_refit_performed'])
        self.assertFalse(review['held_out_scored'])
        self.assertEqual(self.read('status')['state'], 'failed')
        self.assertFalse((self.run_dir / 'completed.json').exists())

    def test_interrupts_preserve_completed_cv_without_ordinary_failure_record(self):
        original_fit = ParameterSensitiveClassifier.fit
        for index, error in enumerate((KeyboardInterrupt(), SystemExit(42))):
            with self.subTest(interruption=type(error).__name__):
                self.run_dir = example.new_run_directory(Path(self.temporary.name) / f'interrupt-{index}')

                def interrupted_fit(estimator, X, y):
                    if len(y) == 10:
                        raise error
                    return original_fit(estimator, X, y)

                with patch.object(ParameterSensitiveClassifier, 'fit', interrupted_fit):
                    with self.assertRaises(type(error)) as caught:
                        self.run_workflow(selection_review=lambda _: [])
                self.assertIs(caught.exception, error)
                self.assertEqual(self.read('status'), {
                    'state': 'interrupted', 'phase': 'final_refit',
                    'exception_type': type(error).__name__, 'resume_supported': False})
                self.assertEqual(self.read('coarse-cv')['fully_valid'], [True])
                self.assertEqual(self.read('selection')['mean_cv_accuracy'], 1.)
                self.assertFalse(self.read('review')['final_refit_performed'])
                self.assertEqual(list((self.run_dir / 'failures').glob('failure-*.json')), [])
                self.assertFalse((self.run_dir / 'completed.json').exists())


class BinaryComplexInputTests(unittest.TestCase):
    def test_complex_features_are_rejected_before_lossy_conversion(self):
        for imaginary in (0., 1.):
            with self.subTest(imaginary=imaginary), warnings.catch_warnings():
                warnings.simplefilter('error')
                with self.assertRaisesRegex(ValueError, 'features must be real'):
                    example.select_pair(np.ones((4, 2), dtype=complex) + imaginary * 1j,
                                        np.array([2, 3, 2, 3]), 2, 3)

    def test_complex_labels_are_rejected_before_lossy_conversion(self):
        for imaginary in (0., 1.):
            with self.subTest(imaginary=imaginary), warnings.catch_warnings():
                warnings.simplefilter('error')
                with self.assertRaisesRegex(ValueError, 'labels must be real'):
                    example.select_pair(np.ones((4, 2)),
                                        np.array([2, 3, 2, 3], dtype=complex) + imaginary * 1j, 2, 3)

    def test_binary_archive_loader_rejects_complex_supplied_test_split(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'complex-test.npz'
            np.savez(path, x_train=np.ones((4, 2)), y_train=[2, 3, 2, 3],
                     x_test=np.ones((4, 2), dtype=complex), y_test=[2, 3, 2, 3])
            with self.assertRaisesRegex(ValueError, 'features must be real'):
                example.load_local_data(path, 2, 3)


class BinaryFeaturePreflightTests(unittest.TestCase):
    def test_nonfinite_and_empty_features_are_rejected_in_supplied_arrays(self):
        labels = np.array([2, 3, 2, 3])
        invalid = [np.full((4, 2), value) for value in (np.nan, np.inf, -np.inf)]
        invalid += [np.empty((4, 0)), np.empty((4, 2, 0))]
        for features in invalid:
            with self.subTest(shape=features.shape), self.assertRaisesRegex(ValueError, 'nonempty and finite'):
                example.select_pair(features, labels, 2, 3)

    def test_binary_and_multiclass_feature_validation_agree(self):
        import tune_mnist_multiclass as multiclass
        labels = np.arange(10)
        valid = np.arange(40, dtype=np.uint16).reshape(10, 2, 2)
        binary_X, binary_y = example.select_pair(valid, labels, 2, 3)
        full_X, full_y = multiclass.validate_split(valid, labels, 'Validation')
        keep = np.isin(full_y, [2, 3])
        np.testing.assert_array_equal(binary_X, full_X[keep])
        np.testing.assert_array_equal(binary_y, full_y[keep])
        self.assertEqual(binary_X.dtype, np.float64)
        invalid = [np.full((10, 2), value) for value in (np.nan, np.inf, -np.inf)]
        invalid += [np.empty((10, 0)), np.empty((10, 2, 0)),
                    np.ones((10, 2), dtype=bool), np.full((10, 2), '1')]
        for features in invalid:
            with self.subTest(dtype=features.dtype, shape=features.shape):
                with self.assertRaises(ValueError):
                    example.select_pair(features, labels, 2, 3)
                with self.assertRaises(ValueError):
                    multiclass.validate_split(features, labels, 'Validation')

    def test_invalid_held_out_features_stop_both_binary_clis_before_run_or_cv(self):
        import adaptive_search
        import tune_mnist_svm as svm
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            archive = folder / 'invalid-test.npz'
            invalid = [np.full((4, 2), value) for value in (np.nan, np.inf, -np.inf)]
            invalid += [np.empty((4, 0)), np.empty((4, 2, 0))]
            for features in invalid:
                np.savez(archive, x_train=np.ones((20, 2)), y_train=np.tile([2, 3], 10),
                         x_test=features, y_test=[2, 3, 2, 3])
                for cli in (example, svm):
                    with self.subTest(cli=cli.__name__, shape=features.shape):
                        with (patch.object(cli, 'new_run_directory') as directory,
                              patch.object(adaptive_search, 'run_adaptive_search') as search,
                              contextlib.redirect_stdout(io.StringIO())):
                            with self.assertRaisesRegex(ValueError, 'nonempty and finite'):
                                cli.main(['--data', str(archive), '--digits', '2', '3', '--run',
                                          '--output-dir', str(folder / 'must-not-exist')])
                        directory.assert_not_called()
                        search.assert_not_called()
            self.assertFalse((folder / 'must-not-exist').exists())


if __name__ == '__main__':
    unittest.main()
