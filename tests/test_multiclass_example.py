"""Small archive and orchestration checks for the ten-digit tuning example."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.model_selection import ParameterGrid, train_test_split


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'tune_mnist_multiclass.py'
SPEC = importlib.util.spec_from_file_location('ensemble_multiclass_tuning_example', EXAMPLE_PATH)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class MulticlassArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'mnist.npz'
        self.X_train = np.arange(400, dtype=np.uint16).reshape(100, 2, 2)
        self.y_train = np.tile(np.arange(10), 10)
        self.X_test = np.arange(1000, 1160, dtype=np.uint16).reshape(40, 2, 2)
        self.y_test = np.tile(np.arange(10), 4)

    def write_archive(self, upper=False):
        np.savez(self.path, **{
            'X_train' if upper else 'x_train': self.X_train, 'y_train': self.y_train,
            'X_test' if upper else 'x_test': self.X_test, 'y_test': self.y_test})

    def test_both_schemas_preserve_splits_and_original_indices(self):
        for upper in (False, True):
            with self.subTest(upper=upper):
                self.write_archive(upper)
                result = example.load_local_multiclass_data(self.path, train_size=60, test_size=20)
                train, test, train_y, test_y, provenance = result
                again = example.load_local_multiclass_data(self.path, train_size=60, test_size=20)
                for first, second in zip(result[:4], again[:4]):
                    assert_array_equal(first, second)
                ti = provenance['subsampling']['train_indices']
                vi = provenance['subsampling']['test_indices']
                assert_array_equal(train, self.X_train[ti].reshape(60, 4))
                assert_array_equal(test, self.X_test[vi].reshape(20, 4))
                assert_array_equal(train_y, self.y_train[ti])
                assert_array_equal(test_y, self.y_test[vi])
                self.assertTrue(np.all(np.diff(ti) > 0))
                self.assertTrue(np.all(np.diff(vi) > 0))
                self.assertTrue(train.max() < test.min())
                self.assertEqual(train.dtype, np.float64)
                self.assertEqual(test.dtype, np.float64)
                self.assertEqual(provenance['source_splits']['training']['rows'], 100)
                self.assertEqual(provenance['source_splits']['test']['rows'], 40)
                self.assertEqual(provenance['selected_splits']['training']['class_counts'],
                                 {str(digit): 6 for digit in range(10)})
                self.assertEqual(provenance['selected_splits']['test']['class_counts'],
                                 {str(digit): 2 for digit in range(10)})

    def test_default_test_selection_keeps_every_supplied_test_row(self):
        self.write_archive()
        _, X_test, _, y_test, provenance = example.load_local_multiclass_data(self.path, train_size=60)
        assert_array_equal(X_test, self.X_test.reshape(40, 4))
        assert_array_equal(y_test, self.y_test)
        assert_array_equal(provenance['subsampling']['test_indices'], np.arange(40))
        self.assertIsNone(provenance['subsampling']['requested_test_size'])

    def test_pooled_archive_and_incompatible_feature_counts_are_rejected(self):
        np.savez(self.path, X=self.X_train, y=self.y_train)
        with self.assertRaisesRegex(ValueError, 'pooled'):
            example.load_local_multiclass_data(self.path, train_size=60)
        self.X_test = np.zeros((40, 5))
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'feature counts differ'):
            example.load_local_multiclass_data(self.path, train_size=60)

    def test_missing_or_invalid_digit_labels_in_either_split_are_rejected(self):
        for field in ('y_train', 'y_test'):
            original = getattr(self, field)
            for invalid in (np.where(original == 9, 8, original),
                            np.where(original == 9, 10, original),
                            np.where(original == 9, 9.5, original),
                            np.where(original == 9, np.nan, original)):
                with self.subTest(field=field, labels=invalid[:10]):
                    setattr(self, field, invalid)
                    self.write_archive()
                    with self.assertRaisesRegex(ValueError, 'all ten'):
                        example.load_local_multiclass_data(self.path, train_size=60)
            setattr(self, field, original)

    def test_shape_nonfinite_and_nonnumeric_features_are_rejected(self):
        for X in (np.zeros(100), np.zeros((100, 0)), np.full((100, 4), np.inf),
                  np.full((100, 4), 'invalid'), np.zeros((99, 4))):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                example.validate_split(X, self.y_train, 'Training split')

    def test_subsampling_enforces_available_size_and_preserves_all_digits(self):
        for size in (0, 9, 101, 60.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                example.stratified_indices(self.y_train, size)
        assert_array_equal(example.stratified_indices(self.y_train, 100), np.arange(100))
        chosen = example.stratified_indices(self.y_train, 10)
        assert_array_equal(np.unique(self.y_train[chosen]), np.arange(10))
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'between 10 and 100; got 6000'):
            example.load_local_multiclass_data(self.path)

    def test_near_full_subsets_keep_requested_rows_and_all_digits_reproducibly(self):
        for total, retained in ((100, 99), (100, 91), (10000, 9999)):
            with self.subTest(total=total, retained=retained):
                y = np.tile(np.arange(10), total // 10)
                chosen = example.stratified_indices(y, retained)
                assert_array_equal(chosen, example.stratified_indices(y, retained))
                self.assertEqual(len(chosen), retained)
                self.assertTrue(np.all(np.diff(chosen) > 0))
                assert_array_equal(np.unique(y[chosen]), np.arange(10))
                retained_counts = np.unique(y[chosen], return_counts=True)[1]
                self.assertLessEqual(np.ptp(retained_counts), 1)

    def test_near_full_imbalanced_subsets_retain_singleton_classes(self):
        y = np.concatenate([np.zeros(11, dtype=int), np.arange(1, 10)])
        for retained in (19, 11):
            with self.subTest(retained=retained):
                chosen = example.stratified_indices(y, retained)
                self.assertEqual(len(chosen), retained)
                self.assertTrue(np.all(np.diff(chosen) > 0))
                assert_array_equal(np.unique(y[chosen]), np.arange(10))
                # All nine singleton source rows must survive, even when the
                # majority class absorbs more removals than its nominal quota.
                assert_array_equal(chosen[y[chosen] != 0], np.arange(11, 20))

    def test_standard_6000_row_subsetting_matches_existing_sklearn_path(self):
        y = np.tile(np.arange(10), 6000)
        expected, _ = train_test_split(np.arange(len(y)), train_size=6000,
                                       stratify=y, random_state=42)
        expected.sort()
        assert_array_equal(example.stratified_indices(y, 6000), expected)

    def test_run_records_provenance_and_passes_raw_folds_to_shared_workflow(self):
        self.write_archive()
        output = Path(self.temporary.name) / 'run'
        sentinel = {'model': object()}
        with patch.object(example, 'run_search_workflow', return_value=sentinel) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                result = example.main([
                    '--data', str(self.path), '--train-size', '60', '--test-size', '20',
                    '--folds', '3', '--jobs', '2', '--smoke-grid', '--no-cache',
                    '--output-dir', str(output), '--run'])
        self.assertIs(result, sentinel)
        pipeline, grid, X_train, y_train, X_test, y_test, folds, run_dir = run.call_args.args
        self.assertEqual(len(ParameterGrid(grid)), 8)
        self.assertEqual((len(y_train), len(y_test), len(folds)), (60, 20, 3))
        self.assertGreater(X_train.mean(), 1)  # Scaling has not happened before CV.
        self.assertFalse(hasattr(pipeline.named_steps['scale'], 'mean_'))
        self.assertEqual(pipeline.named_steps['ovo'].n_jobs, 1)
        self.assertEqual(pipeline.named_steps['ovo'].estimator.M, 5)
        self.assertFalse(pipeline.named_steps['ovo'].estimator.store_base_estimators)
        self.assertEqual(run.call_args.kwargs['jobs'], 2)
        self.assertIs(run.call_args.kwargs['refine'], False)
        self.assertEqual(run.call_args.kwargs['error_policy'], 'nan')
        self.assertEqual(run_dir, output.resolve())
        for train, validation in folds:
            self.assertFalse(np.intersect1d(train, validation).size)
            assert_array_equal(np.unique(y_train[train]), np.arange(10))
            assert_array_equal(np.unique(y_train[validation]), np.arange(10))
        record = json.loads((output / 'data-selection.json').read_text(encoding='utf-8'))
        self.assertEqual(record['n_pairs'], 45)
        self.assertEqual(record['fold_count'], 3)
        self.assertEqual(record['grid_kind'], 'smoke')
        self.assertEqual(len(record['subsampling']['train_indices']), 60)
        self.assertEqual(len(record['subsampling']['test_indices']), 20)
        self.assertEqual(record['selected_splits']['training']['class_counts'],
                         {str(digit): 6 for digit in range(10)})
        self.assertEqual(len(record['source_splits']['training']['fingerprint']['X']['sha256']), 64)

    def test_too_few_rows_per_class_stops_before_search_or_output_creation(self):
        self.write_archive()
        output = Path(self.temporary.name) / 'bad-run'
        with patch.object(example, 'run_search_workflow') as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    example.main(['--data', str(self.path), '--train-size', '10',
                                  '--folds', '2', '--output-dir', str(output), '--run'])
        run.assert_not_called()
        self.assertFalse(output.exists())


class MulticlassPreviewTests(unittest.TestCase):
    def test_default_and_smoke_preview_load_no_data_and_report_45_pair_budget(self):
        for argv, candidates, budget in (([], 810, '1,093,500'),
                                         (['--smoke-grid'], 8, '10,800')):
            with self.subTest(argv=argv):
                output = io.StringIO()
                with patch.object(example, 'load_local_multiclass_data') as load:
                    with contextlib.redirect_stdout(output):
                        example.main(argv + ['--data', 'does-not-exist.npz'])
                load.assert_not_called()
                self.assertIn(f'{candidates} candidates', output.getvalue())
                self.assertIn(budget, output.getvalue())
                self.assertIn('no data were loaded', output.getvalue())

    def test_grid_prefixes_and_smoke_parameter_values(self):
        broad = example.search_grid()
        smoke = example.search_grid(smoke=True)
        self.assertEqual(len(ParameterGrid(broad)), 810)
        self.assertEqual(len(ParameterGrid(smoke)), 8)
        self.assertTrue(all(name.startswith('ovo__estimator__') for name in broad))
        self.assertEqual(smoke['ovo__estimator__lambd'], [2.0 ** -24, 2.0 ** -18])
        self.assertEqual(smoke['ovo__estimator__gamma'], [1e-4, 1e-3])
        self.assertEqual(smoke['ovo__estimator__theta'], [0.1, 0.2])
        self.assertEqual(smoke['ovo__estimator__sv_selection_ratio'], [0.3])


if __name__ == '__main__':
    unittest.main()
