"""Type-preserving checkpoint identities and real subset-rounding regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from ensemble_dwd import EnsembleKernelSVM

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint


class CustomFloat(float):
    pass


class CustomInteger(int):
    pass


class CustomString(str):
    pass


class CustomList(list):
    pass


class CustomTuple(tuple):
    pass


class CustomDictionary(dict):
    pass


class CustomArray(np.ndarray):
    pass


class CustomNumpyFloat(np.float32):
    pass


class CustomPath(type(Path())):
    pass


class ParameterEncodingTests(unittest.TestCase):
    def assert_distinct(self, left, right):
        encoded_left, encoded_right = checkpoint._stable(left), checkpoint._stable(right)
        self.assertNotEqual(encoded_left, encoded_right)
        self.assertNotEqual(checkpoint._digest(encoded_left), checkpoint._digest(encoded_right))

    def test_scalar_types_are_not_collapsed_by_python_equality(self):
        numpy_ratio = np.float32(.7)
        pairs = [(True, 1), (1, 1.), (np.bool_(True), True),
                 (np.int64(1), 1), (np.int32(1), np.int64(1)),
                 (np.float64(.5), .5), (numpy_ratio, float(numpy_ratio)),
                 (np.float32(.5), np.float64(.5)),
                 (np.float64(.5), np.longdouble(.5))]
        for left, right in pairs:
            with self.subTest(left=type(left).__name__, right=type(right).__name__):
                self.assert_distinct(left, right)

    def test_container_path_and_dictionary_keys_preserve_type(self):
        pairs = [([1, 2], (1, 2)), (Path('checkpoint-cache'), 'checkpoint-cache'),
                 ({'nested': [True]}, {'nested': [1]}),
                 ({'nested': (np.int64(1),)}, {'nested': (1,)})]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assert_distinct(left, right)
        # A custom estimator may iterate a dictionary in insertion order.
        self.assert_distinct({'b': 2, 'a': 1}, {'a': 1, 'b': 2})

    def test_dictionary_cannot_impersonate_an_encoded_array(self):
        values = np.array([1., 2., 3.])
        legacy_array_tag = {'array': checkpoint.array_fingerprint(values)}
        self.assert_distinct(values, legacy_array_tag)
        # Even a copy of the current encoding is ordinary user data when encoded.
        self.assert_distinct(values, deepcopy(checkpoint._stable(values)))

    def test_array_layout_and_dtype_remain_part_of_identity(self):
        values = np.arange(12, dtype=np.float64).reshape(3, 4)
        self.assert_distinct(values, np.asfortranarray(values))
        self.assert_distinct(values[:, ::2], values[:, ::2].copy())
        self.assert_distinct(values, values.astype(np.float32))
        # Windows can give these dtypes identical strings and byte layouts.
        self.assert_distinct(values, values.astype(np.longdouble))
        readonly = values.copy()
        readonly.setflags(write=False)
        self.assert_distinct(values, readonly)
        self.assertEqual(checkpoint._stable(values), checkpoint._stable(values.copy()))

    def test_longdouble_is_finite_repeatable_and_preserves_signed_zero(self):
        value = np.longdouble(np.finfo(np.longdouble).max)
        encoded = checkpoint._stable(value)
        self.assertEqual(encoded, checkpoint._stable(np.longdouble(value)))
        self.assertEqual(checkpoint._digest(encoded),
                         checkpoint._digest(checkpoint._stable(value)))
        # The representation must remain valid strict JSON even if longdouble
        # has a wider finite range than Python float on the current platform.
        json.dumps(encoded, allow_nan=False)
        self.assert_distinct(np.longdouble(0.), np.longdouble(-0.))
        self.assert_distinct(0., -0.)

    def test_nonfinite_and_unsupported_arrays_are_rejected(self):
        unsupported = [float('nan'), float('inf'), np.longdouble('inf'),
                       np.array([object()], dtype=object),
                       np.array([(1.,)], dtype=[('value', 'f8')]),
                       np.array([1], dtype=np.dtype('i4', metadata={'unit': 'rows'})),
                       np.arange(2).view(CustomArray)]
        for value in unsupported:
            with self.subTest(kind=type(value).__name__, dtype=getattr(value, 'dtype', None)):
                with self.assertRaises(checkpoint.CheckpointError):
                    checkpoint._stable(value)

    def test_custom_scalar_container_and_key_subclasses_are_rejected(self):
        unsupported = [CustomFloat(.7), CustomInteger(1), CustomString('value'),
                       CustomList([1]), CustomTuple((1,)), CustomDictionary(a=1),
                       CustomNumpyFloat(.7), CustomPath('checkpoint-cache'),
                       {CustomString('a'): 1}, {'nested': [CustomFloat(.7)]}]
        for value in unsupported:
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(checkpoint.CheckpointError):
                    checkpoint._stable(value)


class CheckpointParameterIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        magnitudes = np.linspace(.3, 2., 10)
        self.X = np.column_stack((np.ravel(np.column_stack((-magnitudes, magnitudes))),
                                  np.repeat(np.sin(magnitudes), 2)))
        self.y = np.tile([0, 1], 10)
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.model = EnsembleKernelSVM(M=1, theta=.5, C=1., gamma=.2,
                                       sv_selection_ratio=.8, random_state=42,
                                       cache_dir=None, store_base_estimators=False)
        self.numpy_theta = np.float32(.7)
        self.plain_theta = float(self.numpy_theta)
        self.runtime = {'counter': 1, 'native_threads': 1}
        runtime_patch = patch.object(checkpoint, 'runtime_fingerprint',
                                     side_effect=lambda **kwargs: deepcopy(self.runtime))
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)

    def session(self, name='run', *, resume=False, **kwargs):
        values = dict(run_dir=self.base / name, estimator=self.model, X=self.X, y=self.y,
                      folds=self.folds, jobs=1, error_policy='raise', resume=resume,
                      provenance={'counter': 1})
        values.update(kwargs)
        (Path(values['run_dir']) / 'failures').mkdir(parents=True, exist_ok=True)
        return checkpoint.CheckpointedCV(**values)

    def assert_counts_and_scores(self, result, thetas):
        for candidate_index, theta in enumerate(thetas):
            self.assertIs(type(result.cv_results_['params'][candidate_index]['estimator__theta']),
                          type(theta))
            for fold_index, (train, validation) in enumerate(self.folds):
                direct = clone(self.model).set_params(theta=theta).fit(self.X[train], self.y[train])
                expected_subset_size = max(2, int(len(train) * theta))
                self.assertEqual(len(direct.subset_indices_[0]), expected_subset_size)
                facts = result.checkpoint_summary_['diagnostics'][candidate_index][fold_index]
                self.assertEqual(facts['subset_sizes'], [expected_subset_size])
                self.assertEqual(facts['selected_size'], 8)
                self.assertEqual(result.cv_results_[f'split{fold_index}_test_score'][candidate_index],
                                 direct.score(self.X[validation], self.y[validation]))

    def test_numpy_and_python_theta_fit_separately_in_same_stage(self):
        # NumPy scalar promotion can make these counts equal or different.
        # Parameter types must remain distinct checkpoint identities either way.
        self.assertNotEqual(checkpoint._stable(self.numpy_theta),
                            checkpoint._stable(self.plain_theta))
        thetas = [self.numpy_theta, self.plain_theta]
        for reuse in (False, True):
            with self.subTest(equivalent_reuse=reuse):
                name = 'same-stage-' + str(reuse)
                with patch.object(checkpoint, '_fit_fold', wraps=checkpoint._fit_fold) as fit:
                    with self.session(name, reuse_equivalent_folds=reuse) as store:
                        first = store.search({'theta': thetas}, stage='both-types')
                        repeated = store.search({'theta': thetas}, stage='exact-repeat')
                self.assertEqual(fit.call_count, 4)
                self.assertEqual(first.checkpoint_summary_['new_folds'], 4)
                self.assertEqual(first.checkpoint_summary_['reused_folds'], 0)
                self.assertEqual(repeated.checkpoint_summary_['reused_folds'], 4)
                self.assertEqual(repeated.checkpoint_summary_['new_folds'], 0)
                if reuse:
                    self.assertEqual(first.checkpoint_summary_['executed_folds'], 4)
                    self.assertEqual(first.checkpoint_summary_['equivalent_reused_folds'], 0)
                self.assert_counts_and_scores(first, thetas)
                self.assert_counts_and_scores(repeated, thetas)
                with self.session(name, resume=True, reuse_equivalent_folds=reuse) as store:
                    resumed = store.search({'theta': thetas[::-1]}, stage='resume-reversed')
                self.assertEqual(resumed.checkpoint_summary_['new_folds'], 0)
                self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 4)
                self.assert_counts_and_scores(resumed, thetas[::-1])

    def test_type_change_between_stages_requires_new_fits_then_resumes_exactly(self):
        with patch.object(checkpoint, '_fit_fold', wraps=checkpoint._fit_fold) as fit:
            with self.session(reuse_equivalent_folds=True) as store:
                first = store.search({'theta': [self.numpy_theta]}, stage='numpy')
                self.assertEqual(fit.call_count, 2)
                second = store.search({'theta': [self.plain_theta]}, stage='python')
                self.assertEqual(fit.call_count, 4)
                repeated = store.search({'theta': [self.numpy_theta]}, stage='numpy-repeat')
                self.assertEqual(fit.call_count, 4)
        self.assertEqual(first.checkpoint_summary_['new_folds'], 2)
        self.assertEqual(second.checkpoint_summary_['new_folds'], 2)
        self.assertEqual(second.checkpoint_summary_['reused_folds'], 0)
        self.assertEqual(repeated.checkpoint_summary_['reused_folds'], 2)
        self.assert_counts_and_scores(first, [self.numpy_theta])
        self.assert_counts_and_scores(second, [self.plain_theta])
        with self.session(resume=True, reuse_equivalent_folds=True) as store:
            resumed = store.search({'theta': [self.numpy_theta, self.plain_theta]})
        self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 4)
        self.assertEqual(resumed.checkpoint_summary_['executed_folds'], 0)
        self.assert_counts_and_scores(resumed, [self.numpy_theta, self.plain_theta])

    def test_original_parameter_type_survives_result_for_fresh_full_data_refit(self):
        # Check each possible winning type separately; checkpoint search itself
        # does not refit or reuse CV subsets for the final training population.
        for index, theta in enumerate((self.numpy_theta, self.plain_theta)):
            with self.subTest(theta_type=type(theta).__name__):
                with self.session('refit-' + str(index)) as store:
                    result = store.search({'theta': [theta]})
                selected = result.best_params_['estimator__theta']
                self.assertIs(type(selected), type(theta))
                fresh = clone(self.model).set_params(theta=selected).fit(self.X, self.y)
                direct = clone(self.model).set_params(theta=theta).fit(self.X, self.y)
                expected_subset_size = max(2, int(len(self.X) * theta))
                self.assertEqual(len(direct.subset_indices_[0]), expected_subset_size)
                self.assertEqual(len(fresh.subset_indices_[0]), expected_subset_size)
                np.testing.assert_array_equal(fresh.subset_indices_[0], direct.subset_indices_[0])
                np.testing.assert_array_equal(fresh.selected_indices_, direct.selected_indices_)
                np.testing.assert_array_equal(fresh.decision_function(self.X), direct.decision_function(self.X))

    def test_type_only_estimator_change_rejects_resume(self):
        original = clone(self.model).set_params(theta=self.numpy_theta)
        with self.session(estimator=original):
            pass
        changed = clone(self.model).set_params(theta=self.plain_theta)
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True, estimator=changed):
            pass

    def test_type_only_provenance_changes_reject_resume(self):
        pairs = [(1, True), (1, np.int64(1)), (.5, np.float64(.5)),
                 ([1], (1,)), (Path('cache'), 'cache')]
        for index, (left, right) in enumerate(pairs):
            name = 'provenance-' + str(index)
            with self.subTest(left=type(left).__name__, right=type(right).__name__):
                with self.session(name, provenance={'value': left}):
                    pass
                with self.assertRaises(checkpoint.CheckpointError):
                    with self.session(name, resume=True, provenance={'value': right}):
                        pass

    def test_type_only_runtime_change_rejects_resume(self):
        with self.session():
            pass
        self.runtime['counter'] = True
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True):
            pass

    def test_type_only_runtime_change_is_detected_in_active_coordinator(self):
        with self.session() as store:
            self.runtime['counter'] = True
            with self.assertRaises(checkpoint.CheckpointError):
                store.search({'theta': [.7]})
        self.assertFalse(list((self.base / 'run/cv-checkpoints/folds').glob('*.json')))

    def test_type_only_runtime_change_is_detected_by_worker(self):
        with self.session() as store:
            self.runtime['counter'] = True
            # Isolate the worker's own guard; a normal search also rejects this
            # change in its coordinator before any work is dispatched.
            with patch.object(store, '_check_runtime', return_value=None):
                with self.assertRaises(checkpoint.CheckpointError):
                    store.search({'theta': [.7]})
        self.assertFalse(list((self.base / 'run/cv-checkpoints/folds').glob('*.json')))

    def test_checksum_valid_receipt_with_type_changed_parameters_is_rejected(self):
        with self.session() as store:
            store.search({'theta': [self.numpy_theta]})
        path = next((self.base / 'run/cv-checkpoints/folds').glob('*.json'))
        record = json.loads(path.read_text())
        record['payload']['parameters'] = checkpoint._stable({'theta': self.plain_theta})
        record['sha256'] = checkpoint._digest(record['payload'])
        path.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True) as store:
            store.search({'theta': [self.numpy_theta]})

    def test_malformed_typed_integer_receipt_does_not_pass_python_equality(self):
        with self.session() as store:
            store.search({'M': [1]})
        path = next((self.base / 'run/cv-checkpoints/folds').glob('*.json'))
        record = json.loads(path.read_text())
        record['payload']['parameters'] = ['dict', [['M', ['int', 1.0]]]]
        record['sha256'] = checkpoint._digest(record['payload'])
        path.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True) as store:
            store.search({'M': [1]})

    def test_previous_schema_identity_is_not_silently_migrated(self):
        self.assertEqual(checkpoint.SCHEMA, 2)
        with self.session():
            pass
        path = self.base / 'run/cv-checkpoints/identity.json'
        record = json.loads(path.read_text())
        record['payload']['identity']['schema'] = 1
        record['payload']['identity_sha256'] = checkpoint._digest(record['payload']['identity'])
        record['sha256'] = checkpoint._digest(record['payload'])
        path.write_text(json.dumps(record), encoding='utf-8')
        previous = path.read_bytes()
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True):
            pass
        self.assertEqual(path.read_bytes(), previous)

    def test_previous_schema_fold_receipt_is_rejected(self):
        with self.session() as store:
            store.search({'theta': [.7]})
        path = next((self.base / 'run/cv-checkpoints/folds').glob('*.json'))
        record = json.loads(path.read_text())
        record['payload']['schema'] = 1
        record['sha256'] = checkpoint._digest(record['payload'])
        path.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True) as store:
            store.search({'theta': [.7]})


if __name__ == '__main__':
    unittest.main()
