"""Selection and numerical reports preserve evidence, not an inferred ideal fit."""
from collections import UserList
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.base import is_classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dwd._validation import ExactLabelClassifierMixin
from dwd.kern_md import KernMD
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
from evaluation_reporting import (classification_report, numerical_check_summary,
                                  selection_composition)
from cv_checkpoint import CheckpointedCV, CheckpointError, _fold_equivalence, _row_diagnostics
from final_recovery import _own_data
from search_support import (CandidateEvaluator, accuracy_evidence, data_fingerprint, eligible_accuracy,
                            label_accuracy_score, new_run_directory, score_ranks)
from stage_reporting import _fold_classes
from work_reporting import summarize_records, aggregate_work_summaries


def learner(checks, refinement=.3, compensated=.7, native=.6, wrapper='solver_diagnostics'):
    return {wrapper: {'linear_system_diagnostics': {
        'native_refinement_trials': checks, 'native_refinement_discarded': checks,
        'native_refinement_acceptances': 0, 'native_refinement_seconds': refinement,
        'compensated_residual_checks': checks, 'compensated_residual_seconds': compensated,
        'native_residual_checks': checks, 'native_residual_seconds': native}}}


def ensemble(cache=False):
    return {'cache_hit_': cache, 'subset_sizes': [3] * 5,
            'base_diagnostics_': [learner(index) for index in range(1, 6)],
            'final_estimator': learner(10, wrapper='diagnostics')}


def search_inputs():
    X = np.array([[-3.], [3.], [-2.], [2.], [-2.5], [2.5], [-1.], [1.]])
    model = Pipeline([('scale', StandardScaler()), ('ensemble', EnsembleKernelDWD(
        M=1, theta=1., sv_selection_ratio=1., kernel='linear', random_state=3,
        max_iter=6, stopping='fixed'))])
    folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
    grid = {'ensemble__lambd': [.1], 'ensemble__gamma': [.2],
            'ensemble__theta': [1.], 'ensemble__sv_selection_ratio': [1.]}
    return X, model, folds, grid


class CaseFolded(str):
    def __eq__(self, other):
        if isinstance(other, str):
            return self.casefold() == other.casefold()
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __hash__(self):
        return hash(self.casefold())


def string_sentinel_holdout(sentinel):
    return np.array([sentinel, 'b'] * 4, dtype=np.dtypes.StringDType(na_object=sentinel))


class EvaluationReportingTests(unittest.TestCase):
    def test_complete_string_values_determine_accuracy_counts_and_ranks(self):
        first, second = 'id\0A', 'id\0B'
        carriers = [list, lambda rows: np.array(rows, dtype=object)]
        if hasattr(np.dtypes, 'StringDType'):
            carriers.append(lambda rows: np.array(rows, dtype='T'))
        for labels in ((first, second), ('id', 'id\0')):
            for truth_carrier in carriers:
                actual = truth_carrier(labels)
                for prediction_carrier in carriers:
                    scores, evidence = [], []
                    for values, correct in ((labels[::-1], 0), (labels, 2)):
                        predicted = prediction_carrier(values)
                        model = SimpleNamespace(eligible_=True, predict=Mock(return_value=predicted))
                        score = eligible_accuracy(model, None, actual)
                        model.predict.assert_called_once_with(None)
                        self.assertEqual(score, correct / 2)
                        self.assertEqual(model.accuracy_counts_, {'correct': correct, 'n_validation': 2})
                        self.assertEqual(ExactLabelClassifierMixin.score(model, None, actual), score)
                        self.assertEqual(label_accuracy_score(actual, predicted), score)
                        report = classification_report(actual, predicted)
                        self.assertEqual(report['accuracy'], score)
                        self.assertEqual(report['balanced_accuracy'], score)
                        self.assertEqual(report['per_class'], [
                            {'label': label, 'support': 1, 'predicted_count': 1,
                             'correct': correct // 2, 'recall': score} for label in labels])
                        scores.append(score)
                        evidence.append(accuracy_evidence([[model.accuracy_counts_]]))
                    result = {'params': [{}, {}], 'mean_test_score': scores, 'accuracy_evidence': evidence}
                    self.assertEqual(score_ranks(result).tolist(), [2, 1])

    @unittest.skipUnless(hasattr(np.dtypes, 'StringDType'), 'NumPy has no StringDType')
    def test_string_class_populations_and_predicted_only_support(self):
        a, b, c = 'id\0A', 'id\0B', 'id\0C'
        y = np.array([b, b, a, b, a, c], dtype='T')
        report = classification_report(y[:3], [a, b, c])
        self.assertEqual(report['accuracy'], 1 / 3)
        self.assertEqual(report['balanced_accuracy'], .25)
        self.assertEqual(report['per_class'], [
            {'label': a, 'support': 1, 'predicted_count': 1, 'correct': 0, 'recall': 0.},
            {'label': b, 'support': 2, 'predicted_count': 1, 'correct': 1, 'recall': .5},
            {'label': c, 'support': 0, 'predicted_count': 1, 'correct': 0, 'recall': None}])
        selected = _row_diagnostics([2, 0, 1], y)
        self.assertEqual(selected['class_counts'], [{'label': a, 'count': 1}, {'label': b, 'count': 2}])
        summary = selection_composition({'selected_rows': selected}, y)
        self.assertTrue(summary['available'])
        self.assertEqual([(row['label'], row['training_count'], row['selected_count']) for row in summary['classes']],
                         [(a, 2, 1), (b, 3, 2), (c, 1, 0)])
        self.assertEqual(_fold_classes(y, [([0, 2, 5], [1, 3, 4])])[0]['training_class_count'], 3)
        model = OvOClassifier(EnsembleKernelSVM(random_state=42, sv_selection_ratio=.5))
        proof = _fold_equivalence(model, 'a' * 64, 0, y)
        self.assertEqual(proof['populations'], [{'pair': [a, b], 'n_train': 5},
                                               {'pair': [a, c], 'n_train': 3},
                                               {'pair': [b, c], 'n_train': 4}])
        self.assertEqual(proof['retained_counts'], [2, 1, 2])

    def test_text_sequence_fingerprints_preserve_trailing_nuls(self):
        source = [np.str_('id'), np.str_('id\0')]
        self.assertNotEqual(data_fingerprint([1, 2], source), data_fingerprint([1, 2], source[::-1]))
        self.assertEqual(data_fingerprint([1, 2], source),
                         data_fingerprint([1, 2], np.asarray(source, dtype=object)))
        X = np.array([[0], [1]])
        with self.assertRaisesRegex(ValueError, 'one-dimensional labels'):
            CheckpointedCV('.', EnsembleKernelSVM(), X, source, [([0], [1]), ([1], [0])])
        with self.assertRaisesRegex(CheckpointError, 'one-dimensional labels'):
            _own_data(X, [0, 1], X, source)

    def test_fingerprints_distinguish_supplementary_character_from_literal_surrogates(self):
        char = np.array(['alpha', '\U0001f680\0'], dtype=object)
        pair = np.array(['alpha', '\ud83d\ude80\0'], dtype=object)
        self.assertNotEqual(data_fingerprint([1, 2], char), data_fingerprint([1, 2], pair))

    def test_pandas_sparse_integer_ids_survive_search_and_fingerprinting(self):
        import pandas as pd

        X = np.array([[-3.], [3.], [-2.], [2.]])
        exact = np.array([0, 2**53 + 1] * 2, dtype=np.uint64)
        rounded = np.array([0, 2**53] * 2, dtype=np.uint64)
        source = pd.arrays.SparseArray(exact, fill_value=0)
        for carrier in (source, pd.Index(source), pd.Series(source), pd.array(exact, dtype='UInt64')):
            with self.subTest(carrier=type(carrier)):
                model = CandidateEvaluator(KernMD()).fit(X, carrier)
                self.assertEqual(model.predict(X).tolist(), exact.tolist())
                self.assertEqual(model.training_data_, data_fingerprint(X, exact))
                self.assertNotEqual(model.training_data_, data_fingerprint(X, rounded))
                for held in (rounded, rounded.astype(float)):
                    self.assertEqual(label_accuracy_score(carrier, held), .5)
                    self.assertEqual(eligible_accuracy(model, X, held), .5)
                    self.assertEqual(model.accuracy_counts_, {'correct': 2, 'n_validation': 4})

    def test_sparse_text_fill_cannot_turn_numeric_training_rows_into_strings(self):
        import pandas as pd

        X = np.array([[-4.], [3.], [-1.], [1.], [-3.], [2.]])
        target = pd.arrays.SparseArray(np.array([1, 2, 0, 2, 1, 2]), fill_value=0)
        try:
            target.fill_value = '1'
        except (TypeError, ValueError):
            self.skipTest('This pandas version disallows a text fill on numeric sparse storage')
        expected = [1, 2, '1', 2, 1, 2]
        self.assertEqual(list(target), expected)
        model = CandidateEvaluator(KernMD()).fit(X, ['1', '2'] * 3)
        for carrier in (target, pd.Index(target), pd.Series(target)):
            with self.subTest(carrier=type(carrier)):
                with self.assertRaisesRegex(ValueError, 'mixes numeric and text'):
                    CandidateEvaluator(KernMD()).fit(X, carrier)
                predicted = model.predict(X).tolist()
                correct = sum(actual == prediction for actual, prediction in zip(expected, predicted))
                self.assertEqual(correct, 1)
                self.assertEqual(label_accuracy_score(carrier, predicted), correct / len(X))
                self.assertEqual(eligible_accuracy(model, X, carrier), correct / len(X))
                self.assertEqual(model.accuracy_counts_, {'correct': correct, 'n_validation': len(X)})

    def test_candidate_default_score_accepts_exact_text_and_sample_weights(self):
        X = np.array([[-3.], [3.], [-2.], [2.]])
        dtypes = [object]
        if hasattr(np.dtypes, 'StringDType'):
            dtypes.append(np.dtypes.StringDType())
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                actual = np.array(['longname', 'x\0'] * 2, dtype=dtype)
                changed = np.array(['longname', 'x'] * 2, dtype=dtype)
                model = CandidateEvaluator(KernMD()).fit(X, actual)
                self.assertTrue(is_classifier(model))
                self.assertEqual(model.score(X, changed), .5)
                self.assertEqual(model.score(X, changed, sample_weight=[2, 1, 2, 1]), 2 / 3)

    def test_object_byte_labels_identify_distinct_fitted_classes(self):
        X = np.array([[-3.], [3.], [-2.], [2.]])
        first = np.array([b'a', b'c'] * 2, dtype=object)
        second = np.array([b'b', b'c'] * 2, dtype=object)
        model = CandidateEvaluator(KernMD()).fit(X, first)
        self.assertEqual(model.predict(X).tolist(), first.tolist())
        self.assertEqual(CandidateEvaluator(KernMD()).fit(X, second).predict(X).tolist(), second.tolist())
        self.assertEqual(model.training_data_, data_fingerprint(X, first.copy()))
        self.assertNotEqual(model.training_data_, data_fingerprint(X, second))

        class PlainBytes(bytes):
            pass

        a = np.array([PlainBytes(b'a'), b'c'], dtype=object)
        b = np.array([PlainBytes(b'b'), b'c'], dtype=object)
        self.assertNotEqual(data_fingerprint(X[:2], a), data_fingerprint(X[:2], b))

    def test_mixed_sparse_label_fingerprints_keep_scalar_values_at_native_precision(self):
        from dwd._validation import _SparseTextObservations

        X = [[-1.], [1.]]
        high = np.finfo(np.longdouble).max
        pairs = [(np.bytes_(b'a\xff\0'), np.bytes_(b'a\xff')),
                 (np.str_('a\0'), np.str_('a')),
                 (np.uint64(2**64 - 1), np.uint64(2**64 - 2)),
                 (high, np.nextafter(high, np.longdouble(0))),
                 (b'\\ud800', '\ud800')]
        for left, right in pairs:
            with self.subTest(kind=type(left)):
                a = np.array([b'fixed', left], dtype=object).view(_SparseTextObservations)
                b = np.array([b'fixed', right], dtype=object).view(_SparseTextObservations)
                self.assertEqual(data_fingerprint(X, a), data_fingerprint(X, a.copy()))
                self.assertNotEqual(data_fingerprint(X, a), data_fingerprint(X, b))

    def test_boxed_byte_subclass_fits_with_display_hooks_or_custom_comparisons(self):
        class Display(np.bytes_):
            def __str__(self):
                return 'same display'

        class Compared(np.bytes_):
            def __eq__(self, other):
                if hasattr(self, 'alias'):
                    return self.alias == other
                return super().__eq__(other)

            __hash__ = np.bytes_.__hash__

        X = np.array([[-3.], [3.], [-2.], [2.]])
        for cls in (Display, Compared):
            with self.subTest(cls=cls):
                actual = np.array([cls(b'a'), cls(b'c')] * 2, dtype=object)
                changed = np.array([cls(b'b'), cls(b'c')] * 2, dtype=object)
                direct = KernMD().fit(X, actual)
                model = CandidateEvaluator(KernMD()).fit(X, actual)
                self.assertEqual(model.predict(X).tolist(), direct.predict(X).tolist())
                if cls is Display:
                    self.assertEqual(model.training_data_, data_fingerprint(X, actual.copy()))
                    self.assertNotEqual(model.training_data_, data_fingerprint(X, changed))
                else:
                    self.assertIsNone(model.training_data_)
                    with self.assertRaisesRegex(ValueError, 'Cannot fingerprint object label'):
                        data_fingerprint(X, actual)

    def test_zero_dimensional_byte_boxes_distinguish_fitted_classes(self):
        X = np.array([[-3.], [3.], [-2.], [2.]])
        for box_all in (False, True):
            fingerprints = []
            for first in (b'a', b'b'):
                expected = [first, b'c'] * 2
                labels = np.empty(4, dtype=object)
                for i, value in enumerate(expected):
                    labels[i] = np.array(value) if box_all or i % 2 == 0 else value
                model = CandidateEvaluator(KernMD()).fit(X, labels)
                self.assertEqual(model.predict(X).tolist(), expected)
                self.assertEqual(model.training_data_, data_fingerprint(X, labels.copy()))
                fingerprints.append(model.training_data_)
            self.assertNotEqual(*fingerprints)

    def test_byte_label_containers_bind_type_shape_and_values_used_during_fit(self):
        X = np.array([[-3.], [3.], [-2.], [2.]])

        def object_cell(value):
            cell = np.empty((1, 1), dtype=object)
            cell[0, 0] = [(value, b'suffix')]
            return cell

        carriers = [lambda v: [v], lambda v: (v,), lambda v: [v, b'suffix'],
                    lambda v: (v, b'suffix'), lambda v: np.array([v]), lambda v: np.array([[v]]),
                    lambda v: np.array([v], dtype='S4'), object_cell]
        observed = set()
        for carrier in carriers:
            for first in (b'a', b'b'):
                labels = np.empty(4, dtype=object)
                for i, value in enumerate([first, b'c'] * 2):
                    labels[i] = carrier(value)
                model = CandidateEvaluator(KernMD()).fit(X, labels)
                direct = KernMD().fit(X, labels)
                self.assertEqual(model.training_data_, data_fingerprint(X, deepcopy(labels)))
                self.assertEqual(data_fingerprint(X, model.predict(X)), data_fingerprint(X, direct.predict(X)))
                key = model.training_data_['y']['sha256']
                self.assertNotIn(key, observed)
                observed.add(key)

    def test_byte_label_tree_binds_nonbyte_siblings_and_distinguishes_cycles_from_sharing(self):
        X = [[1.]]

        def fingerprint(row):
            labels = np.empty(1, dtype=object)
            labels[0] = row
            return data_fingerprint(X, labels)

        shared = [b'a\0']
        self.assertEqual(fingerprint([shared, shared]), fingerprint([[b'a\0'], [b'a\0']]))
        self.assertNotEqual(fingerprint([shared, [1]]), fingerprint([shared, (1,)]))
        self.assertNotEqual(fingerprint([shared, np.array([1])]), fingerprint([shared, np.array([[1]])]))
        self.assertNotEqual(fingerprint([shared, b'a\0']), fingerprint([shared, b'a']))
        cyclic = np.empty(1, dtype=object)
        cyclic[0] = [b'a', cyclic]
        with self.assertRaisesRegex(ValueError, 'cyclic label container'):
            fingerprint(cyclic)

    def test_byte_label_target_also_identifies_rows_without_bytes(self):
        X = np.array([[-3.], [3.], [-2.], [2.]])

        def fit(first_class, second_class):
            rows = [first_class, second_class] * 2
            y = np.fromiter(rows, dtype=object, count=len(rows))
            model = CandidateEvaluator(KernMD()).fit(X, y)
            self.assertEqual(model.predict(X).tolist(), rows)
            self.assertEqual(model.training_data_, data_fingerprint(X, y.copy()))
            return model.training_data_

        cases = [((0, b'aa'), (1, Decimal('11')), (1, Decimal('12'))),
                 ((0, b'aa'), (1, Fraction(2, 3)), (1, Fraction(1, 3))),
                 (((0,), (b'xx',)), ((1,), [11]), ((1,), (11,)))]
        for other, first, second in cases:
            with self.subTest(first=first):
                self.assertNotEqual(fit(other, first), fit(other, second))

    def test_mutable_byte_labels_preserve_fitted_class_values_in_fingerprints(self):
        X = np.array([[-3.], [3.], [-2.], [2.]])
        for wrap in (bytearray, lambda b: UserList([np.bytes_(b)])):
            with self.subTest(wrap=wrap):
                identifiers = []
                for name in (b'a', b'b'):
                    rows = [wrap(name), wrap(b'c')] * 2
                    y = np.fromiter(rows, dtype=object, count=len(rows))
                    model = CandidateEvaluator(KernMD()).fit(X, y)
                    self.assertEqual(model.predict(X).tolist(), rows)
                    self.assertIsNotNone(model.training_data_)
                    identifiers.append(model.training_data_)
                self.assertNotEqual(*identifiers)

    def test_one_shot_search_scores_labels_without_a_reusable_identity(self):
        from tune_mnist import run_search_workflow

        class Compared(str):
            def __eq__(self, other):
                return str.__eq__(self, other)

            __hash__ = str.__hash__

        X, model, folds, grid = search_inputs()
        train = np.array(['aa', 'bb'] * 4)
        held = np.array([Compared(s) for s in train.tolist()], dtype=object)
        with self.assertRaisesRegex(ValueError, 'Cannot fingerprint'):
            data_fingerprint(X, held)
        with TemporaryDirectory() as temp:
            folder = new_run_directory(Path(temp) / 'oneshot')
            result = run_search_workflow(model, grid, X, train, X, held, folds, folder, verbose=0)
            manifest = json.loads((folder / 'run.json').read_bytes())
            self.assertIsNone(manifest['data']['held_out'])
            self.assertEqual(manifest['data']['training'], data_fingerprint(X, train))
            self.assertEqual(result['completed']['held_out_accuracy'], result['model'].score(X, held))
            self.assertEqual(result['completed']['held_out_accuracy'], 1.)

    @unittest.skipUnless(hasattr(np.dtypes, 'StringDType'), 'NumPy has no StringDType')
    def test_one_shot_scores_string_dtype_sentinels_with_their_own_equality(self):
        from tune_mnist import run_search_workflow

        X, model, folds, grid = search_inputs()
        train = np.array(['a', 'b'] * 4)
        with TemporaryDirectory() as temp:
            for sentinel, expected in (('A', .5), (CaseFolded('A'), 1.)):
                with self.subTest(sentinel_type=type(sentinel)):
                    held = string_sentinel_holdout(sentinel)
                    self.assertIs(type(held[0]), type(sentinel))
                    folder = new_run_directory(Path(temp) / type(sentinel).__name__)
                    result = run_search_workflow(model, grid, X, train, X, held, folds, folder, verbose=0)
                    self.assertEqual(result['model'].predict(X).tolist(), train.tolist())
                    self.assertEqual(result['completed']['held_out_accuracy'], expected)
                    manifest = json.loads((folder / 'run.json').read_bytes())
                    if type(sentinel) is str:
                        self.assertEqual(manifest['data']['held_out'], data_fingerprint(X, held))
                    else:
                        self.assertIsNone(manifest['data']['held_out'])

    @unittest.skipUnless(hasattr(np.dtypes, 'StringDType'), 'NumPy has no StringDType')
    def test_string_dtype_label_fingerprint_binds_native_policy_and_values(self):
        X = [[0], [1]]
        identities = set()
        for options in ({}, {'coerce': False}, {'na_object': None},
                        {'na_object': float('nan')}, {'na_object': 'A'}):
            with self.subTest(options=options):
                def make(rows):
                    return np.array(rows, dtype=np.dtypes.StringDType(**options))

                rows = ['name', 'name\0']
                fingerprint = data_fingerprint(X, make(rows))
                self.assertEqual(fingerprint, data_fingerprint(X, make(rows)))
                self.assertNotEqual(fingerprint, data_fingerprint(X, make(rows[::-1])))
                key = json.dumps(fingerprint['y'], sort_keys=True)
                self.assertNotIn(key, identities)
                identities.add(key)
                if 'na_object' in options:
                    missing = [options['na_object'], 'name\0']
                    marked = data_fingerprint(X, make(missing))
                    self.assertEqual(marked, data_fingerprint(X, make(missing)))
                    self.assertNotEqual(fingerprint, marked)

    @unittest.skipUnless(hasattr(np.dtypes, 'StringDType'), 'NumPy has no StringDType')
    def test_string_dtype_marker_identity_works_for_native_and_boxed_arrays(self):
        import pandas as pd

        class NoRepr(CaseFolded):
            def __repr__(self):
                raise AssertionError('fingerprinting called the marker display hook')

        def fingerprint(marker, *, nested, occupied=False, required=True):
            values = np.array([marker if occupied else 'label'],
                              dtype=np.dtypes.StringDType(na_object=marker))
            labels = np.fromiter([values], dtype=object) if nested else values
            return data_fingerprint([[0]], labels, required=required)

        for nested in (False, True):
            with self.subTest(nested=nested):
                for occupied in (False, True):
                    marker = NoRepr('A')
                    with self.assertRaisesRegex(ValueError, 'Cannot fingerprint'):
                        fingerprint(marker, nested=nested, occupied=occupied)
                    self.assertIsNone(fingerprint(marker, nested=nested, occupied=occupied, required=False))

                self.assertEqual(fingerprint(object(), nested=nested), fingerprint(object(), nested=nested))
                with self.assertRaisesRegex(ValueError, 'Cannot fingerprint'):
                    fingerprint(object(), nested=nested, occupied=True)
                self.assertIsNone(fingerprint(object(), nested=nested, occupied=True, required=False))

                for empty, nul in ((np.str_(''), np.str_('\0')), (np.bytes_(b''), np.bytes_(b'\0'))):
                    self.assertEqual(np.asarray(empty).tobytes(), np.asarray(nul).tobytes())
                    self.assertNotEqual(fingerprint(empty, nested=nested), fingerprint(nul, nested=nested))

                for marker in (pd.NA, pd.NaT):
                    marked = fingerprint(marker, nested=nested, occupied=True)
                    self.assertEqual(marked, fingerprint(marker, nested=nested, occupied=True))
                    self.assertNotEqual(marked, fingerprint(marker, nested=nested))
                self.assertNotEqual(fingerprint(pd.NA, nested=nested, occupied=True),
                                    fingerprint(pd.NaT, nested=nested, occupied=True))

    def test_sparse_byte_fills_distinguish_held_out_scores_and_replay_identities(self):
        import pandas as pd

        X = np.array([[-3.], [3.], [-2.], [2.]])
        model = CandidateEvaluator(KernMD()).fit(X, np.array([b'a', b'c'] * 2))

        def observations(fill):
            values = pd.arrays.SparseArray([0., 2., 0., 2.], fill_value=0)
            try:
                values.fill_value = fill
            except (TypeError, ValueError):
                self.skipTest('This pandas version disallows a byte fill on numeric sparse storage')
            return values

        first, same, changed = (observations(fill) for fill in (b'a', b'a', b'b'))
        for labels, expected in ((first, .5), (same, .5), (changed, 0.)):
            self.assertEqual(model.score(X, labels), expected)
            self.assertEqual(eligible_accuracy(model, X, labels), expected)
        self.assertEqual(data_fingerprint(X, first), data_fingerprint(X, same))
        self.assertNotEqual(data_fingerprint(X, first), data_fingerprint(X, changed))

    def test_workflows_preserve_mixed_sparse_held_out_score_domain(self):
        import adaptive_search
        import pandas as pd
        from tune_mnist import run_search_workflow

        X, model, folds, grid = search_inputs()
        train = np.array([1, 2] * 4)
        with TemporaryDirectory() as temp:
            for point in (1., 1.25):
                held = pd.arrays.SparseArray([point, 2., 0., 2., point, 2., 1., 2.], fill_value=0)
                try:
                    held.fill_value = '1'
                except (TypeError, ValueError):
                    self.skipTest('This pandas version disallows a text fill on numeric sparse storage')
                with self.subTest(point=point):
                    folder = new_run_directory(Path(temp) / f'legacy-{point}')
                    if point == 1.:
                        result = run_search_workflow(model, grid, X, train, X, held, folds, folder, verbose=0)
                        self.assertEqual(result['completed']['held_out_accuracy'], 7 / 8)
                    else:
                        with self.assertRaisesRegex(ValueError, 'finite integers for scoring'):
                            run_search_workflow(model, grid, X, train, X, held, folds, folder, verbose=0)
                        self.assertFalse((folder / 'completed.json').exists())

                    # Capture the owned holdout before the checkpoint rejects object-dtype training labels.
                    folder = new_run_directory(Path(temp) / f'adaptive-{point}')
                    with patch.object(adaptive_search, 'preflight_final_native_threads'), \
                            patch.object(adaptive_search, 'data_fingerprint', wraps=data_fingerprint) as fingerprint:
                        with self.assertRaisesRegex(ValueError, 'one-dimensional labels'):
                            adaptive_search.run_adaptive_search(
                                model, grid, X, ['a', 'b\0'] * 4, X, held, folds, folder)
                    owned = fingerprint.call_args_list[1].args[1]
                    self.assertFalse(owned.flags.writeable)
                    self.assertEqual(data_fingerprint(X, owned), data_fingerprint(X, held))
                    self.assertNotEqual(data_fingerprint(X, owned), data_fingerprint(X, np.asarray(owned)))
                    if point == 1.:
                        self.assertEqual(label_accuracy_score(owned, train), 7 / 8)
                    else:
                        with self.assertRaisesRegex(ValueError, 'finite integers for scoring'):
                            label_accuracy_score(owned, train)

    def test_mixed_scalar_and_array_labels_in_search_reports(self):
        class StringLabel(str):
            pass

        class NumpyStringLabel(np.str_):
            pass

        dtypes = [str]
        if hasattr(np.dtypes, 'StringDType'):
            dtypes.append(np.dtypes.StringDType(coerce=False, na_object=None))
        for dtype in dtypes:
            for scalar_type in (np.str_, StringLabel, NumpyStringLabel):
                with self.subTest(dtype=dtype, scalar_type=scalar_type):
                    actual = [np.array('q', dtype=dtype), scalar_type('q\0')] * 2
                    swapped = actual[::-1]
                    self.assertEqual(label_accuracy_score(actual, actual), 1.)
                    self.assertEqual(label_accuracy_score(actual, swapped), 0.)
                    report = classification_report(actual, swapped)
                    self.assertEqual(report['accuracy'], 0.)
                    counts = [(r['label'], r['support'], r['correct']) for r in report['per_class']]
                    self.assertEqual(counts, [('q', 2, 0), ('q\0', 2, 0)])
                    self.assertNotEqual(data_fingerprint(range(4), actual),
                                        data_fingerprint(range(4), swapped))

    def test_sequence_wrappers_keep_exact_classes_in_scores_and_checkpoint_identity(self):
        class LabelList(list):
            pass

        X = np.array([[-3.], [3.], [-2.], [2.]])
        folds = [([0, 1], [2, 3]), ([2, 3], [0, 1])]
        for wrapper in (LabelList, UserList):
            with self.subTest(wrapper=wrapper):
                actual = wrapper(['longname', 'x\0'] * 2)
                changed = wrapper(['longname', 'x'] * 2)
                model = CandidateEvaluator(KernMD()).fit(X, actual)
                self.assertEqual(model.predict(X).tolist(), list(actual))
                self.assertEqual(eligible_accuracy(model, X, changed), .5)
                self.assertEqual(model.accuracy_counts_, {'correct': 2, 'n_validation': 4})
                self.assertEqual(label_accuracy_score(actual, changed), .5)
                self.assertEqual(classification_report(actual, changed)['accuracy'], .5)
                self.assertNotEqual(model.training_data_, data_fingerprint(X, changed))
                self.assertEqual(model.training_data_, data_fingerprint(X, list(actual)))
                with self.assertRaisesRegex(ValueError, 'one-dimensional labels'):
                    CheckpointedCV('.', KernMD(), X, actual, folds)
                with self.assertRaisesRegex(CheckpointError, 'one-dimensional labels'):
                    _own_data(X, changed, X, actual)
                self.assertEqual(CheckpointedCV('.', KernMD(), X, changed, folds).y.tolist(),
                                 list(changed))

    def test_search_candidate_uses_the_supplied_label_representation_once(self):
        labels = ['longname', 'x\0'] * 2
        X = np.array([[-3.], [3.], [-2.], [2.]])

        class IteratedLabels(list):
            reads = 0

            def __iter__(self):
                self.reads += 1
                if self.reads > 1:
                    raise AssertionError('repeated label iteration')
                return iter(labels)

        class ArrayLabels(IteratedLabels):
            def __iter__(self):
                raise AssertionError('iteration overrides the array protocol')

            def __array__(self, dtype=None, copy=None):
                self.reads += 1
                if self.reads > 1:
                    raise AssertionError('repeated label conversion')
                return np.array(labels, dtype=object if dtype is None else dtype)

        for wrapper in (IteratedLabels, ArrayLabels):
            with self.subTest(wrapper=wrapper):
                source = wrapper(['unused'] * len(labels))
                model = CandidateEvaluator(KernMD()).fit(X, source)
                self.assertEqual(source.reads, 1)
                self.assertEqual(model.predict(X).tolist(), labels)
                self.assertEqual(model.training_data_, data_fingerprint(X, labels))

    def test_external_workflows_reuse_each_label_input_after_fingerprinting(self):
        import adaptive_search
        from tune_mnist import run_search_workflow

        class Once(list):
            reads = 0

            def __iter__(self):
                self.reads += 1
                if self.reads > 1:
                    raise AssertionError('the workflow consumed a label provider twice')
                return super().__iter__()

        X, model, folds, grid = search_inputs()
        labels = ['longname', 'x\0'] * 4
        held = ['longname', 'x'] * 4
        with TemporaryDirectory() as temp:
            training, validation = Once(labels), Once(held)
            folder = new_run_directory(Path(temp) / 'legacy')
            result = run_search_workflow(model, grid, X, training, X, validation,
                                         folds, folder, verbose=0)
            self.assertEqual(result['model'].predict(X).tolist(), labels)
            self.assertEqual(result['completed']['held_out_accuracy'], .5)
            self.assertEqual((training.reads, validation.reads), (1, 1))

            training, validation = Once(labels), Once(held)
            folder = new_run_directory(Path(temp) / 'adaptive')
            with patch.object(adaptive_search, 'preflight_final_native_threads'):
                with self.assertRaisesRegex(ValueError, 'one-dimensional labels'):
                    adaptive_search.run_adaptive_search(model, grid, X, training, X,
                                                        validation, folds, folder, verbose=0)
            self.assertEqual((training.reads, validation.reads), (1, 1))

    def test_metrics_use_actual_support_and_do_not_confuse_accuracy_with_balance(self):
        actual, predicted = np.array(['A'] * 9 + ['B']), np.array(['A'] * 10)
        report = classification_report(actual, predicted)
        self.assertEqual(report['accuracy'], .9)
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertEqual(report['balanced_accuracy'], balanced_accuracy_score(actual, predicted))
        self.assertEqual([row['recall'] for row in report['per_class']], [1., 0.])
        json.dumps(report, allow_nan=False)

    def test_predicted_only_class_has_unknown_recall_not_zero_in_mean(self):
        report = classification_report([2, 2, 4, 4], [2, 6, 4, 6])
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertIsNone(report['per_class'][-1]['recall'])
        self.assertEqual(report['per_class'][-1]['support'], 0)

    def test_invalid_label_shapes_fail_without_touching_arrays(self):
        for actual, prediction in (([], []), ([1], [1, 2]), ([[1]], [[1]])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                classification_report(actual, prediction)

    def test_selection_distinguishes_mixture_from_class_retention(self):
        labels = np.array(['A'] * 8 + ['B'] * 2)
        facts = {'selected_rows': {'size': 3, 'class_counts': [
            {'label': 'A', 'count': 1}, {'label': 'B', 'count': 2}]}}
        before = deepcopy(facts)
        report = selection_composition(facts, labels)
        self.assertEqual(report['selection_ratio_realized'], .3)
        self.assertEqual(report['classes'][1]['training_fraction'], .2)
        self.assertEqual(report['classes'][1]['selected_fraction'], 2 / 3)
        self.assertEqual(report['classes'][1]['class_retention_fraction'], 1.)
        self.assertEqual(facts, before)

    def test_missing_class_is_zero_only_with_complete_selected_counts(self):
        facts = {'selected_rows': {'size': 2, 'class_counts': [{'label': 0, 'count': 2}]}}
        report = selection_composition(facts, [0, 0, 1, 1])
        self.assertEqual(report['classes'][1]['selected_count'], 0)
        facts['selected_rows']['size'] = 3
        self.assertFalse(selection_composition(facts, [0, 0, 1, 1])['available'])

    def test_unavailable_and_invalid_selection_counts_stay_unknown(self):
        cases = ({}, {'selected_rows': {'size': 1}},
                 {'selected_rows': {'size': 3, 'class_counts': [{'label': 0, 'count': 3}]}},
                 {'selected_rows': {'size': 1, 'class_counts': [{'label': 9, 'count': 1}]}},
                 {'selected_rows': {'size': 2, 'class_counts': [{'label': 0, 'count': 1},
                                                              {'label': 0, 'count': 1}]}})
        for facts in cases:
            with self.subTest(facts=facts):
                self.assertFalse(selection_composition(facts, [0, 0, 1, 1])['available'])

    def test_ovo_training_denominator_and_original_label_mapping(self):
        facts = {'pairwise_estimators': [{'pair': ['ant', 'zebra'], 'diagnostics': {
            'selected_rows': {'size': 2, 'class_counts': [{'label': -1, 'count': 2}]}}}]}
        report = selection_composition(facts, ['ant'] * 3 + ['moth'] * 10 + ['zebra'] * 2)
        pair = report['pairwise'][0]
        self.assertEqual(pair['training_size'], 5)
        self.assertEqual(pair['classes'][0]['label'], -1)
        self.assertEqual(pair['classes'][0]['training_count'], 2)
        self.assertEqual(pair['label_mapping'], [{'encoded': 1, 'original': 'ant'},
                                                 {'encoded': -1, 'original': 'zebra'}])

    def test_all_five_bases_and_final_are_counted_once_without_timer_overlap(self):
        facts = ensemble()
        before = deepcopy(facts)
        report = numerical_check_summary(facts)
        self.assertEqual(len(report['learners']), 6)
        fields = report['by_origin']['current_fit']['fields']
        self.assertEqual(fields['native_refinement_trials']['sum'], 25)
        self.assertEqual(fields['refinement_and_compensated_seconds']['sum'], 6.)
        self.assertAlmostEqual(fields['native_residual_seconds']['sum'], 3.6)
        self.assertEqual(fields['native_refinement_trials']['unknown_learners'], 0)
        self.assertEqual(facts, before)

    def test_cache_hit_bases_remain_historical_final_is_current(self):
        report = numerical_check_summary(ensemble(True))
        groups = report['by_origin']
        self.assertEqual(groups['historical_reference']['learner_records'], 5)
        self.assertEqual(groups['current_fit']['learner_records'], 1)
        self.assertEqual(groups['current_fit']['fields']['native_refinement_trials']['sum'], 10)

    def test_adaptive_counters_cover_all_bases_and_final_without_overlapping_timer(self):
        fields = ('adaptive_residual_probes', 'adaptive_residual_probe_errors',
                  'adaptive_residual_probe_invalidations', 'adaptive_residual_probe_acceptances',
                  'adaptive_native_trials_skipped', 'adaptive_residual_probe_fallbacks',
                  'adaptive_residual_probe_reuses', 'adaptive_residual_history_resets')
        facts = ensemble()
        nodes = [(node, 'solver_diagnostics') for node in facts['base_diagnostics_']]
        nodes.append((facts['final_estimator'], 'diagnostics'))
        for index, (node, wrapper) in enumerate(nodes, 1):
            linear = node[wrapper]['linear_system_diagnostics']
            linear.update({name: index * multiplier for multiplier, name in enumerate(fields, 1)})
            linear['adaptive_residual_probe_seconds'] = 100.
            node[wrapper]['residual_check_order'] = 'adaptive'
        before = deepcopy(facts)
        report = numerical_check_summary(facts)['by_origin']['current_fit']
        self.assertEqual(report['learner_records'], 6)
        for multiplier, name in enumerate(fields, 1):
            self.assertEqual(report['fields'][name], {
                'sum': 21 * multiplier, 'reported_learners': 6, 'unknown_learners': 0})
        self.assertNotIn('adaptive_residual_probe_seconds', report['fields'])
        self.assertNotIn('residual_check_order', report['fields'])
        self.assertEqual(report['fields']['refinement_and_compensated_seconds']['sum'], 6.)
        self.assertEqual(facts, before)

    def test_old_cached_adaptive_counts_remain_unknown(self):
        facts = ensemble(True)
        facts['final_estimator']['diagnostics']['linear_system_diagnostics']['adaptive_native_trials_skipped'] = 7
        groups = numerical_check_summary(facts)['by_origin']
        self.assertEqual(groups['current_fit']['fields']['adaptive_native_trials_skipped'], {
            'sum': 7, 'reported_learners': 1, 'unknown_learners': 0})
        self.assertEqual(groups['historical_reference']['fields']['adaptive_native_trials_skipped'], {
            'sum': None, 'reported_learners': 0, 'unknown_learners': 5})

    def test_older_numeric_summary_missing_counter_retains_unknown_learner_coverage(self):
        record = {'state': 'complete', 'fit_time': 1., 'score_time': .1,
                  'diagnostics': ensemble()}
        summary = summarize_records([(record, False)])
        for group in summary['numerical_checks']['by_origin'].values():
            for name in list(group['fields']):
                if name.startswith('adaptive_'):
                    del group['fields'][name]
        combined = aggregate_work_summaries([summary])['numerical_checks']['by_origin']
        self.assertEqual(combined['current_fit']['fields']['adaptive_native_trials_skipped'], {
            'sum': None, 'reported_learners': 0, 'unknown_learners': 6})
        self.assertEqual(combined['current_fit']['fields']['native_refinement_trials']['sum'], 25)

    def test_unknown_cache_origin_is_not_assumed_new_work(self):
        facts = ensemble()
        del facts['cache_hit_']
        report = numerical_check_summary(facts)
        self.assertEqual(report['by_origin']['unknown_origin']['learner_records'], 5)
        self.assertEqual(report['by_origin']['current_fit']['learner_records'], 1)
        facts.update(cache_hit_=True, cache_status_='miss')
        self.assertEqual(numerical_check_summary(facts)['by_origin']['unknown_origin']['learner_records'], 5)

    def test_recovered_pair_and_replayed_receipts_have_no_new_solver_work(self):
        facts = {'pairwise_estimators': [{'pair': [0, 1], 'work_origin': 'recovered',
                                         'diagnostics': ensemble()}]}
        report = numerical_check_summary(facts)
        self.assertEqual(report['by_origin']['current_fit']['learner_records'], 0)
        self.assertEqual(report['by_origin']['historical_reference']['learner_records'], 6)
        replay = numerical_check_summary(ensemble(), historical=True)
        self.assertEqual(replay['by_origin']['historical_reference']['learner_records'], 6)

    def test_missing_base_records_do_not_turn_missing_values_into_zero(self):
        facts = ensemble()
        facts['base_diagnostics_'] = facts['base_diagnostics_'][:2]
        facts['final_estimator'] = {}
        report = numerical_check_summary(facts)
        self.assertTrue(report['learner_inventory_incomplete'])
        field = report['by_origin']['current_fit']['fields']['native_refinement_trials']
        self.assertEqual(field, {'sum': 3, 'reported_learners': 2, 'unknown_learners': 4})
        unknown = numerical_check_summary({})['by_origin']['current_fit']['fields']['native_refinement_trials']
        self.assertIsNone(unknown['sum'])

    def test_invalid_numeric_facts_and_incomplete_pair_inventory_are_unknown(self):
        facts = learner(1)
        linear = facts['solver_diagnostics']['linear_system_diagnostics']
        for invalid in (True, -1, float('nan'), float('inf'), '1', 10 ** 1000):
            linear['compensated_residual_seconds'] = invalid
            report = numerical_check_summary(facts)
            self.assertIsNone(report['learners'][0]['values']['refinement_and_compensated_seconds'])
        self.assertTrue(numerical_check_summary({'pairwise_estimators': []})['learner_inventory_incomplete'])

    def test_numeric_reporting_error_does_not_erase_committed_work(self):
        record = {'state': 'complete', 'fit_time': 1., 'score_time': .1}
        with patch('work_reporting.numerical_check_summary', side_effect=ValueError('malformed evidence')):
            report = summarize_records([(record, False)])
        self.assertEqual(report['current_worker_seconds']['fit'], 1.)
        self.assertEqual(report['numerical_checks']['evaluations_without_summary'], 1)

    def test_work_summaries_exclude_aliases_and_separate_replay(self):
        record = {'state': 'complete', 'fit_time': 1., 'score_time': .1,
                  'diagnostics': ensemble()}
        alias = {**record, 'equivalent_reuse': {'reuse_seconds': .01}}
        summary = summarize_records([(record, False), (record, True), (alias, False)])
        evidence = summary['numerical_checks']
        self.assertEqual(evidence['aliases_excluded'], 1)
        self.assertEqual(evidence['by_origin']['current_fit']['learner_records'], 6)
        self.assertEqual(evidence['by_origin']['historical_reference']['learner_records'], 6)
        combined = aggregate_work_summaries([summary, summary])['numerical_checks']
        self.assertEqual(combined['by_origin']['current_fit']['fields']['native_refinement_trials']['sum'], 50)
        old = deepcopy(summary)
        del old['numerical_checks']
        self.assertEqual(aggregate_work_summaries([old])['numerical_checks']['evaluations_without_summary'], 3)


if __name__ == '__main__':
    unittest.main()
