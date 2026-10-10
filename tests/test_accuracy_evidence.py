"""Exact accuracy comparisons retain equal fold/seed weighting and one prediction."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
import cv_checkpoint as checkpoint
import adaptive_search
from numerical_confirmation import compare_rankings, _candidate_evidence, _slice_results, _diagnostic_summary
from search_support import (CandidateEvaluator, eligible_accuracy, accuracy_fraction,
                            accuracy_evidence, apply_accuracy_evidence, score_value, score_ranks,
                            score_nonimproving)
from tune_mnist import ranked_valid_indices


class CountingClassifier(ClassifierMixin, BaseEstimator):
    predictions = 0
    fits = 0

    def __init__(self, shift=0, random_state=42):
        self.shift, self.random_state = shift, random_state

    def fit(self, X, y):
        type(self).fits += 1
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        type(self).predictions += 1
        return (np.asarray(X)[:, 0].astype(int) + self.shift) % 2


def evidence(values, sizes=None):
    sizes = [[1200] * len(row) for row in values] if sizes is None else sizes
    return accuracy_evidence([[{'correct': int(value), 'n_validation': int(size)}
                               for value, size in zip(row, ns)] for row, ns in zip(values, sizes)])


def results(items):
    count = len(items)
    n_folds = len(items[0]['fold_means'])
    result = {'params': [{'p': index} for index in range(count)],
              'mean_test_score': np.zeros(count), 'std_test_score': np.zeros(count),
              **{f'split{i}_test_score': np.zeros(count) for i in range(n_folds)}}
    apply_accuracy_evidence(result, items)
    return result


class AccuracyEvidenceTests(unittest.TestCase):
    def test_one_prediction_and_string_labels(self):
        model = CandidateEvaluator(CountingClassifier()).fit(np.arange(6).reshape(-1, 1), np.arange(6) % 2)
        CountingClassifier.predictions = 0
        self.assertEqual(eligible_accuracy(model, np.arange(6).reshape(-1, 1), np.arange(6) % 2), 1.)
        self.assertEqual(CountingClassifier.predictions, 1)
        self.assertEqual(model.accuracy_counts_, {'correct': 6, 'n_validation': 6})
        with patch.object(model, 'predict', return_value=np.array(['a', 'b', 'a'])) as predict:
            self.assertEqual(eligible_accuracy(model, np.zeros((3, 1)), np.array(['a', 'a', 'a'])), 2 / 3)
            self.assertEqual(model.accuracy_counts_, {'correct': 2, 'n_validation': 3})
            predict.assert_called_once()

    def test_generic_or_unrecognized_score_does_not_fabricate_counts(self):
        model = CandidateEvaluator(CountingClassifier()).fit(np.zeros((2, 1)), [0, 1])
        model.accuracy_counts_ = {'correct': 1, 'n_validation': 2}
        with patch('search_support.accuracy_score', return_value=.123456):
            self.assertEqual(eligible_accuracy(model, np.zeros((2, 1)), [0, 1]), .123456)
        self.assertIsNone(model.accuracy_counts_)
        scalar = {'params': [{}, {}], 'mean_test_score': [.123456, .123457],
                  'split0_test_score': [.123456, .123457]}
        self.assertEqual(ranked_valid_indices(scalar).tolist(), [1, 0])

    def test_unequal_folds_keep_equal_fold_then_seed_weighting(self):
        item = evidence([[1, 1], [2, 0]], [[2, 3], [2, 3]])
        exact = ((Fraction(1, 2) + 1) / 2 + (Fraction(1, 3) + 0) / 2) / 2
        record = results([item])
        self.assertEqual(score_value(record, 0), exact)
        self.assertNotEqual(exact, Fraction(4, 10))  # pooled row accuracy is a different metric

    def test_scalar_tolerance_boundary_retains_original_addition_rounding(self):
        self.assertGreater(.201 - .2, .001)
        self.assertTrue(score_nonimproving(.201, .2, .001))
        self.assertEqual(score_nonimproving(.201, .2, .001), .201 <= .2 + .001)
        self.assertTrue(score_nonimproving(Fraction(201, 1000), Fraction(1, 5), .001))
        self.assertFalse(score_nonimproving(Fraction(201, 1000), Fraction(1, 5), .000999))

    def test_scalar_expansion_and_full_strip_keep_tolerance_boundary(self):
        from sklearn.model_selection import ParameterGrid
        from joblib import hash as parameter_hash
        grid = {'ensemble__lambd': [2., 4.], 'ensemble__gamma': [2., 4.]}
        config = adaptive_search.AdaptiveSearchConfig(improvement_tol=.001)
        def scalar(parameters, values):
            return {'params': parameters, 'mean_test_score': np.asarray(values),
                    'split0_test_score': np.asarray(values), 'split1_test_score': np.asarray(values)}
        params = list(ParameterGrid(grid))
        before = scalar(params, [.1] * (len(params) - 1) + [.2])
        pending = adaptive_search.next_search_stage(before, grid, config)
        self.assertEqual(pending['phase'], 'expansion-1')
        after = scalar(params + pending['candidates'],
                       before['mean_test_score'].tolist() + [.201] * len(pending['candidates']))
        answer = adaptive_search.next_search_stage(after, grid, config, pending['state'])
        self.assertEqual(answer['state']['stalled_expansions'], 1)
        leader = {'ensemble__lambd': 1., 'ensemble__gamma': 1.}
        edge = {'ensemble__lambd': 8., 'ensemble__gamma': 1.}
        collection = scalar([leader] + [{'ensemble__lambd': v, 'ensemble__gamma': 1.} for v in (2., 4., 8.)],
                            [.5, .2, .201, .201])
        stages = [{'label': str(index), 'mean_score': .5, 'leader_key': parameter_hash(leader),
                   'levels': [{'axis': 'ensemble__lambd', 'side': 'high', 'from': first, 'to': second}]}
                  for index, (first, second) in enumerate(((2., 4.), (4., 8.)))]
        certificates = adaptive_search._full_strip_certificates(collection, [edge],
            adaptive_search._policy_records(collection), stages, leader, .5, config)
        self.assertTrue(certificates)

    def test_equal_count_totals_tie_despite_float_order(self):
        first = [[1105, 1114, 1126, 1132, 1100], [1119, 1123, 1108, 1121, 1136],
                 [1102, 1140, 1125, 1109, 1102]]
        flat = np.array(first).ravel()
        # Permutations of equal-size folds have the same mathematical mean.
        second = [[1121, 1119, 1102, 1126, 1125], [1105, 1114, 1102, 1109, 1108],
                  [1140, 1132, 1123, 1136, 1100]]
        # Floating reduction order may distinguish these permutations; the
        # count-based comparison must not depend on a NumPy reduction tree.
        items = [evidence(first), evidence(second)]
        record = results(items)
        self.assertEqual(score_value(record, 0), Fraction(int(flat.sum()), 18000))
        self.assertEqual(score_value(record, 0), score_value(record, 1))
        self.assertEqual(score_ranks(record).tolist(), [1, 1])
        self.assertEqual(ranked_valid_indices(record).tolist(), [0, 1])
        self.assertEqual(compare_rankings(record, results(items[::-1]))['state'], 'stable')

    def test_real_difference_smaller_than_float_ulp_is_not_a_tie(self):
        size = 2 ** 60
        record = results([evidence([[size // 2]], [[size]]),
                          evidence([[size // 2 + 1]], [[size]])])
        self.assertEqual(record['mean_test_score'][0], record['mean_test_score'][1])
        self.assertEqual(score_ranks(record).tolist(), [2, 1])
        self.assertEqual(ranked_valid_indices(record).tolist(), [1, 0])
        tied = results([record['accuracy_evidence'][0]] * 2)
        self.assertTrue(compare_rankings(tied, record)['ranking_changed'])

    def test_mixed_and_cross_budget_legacy_evidence_use_one_scalar_policy(self):
        exact = results([evidence([[1]], [[3]])] * 2)
        mixed = deepcopy(exact)
        mixed['accuracy_evidence'][1] = None
        self.assertEqual(ranked_valid_indices(mixed).tolist(), [0, 1])
        self.assertEqual(score_ranks(mixed).tolist(), [1, 1])
        report = compare_rankings(exact, mixed)
        self.assertEqual(report['comparison'], 'reported_scalar_scores')
        self.assertEqual(report['state'], 'stable')
        self.assertEqual(report['baseline_ranks'], [1, 1])
        self.assertEqual(report['confirmation_ranks'], [1, 1])

    def test_invalid_missing_and_tampered_counts(self):
        for counts in ({'correct': True, 'n_validation': 2}, {'correct': 3, 'n_validation': 2},
                       {'correct': 0, 'n_validation': 0}, {'correct': 1.0, 'n_validation': 2}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                accuracy_fraction(counts)
        self.assertIsNone(accuracy_evidence([[None, {'correct': 1, 'n_validation': 2}]]))
        record = results([evidence([[2]], [[3]])])
        record['accuracy_evidence'][0]['mean']['numerator'] += 1
        with self.assertRaises(ValueError):
            score_value(record, 0)

    def test_exact_evidence_survives_accumulation_and_confirmation_slicing(self):
        record = results([evidence([[2, 1]], [[3, 2]])])
        row = {'params': record['params'][0], 'accuracy_evidence': record['accuracy_evidence'][0]}
        for key in ('mean_test_score', 'std_test_score', 'split0_test_score', 'split1_test_score'):
            row[key] = record[key][0]
        for key in ('mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'):
            row[key] = 0.
        accumulated = adaptive_search._results([row], 2)
        sliced = _slice_results(accumulated, [0], record['params'])
        self.assertEqual(score_value(sliced, 0), Fraction(7, 12))
        search = SimpleNamespace(cv_results_=sliced, checkpoint_summary_={'diagnostics': [[{}, {}]]})
        self.assertEqual(_candidate_evidence(search, 0, 2)['accuracy_evidence'], record['accuracy_evidence'][0])

    def test_actual_policy_retains_tie_credit_but_resets_on_a_true_sub_ulp_gain(self):
        from sklearn.model_selection import ParameterGrid
        grid = {'ensemble__lambd': [2., 4., 8.], 'ensemble__gamma': [2., 4., 8.]}
        params = list(ParameterGrid(grid))
        size = 2 ** 60
        items = [evidence([[size // 2 if all(v == 4 for v in p.values()) else size // 4]], [[size]])
                 for p in params]
        initial = results(items)
        initial['params'] = params
        config = adaptive_search.AdaptiveSearchConfig(top_k=1)
        pending = adaptive_search.next_search_stage(initial, grid, config)
        self.assertEqual(pending['phase'], 'local')
        for gain, expected in ((0, 1), (1, 0)):
            new = [evidence([[size // 2 + gain if index == 0 else size // 4]], [[size]])
                   for index, _ in enumerate(pending['candidates'])]
            completed = results(items + new)
            completed['params'] = params + pending['candidates']
            answer = adaptive_search.next_search_stage(completed, grid, config, deepcopy(pending['state']))
            self.assertEqual(answer['state']['stable_rounds'], expected)

    def test_real_checkpoint_counts_resume_and_old_identity_refusal(self):
        X = np.arange(10).reshape(-1, 1)
        y = np.arange(10) % 2
        folds = [(np.arange(4, 10), np.arange(4)), (np.arange(4), np.arange(4, 10))]
        CountingClassifier.fits = CountingClassifier.predictions = 0
        with tempfile.TemporaryDirectory() as folder, patch.object(checkpoint, 'runtime_fingerprint', return_value={'test': 'fixed'}):
            def session(resume=False):
                return checkpoint.CheckpointedCV(folder, CountingClassifier(), X, y, folds,
                    cv_seeds=(42, 1729, 2026), seed_parameter='random_state', resume=resume)
            with session() as store:
                search = store.search({'shift': [0, 1]})
                identity = deepcopy(store.identity)
            self.assertEqual((CountingClassifier.fits, CountingClassifier.predictions), (12, 12))
            self.assertEqual(search.best_index_, 0)
            self.assertEqual(search.seed_results_['accuracy_counts'][0][0],
                             [{'correct': 4, 'n_validation': 4}, {'correct': 6, 'n_validation': 6}])
            with session(True) as store:
                replay = store.search({'shift': [0, 1]})
            self.assertEqual((CountingClassifier.fits, CountingClassifier.predictions), (12, 12))
            self.assertEqual(replay.cv_results_['accuracy_evidence'], search.cv_results_['accuracy_evidence'])
            identity.pop('accuracy_aggregation')
            path = Path(folder) / 'cv-checkpoints' / 'identity.json'
            # Simulate a valid old-format manifest with no new aggregation identity.
            path.unlink()
            checkpoint._write_receipt(path, {'identity': identity, 'identity_sha256': checkpoint._digest(identity)})
            with self.assertRaises(checkpoint.CheckpointError), session(True):
                pass

    def test_row_hashes_distinguish_order_and_set_and_keep_class_counts(self):
        y = np.array(['left', 'right', 'right', 'left'])
        first = checkpoint._row_diagnostics([0, 1, 3], y)
        reverse = checkpoint._row_diagnostics([3, 1, 0], y)
        other = checkpoint._row_diagnostics([0, 1, 2], y)
        self.assertEqual(first['set_sha256'], reverse['set_sha256'])
        self.assertNotEqual(first['order_sha256'], reverse['order_sha256'])
        self.assertNotEqual(first['set_sha256'], other['set_sha256'])
        self.assertEqual(first['class_counts'], [{'label': 'left', 'count': 2}, {'label': 'right', 'count': 1}])
        self.assertNotIn('class_counts', checkpoint._row_diagnostics([0, 1]))

    def test_resigned_wrong_denominator_is_refused_on_replay_and_source_scan(self):
        X = np.arange(8).reshape(-1, 1)
        y = np.arange(8) % 2
        folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
        for reuse in (False, True):
            with self.subTest(reuse=reuse), tempfile.TemporaryDirectory() as folder, \
                    patch.object(checkpoint, 'runtime_fingerprint', return_value={'test': 'fixed'}):
                def session(resume=False):
                    return checkpoint.CheckpointedCV(folder, CountingClassifier(), X, y, folds,
                                                      resume=resume, reuse_equivalent_folds=reuse)
                with session() as store:
                    store.search({'shift': [0]})
                path = next((Path(folder) / 'cv-checkpoints' / 'folds').glob('*.json'))
                record = json.loads(path.read_text())['payload']
                record['accuracy_counts'] = {'correct': 8, 'n_validation': 8}
                path.unlink()
                checkpoint._write_receipt(path, record)
                before = CountingClassifier.fits
                with session(True) as store, self.assertRaisesRegex(checkpoint.CheckpointError, 'accuracy counts'):
                    store.search({'shift': [0]})
                self.assertEqual(CountingClassifier.fits, before)

    def test_worker_cannot_commit_mismatched_count_denominator(self):
        X = np.arange(8).reshape(-1, 1)
        y = np.arange(8) % 2
        folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
        def forged(evaluator, X, y):
            score = eligible_accuracy(evaluator, X, y)
            evaluator.accuracy_counts_ = {'correct': 8, 'n_validation': 8}
            return score
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(checkpoint, 'runtime_fingerprint', return_value={'test': 'fixed'}), \
                patch.object(checkpoint, 'eligible_accuracy', side_effect=forged):
            with checkpoint.CheckpointedCV(folder, CountingClassifier(), X, y, folds) as store, \
                    self.assertRaisesRegex(checkpoint.CheckpointError, 'accuracy counts'):
                store.search({'shift': [0]})
            self.assertEqual(list((Path(folder) / 'cv-checkpoints' / 'folds').glob('*.json')), [])

    def test_stopping_summary_missing_flags_are_unknown(self):
        report = _diagnostic_summary([{'diagnostics': [{'n_iter': 3, 'optimality_met': False,
                                                      'criterion_reached': True}, {'n_iter_': 2}]}], 3)
        self.assertEqual(report['optimality_false'], 1)
        self.assertEqual(report['optimality_unknown'], 1)
        self.assertEqual(report['criterion_reached_true'], 1)
        self.assertEqual(report['criterion_reached_unknown'], 1)
        self.assertEqual(report['converged_unknown'], 2)
        self.assertEqual(report['objective_tolerance_unknown'], 2)


if __name__ == '__main__':
    unittest.main()
