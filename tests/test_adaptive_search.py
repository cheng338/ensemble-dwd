"""Behavioral regression tests for adaptive checkpointed external cross-validation.

A deterministic classifier supplies known response surfaces. These tests exercise
real checkpointed fits and scoring without invoking any DWD or SVM solver.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from ensemble_dwd import IneligibleSubsetError

EXAMPLES_DIRECTORY = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES_DIRECTORY not in sys.path:
    sys.path.insert(0, EXAMPLES_DIRECTORY)
import adaptive_search as example
import cv_checkpoint
from cv_checkpoint import CheckpointedCV
from search_support import new_run_directory


FRACTIONS = (.1, .15, .2, .25, .3)


def stable(value):
    return float(format(float(value), '.12g'))


def bare(parameters):
    return {key.removeprefix('estimator__'): value
            for key, value in parameters.items()}


def key(parameters):
    return tuple(sorted((name, stable(value)) for name, value in bare(parameters).items()))


class ResponseSurfaceClassifier(ClassifierMixin, BaseEstimator):
    """Known accuracies on four-row validation blocks, with visible lifecycle."""
    events = []

    def __init__(self, lambd=1., C=1., gamma=1., M=5, theta=.1,
                 sv_selection_ratio=.1, surface='interior', failure=None,
                 regularization='lambd'):
        self.lambd = lambd
        self.C = C
        self.gamma = gamma
        self.M = M
        self.theta = theta
        self.sv_selection_ratio = sv_selection_ratio
        self.surface = surface
        self.failure = failure
        self.regularization = regularization

    def signature(self):
        return tuple(map(stable, (getattr(self, self.regularization), self.gamma,
                                  self.M, self.theta, self.sv_selection_ratio)))

    def fit(self, X, y):
        rows = tuple(np.asarray(X)[:, 0].astype(int))
        type(self).events.append(('fit', rows, self.signature()))
        reg = self.signature()[0]
        if self.failure == 'numerical':
            raise FloatingPointError('controlled unexpected numerical failure')
        invalid = (self.failure == 'invalid_all'
                   or (self.failure == 'invalid_reg' and reg == 1.)
                   or (self.failure == 'one_fold' and reg == 1. and min(rows) == 4)
                   or (self.failure == 'final' and len(rows) == 20)
                   or (self.failure == 'invalid_expansion' and (reg < 1. or reg > 4.
                                                              or self.gamma < 1. or self.gamma > 4.)))
        if invalid:
            raise IneligibleSubsetError('controlled infeasible selection')
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def quota(self):
        reg, gamma = self.signature()[:2]
        pair = (reg, gamma)
        if self.surface == 'interior':
            return 4 if pair == (4., 4.) else 1
        if self.surface == 'upper_peak':
            return 4 if pair == (16., 4.) else 3 if pair == (4., 4.) else 1
        if self.surface == 'tied_ends':
            return 3 if gamma == 4. and reg in (1., 16.) else 1
        if self.surface == 'tied_cross':
            return 3 if pair in ((1., 4.), (4., 16.)) else 1
        if self.surface == 'flat':
            return 2
        if self.surface == 'refine_three':
            return (4 if pair == (8., 8.) else 3 if pair == (4., 4.)
                    else 2 if pair in ((16., 16.), (64., 64.)) else 0)
        if self.surface == 'invalid_best':
            return 4 if pair == (1., 4.) else 3 if pair == (4., 4.) else 1
        raise AssertionError('Unknown controlled surface')

    def predict(self, X):
        array = np.asarray(X)
        rows = tuple(array[:, 0].astype(int))
        type(self).events.append(('predict', rows, self.signature()))
        if self.failure == 'prediction' and min(rows) >= 100:
            raise FloatingPointError('controlled held-out prediction failure')
        result = array[:, 1].astype(int).copy()
        flip = array[:, 0].astype(int) % 4 >= self.quota()
        result[flip] = 1 - result[flip]
        return result


class RecordedCheckpointedCV(CheckpointedCV):
    """Observe real checkpointed searches without replacing fitting or scoring."""
    calls = []

    def search(self, candidate_grid, *, stage='cv'):
        call = {'parameters': [bare(p) for p in ParameterGrid(candidate_grid)],
                'folds': [(np.asarray(t).copy(), np.asarray(v).copy()) for t, v in self.folds],
                'rows': np.asarray(self.X)[:, 0].astype(int).copy(),
                'refit': False, 'stage': stage, 'search': None}
        type(self).calls.append(call)
        result = super().search(candidate_grid, stage=stage)
        call['search'] = result
        ResponseSurfaceClassifier.events.append(('cv_end', (), ()))
        return result


class AdaptiveSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='adaptive-search-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        train_ids, test_ids = np.arange(20), np.arange(100, 108)
        self.y_train, self.y_test = train_ids % 2, test_ids % 2
        self.X_train = np.column_stack([train_ids, self.y_train])
        self.X_test = np.column_stack([test_ids, self.y_test])
        self.folds = [(np.setdiff1d(train_ids, np.arange(start, start + 4)),
                       np.arange(start, start + 4)) for start in range(0, 20, 4)]
        self.sequence = 0

    def grid(self, reg=(1., 4., 16.), gamma=(1., 4., 16.), *, prefix='ensemble__',
             regularization='lambd'):
        return {prefix + regularization: list(reg), prefix + 'gamma': list(gamma),
                prefix + 'M': [5], prefix + 'theta': [.1],
                prefix + 'sv_selection_ratio': list(FRACTIONS)}

    def execute(self, *, surface='interior', failure=None, grid=None, config=None,
                jobs=1, generator=False, prefix='ensemble__', regularization='lambd'):
        self.sequence += 1
        self.run_dir = new_run_directory(self.root / str(self.sequence))
        ResponseSurfaceClassifier.events = []
        RecordedCheckpointedCV.calls = []
        estimator = ResponseSurfaceClassifier(surface=surface, failure=failure,
                                               regularization=regularization)
        pipeline = Pipeline([(prefix[:-2], estimator)])
        if config is None:
            config = example.AdaptiveSearchConfig(prefix=prefix, regularization=regularization,
                                                   refine=False)
        folds = ((t, v) for t, v in self.folds) if generator else self.folds
        # Serial response surfaces test policy and real fold transactions; DLL
        # discovery has dedicated guard tests and real-estimator validation.
        # The two-worker case retains actual coordinator/worker fingerprints.
        runtime_scope = (patch.object(cv_checkpoint, 'runtime_fingerprint',
                                     return_value={'synthetic_runtime': 'adaptive-policy-v1'})
                         if jobs == 1 else contextlib.nullcontext())
        with patch.object(example, 'CheckpointedCV', RecordedCheckpointedCV), \
                runtime_scope, contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return example.run_adaptive_search(
                pipeline, self.grid(prefix=prefix, regularization=regularization) if grid is None else grid,
                self.X_train, self.y_train, self.X_test, self.y_test, folds, self.run_dir,
                config=config, jobs=jobs, error_policy='nan')

    def assert_protocol(self, result, *, final):
        calls = RecordedCheckpointedCV.calls
        all_parameters = [p for call in calls for p in call['parameters']]
        self.assertEqual(len(all_parameters), len({key(p) for p in all_parameters}))
        self.assertEqual({key(p) for p in all_parameters},
                         {key(p) for p in result['cv_results']['params']})
        for call in calls:
            self.assertIs(call['refit'], False)
            np.testing.assert_array_equal(call['rows'], np.arange(20))
            self.assertEqual(len(call['folds']), 5)
            for actual, expected in zip(call['folds'], self.folds):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
        events = ResponseSurfaceClassifier.events
        fits = [(i, event) for i, event in enumerate(events) if event[0] == 'fit']
        for _, event in fits:
            self.assertLess(max(event[1]), 100)
        final_fits = [(i, e) for i, e in fits if len(e[1]) == 20]
        heldout = [(i, e) for i, e in enumerate(events)
                   if e[0] == 'predict' and min(e[1]) >= 100]
        self.assertEqual(len(final_fits), int(final))
        self.assertEqual(bool(heldout), final)
        if final:
            self.assertGreater(final_fits[0][0], max(i for i, e in enumerate(events) if e[0] == 'cv_end'))
            self.assertGreater(min(i for i, _ in heldout), final_fits[0][0])
        cv_fits = [e for _, e in fits if len(e[1]) == 16]
        self.assertEqual(len(cv_fits), len(all_parameters) * 5)
        self.assertEqual(len(cv_fits), len({(e[1], e[2]) for e in cv_fits}))

    def assert_fraction_plane(self, parameters, expected_pair, *, prefix='ensemble__', reg='lambd'):
        found = [p for p in parameters if (stable(p[prefix + reg]), stable(p[prefix + 'gamma']))
                 == expected_pair]
        self.assertEqual({stable(p[prefix + 'sv_selection_ratio']) for p in found}, set(FRACTIONS))
        self.assertEqual(len(found), 5)
        self.assertTrue(all(p[prefix + 'M'] == 5 and p[prefix + 'theta'] == .1 for p in found))

    def assert_held(self, result):
        self.assertEqual(result['review']['state'], 'needs_review')
        self.assertIsNone(result['model'])
        self.assertIsNone(result['completed'])
        self.assert_protocol(result, final=False)

    def test_interior_exact_budget_uses_training_folds_and_refits_once(self):
        result = self.execute(config=example.AdaptiveSearchConfig(max_candidates=45, refine=False))
        self.assertEqual(result['review']['state'], 'accepted')
        self.assertEqual(result['selection']['mean_cv_accuracy'], 1.)
        self.assertEqual(result['selection']['parameters']['ensemble__lambd'], 4.)
        self.assertEqual(result['completed']['held_out_accuracy'], 1.)
        self.assertEqual(len(RecordedCheckpointedCV.calls), 1)
        self.assert_protocol(result, final=True)

    def test_expansion_retains_accumulated_winner_and_materializes_generator(self):
        result = self.execute(surface='upper_peak', grid=self.grid(reg=(1., 4.)), generator=True)
        self.assertEqual(stable(result['selection']['parameters']['ensemble__lambd']), 16.)
        self.assertEqual(result['selection']['source_stage'], 'expansion-1')
        self.assertIs(result['selected_search'], RecordedCheckpointedCV.calls[1]['search'])
        self.assertIsNot(result['selected_search'], result['last_search'])
        self.assertEqual(len(result['cv_results']['params']), 60)
        for reg in (16., 64.):
            for gamma in (1., 4., 16.):
                self.assert_fraction_plane(result['cv_results']['params'], (reg, gamma))
        self.assert_protocol(result, final=True)

    def test_all_tied_boundary_directions_include_new_corners(self):
        for surface, pairs in [('tied_ends', [(.25, 4.), (64., 4.)]),
                               ('tied_cross', [(.25, 4.), (4., 64.), (.25, 64.)])]:
            with self.subTest(surface=surface):
                result = self.execute(surface=surface)
                self.assertEqual(len(RecordedCheckpointedCV.calls), 2)
                for pair in pairs:
                    self.assert_fraction_plane(result['cv_results']['params'], pair)
                self.assert_protocol(result, final=True)

    def test_two_complete_flat_expansions_record_plateau_without_optimum_claim(self):
        result = self.execute(surface='flat', grid=self.grid(reg=(1., 4.), gamma=(1., 4.)))
        self.assertEqual(len(RecordedCheckpointedCV.calls), 3)
        self.assertEqual(result['review']['expansion_rounds'], 2)
        self.assertIn('empirical boundary plateau', result['review']['stopping_reason'])
        self.assertIn('not an optimality certificate', result['review']['stopping_reason'])
        self.assertIs(result['review']['plateau_is_optimality_certificate'], False)
        self.assertEqual(len(result['cv_results']['params']), 180)
        self.assert_protocol(result, final=True)

    def test_budget_does_not_partially_evaluate_initial_grid_or_expansion_plane(self):
        for budget, expected_candidates in ((19, 0), (25, 20)):
            with self.subTest(budget=budget):
                result = self.execute(surface='flat', grid=self.grid(reg=(1., 4.), gamma=(1., 4.)),
                    config=example.AdaptiveSearchConfig(max_candidates=budget, refine=False))
                self.assertEqual(len(result['cv_results']['params']), expected_candidates)
                self.assertIn('candidate_budget', str(result['review']['reasons']))
                self.assert_held(result)

    def test_unresolved_boundary_at_expansion_limit_holds_final_fit(self):
        result = self.execute(surface='upper_peak', grid=self.grid(reg=(1., 4.)),
            config=example.AdaptiveSearchConfig(max_expansions=0, refine=False))
        self.assertIn('expansion_budget', str(result['review']['reasons']))
        self.assert_held(result)

    def test_missing_refinement_budget_holds_even_for_interior_winner(self):
        result = self.execute(config=example.AdaptiveSearchConfig(max_candidates=45, refine=True))
        self.assertIn('refinement', str(result['review']['reasons']))
        self.assert_held(result)

    def test_refinement_uses_three_distinct_pairs_and_all_fixed_fractions(self):
        axes = (1., 4., 16., 64., 256.)
        result = self.execute(surface='refine_three', grid=self.grid(reg=axes, gamma=axes),
            config=example.AdaptiveSearchConfig(refine=True, top_k=3, max_candidates=500))
        self.assertEqual(stable(result['selection']['parameters']['ensemble__lambd']), 8.)
        self.assertEqual(stable(result['selection']['parameters']['ensemble__gamma']), 8.)
        for value in (2., 8., 32., 128.):
            self.assert_fraction_plane(result['cv_results']['params'], (value, value))
        local_calls = [call for call in RecordedCheckpointedCV.calls
                       if call['stage'] == 'local' or call['stage'].startswith('local-')]
        self.assertGreaterEqual(len(local_calls), 3)
        self.assertEqual(result['review']['refinement_rounds'], len(local_calls))
        self.assertGreaterEqual(result['review']['stable_refinement_rounds'], 2)
        self.assertTrue(all(item['lower_valid'] and item['upper_valid'] and item['within_factor']
                            for item in result['review']['winner_bracket'].values()))
        self.assertIn('stable completed local rounds', result['review']['stopping_reason'])
        self.assert_protocol(result, final=True)

    def test_one_invalid_fold_disqualifies_a_higher_scoring_candidate(self):
        result = self.execute(surface='invalid_best', failure='one_fold')
        self.assertEqual(result['selection']['parameters']['ensemble__lambd'], 4.)
        invalid = [i for i, p in enumerate(result['cv_results']['params']) if p['ensemble__lambd'] == 1.]
        self.assertTrue(np.isnan(result['cv_results']['mean_test_score'][invalid]).all())
        self.assert_protocol(result, final=True)

    def test_all_invalid_folds_fail_without_refit_or_test_scoring(self):
        with self.assertRaisesRegex(ValueError, 'finite scores on every validation fold'):
            self.execute(failure='invalid_all')
        self.assertFalse(any(e[0] == 'fit' and len(e[1]) == 20 for e in ResponseSurfaceClassifier.events))
        self.assertFalse(any(e[0] == 'predict' and min(e[1]) >= 100 for e in ResponseSurfaceClassifier.events))
        self.assertFalse((self.run_dir / 'completed.json').exists())

    def test_unexpected_numerical_failure_propagates_despite_nan_policy(self):
        with self.assertRaisesRegex(FloatingPointError, 'unexpected numerical failure'):
            self.execute(failure='numerical')
        self.assertFalse(any(e[0] == 'fit' and len(e[1]) == 20 for e in ResponseSurfaceClassifier.events))
        self.assertFalse(any(e[0] == 'predict' for e in ResponseSurfaceClassifier.events))

    def test_failed_refit_never_scores_heldout(self):
        with self.assertRaises(IneligibleSubsetError):
            self.execute(failure='final')
        self.assertEqual(sum(e[0] == 'fit' and len(e[1]) == 20 for e in ResponseSurfaceClassifier.events), 1)
        self.assertFalse(any(e[0] == 'predict' and min(e[1]) >= 100 for e in ResponseSurfaceClassifier.events))
        self.assertFalse((self.run_dir / 'completed.json').exists())

    def test_prediction_failure_preserves_successful_refit_receipt(self):
        with self.assertRaisesRegex(FloatingPointError, 'held-out prediction failure'):
            self.execute(failure='prediction')
        review = json.loads((self.run_dir / 'review.json').read_text(encoding='utf-8'))
        self.assertIs(review['final_refit_performed'], True)
        self.assertIs(review['held_out_scored'], False)
        self.assertFalse((self.run_dir / 'completed.json').exists())

    def test_wide_initial_spacing_expands_by_at_most_one_decade(self):
        axes = (1., 100., 10000.)
        result = self.execute(surface='flat', grid=self.grid(reg=axes, gamma=axes),
            config=example.AdaptiveSearchConfig(max_expansions=1, max_candidates=150, refine=False))
        self.assertEqual(len(RecordedCheckpointedCV.calls), 2)
        parameters = result['cv_results']['params']
        for pair in ((.1, 100.), (100000., 100.), (.1, .1), (100000., 100000.)):
            self.assert_fraction_plane(parameters, pair)
        for axis in ('ensemble__lambd', 'ensemble__gamma'):
            self.assertEqual({stable(p[axis]) for p in parameters}, {.1, 1., 100., 10000., 100000.})
        self.assert_held(result)

    def test_invalid_new_outer_candidates_do_not_certify_a_plateau(self):
        result = self.execute(surface='flat', failure='invalid_expansion',
            grid=self.grid(reg=(1., 4.), gamma=(1., 4.)),
            config=example.AdaptiveSearchConfig(max_candidates=80, refine=False))
        self.assertIn('candidate_budget', result['review']['stopping_reason'])
        self.assertIs(result['review']['plateau_is_optimality_certificate'], False)
        self.assertEqual(result['review']['expansion_rounds'], 1)
        invalid = [i for i, p in enumerate(result['cv_results']['params'])
                   if p['ensemble__lambd'] < 1. or p['ensemble__lambd'] > 4.
                   or p['ensemble__gamma'] < 1. or p['ensemble__gamma'] > 4.]
        self.assertTrue(invalid)
        self.assertTrue(np.isnan(result['cv_results']['mean_test_score'][invalid]).all())
        self.assertTrue(any(item['issue'] == 'invalid_conditional_neighbor'
                            for item in result['policy_state']['boundary_issues']))
        self.assertIs(result['review']['final_refit_performed'], False)
        self.assertIs(result['review']['held_out_scored'], False)
        self.assert_held(result)

    def test_C_parameter_custom_prefix_and_two_workers_match_serial(self):
        options = dict(prefix='classifier__', regularization='C')
        grid = self.grid(**options)
        grid['classifier__sv_selection_ratio'] = [.1]
        options['grid'] = grid
        serial = self.execute(**options)
        parallel = self.execute(jobs=2, **options)
        self.assertEqual(serial['selection']['parameters'], parallel['selection']['parameters'])
        self.assertEqual([key(p) for p in serial['cv_results']['params']],
                         [key(p) for p in parallel['cv_results']['params']])
        np.testing.assert_array_equal(serial['cv_results']['mean_test_score'],
                                      parallel['cv_results']['mean_test_score'])
        self.assertEqual(serial['completed']['held_out_accuracy'], parallel['completed']['held_out_accuracy'])
        self.assertEqual(parallel['selection']['parameters']['classifier__C'], 4.)
        self.assertEqual(len(RecordedCheckpointedCV.calls), 1)


if __name__ == '__main__':
    unittest.main()
