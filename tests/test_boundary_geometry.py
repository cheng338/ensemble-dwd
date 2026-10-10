"""Boundary extension must preserve coordinates and refuse only true limits."""
from decimal import Context, Decimal, Inexact, ROUND_DOWN, localcontext
import json
import math
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search as search
import cv_checkpoint
from ratio_refinement import RatioRefinementConfig, next_ratio_stage


def advance(value, steps):
    for _ in range(steps):
        value = math.nextafter(value, math.inf)
    return value


def previous_extension(values, side):
    """The preceding log/exp path, retained as an identity regression oracle."""
    ordered = sorted(values)
    edge = ordered[0] if side == 'low' else ordered[-1]
    if len(ordered) == 1:
        step = math.log(10.)
    elif side == 'low':
        step = math.log(ordered[1]) - math.log(edge)
    else:
        step = math.log(edge) - math.log(ordered[-2])
    step = min(step, math.log(10.))
    try:
        value = math.exp(math.log(edge) + (-step if side == 'low' else step))
    except OverflowError:
        return None
    return value if (math.isfinite(value) and value > 0
                     and (value < edge if side == 'low' else value > edge)) else None


def reference_extension(values, side):
    ordered = sorted(values)
    edge = ordered[0] if side == 'low' else ordered[-1]
    with localcontext(Context(prec=1000)):
        exact_edge = Decimal.from_float(edge)
        if len(ordered) == 1:
            factor = Decimal(10)
        else:
            neighbor = Decimal.from_float(ordered[1] if side == 'low' else ordered[-2])
            factor = min(neighbor / exact_edge if side == 'low' else exact_edge / neighbor,
                         Decimal(10))
        value = float(exact_edge / factor if side == 'low' else exact_edge * factor)
    return value if (math.isfinite(value) and value > 0
                     and (value < edge if side == 'low' else value > edge)) else None


def responses(parameters, peak):
    scores = np.array([1. if all(p[name] == value for name, value in peak.items())
                       else .5 for p in parameters])
    return {'params': parameters, 'mean_test_score': scores,
            'split0_test_score': scores.copy(), 'split1_test_score': scores.copy()}


class CountingBoundaryClassifier(ClassifierMixin, BaseEstimator):
    calls = 0
    interrupt_at = None

    def __init__(self, lambd=1., gamma=1.):
        self.lambd = lambd
        self.gamma = gamma

    def fit(self, X, y):
        type(self).calls += 1
        if type(self).calls == type(self).interrupt_at:
            raise KeyboardInterrupt('controlled boundary checkpoint interruption')
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(int)


class BoundaryGeometryTests(unittest.TestCase):
    def test_successful_existing_extensions_keep_identical_bits(self):
        generator = random.Random(91009)
        cases = [[1.], [1., 4.], [1e-300, 1e300], [math.ulp(0.), 10 * math.ulp(0.)]]
        for _ in range(500):
            bits = sorted(generator.sample(range(1, 0x7ff0000000000000), 2))
            cases.append([struct.unpack('>d', struct.pack('>Q', value))[0] for value in bits])
        successes = 0
        with patch.object(search, '_rounded_geometric_extension',
                          side_effect=AssertionError('successful extension must not fall back')):
            for values in cases:
                for side in ('low', 'high'):
                    expected = previous_extension(values, side)
                    if expected is not None:
                        self.assertEqual(search._outside(values, side).hex(), expected.hex())
                        self.assertEqual(search._outside(values[::-1], side).hex(), expected.hex())
                        successes += 1
        self.assertGreater(successes, 900)

    def test_exact_fallback_matches_independent_high_precision_reference(self):
        cases = [[value] for value in (math.ulp(0.), sys.float_info.min, 1e-300,
                                       1e-9, 1., 1e308, sys.float_info.max)]
        cases += [[value, advance(value, steps)]
                  for value in (math.ulp(0.), math.ulp(0.) * 32,
                                math.nextafter(sys.float_info.min, 0.), sys.float_info.min,
                                1e-300, 1e-9, math.nextafter(1., 0.), 1., 1e308)
                  for steps in (1, 2, 3, 4, 16)]
        cases += [[1e-300, 1e300], [math.ulp(0.), sys.float_info.max]]
        for values in cases:
            for side in ('low', 'high'):
                with self.subTest(values=values, side=side):
                    expected = reference_extension(values, side)
                    if expected is None:
                        with self.assertRaises(ValueError):
                            search._rounded_geometric_extension(values, side)
                    else:
                        self.assertEqual(search._rounded_geometric_extension(values, side), expected)

    def test_narrow_normal_and_extreme_extensions_recover_without_nextafter_choice(self):
        for lower in (1e-300, 1e-9, 1e308):
            values = [lower, advance(lower, 4)]
            for side in ('low', 'high'):
                edge = values[0] if side == 'low' else values[-1]
                expected = reference_extension(values, side)
                self.assertIsNotNone(expected)
                # Force loss in the fast path even on libm variants whose
                # logarithms happen to distinguish this interval.
                with patch.object(search.math, 'exp', return_value=edge), \
                        patch.object(search, '_rounded_geometric_extension',
                                     wraps=search._rounded_geometric_extension) as fallback:
                    self.assertEqual(search._outside(values, side), expected)
                    self.assertEqual(fallback.call_count, 1)

    def test_geometric_step_keeps_one_decade_cap(self):
        self.assertEqual(search._rounded_geometric_extension([1., 1e100], 'low'), .1)
        self.assertEqual(search._rounded_geometric_extension([1e-100, 1.], 'high'), 10.)
        self.assertEqual(search._rounded_geometric_extension([1., 2.], 'low'), .5)
        self.assertEqual(search._rounded_geometric_extension([1., 2.], 'high'), 4.)

    def test_true_float_limits_still_refuse(self):
        for values, side in (([math.ulp(0.)], 'low'), ([sys.float_info.max], 'high'),
                             ([math.ulp(0.), math.ulp(0.) * 2], 'low'),
                             ([math.nextafter(sys.float_info.max, 0.), sys.float_info.max], 'high')):
            with self.subTest(values=values, side=side), self.assertRaisesRegex(
                    ValueError, 'outside representable positive arithmetic'):
                search._outside(values, side)

    def test_invalid_grid_values_are_rejected_before_geometric_work(self):
        config = search.AdaptiveSearchConfig(prefix='')
        with patch.object(search, '_rounded_geometric_extension',
                          side_effect=AssertionError('invalid inputs reached fallback')):
            for value in (0., -1., math.nan, math.inf, -math.inf, True, '1'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    search._initial_grid({'lambd': [value], 'gamma': [1.]}, config)

    def test_fallback_ignores_caller_decimal_context(self):
        values = [1e-9, advance(1e-9, 4)]
        expected = reference_extension(values, 'low')
        with localcontext(Context(prec=2, rounding=ROUND_DOWN, traps=[Inexact])), \
                patch.object(search.math, 'exp', return_value=values[0]):
            self.assertEqual(search._outside(values, 'low'), expected)

    def test_adaptive_boundary_stage_and_partial_resume_preserve_identity(self):
        lower, upper = 1e-9, math.nextafter(1e-9, math.inf)
        grid = {'lambd': [lower, upper], 'gamma': [lower, upper]}
        config = search.AdaptiveSearchConfig(prefix='')
        parameters = list(search.ParameterGrid(grid))
        peak = {'lambd': lower, 'gamma': lower}
        results = responses(parameters, peak)
        before = search.parameter_hash(results)
        proposal = search.next_search_stage(results, grid, config)
        self.assertIsNone(proposal['review'])
        self.assertEqual(proposal['phase'], 'expansion-1')
        self.assertEqual(len(proposal['candidates']), 5)
        new_keys = {search.parameter_hash(p) for p in proposal['candidates']}
        self.assertEqual(len(new_keys), 5)
        self.assertTrue(new_keys.isdisjoint(search.parameter_hash(p) for p in parameters))
        self.assertTrue(any(p['lambd'] < lower for p in proposal['candidates']))
        self.assertTrue(any(p['gamma'] < lower for p in proposal['candidates']))
        restored = json.loads(json.dumps(proposal['state']))
        self.assertEqual(search.next_search_stage(results, grid, config, restored), proposal)
        partial = responses(parameters + proposal['candidates'][:1], peak)
        resumed = search.next_search_stage(partial, grid, config, restored)
        self.assertEqual(resumed['candidates'], proposal['candidates'][1:])
        self.assertEqual(search.parameter_hash(results), before)

    def test_optional_ratio_policy_uses_extension_and_replays_pending_candidates(self):
        lower = 1e-9
        axis = [lower, advance(lower, 4), advance(lower, 8)]
        grid = {'lambd': axis, 'gamma': axis, 'sv_selection_ratio': [.2, .3, .4]}
        config = search.AdaptiveSearchConfig(prefix='', top_k=1)
        ratio_config = RatioRefinementConfig()
        parameters = list(search.ParameterGrid(grid))
        peak = {'lambd': lower, 'gamma': lower, 'sv_selection_ratio': .3}
        results = responses(parameters, peak)
        proposal = next_ratio_stage(results, grid, config, ratio_config)
        self.assertIsNone(proposal['review'])
        self.assertFalse(proposal['state']['parameter_limits'])
        self.assertTrue(any(p['lambd'] < lower for p in proposal['candidates']))
        self.assertTrue(any(p['gamma'] < lower for p in proposal['candidates']))
        restored = json.loads(json.dumps(proposal['state']))
        self.assertEqual(next_ratio_stage(results, grid, config, ratio_config, restored), proposal)
        partial = responses(parameters + proposal['candidates'][:1], peak)
        resumed = next_ratio_stage(partial, grid, config, ratio_config, restored)
        self.assertEqual(resumed['candidates'], proposal['candidates'][1:])

    def test_recovered_extensions_do_not_bypass_complete_stage_budget(self):
        lower, upper = 1e-9, math.nextafter(1e-9, math.inf)
        grid = {'lambd': [lower, upper], 'gamma': [lower, upper]}
        parameters = list(search.ParameterGrid(grid))
        results = responses(parameters, {'lambd': lower, 'gamma': lower})
        # Four old candidates plus five required exterior candidates exceed
        # this budget: recovery must not admit a partial expansion strip.
        config = search.AdaptiveSearchConfig(prefix='', max_candidates=8)
        proposal = search.next_search_stage(results, grid, config)
        self.assertEqual(proposal['candidates'], [])
        self.assertEqual(proposal['review']['state'], 'needs_review')
        self.assertIn('candidate_budget:', proposal['review']['stopping_reason'])

    def test_generated_boundary_and_original_point_keep_distinct_fold_receipts(self):
        lower, upper = 1e-9, math.nextafter(1e-9, math.inf)
        exterior = search._outside([lower, upper], 'low')
        self.assertLess(exterior, lower)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'failures').mkdir()
            X = np.array([[0.], [1.]] * 4)
            y = X[:, 0].astype(int)
            folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
            grid = {'lambd': [exterior, lower], 'gamma': [upper]}
            settings = dict(run_dir=root, estimator=CountingBoundaryClassifier(), X=X, y=y,
                            folds=folds, jobs=1, provenance={'boundary': 'fixed-test'})
            CountingBoundaryClassifier.calls, CountingBoundaryClassifier.interrupt_at = 0, 3
            try:
                with patch.object(cv_checkpoint, 'runtime_fingerprint', return_value={'runtime': 'fixed-test'}):
                    with self.assertRaises(KeyboardInterrupt), cv_checkpoint.CheckpointedCV(**settings) as checkpoint:
                        checkpoint.search(grid)
                    CountingBoundaryClassifier.calls, CountingBoundaryClassifier.interrupt_at = 0, None
                    with cv_checkpoint.CheckpointedCV(**settings, resume=True) as checkpoint:
                        resumed = checkpoint.search(grid)
                    self.assertEqual(CountingBoundaryClassifier.calls, 2)
                    self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 2)
                    self.assertEqual(resumed.checkpoint_summary_['new_folds'], 2)
                    self.assertEqual([p['estimator__lambd'] for p in resumed.cv_results_['params']],
                                     [exterior, lower])
                    CountingBoundaryClassifier.calls = 0
                    with cv_checkpoint.CheckpointedCV(**settings, resume=True) as checkpoint:
                        replayed = checkpoint.search(grid)
                    self.assertEqual(CountingBoundaryClassifier.calls, 0)
                    self.assertEqual(replayed.checkpoint_summary_['reused_folds'], 4)
                    self.assertEqual(len(list((root / 'cv-checkpoints/folds').glob('*.json'))), 4)
            finally:
                CountingBoundaryClassifier.calls, CountingBoundaryClassifier.interrupt_at = 0, None


if __name__ == '__main__':
    unittest.main()
