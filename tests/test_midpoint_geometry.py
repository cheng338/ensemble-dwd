"""Log-coordinate refinement and checkpoint identity at float resolution."""
from decimal import Context, Decimal, Inexact, ROUND_DOWN, localcontext
from contextlib import redirect_stdout
import io
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


def reference_midpoint(first, second):
    # Explicit context: the reference must not inherit the caller's precision.
    with localcontext(Context(prec=250)):
        return float((Decimal.from_float(first) * Decimal.from_float(second)).sqrt())


def expected_midpoint(first, second):
    # The implementation deliberately preserves each platform's valid fast path.
    lower, upper = sorted((first, second))
    previous = math.exp((math.log(lower) + math.log(upper)) / 2.)
    return previous if lower < previous < upper else reference_midpoint(lower, upper)


def responses(parameters, peak):
    scores = np.array([1. if all(p[name] == value for name, value in peak.items())
                       else .75 for p in parameters])
    return {'params': parameters, 'mean_test_score': scores,
            **{f'split{fold}_test_score': scores.copy() for fold in range(5)}}


class CountingClassifier(ClassifierMixin, BaseEstimator):
    """Only count stub fits; no numerical solver participates in these tests."""
    calls = 0
    interrupt_at = None

    def __init__(self, lambd=1., gamma=1.):
        self.lambd = lambd
        self.gamma = gamma

    def fit(self, X, y):
        type(self).calls += 1
        if type(self).calls == type(self).interrupt_at:
            raise KeyboardInterrupt('controlled midpoint checkpoint interruption')
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(int)


class MidpointGeometryTests(unittest.TestCase):
    def test_existing_successful_points_keep_identical_float_values(self):
        generator = random.Random(7007)
        cases = [(1., 4.), (1e-8, 1e-6), (1e-300, 1e300),
                 (math.ulp(0.), 10 * math.ulp(0.))]
        for _ in range(300):
            bits = sorted(generator.sample(range(1, 0x7ff0000000000000), 2))
            cases.append(tuple(struct.unpack('>d', struct.pack('>Q', value))[0] for value in bits))
        successes = 0
        with patch.object(search, '_rounded_geometric_midpoint', side_effect=AssertionError('fast path fell back')):
            for lower, upper in cases:
                previous = math.exp((math.log(lower) + math.log(upper)) / 2.)
                if lower < previous < upper:
                    self.assertEqual(search._midpoint(lower, upper).hex(), previous.hex())
                    self.assertEqual(search._midpoint(upper, lower).hex(), previous.hex())
                    successes += 1
        self.assertGreater(successes, 300)

    def test_narrow_fallback_at_normal_and_extreme_exponents(self):
        for start in (1e-8, 1e-300, 1e308):
            with self.subTest(start=start):
                upper = advance(start, 4)
                expected = expected_midpoint(start, upper)
                self.assertLess(start, expected)
                self.assertLess(expected, upper)
                self.assertEqual(search._midpoint(start, upper), expected)
                self.assertEqual(search._midpoint(upper, start), expected)
                # Exercise the fallback even on libm versions whose fast path
                # happens to produce an interior point for this interval.
                with patch.object(search.math, 'exp', return_value=start), \
                        patch.object(search, '_rounded_geometric_midpoint',
                                     wraps=search._rounded_geometric_midpoint) as fallback:
                    self.assertEqual(search._midpoint(start, upper), reference_midpoint(start, upper))
                    self.assertEqual(fallback.call_count, 1)

    def test_exact_fallback_matches_independent_reference(self):
        smallest = math.ulp(0.)
        cases = [(smallest, sys.float_info.max), (smallest, smallest * 3),
                 (smallest * 2, smallest * 5), (1e-300, 1e300)]
        starts = (smallest, smallest * 32, math.nextafter(sys.float_info.min, 0.),
                  sys.float_info.min, math.nextafter(1., 0.), 1., 2., 1e-8, 1e-300, 1e308)
        cases += [(start, advance(start, steps)) for start in starts for steps in (2, 3, 4, 16)]
        for lower, upper in cases:
            with self.subTest(lower=lower, upper=upper):
                self.assertEqual(search._rounded_geometric_midpoint(lower, upper),
                                 reference_midpoint(lower, upper))

    def test_equal_and_adjacent_endpoints_have_no_interior(self):
        starts = (math.ulp(0.), sys.float_info.min, 1e-300, 1e-8, 1., 1e308,
                  math.nextafter(sys.float_info.max, 0.))
        for start in starts:
            self.assertIsNone(search._midpoint(start, start))
            upper = math.nextafter(start, math.inf)
            self.assertIsNone(search._midpoint(start, upper))
            self.assertIsNone(search._midpoint(upper, start))

    def test_subnormal_and_binade_two_step_interiors(self):
        for start in (math.ulp(0.), math.ulp(0.) * 2,
                      math.nextafter(sys.float_info.min, 0.), math.nextafter(1., 0.)):
            upper = advance(start, 2)
            expected = math.nextafter(start, math.inf)
            self.assertEqual(search._midpoint(start, upper), expected)

    def test_caller_decimal_context_does_not_affect_fallback(self):
        lower, upper = 1e-8, advance(1e-8, 4)
        expected = reference_midpoint(lower, upper)
        with localcontext(Context(prec=2, rounding=ROUND_DOWN, traps=[Inexact])), \
                patch.object(search.math, 'exp', return_value=lower):
            self.assertEqual(search._midpoint(lower, upper), expected)

    def test_fallback_bisection_is_bounded_over_full_float_range(self):
        # Positive finite binary64 patterns occupy less than 2**63 positions.
        with patch.object(search.struct, 'unpack', wraps=struct.unpack) as unpack:
            value = search._rounded_geometric_midpoint(math.ulp(0.), sys.float_info.max)
        # Two endpoint reads and two final neighbors, plus at most 63 bisections.
        self.assertLessEqual(unpack.call_count, 67)
        self.assertEqual(value, reference_midpoint(math.ulp(0.), sys.float_info.max))

    def test_neighbors_preserve_endpoints_and_add_missing_interiors(self):
        axis = [1e-8, advance(1e-8, 4), advance(1e-8, 8)]
        original = axis.copy()
        result = search._neighbors(axis[1], axis)
        self.assertEqual(axis, original)
        self.assertIn(axis[1], result)
        self.assertEqual(result, [expected_midpoint(axis[0], axis[1]), axis[1],
                                  expected_midpoint(axis[1], axis[2])])
        self.assertEqual(len({search.parameter_hash({'gamma': value}) for value in axis + result}), 5)

    def test_actual_policy_completes_two_rounds_and_replays_pending_proposal(self):
        for start, steps in ((1e-8, 4), (1e308, 8), (1e-300, 8)):
            axis = [start, advance(start, steps), advance(start, 2 * steps)]
            grid = {'lambd': axis, 'gamma': axis, 'sv_selection_ratio': [.3]}
            config = search.AdaptiveSearchConfig(prefix='')
            parameters = list(search.ParameterGrid(grid))
            peak = {'lambd': axis[1], 'gamma': axis[1]}
            state = None
            for round_index in range(3):
                result = responses(parameters, peak)
                proposal = search.next_search_stage(result, grid, config, state)
                state = proposal['state']
                if round_index < 2:
                    self.assertIsNone(proposal['review'])
                    self.assertTrue(proposal['candidates'])
                    saved_state = json.loads(json.dumps(state))
                    resumed = search.next_search_stage(result, grid, config, saved_state)
                    self.assertEqual(resumed, proposal)
                    # Saving one newly finished candidate must preserve the rest.
                    partial = parameters + proposal['candidates'][:1]
                    pending = search.next_search_stage(responses(partial, peak), grid, config, saved_state)
                    self.assertEqual(pending['candidates'], proposal['candidates'][1:])
                    parameters.extend(proposal['candidates'])
                else:
                    self.assertEqual(proposal['review']['state'], 'accepted')
                    self.assertEqual(state['stable_rounds'], 2)
                    self.assertEqual(state['best_parameters']['lambd'], axis[1])

    def test_optional_ratio_policy_uses_same_midpoints_and_resumes(self):
        axis = [1e-8, advance(1e-8, 4), advance(1e-8, 8)]
        grid = {'lambd': axis, 'gamma': axis, 'sv_selection_ratio': [.2, .3, .4]}
        config = search.AdaptiveSearchConfig(prefix='')
        ratio_config = RatioRefinementConfig()
        parameters = list(search.ParameterGrid(grid))
        result = responses(parameters, {'lambd': axis[1], 'gamma': axis[1], 'sv_selection_ratio': .3})
        proposal = next_ratio_stage(result, grid, config, ratio_config)
        self.assertIsNone(proposal['review'])
        self.assertTrue(proposal['candidates'])
        self.assertFalse(proposal['state']['parameter_limits'])
        midpoint = expected_midpoint(axis[0], axis[1])
        self.assertTrue(any(p['gamma'] == midpoint for p in proposal['candidates']))
        restored = next_ratio_stage(result, grid, config, ratio_config,
                                    json.loads(json.dumps(proposal['state'])))
        self.assertEqual(restored, proposal)

    def test_nearby_candidates_keep_distinct_checkpoints_after_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'failures').mkdir()
            X = np.array([[0.], [1.]] * 4)
            y = X[:, 0].astype(int)
            folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
            lower, upper = 1e-8, advance(1e-8, 4)
            midpoint = search._midpoint(lower, upper)
            grid = {'lambd': [lower, midpoint], 'gamma': [upper]}
            settings = dict(run_dir=root, estimator=CountingClassifier(), X=X, y=y,
                            folds=folds, jobs=1, provenance={'geometry': 'fixed-test'})
            CountingClassifier.calls, CountingClassifier.interrupt_at = 0, 3
            try:
                with patch.object(cv_checkpoint, 'runtime_fingerprint', return_value={'runtime': 'fixed-test'}):
                    with self.assertRaises(KeyboardInterrupt), cv_checkpoint.CheckpointedCV(**settings) as checkpoint:
                        checkpoint.search(grid)
                    self.assertEqual(CountingClassifier.calls, 3)
                    CountingClassifier.calls, CountingClassifier.interrupt_at = 0, None
                    with cv_checkpoint.CheckpointedCV(**settings, resume=True) as checkpoint:
                        resumed = checkpoint.search(grid)
                    self.assertEqual(CountingClassifier.calls, 2)
                    self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 2)
                    self.assertEqual(resumed.checkpoint_summary_['new_folds'], 2)
                    self.assertEqual([p['estimator__lambd'] for p in resumed.cv_results_['params']], [lower, midpoint])
                    CountingClassifier.calls = 0
                    with cv_checkpoint.CheckpointedCV(**settings, resume=True) as checkpoint:
                        replayed = checkpoint.search(grid)
                    self.assertEqual(CountingClassifier.calls, 0)
                    self.assertEqual(replayed.checkpoint_summary_['reused_folds'], 4)
                    self.assertEqual(len(list((root / 'cv-checkpoints/folds').glob('*.json'))), 4)
            finally:
                CountingClassifier.calls, CountingClassifier.interrupt_at = 0, None

    def test_changed_workflow_source_refuses_resume_before_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'failures').mkdir()
            X = np.array([[0.], [1.]] * 4)
            y = X[:, 0].astype(int)
            folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
            grid = {'lambd': [1., 4., 16.], 'gamma': [1., 4., 16.]}
            config = search.AdaptiveSearchConfig(prefix='', refine=False, max_expansions=0)
            original_read = Path.read_bytes
            workflow = Path(search.__file__)

            def previous_workflow(path):
                data = original_read(path)
                return data + b'\n# Earlier workflow identity\n' if path == workflow else data

            CountingClassifier.calls, CountingClassifier.interrupt_at = 0, None
            try:
                with patch.object(cv_checkpoint, 'runtime_fingerprint', return_value={'runtime': 'fixed-test'}), redirect_stdout(io.StringIO()):
                    with patch.object(Path, 'read_bytes', previous_workflow):
                        search.run_adaptive_search(CountingClassifier(), grid, X, y, X, y, folds, root, config=config)
                    previous_status = (root / 'status.json').read_bytes()
                    calls = CountingClassifier.calls
                    with self.assertRaises(cv_checkpoint.CheckpointError):
                        search.run_adaptive_search(CountingClassifier(), grid, X, y, X, y, folds, root, config=config, resume=True)
                    self.assertEqual(CountingClassifier.calls, calls)
                    self.assertEqual((root / 'status.json').read_bytes(), previous_status)
            finally:
                CountingClassifier.calls, CountingClassifier.interrupt_at = 0, None


if __name__ == '__main__':
    unittest.main()
