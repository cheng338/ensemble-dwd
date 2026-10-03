"""Pure search-policy checks using known fold scores, without estimator fits."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.model_selection import ParameterGrid

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
import adaptive_search as policy


def grid(values=(1., 4., 16.), gamma=None, fractions=(.3,)):
    return {'ensemble__lambd': list(values), 'ensemble__gamma': list(values if gamma is None else gamma),
            'ensemble__M': [5], 'ensemble__theta': [.1], 'ensemble__sv_selection_ratio': list(fractions)}


def pair(params):
    return tuple(float(format(params['ensemble__' + name], '.12g')) for name in ('lambd', 'gamma'))


def result(rows):
    return {'params': [row[0] for row in rows],
            'mean_test_score': np.asarray([row[1] for row in rows]),
            **{f'split{i}_test_score': np.asarray([row[1] for row in rows]) for i in range(5)}}


def drive(surface, initial=None, config=None, max_stages=50):
    initial = grid() if initial is None else initial
    config = policy.AdaptiveSearchConfig(top_k=1) if config is None else config
    rows, state, stages = [], None, []
    for _ in range(max_stages):
        answer = policy.next_search_stage(result(rows), initial, config, state)
        state = answer['state']
        if answer['review'] is not None:
            return answer, rows, stages
        stages.append((answer['phase'], deepcopy(answer['candidates'])))
        rows.extend((p, surface(p)) for p in answer['candidates'])
    raise AssertionError('Policy did not finish within the bounded synthetic stage count.')


class AdaptivePolicyTests(unittest.TestCase):
    def test_second_midpoint_peak_is_found_and_refined_repeatedly(self):
        def surface(p):
            regularization, gamma = pair(p)
            if gamma != 1.:
                return .1
            if math.isclose(math.log10(regularization), 1.5, abs_tol=1e-11):
                return 1.
            return .75 if regularization == 1000. else .5 if regularization == 1. else .25
        answer, rows, stages = drive(surface, grid((1e-6, 1., 1e6)))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(answer['state']['best_mean_score'], 1.)
        self.assertTrue(answer['state']['winner_bracket_resolved'])
        self.assertGreaterEqual(answer['state']['stable_rounds'], 2)
        self.assertGreater(answer['state']['refinement_rounds'], 2)
        self.assertEqual(len(stages[0][1]), 9)
        self.assertLess(len(rows), 500)

    def test_missing_cross_combination_is_repaired_without_new_axis(self):
        surface = lambda p: .75 if pair(p) == (8., 8.) else .5 if pair(p) == (4., 4.) else .25
        initial = grid()
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        config = policy.AdaptiveSearchConfig(top_k=1)
        first = policy.next_search_stage(result(rows), initial, config)
        self.assertEqual(first['phase'], 'local')
        rows.extend((p, surface(p)) for p in first['candidates'])
        repair = policy.next_search_stage(result(rows), initial, config, first['state'])
        self.assertIsNone(repair['review'])
        self.assertEqual(repair['state']['pending']['kind'], 'conditional_repair')
        self.assertIn((16., 8.), [pair(p) for p in repair['candidates']])
        self.assertIn((8., 16.), [pair(p) for p in repair['candidates']])
        axes = [{p[axis] for p, _ in rows} for axis in config.axes]
        self.assertTrue(all(all(p[axis] in values for axis, values in zip(config.axes, axes))
                            for p in repair['candidates']))
        self.assertFalse(repair['state']['winner_bracket_resolved'])

    def test_invalid_outer_attempts_cannot_establish_interior_or_plateau(self):
        def surface(p):
            return .5 if all(1 <= value <= 4 for value in pair(p)) else float('nan')
        answer, rows, _ = drive(surface, grid((1., 4.)),
            policy.AdaptiveSearchConfig(refine=False, max_candidates=40))
        self.assertEqual(answer['review']['state'], 'needs_review')
        self.assertEqual(answer['state']['stable_rounds'], 0)
        self.assertFalse(answer['state']['boundary_plateau_evidence'])
        self.assertLessEqual(len(rows), 40)

    def test_all_tied_parameter_tuples_include_fraction_specific_edge(self):
        initial = grid(fractions=(.1, .3))
        def surface(p):
            return .75 if ((pair(p) == (4., 4.) and p['ensemble__sv_selection_ratio'] == .1)
                           or (pair(p) == (16., 4.) and p['ensemble__sv_selection_ratio'] == .3)) else .25
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        answer = policy.next_search_stage(result(rows), initial, policy.AdaptiveSearchConfig())
        self.assertEqual(answer['phase'], 'expansion-1')
        self.assertTrue(any(pair(p) == (64., 4.) and p['ensemble__sv_selection_ratio'] == .3
                            for p in answer['candidates']))

    def test_default_requires_two_actual_local_stages(self):
        answer, _, stages = drive(lambda p: .75 if pair(p) == (4., 4.) else .25,
                                  grid((2., 4., 8.)))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(answer['state']['stable_rounds'], 2)
        self.assertEqual([label for label, _ in stages], ['coarse', 'local', 'local-2'])

    def test_no_refine_accepts_valid_wide_conditional_neighbors(self):
        answer, rows, stages = drive(lambda p: .75 if pair(p) == (4., 4.) else .25,
            config=policy.AdaptiveSearchConfig(refine=False))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertIn('refinement disabled', answer['review']['stopping_reason'])
        self.assertEqual(len(stages), 1)
        self.assertEqual(len(rows), 9)
        self.assertFalse(answer['state']['winner_bracket_resolved'])

    def test_flat_no_refine_preserves_valid_full_strip_plateau(self):
        answer, _, _ = drive(lambda p: .5, grid((1., 4.)),
                            policy.AdaptiveSearchConfig(refine=False))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(answer['state']['expansion_rounds'], 2)
        self.assertTrue(answer['state']['boundary_plateau_evidence'])
        self.assertTrue(all('full_strip_plateau' in item['basis']
                            for item in answer['state']['boundary_plateau_evidence']))
        self.assertIn('not an optimality certificate', answer['review']['stopping_reason'])

    def test_plateau_disabled_holds_at_expansion_budget(self):
        answer, _, _ = drive(lambda p: .5, grid((1., 4.)),
                            policy.AdaptiveSearchConfig(refine=False, allow_boundary_plateau=False))
        self.assertEqual(answer['review']['state'], 'needs_review')
        self.assertIn('expansion_budget', answer['review']['reasons'][0])

    def test_plateau_invalid_same_slice_evidence_reopens(self):
        initial = grid((1., 4.))
        config = policy.AdaptiveSearchConfig(refine=False)
        accepted, rows, _ = drive(lambda p: .5, initial, config)
        changed = list(rows)
        certificate = accepted['state']['boundary_plateau_evidence'][0]
        key = certificate['evidence_keys'][0]
        i = next(i for i, (p, _) in enumerate(changed) if policy.parameter_hash(p) == key)
        changed[i] = (changed[i][0], float('nan'))
        reopened = policy.next_search_stage(result(changed), initial, config, accepted['state'])
        self.assertNotEqual((reopened['review'] or {}).get('state'), 'accepted')
        self.assertFalse(reopened['state']['boundary_plateau_evidence'])

    def test_plateau_scores_are_checked_at_new_corner_slices(self):
        # Old gamma=1 evidence is flat; the newly created gamma=.0625 slice
        # improves outward. Full-strip completion must not transfer old scores.
        initial = grid((1., 4.))
        config = policy.AdaptiveSearchConfig(refine=False)
        state, rows = None, []
        for _ in range(3):
            proposal = policy.next_search_stage(result(rows), initial, config, state)
            self.assertIsNone(proposal['review'])
            state = proposal['state']
            for p in proposal['candidates']:
                reg, gamma = pair(p)
                score = .5 if gamma != .0625 or reg == .0625 else .25
                rows.append((p, score))
        next_ = policy.next_search_stage(result(rows), initial, config, state)
        self.assertNotEqual((next_['review'] or {}).get('state'), 'accepted')
        self.assertTrue(any(pair(item['parameters'])[1] == .0625
                            for item in next_['state']['boundary_issues']))

    def test_intervening_tail_improvement_voids_full_strip_certificate(self):
        config = policy.AdaptiveSearchConfig(refine=False)
        accepted, rows, _ = drive(lambda p: .5, grid((1., 4.)), config)
        leader = accepted['state']['best_parameters']
        anchor = min((p for p, _ in rows), key=lambda p: pair(p))
        inner = sorted({p['ensemble__lambd'] for p, _ in rows})[1]
        inserted = {**anchor, 'ensemble__lambd': policy._midpoint(anchor['ensemble__lambd'], inner)}
        rows.append((inserted, .4))
        results = result(rows)
        certificates = policy._full_strip_certificates(results, [anchor], policy._policy_records(results),
            accepted['state']['full_strip_stages'], leader, .5, config)
        target = policy._slice_key(anchor, 'ensemble__lambd', 'low')
        self.assertFalse(any(item['slice_key'] == target for item in certificates))

    def test_partial_pending_stage_never_earns_stability_credit(self):
        initial = grid((2., 4., 8.))
        rows = [(p, .75 if pair(p) == (4., 4.) else .25) for p in ParameterGrid(initial)]
        config = policy.AdaptiveSearchConfig(top_k=1)
        first = policy.next_search_stage(result(rows), initial, config)
        rows.append((first['candidates'][0], .25))
        partial = policy.next_search_stage(result(rows), initial, config, first['state'])
        self.assertEqual(partial['phase'], first['phase'])
        self.assertEqual(partial['state']['stable_rounds'], 0)
        self.assertEqual(len(partial['candidates']), len(first['candidates']) - 1)

    def test_changed_baseline_prevents_stability_credit(self):
        initial = grid((2., 4., 8.))
        rows = [(p, .75 if pair(p) == (4., 4.) else .25) for p in ParameterGrid(initial)]
        config = policy.AdaptiveSearchConfig(top_k=1)
        first = policy.next_search_stage(result(rows), initial, config)
        rows.extend((p, .25) for p in first['candidates'])
        rows[0] = (rows[0][0], .3)
        next_ = policy.next_search_stage(result(rows), initial, config, first['state'])
        self.assertEqual(next_['state']['stable_rounds'], 0)

    def test_invalid_local_trials_do_not_earn_stable_round(self):
        initial = grid((2., 4., 8.))
        rows = [(p, .75 if pair(p) == (4., 4.) else .25) for p in ParameterGrid(initial)]
        config = policy.AdaptiveSearchConfig(top_k=1)
        first = policy.next_search_stage(result(rows), initial, config)
        rows.extend((p, float('nan')) for p in first['candidates'])
        next_ = policy.next_search_stage(result(rows), initial, config, first['state'])
        self.assertEqual(next_['state']['stable_rounds'], 0)
        self.assertIsNone(next_['review'])
        self.assertFalse(next_['state']['winner_bracket_resolved'])

    def test_every_refined_pair_keeps_all_fixed_field_choices(self):
        initial = grid(fractions=(.1, .3))
        initial['ensemble__theta'] = [.1, .15]
        rows = [(p, .75 if pair(p) == (4., 4.) else .25) for p in ParameterGrid(initial)]
        answer = policy.next_search_stage(result(rows), initial, policy.AdaptiveSearchConfig(top_k=1))
        grouped = {}
        for p in answer['candidates']:
            grouped.setdefault(pair(p), set()).add((p['ensemble__theta'], p['ensemble__sv_selection_ratio']))
        self.assertTrue(grouped)
        self.assertTrue(all(values == {(.1, .1), (.1, .3), (.15, .1), (.15, .3)}
                            for values in grouped.values()))

    def test_candidate_budget_holds_whole_stage_without_partial_proposal(self):
        for limit in (8, 9):
            answer, rows, _ = drive(lambda p: .75 if pair(p) == (4., 4.) else .25,
                config=policy.AdaptiveSearchConfig(top_k=1, max_candidates=limit))
            self.assertEqual(answer['review']['state'], 'needs_review')
            self.assertIn('candidate_budget', answer['review']['reasons'][0])
            self.assertFalse(answer['candidates'])
            self.assertEqual(len(rows), 0 if limit == 8 else 9)

    def test_duplicate_initial_candidates_are_not_scheduled_twice(self):
        initial = grid((1., 4., 4., 16.), fractions=(.3, .3))
        first = policy.next_search_stage(result([]), initial, policy.AdaptiveSearchConfig())
        self.assertEqual(len(first['candidates']), 9)

    def test_state_config_and_schema_are_bound(self):
        initial = grid()
        first = policy.next_search_stage(result([]), initial, policy.AdaptiveSearchConfig())
        with self.assertRaisesRegex(ValueError, 'different settings'):
            policy.next_search_stage(result([]), initial, policy.AdaptiveSearchConfig(top_k=1), first['state'])
        state = deepcopy(first['state'])
        state['schema_version'] = 999
        with self.assertRaisesRegex(ValueError, 'schema'):
            policy.next_search_stage(result([]), initial, policy.AdaptiveSearchConfig(), state)

    def test_new_configuration_rejects_invalid_controls(self):
        for kwargs in ({'bracket_factor': 1}, {'bracket_factor': float('nan')},
                       {'stable_refinement_rounds': 0}, {'allow_boundary_plateau': 'yes'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                policy.AdaptiveSearchConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
