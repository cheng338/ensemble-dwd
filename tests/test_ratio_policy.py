"""Synthetic evidence for bounded, conditional selection-ratio refinement."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest

from joblib import hash as parameter_hash
import numpy as np
from sklearn.model_selection import ParameterGrid

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
from adaptive_search import AdaptiveSearchConfig
from ratio_refinement import RatioRefinementConfig, next_ratio_stage, validate_ratio_grid
from search_support import json_safe


def grid(coefficients=(1., 2., 4.), ratios=(.2, .4, .6), *, regularization='lambd'):
    return {'ensemble__' + regularization: list(coefficients),
            'ensemble__gamma': list(coefficients),
            'ensemble__sv_selection_ratio': list(ratios),
            'ensemble__M': [5], 'ensemble__theta': [.1]}


def result(rows):
    return {'params': [row[0] for row in rows],
            'mean_test_score': np.asarray([row[1] for row in rows]),
            'split0_test_score': np.asarray([row[1] for row in rows]),
            'split1_test_score': np.asarray([row[1] for row in rows])}


def surface(parameters):
    return (.9 - abs(math.log2(parameters['ensemble__lambd']) - 1) * .1
            - abs(math.log2(parameters['ensemble__gamma']) - 1) * .1
            - abs(parameters['ensemble__sv_selection_ratio'] - .4) * .1)


def drive(score=surface, initial=None, config=None, ratio_config=None):
    initial = grid() if initial is None else initial
    config = AdaptiveSearchConfig(top_k=1) if config is None else config
    ratio_config = (RatioRefinementConfig(max_rounds=8, ratio_tol=.1)
                    if ratio_config is None else ratio_config)
    rows = [(parameters, score(parameters)) for parameters in ParameterGrid(initial)]
    state, stages = None, []
    for _ in range(ratio_config.max_rounds + 2):
        answer = next_ratio_stage(result(rows), initial, config, ratio_config, state)
        state = answer['state']
        if answer['review'] is not None:
            return answer, rows, stages
        stages.append(deepcopy(answer['candidates']))
        rows.extend((parameters, score(parameters)) for parameters in answer['candidates'])
    raise AssertionError('Ratio policy failed to stop within its round budget.')


class RatioPolicyTests(unittest.TestCase):
    def test_validation_and_inferred_bounds(self):
        config = AdaptiveSearchConfig()
        self.assertEqual(validate_ratio_grid(grid(), config, RatioRefinementConfig()), (.2, .6))
        self.assertEqual(validate_ratio_grid(grid(), config,
                         RatioRefinementConfig(bounds=(.1, 1))), (.1, 1.))
        for ratios in ((.2,), (.2, .2), (False, .2), (.2, np.float64(.4)),
                       (.2, np.int64(1)), (.2, 0), (.2, 1.1), (.2, math.nan)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                validate_ratio_grid(grid(ratios=ratios), config, RatioRefinementConfig())
        with self.assertRaises(ValueError):
            validate_ratio_grid(grid(), config, RatioRefinementConfig(bounds=(.3, .8)))
        with self.assertRaises(ValueError):
            validate_ratio_grid(grid(), replace(config, refine=False), RatioRefinementConfig())

    def test_invalid_configuration(self):
        for settings in ({'max_rounds': 0}, {'max_rounds': True}, {'stable_rounds': 1.5},
                         {'ratio_tol': 0}, {'ratio_tol': math.inf}, {'ratio_tol': False},
                         {'bounds': (0, 1)}, {'bounds': (.4, .2)}, {'bounds': (.1, 1.1)},
                         {'bounds': (np.float64(.1), .5)}, {'bounds': 'invalid'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RatioRefinementConfig(**settings)

    def test_two_stable_rounds_are_actual_completed_local_evaluations(self):
        answer, _, stages = drive(ratio_config=RatioRefinementConfig(max_rounds=2, ratio_tol=.2))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(len(stages), 2)
        self.assertEqual(answer['state']['stable_rounds'], 2)
        self.assertTrue(answer['state']['winner_bracket_resolved'])
        self.assertFalse(answer['review']['plateau_is_optimality_certificate'])
        self.assertFalse(answer['review']['final_refit_performed'])

    def test_default_ratio_menu_and_policy_accept_after_three_rounds(self):
        initial = grid(ratios=(.1, .15, .2, .25, .3))

        def score(parameters):
            return (.9 - abs(math.log2(parameters['ensemble__lambd']) - 1) * .1
                    - abs(math.log2(parameters['ensemble__gamma']) - 1) * .1
                    - abs(parameters['ensemble__sv_selection_ratio'] - .2) * .1)

        answer, _, stages = drive(score, initial,
                                  ratio_config=RatioRefinementConfig())
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(len(stages), 3)
        self.assertEqual(answer['state']['rounds'], 3)
        bracket = answer['state']['winner_bracket']['ensemble__sv_selection_ratio']
        self.assertLessEqual(.2 - bracket['lower'], .01)
        self.assertLessEqual(bracket['upper'] - .2, .01)

    def test_second_round_ratio_optimum_is_reached_and_confirmed(self):
        def score(parameters):
            return (1. - abs(parameters['ensemble__sv_selection_ratio'] - .3)
                    - abs(math.log2(parameters['ensemble__lambd']) - 1) * .1
                    - abs(math.log2(parameters['ensemble__gamma']) - 1) * .1)
        initial = grid(ratios=(.2, .6, .8))
        # The initial winner must be .6 to place .3 at a second midpoint.
        def second_round_surface(parameters):
            ratio = parameters['ensemble__sv_selection_ratio']
            if ratio == .2:
                return score(parameters) - .25
            return score(parameters)
        answer, _, stages = drive(second_round_surface, initial)
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertFalse(any(math.isclose(p['ensemble__sv_selection_ratio'], .3)
                             for p in stages[0]))
        self.assertTrue(any(math.isclose(p['ensemble__sv_selection_ratio'], .3)
                            for p in stages[1]))
        self.assertAlmostEqual(answer['state']['best_parameters']['ensemble__sv_selection_ratio'], .3)
        self.assertGreaterEqual(len(stages), 4)

    def test_new_winner_resets_stability(self):
        initial = grid()
        config = AdaptiveSearchConfig(top_k=1)
        ratio_config = RatioRefinementConfig(max_rounds=6, ratio_tol=.2)
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        first = next_ratio_stage(result(rows), initial, config, ratio_config)
        rows.extend((p, surface(p)) for p in first['candidates'])
        second = next_ratio_stage(result(rows), initial, config, ratio_config, first['state'])
        self.assertEqual(second['state']['stable_rounds'], 1)
        chosen = parameter_hash(second['candidates'][0])
        rows.extend((p, 1. if parameter_hash(p) == chosen else surface(p))
                    for p in second['candidates'])
        third = next_ratio_stage(result(rows), initial, config, ratio_config, second['state'])
        self.assertEqual(third['state']['stable_rounds'], 0)
        self.assertIsNone(third['review'])

    def test_joint_coefficient_and_ratio_patch_and_new_slice_bracket(self):
        initial = grid(ratios=(.2, .6))
        rows = [(p, .8 if p['ensemble__lambd'] == p['ensemble__gamma'] == 2. else .1)
                for p in ParameterGrid(initial)]
        config = AdaptiveSearchConfig(top_k=1)
        ratio_config = RatioRefinementConfig(max_rounds=6, ratio_tol=.2)
        first = next_ratio_stage(result(rows), initial, config, ratio_config)
        matching = [p for p in first['candidates']
                    if math.isclose(p['ensemble__lambd'], math.sqrt(2))
                    and math.isclose(p['ensemble__gamma'], math.sqrt(2))
                    and p['ensemble__sv_selection_ratio'] == .4]
        self.assertEqual(len(matching), 1)
        winner = parameter_hash(matching[0])
        rows.extend((p, .9 if parameter_hash(p) == winner else .1) for p in first['candidates'])
        second = next_ratio_stage(result(rows), initial, config, ratio_config, first['state'])
        self.assertFalse(second['state']['winner_bracket_resolved'])
        self.assertIsNone(second['state']['winner_bracket']['ensemble__lambd']['lower'])
        self.assertTrue(any(p['ensemble__lambd'] == 1.
                            and p['ensemble__sv_selection_ratio'] == .4
                            for p in second['candidates']))

    def test_explicit_bounds_are_evaluated_before_range_boundary_is_reported(self):
        def score(parameters):
            return (surface(parameters) + parameters['ensemble__sv_selection_ratio'])
        answer, rows, stages = drive(score, ratio_config=RatioRefinementConfig(
                                    bounds=(.1, .8), max_rounds=7, ratio_tol=.1))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(answer['state']['best_parameters']['ensemble__sv_selection_ratio'], .8)
        self.assertTrue(any(p['ensemble__sv_selection_ratio'] == .8 for p in stages[0]))
        boundaries = answer['review']['range_boundary']
        self.assertEqual([(item['side'], item['coordinate']) for item in boundaries], [('high', .8)])
        self.assertTrue(all(.1 <= p['ensemble__sv_selection_ratio'] <= .8 for p, _ in rows))

    def test_observed_edge_is_not_declared_boundary_under_wider_bounds(self):
        initial = grid()
        rows = [(p, surface(p) - p['ensemble__sv_selection_ratio']) for p in ParameterGrid(initial)]
        answer = next_ratio_stage(result(rows), initial, AdaptiveSearchConfig(top_k=1),
                                  RatioRefinementConfig(bounds=(.1, .8)))
        ratio_issues = [item for item in answer['state']['boundary_directions']
                        if item['axis'].endswith('sv_selection_ratio')]
        self.assertTrue(any(item['side'] == 'low' and item['issue'] == 'missing_neighbor'
                            for item in ratio_issues))
        self.assertFalse(any(item['issue'] == 'declared_range_boundary' for item in ratio_issues))

    def test_flat_ties_preserve_original_winner_and_stop_within_budget(self):
        initial = grid()
        answer, rows, stages = drive(lambda p: .5, initial,
                                    ratio_config=RatioRefinementConfig(max_rounds=3, ratio_tol=.2))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(answer['state']['best_parameters'], rows[0][0])
        self.assertEqual(len(stages), 2)

    def test_invalid_neighbors_are_narrowed_not_used_as_support(self):
        initial = grid()
        config = AdaptiveSearchConfig(top_k=1)
        ratio_config = RatioRefinementConfig(max_rounds=2, ratio_tol=.2)
        score = lambda p: surface(p) if p['ensemble__sv_selection_ratio'] == .4 else math.nan
        answer, _, stages = drive(score, initial, config, ratio_config)
        self.assertEqual(answer['review']['state'], 'needs_review')
        self.assertEqual(answer['state']['stable_rounds'], 0)
        self.assertFalse(answer['state']['winner_bracket_resolved'])
        self.assertTrue(any(math.isclose(p['ensemble__sv_selection_ratio'], .35) for p in stages[1]))
        self.assertTrue(any(math.isclose(p['ensemble__sv_selection_ratio'], .45) for p in stages[1]))

    def test_invalid_nearest_neighbor_is_not_skipped_for_farther_valid_one(self):
        initial = grid(ratios=(.1, .2, .4, .6, .8))
        rows = [(p, math.nan if p['ensemble__sv_selection_ratio'] == .2 else surface(p))
                for p in ParameterGrid(initial)]
        answer = next_ratio_stage(result(rows), initial, AdaptiveSearchConfig(top_k=1),
                                  RatioRefinementConfig(ratio_tol=.5))
        bracket = answer['state']['winner_bracket']['ensemble__sv_selection_ratio']
        self.assertEqual(bracket['lower'], .2)
        self.assertFalse(bracket['lower_valid'])
        self.assertTrue(any(math.isclose(p['ensemble__sv_selection_ratio'], .3)
                            for p in answer['candidates']))
        self.assertFalse(any(math.isclose(p['ensemble__sv_selection_ratio'], .25)
                             for p in answer['candidates']))

    def test_top_k_uses_complete_tuples_and_never_global_cartesian_fixed_menu(self):
        initial = grid()
        initial['ensemble__M'] = [5, 7, 9]
        rows = [(p, surface(p) + (0. if p['ensemble__M'] == 5 else -.5))
                for p in ParameterGrid(initial)]
        answer = next_ratio_stage(result(rows), initial, AdaptiveSearchConfig(top_k=2),
                                  RatioRefinementConfig())
        self.assertLessEqual(len(answer['candidates']), 2 * 26)
        self.assertEqual({p['ensemble__M'] for p in answer['candidates']}, {5})
        descriptors = answer['state']['pending']['candidates']
        self.assertEqual(len({item['parent_key'] for item in descriptors}), 2)
        parents = {parameter_hash(p): p for p, _ in rows}
        self.assertEqual(len({parents[item['parent_key']]['ensemble__sv_selection_ratio']
                              for item in descriptors}), 2)

    def test_complete_stage_budget_counts_original_rows(self):
        initial = grid()
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        config = AdaptiveSearchConfig(top_k=1)
        proposal = next_ratio_stage(result(rows), initial, config, RatioRefinementConfig())
        limit = len(rows) + len(proposal['candidates']) - 1
        held = next_ratio_stage(result(rows), initial, replace(config, max_candidates=limit),
                                RatioRefinementConfig())
        self.assertEqual(held['review']['state'], 'needs_review')
        self.assertIn('candidate_budget', held['review']['stopping_reason'])
        self.assertEqual(held['candidates'], [])
        self.assertEqual(held['state']['rounds'], 0)

    def test_incomplete_stage_only_returns_missing_and_earns_no_credit(self):
        initial = grid()
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        config, ratio_config = AdaptiveSearchConfig(top_k=1), RatioRefinementConfig()
        first = next_ratio_stage(result(rows), initial, config, ratio_config)
        completed = first['candidates'][:3]
        rows.extend((p, surface(p)) for p in completed)
        partial = next_ratio_stage(result(rows), initial, config, ratio_config, first['state'])
        self.assertEqual(partial['phase'], first['phase'])
        self.assertEqual(partial['state']['stable_rounds'], 0)
        self.assertEqual(partial['state']['rounds'], 0)
        self.assertEqual([parameter_hash(p) for p in partial['candidates']],
                         [parameter_hash(p) for p in first['candidates'][3:]])

    def test_json_state_replay_preserves_fixed_parameter_types(self):
        initial = grid()
        initial['ensemble__random_state'] = [np.int64(17)]
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        config, ratio_config = AdaptiveSearchConfig(top_k=1), RatioRefinementConfig()
        first = next_ratio_stage(result(rows), initial, config, ratio_config)
        saved = json.loads(json.dumps(json_safe(first['state'])))
        replay = next_ratio_stage(result(rows), initial, config, ratio_config, saved)
        self.assertEqual([parameter_hash(p) for p in replay['candidates']],
                         [parameter_hash(p) for p in first['candidates']])
        self.assertTrue(all(type(p['ensemble__random_state']) is np.int64 for p in replay['candidates']))
        self.assertEqual(first['state'], replay['state'])

    def test_modified_state_and_policy_are_rejected(self):
        initial = grid()
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        config, ratio_config = AdaptiveSearchConfig(top_k=1), RatioRefinementConfig()
        first = next_ratio_stage(result(rows), initial, config, ratio_config)
        altered = deepcopy(first['state'])
        altered['stable_rounds'] = 100
        with self.assertRaisesRegex(ValueError, 'checksum'):
            next_ratio_stage(result(rows), initial, config, ratio_config, altered)
        with self.assertRaisesRegex(ValueError, 'different settings'):
            next_ratio_stage(result(rows), initial, config, replace(ratio_config, ratio_tol=.2), first['state'])

    def test_modified_previous_scores_are_rejected_without_stable_credit(self):
        initial = grid()
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        config, ratio_config = AdaptiveSearchConfig(top_k=1), RatioRefinementConfig()
        first = next_ratio_stage(result(rows), initial, config, ratio_config)
        rows[0] = (rows[0][0], rows[0][1] + .01)
        changed = next_ratio_stage(result(rows), initial, config, ratio_config, first['state'])
        self.assertEqual(changed['review']['state'], 'needs_review')
        self.assertIn('changed_evidence', changed['review']['stopping_reason'])
        self.assertEqual(changed['state']['stable_rounds'], 0)

    def test_repeated_terminal_state_has_same_result(self):
        answer, rows, _ = drive(ratio_config=RatioRefinementConfig(max_rounds=2, ratio_tol=.2))
        repeated = next_ratio_stage(result(rows), grid(), AdaptiveSearchConfig(top_k=1),
                                    RatioRefinementConfig(max_rounds=2, ratio_tol=.2),
                                    json.loads(json.dumps(answer['state'])))
        self.assertEqual(answer, repeated)

    def test_float_exhaustion_holds_without_partial_stage(self):
        adjacent = math.nextafter(.5, 1.)
        initial = grid(ratios=(.5, adjacent))
        rows = [(p, surface(p)) for p in ParameterGrid(initial)]
        answer = next_ratio_stage(result(rows), initial, AdaptiveSearchConfig(top_k=1),
                                  RatioRefinementConfig(ratio_tol=.1))
        self.assertEqual(answer['review']['state'], 'needs_review')
        self.assertIn('parameter_limit', answer['review']['stopping_reason'])
        self.assertEqual(answer['candidates'], [])

    def test_svm_C_axis_and_fixed_parameters_are_preserved(self):
        initial = grid(regularization='C')
        initial['ensemble__M'] = [5]
        initial['ensemble__base_class_weight'] = ['balanced']
        rows = [(p, .8 if p['ensemble__C'] == p['ensemble__gamma'] == 2. else .1)
                for p in ParameterGrid(initial)]
        answer = next_ratio_stage(result(rows), initial, AdaptiveSearchConfig(regularization='C', top_k=1),
                                  RatioRefinementConfig())
        self.assertTrue(answer['candidates'])
        self.assertTrue(all('ensemble__C' in p and 'ensemble__lambd' not in p
                            and type(p['ensemble__M']) is int
                            and p['ensemble__base_class_weight'] == 'balanced'
                            for p in answer['candidates']))

    def test_ineligible_and_duplicate_records_cannot_be_accepted(self):
        initial = grid()
        rows = [(p, math.nan) for p in ParameterGrid(initial)]
        answer = next_ratio_stage(result(rows), initial, AdaptiveSearchConfig(), RatioRefinementConfig())
        self.assertEqual(answer['review']['state'], 'needs_review')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            next_ratio_stage(result(rows + rows[:1]), initial, AdaptiveSearchConfig(), RatioRefinementConfig())


if __name__ == '__main__':
    unittest.main()
