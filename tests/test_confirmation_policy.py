"""Policy and provenance checks for optional DWD iteration confirmation."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from joblib import hash as parameter_hash
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
from adaptive_search import AdaptiveSearchConfig
from cv_checkpoint import CheckpointError
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier
from numerical_confirmation import (NumericalConfirmationConfig, _diagnostic_summary,
                                    compare_rankings, run_numerical_confirmation,
                                    validate_confirmation)


def pipeline(max_iter=3):
    return Pipeline([('scale', StandardScaler()), ('ensemble',
        EnsembleKernelDWD(M=1, theta=.5, max_iter=max_iter, random_state=17))])


def results(means, parameters=None):
    parameters = ([{'ensemble__lambd': value, 'ensemble__gamma': .1}
                   for value in range(1, len(means) + 1)]
                  if parameters is None else parameters)
    values = np.asarray(means, dtype=float)
    return {'params': deepcopy(parameters), 'mean_test_score': values,
            'split0_test_score': values.copy(), 'split1_test_score': values.copy(),
            'mean_fit_time': np.ones(len(values)), 'mean_score_time': np.ones(len(values)) * .1}


def search_record(record):
    record = deepcopy(record)
    record['params'] = [{'estimator__' + name: value for name, value in candidate.items()}
                        for candidate in record['params']]
    return SimpleNamespace(cv_results_=record, n_splits_=2, checkpoint_summary_={
        'diagnostics': [[{'n_iter_': 3, 'converged_': False,
                          'termination_reason_': 'max_iter'}] * 2 for _ in record['params']],
        'new_folds': 2 * len(record['params']), 'reused_folds': 0, 'wall_seconds': .01})


class DerivedDWD(EnsembleKernelDWD):
    pass


class ConfirmationPolicyTests(unittest.TestCase):
    def test_configuration_requires_explicit_larger_integer_budget(self):
        with self.assertRaises(TypeError):
            NumericalConfirmationConfig()
        for kwargs in ({'max_iter': 0}, {'max_iter': True}, {'max_iter': 2.5},
                       {'max_iter': np.bool_(True)}, {'max_iter': 5, 'top_k': 1},
                       {'max_iter': 5, 'top_k': True}, {'max_iter': 5, 'top_k': 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NumericalConfirmationConfig(**kwargs)
        value = NumericalConfirmationConfig(np.int64(5), np.int64(2))
        self.assertEqual(value.top_k, 2)

    def test_validation_preserves_profile_and_accepts_binary_and_nested_paths(self):
        config = NumericalConfirmationConfig(6)
        original = pipeline()
        before = parameter_hash(original)
        profile = validate_confirmation(original, {'ensemble__lambd': [1.]},
                                        AdaptiveSearchConfig(), config)
        self.assertEqual(profile['parameter'], 'ensemble__max_iter')
        self.assertEqual(profile['max_iter'], 3)
        self.assertEqual(profile['confirmation_max_iter'], 6)
        self.assertEqual(profile['stopping'], 'objective')
        self.assertEqual(profile['obj_tol'], 1e-5)
        self.assertEqual(parameter_hash(original), before)
        multiclass = Pipeline([('scale', StandardScaler()), ('ovo',
            OvOClassifier(EnsembleKernelDWD(max_iter=3)))])
        nested = validate_confirmation(multiclass, {},
                    AdaptiveSearchConfig(prefix='ovo__estimator__'), config)
        self.assertEqual(nested['parameter'], 'ovo__estimator__max_iter')
        direct = validate_confirmation(EnsembleKernelDWD(max_iter=3), {},
                    AdaptiveSearchConfig(prefix=''), config)
        self.assertEqual(direct['parameter'], 'max_iter')

    def test_invalid_target_caps_prefixes_and_grid_replacements(self):
        config = NumericalConfirmationConfig(6)
        invalid = [pipeline(0), pipeline(True), pipeline(3.), pipeline(6), pipeline(7),
                   Pipeline([('ensemble', EnsembleKernelSVM())]),
                   Pipeline([('ensemble', DerivedDWD(max_iter=3))])]
        for original in invalid:
            with self.subTest(original=original), self.assertRaises(ValueError):
                validate_confirmation(original, {}, AdaptiveSearchConfig(), config)
        for prefix in ('ensemble', 'ensemble____'):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                validate_confirmation(pipeline(), {}, AdaptiveSearchConfig(prefix=prefix), config)
        for name in ('ensemble__max_iter', 'ensemble', 'steps', 'ensemble__steps'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_confirmation(pipeline(), {name: [3]}, AdaptiveSearchConfig(), config)
        with self.assertRaises(ValueError):
            validate_confirmation(pipeline(), {}, AdaptiveSearchConfig(regularization='C'), config)
        with self.assertRaises(TypeError):
            validate_confirmation(pipeline(), {}, AdaptiveSearchConfig(), {'max_iter': 6})

    def test_nested_estimator_replacements_are_rejected(self):
        original = Pipeline([('ovo', OvOClassifier(EnsembleKernelDWD(max_iter=3)))])
        for name in ('ovo', 'ovo__estimator', 'steps', 'ovo__steps',
                     'ovo__estimator__max_iter'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_confirmation(original, {name: [None]},
                    AdaptiveSearchConfig(prefix='ovo__estimator__'), NumericalConfirmationConfig(6))

    def test_rank_gate_checks_ties_and_competitors_below_unchanged_winner(self):
        baseline = results([.9, .8, .8])
        self.assertEqual(compare_rankings(baseline, results([.95, .85, .85]))['state'], 'stable')
        for after in ([.9, .8, .7], [.9, .7, .8], [.9, .9, .9], [.8, .9, .9]):
            with self.subTest(after=after):
                report = compare_rankings(baseline, results(after))
                self.assertEqual(report['state'], 'needs_review')
                self.assertTrue(report['ranking_changed'])
        before = results([.9, .8, .7])
        self.assertTrue(compare_rankings(before, results([.9, .7, .8]))['ranking_changed'])
        self.assertTrue(compare_rankings(results([.9, .8]),
            results([.9, np.nextafter(.9, 0.)]))['state'] == 'stable')
        self.assertTrue(compare_rankings(results([.9, .9]),
            results([.9, np.nextafter(.9, 0.)]))['ranking_changed'])

    def test_invalid_fold_held_even_if_aggregate_is_finite(self):
        after = results([.9, .8])
        after['split1_test_score'][1] = np.nan
        report = compare_rankings(results([.9, .8]), after)
        self.assertEqual(report['state'], 'needs_review')
        self.assertEqual(report['eligible'].tolist(), [True, False])
        self.assertEqual(report['confirmation_order'], [0])
        with self.assertRaises(ValueError):
            compare_rankings(after, results([.9, .8]))

    def test_diagnostic_counts_visit_original_learners_once(self):
        binary = {'n_iter_': [2, 2, 3], 'converged_': [True, True, False],
                  'base_diagnostics_': [
                      {'n_iter': 2, 'converged': True, 'termination_reason': 'objective_tolerance',
                       'objective_tolerance_met': True},
                      {'n_iter': 2, 'converged': False, 'termination_reason': 'objective_tolerance',
                       'objective_tolerance_met': True}],
                  'final_estimator': {'n_iter_': 3, 'converged_': False,
                                      'termination_reason_': 'max_iter'}}
        multiclass = {'pairwise_estimators': [{'pair': [0, 1], 'diagnostics': binary},
                                             {'pair': [0, 2], 'diagnostics': binary}]}
        report = _diagnostic_summary([{'diagnostics': [[multiclass]]}], 3)
        self.assertEqual(report['learner_records'], 6)
        self.assertEqual(report['converged_true'], 2)
        self.assertEqual(report['converged_false'], 4)
        self.assertEqual(report['objective_tolerance_true'], 4)
        self.assertEqual(report['iteration_cap_reached'], 2)
        self.assertEqual(report['termination_reasons'], {'max_iter': 2, 'objective_tolerance': 4})

    def run_mock(self, before, after, folder, *, resume=False):
        original = pipeline()
        baseline = search_record(before)
        rows = [{'params': candidate, 'stage': 'coarse', 'stage_index': index}
                for index, candidate in enumerate(before['params'])]
        captured = {}

        class Checkpoint:
            def __init__(self, child_dir, estimator, X, y, folds, **kwargs):
                captured.update(directory=child_dir, estimator=estimator, folds=folds, **kwargs)
                self.identity_sha256 = 'child-context'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def search(self, grid, *, stage):
                captured.update(grid=grid, stage=stage)
                selected = [{name: value[0] for name, value in candidate.items()} for candidate in grid]
                return search_record(results(after, selected))

        folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))]
        with patch('numerical_confirmation.CheckpointedCV', Checkpoint):
            report = run_numerical_confirmation(
                pipeline=original, results=before, rows=rows, searches={'coarse': baseline},
                folds=folds, X=np.ones((4, 1)), y=np.array([0, 1, 0, 1]), run_dir=folder,
                config=NumericalConfirmationConfig(6, top_k=3), search_config=AdaptiveSearchConfig(),
                jobs=2, error_policy='nan', resume=resume, cv_seeds=None, seed_parameter=None,
                reuse_equivalent_folds=True, parent_identity='original-context')
        return report, captured, original

    def test_shortlist_freezes_stable_ranking_and_changes_only_cloned_cap(self):
        parameters = [{'ensemble__lambd': np.float64(index + 1),
                       'ensemble__gamma': np.float32(.1),
                       'ensemble__kernel_kws': {'shape': [1, 2]}}
                      for index in range(4)]
        before = results([.8, .9, .9, np.nan], parameters)
        original_hash = parameter_hash(before)
        with tempfile.TemporaryDirectory() as folder:
            report, captured, original = self.run_mock(before, [.95, .95, .85], folder)
        self.assertEqual(report['shortlist_indices'], [1, 2, 0])
        self.assertEqual(report['state'], 'stable')
        self.assertEqual(report['baseline_ranking'], [1, 2, 0])
        self.assertEqual(report['confirmation_ranking'], [1, 2, 0])
        self.assertEqual(report['shortlist'][0]['baseline_rank'], 1)
        self.assertEqual(report['shortlist'][1]['confirmation_rank'], 1)
        self.assertEqual(captured['estimator'].get_params()['ensemble__max_iter'], 6)
        self.assertEqual(original.get_params()['ensemble__max_iter'], 3)
        self.assertEqual(parameter_hash(before), original_hash)
        self.assertEqual(captured['directory'].name, 'numerical-confirmation')
        self.assertFalse(captured['resume'])
        self.assertEqual(captured['jobs'], 2)
        self.assertEqual(captured['provenance']['parent_identity_sha256'], 'original-context')
        self.assertEqual(captured['provenance']['shortlist_parameter_hashes'],
                         [parameter_hash(parameters[index]) for index in (1, 2, 0)])
        for candidate, index in zip(captured['grid'], (1, 2, 0), strict=True):
            selected = {name: values[0] for name, values in candidate.items()}
            self.assertEqual(parameter_hash(selected), parameter_hash(parameters[index]))
            self.assertNotIn('ensemble__max_iter', selected)
        self.assertEqual(report['checkpoint_summary']['executed_folds'], 6)
        self.assertFalse(report['held_out_scored'])

    def test_parent_resume_creates_new_child_only_until_child_identity_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            _, captured, _ = self.run_mock(results([.9, .8]), [.9, .8], folder, resume=True)
            self.assertFalse(captured['resume'])
            child = Path(folder) / 'numerical-confirmation' / 'cv-checkpoints'
            child.mkdir(parents=True)
            (child / 'identity.json').write_text('{}', encoding='utf-8')
            _, captured, _ = self.run_mock(results([.9, .8]), [.9, .8], folder, resume=True)
            self.assertTrue(captured['resume'])
            _, captured, _ = self.run_mock(results([.9, .8]), [.9, .8], folder, resume=False)
            self.assertFalse(captured['resume'])

    def test_missing_child_identity_with_old_receipts_is_not_recreated(self):
        with tempfile.TemporaryDirectory() as folder:
            child = Path(folder) / 'numerical-confirmation' / 'cv-checkpoints' / 'folds'
            child.mkdir(parents=True)
            saved = child / 'old.json'
            saved.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(CheckpointError, 'without their checkpoint identity'):
                self.run_mock(results([.9, .8]), [.9, .8], folder, resume=True)
            self.assertEqual(saved.read_text(encoding='utf-8'), '{}')
            self.assertFalse((child.parent / 'identity.json').exists())

    def test_candidate_numerical_overrides_are_preserved_and_reported(self):
        params = [{'ensemble__lambd': 1., 'ensemble__stopping': 'fixed', 'ensemble__obj_tol': 0.},
                  {'ensemble__lambd': 2., 'ensemble__tol': 1e-5}]
        with tempfile.TemporaryDirectory() as folder:
            report, captured, _ = self.run_mock(results([.9, .8], params), [.9, .8], folder)
        self.assertEqual(report['profile']['stopping'], 'objective')
        self.assertEqual(report['shortlist'][0]['profile']['stopping'], 'fixed')
        self.assertEqual(report['shortlist'][0]['profile']['obj_tol'], 0.)
        self.assertEqual(report['shortlist'][1]['profile']['tol'], 1e-5)
        for actual, expected in zip(captured['grid'], params, strict=True):
            self.assertEqual({key: value[0] for key, value in actual.items()}, expected)

    def test_insufficient_shortlist_does_not_start_confirmation(self):
        before = results([.9, np.nan])
        with tempfile.TemporaryDirectory() as folder:
            report, captured, _ = self.run_mock(before, [], folder)
        self.assertEqual(report['state'], 'needs_review')
        self.assertEqual(report['candidate_count'], 1)
        self.assertEqual(captured, {})

    def test_mismatched_original_evidence_is_rejected_before_dispatch(self):
        before = results([.9, .8])
        with tempfile.TemporaryDirectory() as folder:
            with patch('numerical_confirmation.bare_parameters', return_value={'wrong': 1}):
                with self.assertRaisesRegex(ValueError, 'baseline does not match'):
                    self.run_mock(before, [.9, .8], folder)

    def test_duplicate_tuples_are_not_counted_as_distinct_competitors(self):
        before = results([.9, .9], [{'ensemble__lambd': 1.}] * 2)
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'duplicate'):
            self.run_mock(before, [.9, .9], folder)

    def test_stage_summary_omits_nonshortlist_diagnostics_but_keeps_costs(self):
        original_record = search_record
        calls = []

        def annotated(record):
            answer = original_record(record)
            if not calls:
                answer.checkpoint_summary_['diagnostics'][0][0]['fact'] = 'selected learner detail'
                answer.checkpoint_summary_['diagnostics'][3][0]['fact'] = 'unselected detail' * 10000
                answer.checkpoint_summary_['seed_results'] = {'unselected detail': [0] * 10000}
                answer.checkpoint_summary_['candidate_count'] = 4
            calls.append(True)
            return answer

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(sys.modules[__name__], 'search_record', side_effect=annotated):
                report, _, _ = self.run_mock(results([.9, .8, .7, .6]), [.9, .8, .7], folder)
        summary = report['baseline_stage_summaries']['coarse']['checkpoint_summary']
        self.assertEqual(summary['candidate_count'], 4)
        self.assertEqual(summary['new_folds'], 8)
        self.assertEqual(summary['wall_seconds'], .01)
        self.assertNotIn('diagnostics', summary)
        self.assertNotIn('seed_results', summary)
        self.assertEqual(report['shortlist'][0]['baseline']['diagnostics'][0]['fact'],
                         'selected learner detail')
        self.assertNotIn('unselected detail', str(report))


if __name__ == '__main__':
    unittest.main()
