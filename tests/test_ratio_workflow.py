"""Bounded ratio stages preserve CV evidence, sampling seeds and final fitting."""
import contextlib
from dataclasses import replace
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import cv_checkpoint
import tune_mnist
import tune_mnist_multiclass
import tune_mnist_svm
from ratio_refinement import RatioRefinementConfig, next_ratio_stage
from search_support import new_run_directory


def accepted_review():
    return {'state': 'accepted', 'reasons': [],
            'stopping_reason': 'controlled lifecycle acceptance'}


class RatioSurface(ClassifierMixin, BaseEstimator):
    """A controlled winner for testing candidate selection, not model accuracy."""
    fits = []

    def __init__(self, C=1., gamma=.5, sv_selection_ratio=.5, random_state=42):
        self.C = C
        self.gamma = gamma
        self.sv_selection_ratio = sv_selection_ratio
        self.random_state = random_state

    def fit(self, X, y):
        type(self).fits.append((len(y), self.sv_selection_ratio, self.random_state))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        labels = (np.asarray(X)[:, 0] > 0).astype(int)
        return labels if self.sv_selection_ratio == .54 else 1 - labels


class EndpointRatioSurface(RatioSurface):
    def predict(self, X):
        labels = (np.asarray(X)[:, 0] > 0).astype(int)
        winner = (self.C == 1. and self.gamma == .5 and self.sv_selection_ratio == .4)
        return labels if winner else 1 - labels


class RatioWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='ratio-workflow-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        x = np.linspace(-3., 3., 40)
        self.X = np.column_stack((x, x * x))
        self.y = (x > 0).astype(int)
        self.test_X = np.array([[-2.1, 4.41], [-.3, .09], [.4, .16], [1.9, 3.61]])
        self.test_y = np.array([0, 0, 1, 1])
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.pipeline = Pipeline([('scale', StandardScaler()), ('ensemble', EnsembleKernelSVM(
            M=1, theta=.5, C=1., gamma=.5, random_state=42, store_base_estimators=False))])
        self.grid = {'ensemble__C': [1.], 'ensemble__gamma': [.5],
                     'ensemble__sv_selection_ratio': [.51, .55]}
        self.generated = [{'ensemble__C': 1., 'ensemble__gamma': .5,
                           'ensemble__sv_selection_ratio': ratio} for ratio in (.525, .54)]
        self.config = adaptive_search.AdaptiveSearchConfig(
            regularization='C', prefix='ensemble__', refine=True, max_candidates=4)
        self.ratio = RatioRefinementConfig(max_rounds=2, ratio_tol=.02, stable_rounds=1)
        self.runtime_patch = patch.object(cv_checkpoint, 'runtime_fingerprint',
                                          return_value={'test_runtime': 'ratio-lifecycle-v1'})
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def execute(self, folder, *, resume=False, enabled=True, ratio=None, seeds=(7, 13),
                reuse=True, jobs=1, hold=False, refit_seed=97):
        # Policy acceptance is controlled here so that these tests isolate the
        # lifecycle. CV fitting, candidate selection, receipts and final fitting
        # all use the actual implementations and real ensemble estimators.
        coefficient_stages = [
            {'phase': 'coarse', 'candidates': list(ParameterGrid(self.grid)),
             'state': {}, 'review': None},
            {'phase': 'done', 'candidates': [], 'state': {}, 'review': accepted_review()}]
        review = ({'state': 'needs_review', 'reasons': ['ratio_round_budget'],
                   'stopping_reason': 'ratio_round_budget'} if hold else accepted_review())
        ratio_stages = [
            {'phase': 'ratio-local-1', 'candidates': list(self.generated),
             'state': {}, 'review': None},
            {'phase': 'done', 'candidates': [], 'state': {}, 'review': review}]
        seed_options = {} if seeds is None else {'cv_seeds': seeds, 'refit_seed': refit_seed}
        with patch.object(adaptive_search, 'next_search_stage', side_effect=coefficient_stages), \
                patch.object(adaptive_search, 'next_ratio_stage', side_effect=ratio_stages) as ratio_policy, \
                contextlib.redirect_stdout(io.StringIO()):
            result = adaptive_search.run_adaptive_search(
                self.pipeline, self.grid, self.X, self.y, self.test_X, self.test_y,
                self.folds, folder, config=self.config, resume=resume, jobs=jobs,
                ratio_refinement=(self.ratio if ratio is None else ratio) if enabled else None,
                reuse_equivalent_folds=reuse, **seed_options)
        if not enabled:
            ratio_policy.assert_not_called()
        return result

    @staticmethod
    def fold_receipts(folder):
        return [json.loads(path.read_text(encoding='utf-8'))['payload']
                for path in (folder / 'cv-checkpoints/folds').glob('*.json')]

    def test_generated_ratios_reuse_only_matching_seeds_and_refit_once(self):
        folder = new_run_directory(self.root / 'reuse')
        with patch.object(adaptive_search, 'clone', wraps=adaptive_search.clone) as final_clone:
            result = self.execute(folder)
        self.assertEqual(final_clone.call_count, 1)
        self.assertEqual([stage['name'] for stage in result['stages']], ['coarse', 'ratio-local-1'])
        self.assertEqual(len(result['cv_results']['params']), 4)
        self.assertEqual(sum(stage['checkpoint']['executed_folds'] for stage in result['stages']), 8)
        self.assertEqual(sum(stage['checkpoint']['equivalent_reused_folds'] for stage in result['stages']), 8)
        records = self.fold_receipts(folder)
        self.assertEqual(len(records), 16)
        aliases = [record for record in records if 'equivalent_reuse' in record]
        self.assertEqual(len(aliases), 8)
        for record in aliases:
            current = dict(record['parameters'][1])['ensemble__random_state']
            source = dict(record['equivalent_reuse']['source_parameters'][1])['ensemble__random_state']
            self.assertEqual(current, source)
        final = result['model'].named_steps['ensemble']
        chosen_ratio = result['selection']['parameters']['ensemble__sv_selection_ratio']
        self.assertEqual(final.sv_selection_ratio, chosen_ratio)
        self.assertEqual(final.random_state, 97)
        self.assertEqual(len(final.selected_indices_), int(len(self.y) * chosen_ratio))
        np.testing.assert_array_equal(result['model'].named_steps['scale'].mean_, self.X.mean(axis=0))
        expected = clone(self.pipeline).set_params(**result['selection']['parameters'],
                                                  ensemble__random_state=97)
        with threadpool_limits(limits=1):
            expected.fit(self.X, self.y)
            expected_scores = expected.decision_function(self.test_X)
            actual_scores = result['model'].decision_function(self.test_X)
        np.testing.assert_array_equal(expected.named_steps['ensemble'].selected_indices_, final.selected_indices_)
        np.testing.assert_array_equal(expected_scores, actual_scores)
        self.assertEqual(self.pipeline.named_steps['ensemble'].random_state, 42)

    def test_interrupted_ratio_stage_resumes_missing_fold_seed_receipts_only(self):
        folder = new_run_directory(self.root / 'interrupted')
        original = cv_checkpoint._fit_fold
        attempts = 0

        def interrupt_third_ratio_fit(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 11:
                raise KeyboardInterrupt('controlled partial ratio stage')
            return original(*args, **kwargs)

        with patch.object(cv_checkpoint, '_fit_fold', interrupt_third_ratio_fit), \
                self.assertRaises(KeyboardInterrupt):
            self.execute(folder, reuse=False)
        status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['state'], 'interrupted')
        self.assertEqual(status['phase'], 'ratio-local-1_cv')
        completed = {path.name: path.read_bytes() for path in (folder / 'cv-checkpoints/folds').glob('*.json')}
        self.assertEqual(len(completed), 10)
        with patch.object(cv_checkpoint, '_fit_fold', wraps=original) as remaining:
            result = self.execute(folder, resume=True, reuse=False)
        self.assertEqual(remaining.call_count, 6)
        self.assertEqual(result['stages'][0]['checkpoint']['reused_folds'], 8)
        self.assertEqual(result['stages'][1]['checkpoint']['reused_folds'], 2)
        for name, contents in completed.items():
            self.assertEqual((folder / 'cv-checkpoints/folds' / name).read_bytes(), contents)
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('completed CV fit')), \
                patch.object(adaptive_search, 'clone', side_effect=AssertionError('completed final fit')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('completed test scoring')):
            replay = self.execute(folder, resume=True, reuse=False)
        self.assertTrue(replay['resumed_completed'])
        self.assertIsNone(replay['model'])
        self.assertEqual(replay['completed'], result['completed'])

    def test_new_ratio_winner_replaces_coarse_selection_and_is_refitted_once(self):
        self.pipeline = Pipeline([('ensemble', RatioSurface())])
        RatioSurface.fits = []
        result = self.execute(new_run_directory(self.root / 'new-winner'), reuse=False)
        self.assertEqual(result['selection']['source_stage'], 'ratio-local-1')
        self.assertEqual(result['selection']['parameters']['ensemble__sv_selection_ratio'], .54)
        self.assertEqual(result['model'].named_steps['ensemble'].sv_selection_ratio, .54)
        self.assertEqual([fit for fit in RatioSurface.fits if fit[0] == 40], [(40, .54, 97)])
        self.assertEqual(result['completed']['held_out_accuracy'], 1.)

    def test_actual_ratio_policy_accepts_endpoint_and_restores_completed_mirror(self):
        folder = new_run_directory(self.root / 'actual-policy')
        pipeline = Pipeline([('ensemble', EndpointRatioSurface())])
        grid = {'ensemble__C': [.5, 1., 2.], 'ensemble__gamma': [.25, .5, 1.],
                'ensemble__sv_selection_ratio': [.4, .6, .8]}
        config = replace(self.config, max_candidates=100, top_k=1)
        ratio = RatioRefinementConfig(max_rounds=3, ratio_tol=.05, stable_rounds=2)
        old_bracket = {'previous_coefficient_winner': True}
        old_plateau = [{'axis': 'ensemble__gamma', 'side': 'high', 'basis': 'previous winner'}]
        state = {'winner_bracket': old_bracket, 'boundary_plateau_evidence': old_plateau}
        stages = [
            {'phase': 'coarse', 'candidates': list(ParameterGrid(grid)), 'state': state, 'review': None},
            {'phase': 'done', 'candidates': [], 'state': state, 'review': accepted_review()}]
        accepted = False

        def observe_actual_ratio_policy(*args, **kwargs):
            nonlocal accepted
            proposal = next_ratio_stage(*args, **kwargs)
            accepted = proposal['review'] is not None and proposal['review']['state'] == 'accepted'
            return proposal

        def final_clone(*args, **kwargs):
            self.assertTrue(accepted, 'Final fitting started before ratio policy acceptance.')
            return clone(*args, **kwargs)

        def final_score(*args, **kwargs):
            self.assertTrue(accepted, 'Held-out scoring started before ratio policy acceptance.')
            return accuracy_score(*args, **kwargs)

        EndpointRatioSurface.fits = []
        with patch.object(adaptive_search, 'next_search_stage', side_effect=stages), \
                patch.object(adaptive_search, 'next_ratio_stage', side_effect=observe_actual_ratio_policy), \
                patch.object(adaptive_search, 'clone', side_effect=final_clone) as final_fit, \
                patch.object(adaptive_search, 'accuracy_score', side_effect=final_score) as test_score, \
                contextlib.redirect_stdout(io.StringIO()):
            result = adaptive_search.run_adaptive_search(
                pipeline, grid, self.X, self.y, self.test_X, self.test_y, self.folds,
                folder, config=config, ratio_refinement=ratio)
        self.assertTrue(accepted)
        self.assertEqual(final_fit.call_count, 1)
        self.assertEqual(test_score.call_count, 1)
        self.assertEqual([fit for fit in EndpointRatioSurface.fits if fit[0] == 40], [(40, .4, 42)])
        self.assertEqual(result['review']['state'], 'accepted')
        self.assertEqual(result['review']['ratio_refinement']['completed_rounds'], 2)
        self.assertEqual(result['review']['ratio_refinement']['stable_rounds'], 2)
        self.assertEqual(result['review']['winner_bracket'], result['ratio_policy_state']['winner_bracket'])
        self.assertIn('ensemble__sv_selection_ratio', result['review']['winner_bracket'])
        self.assertEqual(result['review']['boundary_directions'], [('ensemble__sv_selection_ratio', 'low')])
        self.assertEqual(result['review']['boundary_plateau_evidence'], [])
        self.assertEqual(result['review']['coefficient_search']['winner_bracket'], old_bracket)
        self.assertEqual(result['review']['coefficient_search']['boundary_plateau_evidence'], old_plateau)
        mirror = folder / 'ratio-refinement.json'
        original_mirror = json.loads(mirror.read_text(encoding='utf-8'))
        mirror.unlink()
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('completed CV fit')), \
                patch.object(adaptive_search, 'next_ratio_stage', side_effect=AssertionError('completed policy')), \
                patch.object(adaptive_search, 'clone', side_effect=AssertionError('completed final fit')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('completed test scoring')), \
                contextlib.redirect_stdout(io.StringIO()):
            replay = adaptive_search.run_adaptive_search(
                pipeline, grid, self.X, self.y, self.test_X, self.test_y, self.folds,
                folder, config=config, ratio_refinement=ratio, resume=True)
        self.assertTrue(replay['resumed_completed'])
        self.assertEqual(json.loads(mirror.read_text(encoding='utf-8')), original_mirror)
        self.assertEqual(replay['ratio_policy_state'], result['ratio_policy_state'])

    def test_changed_ratio_settings_refuse_resume_before_work_or_status_replacement(self):
        folder = new_run_directory(self.root / 'identity')
        self.execute(folder)
        status = (folder / 'status.json').read_bytes()
        variations = [
            {'enabled': False},
            {'ratio': replace(self.ratio, bounds=(.50, .60))},
            {'ratio': replace(self.ratio, max_rounds=3)},
            {'ratio': replace(self.ratio, ratio_tol=.01)},
            {'ratio': replace(self.ratio, stable_rounds=2)},
            {'seeds': (7, 19)},
            {'refit_seed': 98}]
        for options in variations:
            with self.subTest(options=options), \
                    patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('changed identity fit')), \
                    self.assertRaises(cv_checkpoint.CheckpointError):
                self.execute(folder, resume=True, **options)
            self.assertEqual((folder / 'status.json').read_bytes(), status)

    def test_unresolved_ratio_review_holds_final_refit_and_test_scoring(self):
        folder = new_run_directory(self.root / 'held')
        with patch.object(adaptive_search, 'clone', side_effect=AssertionError('held final fit')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('held test scoring')):
            result = self.execute(folder, hold=True)
        self.assertEqual(result['review']['state'], 'needs_review')
        self.assertIn('ratio_round_budget', result['review']['reasons'])
        self.assertIsNone(result['model'])
        self.assertIsNone(result['completed'])
        self.assertEqual(len(result['cv_results']['params']), 4)
        self.assertFalse((folder / 'refit.json').exists())
        self.assertFalse((folder / 'completed.json').exists())

    def test_disabled_option_keeps_only_original_stage_and_original_seed(self):
        result = self.execute(new_run_directory(self.root / 'disabled'), enabled=False, seeds=None)
        self.assertEqual([stage['name'] for stage in result['stages']], ['coarse'])
        self.assertEqual(result['cv_results']['params'], list(ParameterGrid(self.grid)))
        self.assertEqual(result['model'].named_steps['ensemble'].random_state, 42)
        self.assertNotIn('cv_seeds', result['selection'])
        manifest = json.loads((result['run_dir'] / 'run.json').read_text(encoding='utf-8'))
        self.assertNotIn('ratio_refinement', manifest)

    def test_parallel_ratio_stage_matches_serial_scores_and_selection(self):
        # Child processes discover their own runtime. Use the same real
        # discovery in the coordinator so its identity can be checked there.
        self.runtime_patch.stop()
        serial = self.execute(new_run_directory(self.root / 'serial'), reuse=False)
        parallel = self.execute(new_run_directory(self.root / 'parallel'), reuse=False, jobs=2)
        for key in ('params', 'split0_test_score', 'split1_test_score', 'mean_test_score', 'std_test_score'):
            np.testing.assert_equal(serial['cv_results'][key], parallel['cv_results'][key])
        self.assertEqual(serial['selection']['parameters'], parallel['selection']['parameters'])
        np.testing.assert_array_equal(serial['last_search'].seed_results_['scores'],
                                      parallel['last_search'].seed_results_['scores'])

    def test_bad_ratio_grid_or_disabled_refinement_rejects_before_fitting(self):
        original = self.grid
        invalid_grids = [
            {name: values for name, values in original.items() if not name.endswith('sv_selection_ratio')},
            {**original, 'ensemble__sv_selection_ratio': [.5]},
            {**original, 'ensemble__sv_selection_ratio': [.5, .5]},
            {**original, 'ensemble__sv_selection_ratio': [False, .5]},
            {**original, 'ensemble__sv_selection_ratio': [0., .5]},
            {**original, 'ensemble__sv_selection_ratio': [.5, 1.1]},
            {**original, 'ensemble__sv_selection_ratio': [.5, np.nan]}]
        for number, grid in enumerate(invalid_grids):
            self.grid = grid
            with self.subTest(grid=grid), \
                    patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('invalid ratio grid fit')), \
                    self.assertRaises(ValueError):
                self.execute(new_run_directory(self.root / ('invalid-' + str(number))))
        self.grid = original
        self.config = replace(self.config, refine=False)
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('disabled refinement fit')), \
                self.assertRaises(ValueError):
            self.execute(new_run_directory(self.root / 'no-refine'))


class GeneratedRatioScoreTests(unittest.TestCase):
    def test_actual_policy_candidates_match_manual_dwd_and_svm_seed_fold_scores(self):
        X, y = make_classification(n_samples=36, n_features=4, n_informative=3,
                                   n_redundant=0, n_clusters_per_class=1, random_state=91)
        folds = list(StratifiedKFold(2, shuffle=True, random_state=29).split(X, y))
        for kind in ('svm', 'dwd'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='ratio-manual-') as temporary:
                regularization = 'C' if kind == 'svm' else 'lambd'
                center = 1. if kind == 'svm' else .05
                grid = {'ensemble__' + regularization: [center / 2, center, center * 2],
                        'ensemble__gamma': [.1, .2, .4], 'ensemble__sv_selection_ratio': [.4, .6, .8]}
                parameters = list(ParameterGrid(grid))
                means = np.array([.9 if (row['ensemble__' + regularization] == center
                                         and row['ensemble__gamma'] == .2
                                         and row['ensemble__sv_selection_ratio'] == .6) else .5
                                  for row in parameters])
                # These artificial scores select a bounded neighborhood. Only
                # the resulting real-estimator scores are compared below.
                evidence = {'params': parameters, 'mean_test_score': means,
                            'split0_test_score': means, 'split1_test_score': means}
                config = adaptive_search.AdaptiveSearchConfig(regularization=regularization,
                    max_candidates=200, top_k=1, bracket_factor=2.)
                proposal = next_ratio_stage(evidence, grid, config,
                    RatioRefinementConfig(max_rounds=2, ratio_tol=.05, stable_rounds=1))
                generated = [row for row in proposal['candidates']
                             if row['ensemble__' + regularization] == center
                             and row['ensemble__gamma'] == .2
                             and row['ensemble__sv_selection_ratio'] not in grid['ensemble__sv_selection_ratio']]
                self.assertGreaterEqual(len(generated), 2)
                options = dict(M=1, theta=.5, gamma=.2, random_state=7,
                               sv_selection_ratio=.6, store_base_estimators=False)
                base = (EnsembleKernelSVM(C=1., **options) if kind == 'svm' else
                        EnsembleKernelDWD(lambd=.05, max_iter=3, stopping='fixed', **options))
                pipeline = Pipeline([('scale', StandardScaler()), ('ensemble', base)])
                candidate_grid = [{name: [value] for name, value in row.items()} for row in generated]
                for seeds in (None, (7, 13)):
                    seed_options = {} if seeds is None else {'cv_seeds': seeds,
                                                            'seed_parameter': 'ensemble__random_state'}
                    name = 'single' if seeds is None else 'multiple'
                    with threadpool_limits(limits=1), cv_checkpoint.CheckpointedCV(
                            Path(temporary) / name, pipeline, X, y, folds,
                            error_policy='raise', **seed_options) as checkpoint:
                        search = checkpoint.search(candidate_grid, stage='ratio-local-1')
                        expected = []
                        for row in generated:
                            by_seed = []
                            for seed in (7,) if seeds is None else seeds:
                                scores = []
                                for training, validation in folds:
                                    model = clone(pipeline).set_params(**row, ensemble__random_state=seed)
                                    model.fit(X[training], y[training])
                                    predictions = model.predict(X[validation])
                                    scores.append(Fraction(int(np.count_nonzero(y[validation] == predictions)), len(validation)))
                                by_seed.append(scores)
                            expected.append(by_seed)
                    exact_folds = np.asarray(expected, dtype=object).mean(axis=1)
                    expected = np.asarray(expected, dtype=float)
                    if seeds is not None:
                        np.testing.assert_array_equal(search.seed_results_['scores'], expected)
                    fold_means = np.asarray(exact_folds, dtype=float)
                    for fold in range(2):
                        np.testing.assert_array_equal(search.cv_results_[f'split{fold}_test_score'],
                                                      fold_means[:, fold])
                    np.testing.assert_array_equal(search.cv_results_['mean_test_score'],
                                                  np.asarray(exact_folds.mean(axis=1), dtype=float))


class RatioCommandLineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='ratio-cli-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def loader_values(module):
        if module is tune_mnist_multiclass:
            y = np.repeat(np.arange(10), 2)
            X = np.column_stack((y, np.arange(20))).astype(float)
            return 'load_local_multiclass_data', (X, X.copy(), y, y.copy(), {})
        y = np.tile([0, 1], 10)
        X = np.column_stack((y, np.arange(20))).astype(float)
        return 'load_local_data', (X, X[:4], y, y[:4])

    def test_three_examples_forward_enabled_policy_and_preserve_disabled_default(self):
        for module in (tune_mnist, tune_mnist_svm, tune_mnist_multiclass):
            for mode in ('disabled', 'default', 'explicit'):
                with self.subTest(module=module.__name__, mode=mode):
                    arguments = ['--data', 'unused.npz', '--run', '--no-cache', '--output-dir',
                                 str(self.root / (module.__name__ + mode))]
                    if mode != 'disabled':
                        arguments += ['--refine-ratio']
                    if mode == 'explicit':
                        arguments += ['--ratio-bounds', '.1', '.9', '--max-ratio-rounds', '4', '--ratio-tol', '.02']
                    if module is tune_mnist_multiclass:
                        arguments += ['--folds', '2', '--train-size', '20']
                    loader, values = self.loader_values(module)
                    held = {'model': None, 'review': {'reasons': ['controlled hold']}}
                    with patch.object(module, loader, return_value=values), \
                            patch.object(adaptive_search, 'run_adaptive_search', return_value=held) as run, \
                            contextlib.redirect_stdout(io.StringIO()):
                        module.main(arguments)
                    ratio = run.call_args.kwargs.get('ratio_refinement')
                    if mode == 'disabled':
                        self.assertIsNone(ratio)
                    else:
                        self.assertIsInstance(ratio, RatioRefinementConfig)
                        self.assertEqual(ratio.max_rounds, 4 if mode == 'explicit' else 3)
                        self.assertEqual(ratio.ratio_tol, .02 if mode == 'explicit' else .01)
                        if mode == 'explicit':
                            self.assertEqual(tuple(ratio.bounds), (.1, .9))

    def test_preview_keeps_initial_candidate_and_seed_fit_counts(self):
        for module in (tune_mnist, tune_mnist_svm, tune_mnist_multiclass):
            for options, fits in (([], 625), (['--cv-seeds', '7', '13'], 1250)):
                with self.subTest(module=module.__name__, options=options):
                    loader, _ = self.loader_values(module)
                    output = io.StringIO()
                    with patch.object(module, loader) as load, \
                            patch.object(adaptive_search, 'run_adaptive_search') as run, \
                            contextlib.redirect_stdout(output):
                        module.main(['--data', 'missing.npz', '--refine-ratio', *options])
                    load.assert_not_called()
                    run.assert_not_called()
                    text = output.getvalue().replace(',', '')
                    self.assertIn('125 candidates', text)
                    self.assertRegex(text, rf'\b{fits}\s+(?:ensemble|OVO) CV fits\b')
                    self.assertIn('500 unique candidates total', text)
                    self.assertIn('ratio', text.lower())
                    self.assertIn('Prepared only', text)

    def test_invalid_options_reject_before_data_loading_or_search(self):
        invalid = [['--ratio-bounds', '.1', '.9'], ['--max-ratio-rounds', '2'], ['--ratio-tol', '.02'],
                   ['--refine-ratio', '--no-refine'], ['--refine-ratio', '--ratio-bounds', '0', '.9'],
                   ['--refine-ratio', '--ratio-bounds', '.9', '.1'],
                   ['--refine-ratio', '--ratio-bounds', '.1', '1.1'],
                   ['--refine-ratio', '--ratio-bounds', '.4', '.6'],
                   ['--refine-ratio', '--max-ratio-rounds', '0'],
                   ['--refine-ratio', '--ratio-tol', '0'],
                   ['--refine-ratio', '--ratio-tol', 'nan']]
        for module in (tune_mnist, tune_mnist_svm, tune_mnist_multiclass):
            loader, _ = self.loader_values(module)
            cases = invalid + ([['--refine-ratio', '--smoke-grid']] if module is tune_mnist_multiclass else [])
            for options in cases:
                with self.subTest(module=module.__name__, options=options), \
                        patch.object(module, loader) as load, \
                        patch.object(adaptive_search, 'run_adaptive_search') as run, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        module.main(['--data', 'missing.npz', '--run', *options])
                    load.assert_not_called()
                    run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
