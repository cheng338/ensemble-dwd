"""Higher-budget confirmation stays separate from selection and final fitting."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelDWD, IneligibleSelectionError, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import cv_checkpoint
import tune_mnist
import tune_mnist_multiclass
import tune_mnist_svm
from numerical_confirmation import NumericalConfirmationConfig
from ratio_refinement import RatioRefinementConfig
from search_support import json_safe, new_run_directory


def accepted_review():
    return {'state': 'accepted', 'reasons': [],
            'stopping_reason': 'controlled lifecycle acceptance'}


class ConfirmationWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='confirmation-workflow-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        x = np.linspace(-3., 3., 40)
        self.X = np.column_stack((x, x * x))
        self.y = (x > 0).astype(int)
        self.test_X = np.array([[-2.1, 4.41], [-.3, .09], [.4, .16], [1.9, 3.61]])
        self.test_y = np.array([0, 0, 1, 1])
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.pipeline = Pipeline([('scale', StandardScaler()), ('ensemble', EnsembleKernelDWD(
            M=1, theta=.5, lambd=.05, gamma=.5, random_state=42,
            max_iter=2, stopping='fixed', store_base_estimators=False))])
        # Both ratios select ten rows per CV training fold. Full fitting uses
        # twenty versus twenty-one rows, so it must remain a fresh fit.
        self.grid = {'ensemble__lambd': [.05], 'ensemble__gamma': [.5],
                     'ensemble__sv_selection_ratio': [.51, .525]}
        self.config = adaptive_search.AdaptiveSearchConfig(
            regularization='lambd', prefix='ensemble__', refine=False, max_candidates=2)
        self.confirmation = NumericalConfirmationConfig(max_iter=4, top_k=2)
        self.runtime_patch = patch.object(cv_checkpoint, 'runtime_fingerprint',
                                          return_value={'test_runtime': 'confirmation-lifecycle-v1'})
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def execute(self, folder, *, enabled=True, confirmation=None, resume=False,
                seeds=(7, 13), refit_seed=97, reuse=True, jobs=1, grid=None,
                original_hold=False):
        grid = self.grid if grid is None else grid
        review = ({'state': 'needs_review', 'reasons': ['controlled coefficient hold'],
                   'stopping_reason': 'controlled coefficient hold'}
                  if original_hold else accepted_review())
        proposals = [
            {'phase': 'coarse', 'candidates': list(ParameterGrid(grid)), 'state': {}, 'review': None},
            {'phase': 'done', 'candidates': [], 'state': {}, 'review': review}]
        seed_options = {} if seeds is None else {'cv_seeds': seeds, 'refit_seed': refit_seed}
        with patch.object(adaptive_search, 'next_search_stage', side_effect=proposals), \
                contextlib.redirect_stdout(io.StringIO()):
            return adaptive_search.run_adaptive_search(
                self.pipeline, grid, self.X, self.y, self.test_X, self.test_y,
                self.folds, folder, config=self.config, resume=resume, jobs=jobs,
                numerical_confirmation=(self.confirmation if confirmation is None else confirmation)
                if enabled else None, reuse_equivalent_folds=reuse, **seed_options)

    @staticmethod
    def receipts(folder):
        return {path.name: json.loads(path.read_text(encoding='utf-8'))['payload']
                for path in (folder / 'cv-checkpoints/folds').glob('*.json')}

    def test_stable_confirmation_preserves_original_selection_budget_and_final_seed(self):
        folder = new_run_directory(self.root / 'stable')
        result = self.execute(folder)
        report = result['numerical_confirmation']
        self.assertEqual(report['state'], 'stable')
        self.assertFalse(report['ranking_changed'])
        self.assertEqual(result['cv_results']['params'], list(ParameterGrid(self.grid)))
        self.assertEqual(result['selection']['parameters'], list(ParameterGrid(self.grid))[0])
        self.assertEqual([stage['name'] for stage in result['stages']], ['coarse'])
        final = result['model'].named_steps['ensemble']
        self.assertEqual(final.max_iter, 2)
        self.assertEqual(final.random_state, 97)
        self.assertEqual(final.sv_selection_ratio, .51)
        self.assertEqual(len(final.selected_indices_), 20)
        self.assertEqual(self.pipeline.named_steps['ensemble'].max_iter, 2)
        expected = clone(self.pipeline).set_params(**result['selection']['parameters'],
                                                   ensemble__random_state=97)
        with threadpool_limits(limits=1):
            expected.fit(self.X, self.y)
            np.testing.assert_array_equal(expected.decision_function(self.test_X),
                                          result['model'].decision_function(self.test_X))
        np.testing.assert_array_equal(expected.named_steps['ensemble'].selected_indices_,
                                      final.selected_indices_)
        original = self.receipts(folder)
        higher = self.receipts(folder / 'numerical-confirmation')
        self.assertEqual(len(original), 8)
        self.assertEqual(len(higher), 8)
        self.assertTrue({row['identity_sha256'] for row in original.values()}.isdisjoint(
                        {row['identity_sha256'] for row in higher.values()}))
        # Exact-row reuse remains local to this higher-budget checkpoint.
        aliases = [row for row in higher.values() if 'equivalent_reuse' in row]
        self.assertEqual(len(aliases), 4)
        for row in aliases:
            source = row['equivalent_reuse']['source_task_sha256']
            self.assertIn(source + '.json', higher)
            self.assertNotIn(source + '.json', original)
        self.assertEqual(report['checkpoint_summary']['executed_folds'], 4)
        self.assertEqual(report['checkpoint_summary']['equivalent_reused_folds'], 4)

    def test_default_disabled_adds_no_confirmation_files_or_manifest_option(self):
        folder = new_run_directory(self.root / 'disabled')
        result = self.execute(folder, enabled=False, seeds=None)
        self.assertNotIn('numerical_confirmation', result)
        self.assertFalse((folder / 'numerical-confirmation').exists())
        self.assertFalse((folder / 'numerical-confirmation.json').exists())
        manifest = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        self.assertNotIn('numerical_confirmation', manifest)
        self.assertEqual(result['model'].named_steps['ensemble'].max_iter, 2)
        self.assertEqual(result['model'].named_steps['ensemble'].random_state, 42)

    def test_rank_reversal_and_tie_changes_hold_without_final_fit_or_test_scoring(self):
        # Controlled predictions exercise weak-order comparison independently
        # of chance score ties. CV/checkpoint/fitting still use real DWD models.
        for kind in ('reverse', 'split_tie', 'merge_tie'):
            def controlled_predictions(model, X):
                y = (np.asarray(X)[:, 0] > 0).astype(int)
                first = model.sv_selection_ratio == .51
                if kind == 'reverse':
                    correct = first if model.max_iter == 2 else not first
                elif kind == 'split_tie':
                    correct = True if model.max_iter == 2 else first
                else:
                    correct = first if model.max_iter == 2 else True
                return y if correct else 1 - y

            with self.subTest(kind=kind), \
                    patch.object(EnsembleKernelDWD, 'predict', controlled_predictions), \
                    patch.object(adaptive_search, 'clone', side_effect=AssertionError('held final fit')), \
                    patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('held test scoring')):
                result = self.execute(new_run_directory(self.root / kind), reuse=False, seeds=None)
            self.assertIsNone(result['model'])
            self.assertIsNone(result['completed'])
            self.assertEqual(result['review']['state'], 'needs_review')
            self.assertTrue(result['numerical_confirmation']['ranking_changed'])
            self.assertEqual(result['selection']['candidate_index'], 0)
            self.assertFalse((result['run_dir'] / 'refit.json').exists())
            if kind != 'reverse':
                self.assertEqual(result['numerical_confirmation']['baseline_ranking'],
                                 result['numerical_confirmation']['confirmation_ranking'])

    def test_high_budget_ineligibility_holds_and_is_not_averaged_away(self):
        original_fit = EnsembleKernelDWD.fit

        def ineligible_at_higher_budget(model, X, y):
            if model.max_iter == 4 and model.sv_selection_ratio > .52 and model.random_state == 13:
                raise IneligibleSelectionError('controlled higher-budget ineligible selection')
            return original_fit(model, X, y)

        with patch.object(EnsembleKernelDWD, 'fit', ineligible_at_higher_budget), \
                patch.object(adaptive_search, 'clone', side_effect=AssertionError('held final fit')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('held test scoring')):
            result = self.execute(new_run_directory(self.root / 'ineligible'), reuse=False)
        report = result['numerical_confirmation']
        self.assertEqual(result['review']['state'], 'needs_review')
        self.assertEqual(report['state'], 'needs_review')
        self.assertTrue(any(not row['eligible'] for row in report['shortlist']))
        self.assertTrue(np.isnan(np.asarray(
            report['cv_results']['confirmation']['mean_test_score'], dtype=float)).any())
        records = self.receipts(result['run_dir'] / 'numerical-confirmation')
        self.assertEqual(sum(row['state'] == 'ineligible' for row in records.values()), 2)
        self.assertIsNone(result['completed'])

    def test_one_eligible_candidate_cannot_claim_comparative_confirmation(self):
        grid = {**self.grid, 'ensemble__sv_selection_ratio': [.51]}
        folder = new_run_directory(self.root / 'one-candidate')
        with patch.object(adaptive_search, 'clone', side_effect=AssertionError('held final fit')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('held test scoring')):
            result = self.execute(folder, grid=grid)
        self.assertEqual(result['review']['state'], 'needs_review')
        self.assertEqual(result['numerical_confirmation']['state'], 'needs_review')
        self.assertEqual(self.receipts(folder / 'numerical-confirmation'), {})
        self.assertIsNone(result['completed'])

    def test_unaccepted_parameter_search_never_starts_confirmation(self):
        folder = new_run_directory(self.root / 'original-hold')
        result = self.execute(folder, original_hold=True)
        self.assertEqual(result['review']['state'], 'needs_review')
        self.assertFalse((folder / 'numerical-confirmation').exists())
        self.assertIsNone(result['completed'])

    def test_ratio_stage_acceptance_precedes_confirmation_and_preserves_each_stage_evidence(self):
        generated = {'ensemble__lambd': .05, 'ensemble__gamma': .5,
                     'ensemble__sv_selection_ratio': .54}
        ratio_config = RatioRefinementConfig(bounds=(.5, .6), max_rounds=1,
                                              ratio_tol=.02, stable_rounds=1)
        config = replace(self.config, refine=True, max_candidates=3)
        original_fit_fold = cv_checkpoint._fit_fold

        def controlled_predictions(model, X):
            labels = (np.asarray(X)[:, 0] > 0).astype(int)
            return labels if model.sv_selection_ratio == .54 else 1 - labels

        for held in (False, True):
            with self.subTest(ratio_held=held):
                folder = new_run_directory(self.root / ('ratio-held' if held else 'ratio-accepted'))
                coefficient_stages = [
                    {'phase': 'coarse', 'candidates': list(ParameterGrid(self.grid)),
                     'state': {}, 'review': None},
                    {'phase': 'done', 'candidates': [], 'state': {}, 'review': accepted_review()}]
                ratio_review = ({'state': 'needs_review', 'reasons': ['controlled ratio hold'],
                                 'stopping_reason': 'controlled ratio hold'}
                                if held else accepted_review())
                ratio_stages = iter([
                    {'phase': 'ratio-local-1', 'candidates': [generated], 'state': {}, 'review': None},
                    {'phase': 'done', 'candidates': [], 'state': {}, 'review': ratio_review}])
                ratio_accepted = False
                confirmation_fits = 0

                def next_ratio(*args, **kwargs):
                    nonlocal ratio_accepted
                    proposal = next(ratio_stages)
                    ratio_accepted = (proposal['review'] is not None
                                      and proposal['review']['state'] == 'accepted')
                    return proposal

                def observe_confirmation_order(*args, **kwargs):
                    nonlocal confirmation_fits
                    if 'numerical-confirmation' in Path(args[0]).parts:
                        self.assertTrue(ratio_accepted)
                        confirmation_fits += 1
                    return original_fit_fold(*args, **kwargs)

                with patch.object(adaptive_search, 'next_search_stage', side_effect=coefficient_stages), \
                        patch.object(adaptive_search, 'next_ratio_stage', side_effect=next_ratio), \
                        patch.object(EnsembleKernelDWD, 'predict', controlled_predictions), \
                        patch.object(cv_checkpoint, '_fit_fold', observe_confirmation_order), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = adaptive_search.run_adaptive_search(
                        self.pipeline, self.grid, self.X, self.y, self.test_X, self.test_y,
                        self.folds, folder, config=config, cv_seeds=(7, 13), refit_seed=97,
                        ratio_refinement=ratio_config, numerical_confirmation=self.confirmation)
                report = result['numerical_confirmation']
                self.assertEqual(result['selection']['parameters'], generated)
                if held:
                    self.assertEqual(report['state'], 'not_started')
                    self.assertEqual(confirmation_fits, 0)
                    self.assertFalse((folder / 'numerical-confirmation').exists())
                    self.assertIsNone(result['completed'])
                    self.assertIsNone(result['model'])
                    continue
                self.assertEqual(report['state'], 'stable')
                self.assertEqual(confirmation_fits, 8)
                self.assertEqual(report['shortlist_indices'], [2, 0])
                self.assertEqual([(row['source_stage'], row['source_stage_index'])
                                  for row in report['shortlist']], [('ratio-local-1', 0), ('coarse', 0)])
                searches = {'ratio-local-1': result['selected_search'],
                            'coarse': result['coarse_search']}
                for row in report['shortlist']:
                    stage = searches[row['source_stage']]
                    index = row['source_stage_index']
                    self.assertEqual(row['baseline']['mean_cv_accuracy'],
                                     stage.cv_results_['mean_test_score'][index])
                    self.assertEqual(row['baseline']['diagnostics'],
                                     json_safe(stage.checkpoint_summary_['diagnostics'][index]))
                    np.testing.assert_array_equal(row['baseline']['seed_results']['scores'],
                                                  stage.seed_results_['scores'][index])
                final = result['model'].named_steps['ensemble']
                self.assertEqual(final.sv_selection_ratio, .54)
                self.assertEqual(final.max_iter, 2)
                self.assertEqual(final.random_state, 97)
                self.assertEqual(len(final.selected_indices_), 21)

    def test_objective_tolerance_can_leave_higher_budget_unused_without_convergence_claim(self):
        self.pipeline.set_params(ensemble__max_iter=10, ensemble__stopping='objective',
                                 ensemble__obj_tol=1e6)
        result = self.execute(new_run_directory(self.root / 'early-objective'),
            confirmation=NumericalConfirmationConfig(max_iter=20, top_k=2), seeds=None, reuse=False)
        report = result['numerical_confirmation']
        self.assertEqual(report['state'], 'stable')
        self.assertEqual(report['profile']['stopping'], 'objective')
        self.assertEqual(report['profile']['obj_tol'], 1e6)
        for context in ('baseline', 'confirmation'):
            summary = report['diagnostics_summary'][context]
            self.assertGreater(summary['objective_tolerance_true'], 0)
            self.assertEqual(summary['iteration_cap_reached'], 0)
        np.testing.assert_equal(report['cv_results']['baseline']['mean_test_score'],
                                report['cv_results']['confirmation']['mean_test_score'])
        self.assertFalse(result['review']['numerical_confirmation']['stability_is_convergence_certificate'])
        self.assertEqual(result['model'].named_steps['ensemble'].max_iter, 10)

    def test_invalid_cap_and_estimator_path_changes_reject_before_cv_or_run_manifest(self):
        cases = [({'confirmation': NumericalConfirmationConfig(max_iter=2, top_k=2)}, {}),
                 ({}, {'ensemble__max_iter': [2]}),
                 ({}, {'ensemble': [EnsembleKernelDWD(max_iter=2)]})]
        for index, (options, additions) in enumerate(cases):
            folder = new_run_directory(self.root / ('invalid-' + str(index)))
            with self.subTest(index=index), \
                    patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('invalid confirmation fit')), \
                    self.assertRaises(ValueError):
                self.execute(folder, grid={**self.grid, **additions}, **options)
            self.assertFalse((folder / 'run.json').exists())
            self.assertEqual(self.receipts(folder), {})

    def test_interruption_resumes_missing_high_budget_receipts_and_restores_completed_mirror(self):
        folder = new_run_directory(self.root / 'interrupt')
        original = cv_checkpoint._fit_fold
        high_attempts = 0

        def interrupt_third_high_budget(*args, **kwargs):
            nonlocal high_attempts
            if 'numerical-confirmation' in Path(args[0]).parts:
                high_attempts += 1
                if high_attempts == 3:
                    raise KeyboardInterrupt('controlled partial numerical confirmation')
            return original(*args, **kwargs)

        with patch.object(cv_checkpoint, '_fit_fold', interrupt_third_high_budget), \
                self.assertRaises(KeyboardInterrupt):
            self.execute(folder, reuse=False)
        self.assertEqual(json.loads((folder / 'status.json').read_text())['state'], 'interrupted')
        before = {str(path.relative_to(folder)): path.read_bytes()
                  for path in folder.glob('**/cv-checkpoints/folds/*.json')}
        self.assertEqual(len(self.receipts(folder)), 8)
        self.assertEqual(len(self.receipts(folder / 'numerical-confirmation')), 2)
        with patch.object(cv_checkpoint, '_fit_fold', wraps=original) as remaining:
            result = self.execute(folder, resume=True, reuse=False)
        self.assertEqual(remaining.call_count, 6)
        self.assertEqual(result['numerical_confirmation']['checkpoint_summary']['reused_folds'], 2)
        for path, contents in before.items():
            self.assertEqual((folder / path).read_bytes(), contents)
        mirror = folder / 'numerical-confirmation.json'
        saved = mirror.read_bytes()
        mirror.unlink()
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('completed CV fit')), \
                patch.object(adaptive_search, 'clone', side_effect=AssertionError('completed final fit')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('completed test scoring')):
            replay = self.execute(folder, resume=True, reuse=False)
        self.assertTrue(replay['resumed_completed'])
        self.assertIsNone(replay['model'])
        self.assertEqual(replay['completed'], result['completed'])
        self.assertEqual(mirror.read_bytes(), saved)

    def test_changed_confirmation_or_seed_identity_cannot_rewrite_completed_state(self):
        folder = new_run_directory(self.root / 'identity')
        self.execute(folder)
        preserved = {name: (folder / name).read_bytes()
                     for name in ('status.json', 'run.json', 'numerical-confirmation.json')}
        cases = ({'confirmation': NumericalConfirmationConfig(max_iter=5, top_k=2)},
                 {'confirmation': NumericalConfirmationConfig(max_iter=4, top_k=3)},
                 {'enabled': False}, {'seeds': (13, 7)}, {'refit_seed': 98})
        for options in cases:
            with self.subTest(options=options), \
                    patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('mismatched identity fit')), \
                    self.assertRaises(cv_checkpoint.CheckpointError):
                self.execute(folder, resume=True, **options)
            for name, contents in preserved.items():
                self.assertEqual((folder / name).read_bytes(), contents)

    def test_real_parallel_confirmation_matches_serial_scores_and_ranking(self):
        # Workers discover their own actual runtime; use actual discovery in
        # their coordinator too so runtime validation is exercised honestly.
        self.runtime_patch.stop()
        serial = self.execute(new_run_directory(self.root / 'serial'), reuse=False)
        parallel = self.execute(new_run_directory(self.root / 'parallel'), reuse=False, jobs=2)
        for mode in ('baseline', 'confirmation'):
            left = serial['numerical_confirmation']['cv_results'][mode]
            right = parallel['numerical_confirmation']['cv_results'][mode]
            for key in ('params', 'split0_test_score', 'split1_test_score', 'mean_test_score'):
                np.testing.assert_equal(left[key], right[key])
        self.assertEqual(serial['selection'], parallel['selection'])
        self.assertEqual(serial['numerical_confirmation']['confirmation_ranking'],
                         parallel['numerical_confirmation']['confirmation_ranking'])


class ConfirmationManualParityTests(unittest.TestCase):
    def test_binary_and_ovo_higher_cap_scores_match_independent_single_and_multiseed_loops(self):
        for multiclass in (False, True):
            X, y = make_classification(n_samples=72 if multiclass else 48, n_features=4,
                n_informative=3, n_redundant=0, n_classes=3 if multiclass else 2,
                n_clusters_per_class=1, class_sep=1.5, random_state=91)
            folds = list(StratifiedKFold(2, shuffle=True, random_state=29).split(X, y))
            base = EnsembleKernelDWD(M=1, theta=.5, lambd=.05, gamma=.2, random_state=7,
                max_iter=2, stopping='fixed', sv_selection_ratio=.8, store_base_estimators=False)
            pipeline = Pipeline([('scale', StandardScaler()),
                                 ('model', OvOClassifier(base, n_jobs=1) if multiclass else base)])
            prefix = 'model__estimator__' if multiclass else 'model__'
            grid = {prefix + 'lambd': [.03, .1], prefix + 'gamma': [.2]}
            config = adaptive_search.AdaptiveSearchConfig(prefix=prefix, max_candidates=2, refine=False)
            for seeds in (None, (7, 13)):
                with self.subTest(multiclass=multiclass, seeds=seeds), \
                        tempfile.TemporaryDirectory(prefix='confirmation-manual-') as temporary:
                    stages = [
                        {'phase': 'coarse', 'candidates': list(ParameterGrid(grid)), 'state': {}, 'review': None},
                        {'phase': 'done', 'candidates': [], 'state': {}, 'review': accepted_review()}]
                    options = {} if seeds is None else {'cv_seeds': seeds, 'refit_seed': 7}
                    with patch.object(adaptive_search, 'next_search_stage', side_effect=stages), \
                            contextlib.redirect_stdout(io.StringIO()):
                        result = adaptive_search.run_adaptive_search(pipeline, grid, X, y, X[:6], y[:6],
                            folds, new_run_directory(Path(temporary) / 'run'), config=config,
                            numerical_confirmation=NumericalConfirmationConfig(max_iter=4, top_k=2), **options)
                    report = result['numerical_confirmation']
                    expected = []
                    with threadpool_limits(limits=1):
                        for row in report['shortlist']:
                            candidate = []
                            for seed in (7,) if seeds is None else seeds:
                                scores = []
                                for training, validation in folds:
                                    model = clone(pipeline).set_params(**row['parameters'],
                                        **{prefix + 'max_iter': 4, prefix + 'random_state': seed})
                                    model.fit(X[training], y[training])
                                    scores.append(accuracy_score(y[validation], model.predict(X[validation])))
                                candidate.append(scores)
                            expected.append(candidate)
                    expected = np.asarray(expected)
                    confirmed = report['cv_results']['confirmation']
                    for fold in range(2):
                        np.testing.assert_array_equal(confirmed[f'split{fold}_test_score'],
                                                      expected.mean(axis=1)[:, fold])
                    np.testing.assert_array_equal(confirmed['mean_test_score'], expected.mean(axis=1).mean(axis=1))
                    if seeds is not None:
                        for index, row in enumerate(report['shortlist']):
                            np.testing.assert_array_equal(row['confirmation']['seed_results']['scores'],
                                                          expected[index])
                    self.assertEqual(pipeline.get_params()[prefix + 'max_iter'], 2)
                    if multiclass:
                        diagnostics = report['shortlist'][0]['confirmation']['diagnostics']
                        self.assertIn('pairwise_estimators', json.dumps(diagnostics))


class ConfirmationCommandLineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='confirmation-cli-')
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

    def test_dwd_examples_forward_explicit_option_and_preserve_disabled_default(self):
        for module in (tune_mnist, tune_mnist_multiclass):
            for mode in ('disabled', 'default-shortlist', 'explicit-shortlist'):
                with self.subTest(module=module.__name__, mode=mode):
                    args = ['--data', 'unused.npz', '--run', '--no-cache', '--output-dir',
                            str(self.root / (module.__name__ + mode))]
                    if mode != 'disabled':
                        args += ['--confirm-max-iter', '200']
                    if mode == 'explicit-shortlist':
                        args += ['--confirm-top-k', '4']
                    if module is tune_mnist_multiclass:
                        args += ['--folds', '2', '--train-size', '20']
                    loader, values = self.loader_values(module)
                    held = {'model': None, 'review': {'reasons': ['controlled hold']}}
                    with patch.object(module, loader, return_value=values), \
                            patch.object(adaptive_search, 'run_adaptive_search', return_value=held) as run, \
                            contextlib.redirect_stdout(io.StringIO()):
                        module.main(args)
                    policy = run.call_args.kwargs.get('numerical_confirmation')
                    if mode == 'disabled':
                        self.assertIsNone(policy)
                    else:
                        self.assertIsInstance(policy, NumericalConfirmationConfig)
                        self.assertEqual(policy.max_iter, 200)
                        self.assertEqual(policy.top_k, 4 if mode == 'explicit-shortlist' else 3)

    def test_preview_counts_extra_confirmation_work_without_loading_data(self):
        for module in (tune_mnist, tune_mnist_multiclass):
            for options, extra_fits in (([], 15), (['--cv-seeds', '7', '13'], 30)):
                with self.subTest(module=module.__name__, options=options):
                    loader, _ = self.loader_values(module)
                    stream = io.StringIO()
                    with patch.object(module, loader) as load, \
                            patch.object(adaptive_search, 'run_adaptive_search') as run, \
                            contextlib.redirect_stdout(stream):
                        module.main(['--confirm-max-iter', '200', *options])
                    load.assert_not_called()
                    run.assert_not_called()
                    lines = [line for line in stream.getvalue().replace(',', '').splitlines()
                             if 'confirmation' in line.lower()]
                    self.assertTrue(any(str(extra_fits) in line and 'fits' in line for line in lines))
                    self.assertIn('125 candidates', stream.getvalue())
                    self.assertIn('Prepared only', stream.getvalue())

    def test_invalid_confirmation_options_fail_before_loading_data(self):
        invalid = [['--confirm-top-k', '3'], ['--confirm-max-iter', '0'],
                   ['--confirm-max-iter', '-1'], ['--confirm-max-iter', '100'],
                   ['--confirm-max-iter', '50'], ['--confirm-max-iter', '200', '--confirm-top-k', '1'],
                   ['--confirm-max-iter', '200', '--confirm-top-k', '0'],
                   ['--confirm-max-iter', '1.5']]
        for module in (tune_mnist, tune_mnist_multiclass):
            loader, _ = self.loader_values(module)
            for options in invalid:
                with self.subTest(module=module.__name__, options=options), \
                        patch.object(module, loader) as load, \
                        patch.object(adaptive_search, 'run_adaptive_search') as run, \
                        contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        module.main(['--data', 'missing.npz', '--run', *options])
                    load.assert_not_called()
                    run.assert_not_called()

    def test_svm_cli_does_not_advertise_or_accept_dwd_iteration_confirmation(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as help_exit:
            tune_mnist_svm.main(['--help'])
        self.assertEqual(help_exit.exception.code, 0)
        self.assertNotIn('--confirm-max-iter', stream.getvalue())
        with patch.object(tune_mnist_svm, 'load_local_data') as load, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tune_mnist_svm.main(['--confirm-max-iter', '200'])
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
