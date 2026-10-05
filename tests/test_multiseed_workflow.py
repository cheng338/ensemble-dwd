"""Multi-seed example searches agree with independent real-estimator loops."""
import contextlib
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

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import cv_checkpoint
import tune_mnist
import tune_mnist_multiclass
import tune_mnist_svm
from search_support import new_run_directory


def real_problem(kind='svm', *, multiclass=False):
    """Small nontrivial fixed data; no study observations or downloaded files."""
    X, y = make_classification(
        n_samples=48 if not multiclass else 54, n_features=4, n_informative=3,
        n_redundant=0, n_classes=3 if multiclass else 2, n_clusters_per_class=1,
        class_sep=1.0, random_state=91)
    options = dict(M=2, theta=.5, gamma=.2, random_state=7,
                   sv_selection_ratio=.8, store_base_estimators=False)
    base = (EnsembleKernelDWD(lambd=.05, max_iter=5, stopping='fixed', **options)
            if kind == 'dwd' else EnsembleKernelSVM(C=1., **options))
    model = OvOClassifier(base, n_jobs=1) if multiclass else base
    prefix = 'model__estimator__' if multiclass else 'model__'
    pipeline = Pipeline([('scale', StandardScaler()), ('model', model)])
    grid = {prefix + ('lambd' if kind == 'dwd' else 'C'):
            [.03, .1] if kind == 'dwd' else [.5, 2.]}
    folds = list(StratifiedKFold(3, shuffle=True, random_state=29).split(X, y))
    return pipeline, grid, prefix + 'random_state', X, y, folds


class RealMultiSeedCVTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='multiseed-real-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def search(self, name, problem, *, seeds=(7, 13), jobs=1, explicit=True):
        estimator, grid, seed_parameter, X, y, folds = problem
        options = dict(jobs=jobs, error_policy='raise')
        if explicit:
            options.update(cv_seeds=seeds, seed_parameter=seed_parameter)
        with cv_checkpoint.CheckpointedCV(self.root / name, estimator, X, y, folds,
                                          **options) as checkpoint:
            return checkpoint.search(grid)

    def assert_manual_scores(self, kind, multiclass=False):
        problem = real_problem(kind, multiclass=multiclass)
        estimator, grid, seed_parameter, X, y, folds = problem
        seeds = (7, 13)
        expected = []
        # This reference deliberately does not call the checkpoint's scorer,
        # candidate evaluator, aggregation or cloning helpers.
        with threadpool_limits(limits=1):
            for parameters in ParameterGrid(grid):
                seed_scores = []
                for seed in seeds:
                    fold_scores = []
                    for training, validation in folds:
                        model = clone(estimator).set_params(
                            **parameters, **{seed_parameter: seed})
                        model.fit(X[training], y[training])
                        fold_scores.append(accuracy_score(y[validation], model.predict(X[validation])))
                    seed_scores.append(fold_scores)
                expected.append(seed_scores)
        expected = np.asarray(expected)
        search = self.search(kind + str(multiclass), problem, seeds=seeds)
        seed_results = search.seed_results_
        self.assertEqual(seed_results['axes'], ['candidate', 'seed', 'fold'])
        self.assertEqual(seed_results['seeds'], list(seeds))
        self.assertEqual(seed_results['seed_parameter'], seed_parameter)
        self.assertEqual(seed_results['candidate_params'], list(ParameterGrid(grid)))
        self.assertEqual(seed_results['aggregation_policy'], 'mean-seeds-within-fold-v1')
        np.testing.assert_array_equal(seed_results['scores'], expected)
        fold_means = expected.mean(axis=1)
        for fold in range(len(folds)):
            np.testing.assert_array_equal(search.cv_results_[f'split{fold}_test_score'], fold_means[:, fold])
        np.testing.assert_array_equal(search.cv_results_['mean_test_score'], fold_means.mean(axis=1))
        np.testing.assert_array_equal(search.cv_results_['std_test_score'], fold_means.std(axis=1))
        self.assertEqual(len(search.cv_results_['params']), len(ParameterGrid(grid)))
        self.assertTrue(all('estimator__' + seed_parameter not in parameters
                            for parameters in search.cv_results_['params']))

    def test_svm_matches_manual_seed_and_fold_loop(self):
        self.assert_manual_scores('svm')

    def test_dwd_matches_manual_seed_and_fold_loop(self):
        self.assert_manual_scores('dwd')

    def test_ovo_matches_manual_seed_and_fold_loop(self):
        self.assert_manual_scores('svm', multiclass=True)

    def test_real_parallel_workers_match_serial_seed_scores(self):
        problem = real_problem()
        serial = self.search('serial', problem)
        parallel = self.search('parallel', problem, jobs=2)
        np.testing.assert_array_equal(serial.seed_results_['scores'], parallel.seed_results_['scores'])
        for key in ('params', 'split0_test_score', 'split1_test_score', 'split2_test_score',
                    'mean_test_score', 'std_test_score', 'rank_test_score'):
            np.testing.assert_equal(serial.cv_results_[key], parallel.cv_results_[key])
        self.assertEqual(serial.best_params_, parallel.best_params_)

    def test_explicit_one_seed_preserves_ordinary_single_seed_scores(self):
        for kind in ('svm', 'dwd'):
            with self.subTest(kind=kind):
                problem = real_problem(kind)
                original = self.search(kind + '-ordinary', problem, explicit=False)
                explicit = self.search(kind + '-explicit', problem, seeds=(7,))
                self.assertFalse(hasattr(original, 'seed_results_'))
                for key in ('params', 'split0_test_score', 'split1_test_score', 'split2_test_score',
                            'mean_test_score', 'std_test_score', 'rank_test_score'):
                    np.testing.assert_equal(original.cv_results_[key], explicit.cv_results_[key])
                self.assertEqual(original.best_params_, explicit.best_params_)


class MultiSeedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='multiseed-workflow-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        x = np.linspace(-3., 3., 40)
        self.X = np.column_stack((x, x * x))
        self.y = (x > 0).astype(int)
        self.X_test = np.array([[-2.1, 4.41], [-.3, .09], [.4, .16], [1.9, 3.61]])
        self.y_test = np.array([0, 0, 1, 1])
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.pipeline = Pipeline([('scale', StandardScaler()), ('ensemble', EnsembleKernelSVM(
            M=1, theta=.5, C=1., gamma=.5, random_state=42, store_base_estimators=False))])
        # On 20-row CV training folds these retain the same ten rows. On all
        # 40 rows they retain twenty versus twenty-one rows.
        self.grid = {'ensemble__C': [1.], 'ensemble__gamma': [.5],
                     'ensemble__sv_selection_ratio': [.51, .525]}
        self.config = adaptive_search.AdaptiveSearchConfig(
            regularization='C', prefix='ensemble__', refine=False, max_candidates=2)
        runtime = patch.object(cv_checkpoint, 'runtime_fingerprint',
                               return_value={'test_runtime': 'multiseed-lifecycle-v1'})
        runtime.start()
        self.addCleanup(runtime.stop)

    def execute(self, folder, *, resume=False, seeds=(7, 13), refit_seed=97, reuse=True,
                grid=None):
        chosen_grid = self.grid if grid is None else grid
        # Control only policy acceptance; real CV fits, scoring, selection,
        # checkpoint transactions and the full-data final fit remain active.
        proposals = [
            {'phase': 'coarse', 'candidates': list(ParameterGrid(chosen_grid)),
             'state': {}, 'review': None},
            {'phase': 'done', 'candidates': [], 'state': {},
             'review': {'state': 'accepted', 'reasons': [],
                        'stopping_reason': 'controlled lifecycle review'}}]
        with patch.object(adaptive_search, 'next_search_stage', side_effect=proposals), \
                contextlib.redirect_stdout(io.StringIO()):
            return adaptive_search.run_adaptive_search(
                self.pipeline, chosen_grid, self.X, self.y, self.X_test, self.y_test,
                self.folds, folder, config=self.config, resume=resume,
                cv_seeds=seeds, refit_seed=refit_seed, reuse_equivalent_folds=reuse)

    def test_equivalent_ratios_reuse_within_seed_and_refit_original_ratio_with_declared_seed(self):
        folder = new_run_directory(self.root / 'reuse')
        with patch.object(adaptive_search, 'clone', wraps=adaptive_search.clone) as final_clone:
            result = self.execute(folder)
        self.assertEqual(final_clone.call_count, 1)
        summary = result['coarse_search'].checkpoint_summary_
        self.assertEqual(summary['executed_folds'], 4)
        self.assertEqual(summary['equivalent_reused_folds'], 4)
        seed_results = result['coarse_search'].seed_results_
        records = [json.loads(path.read_text(encoding='utf-8'))['payload']
                   for path in (folder / 'cv-checkpoints/folds').glob('*.json')]
        self.assertEqual(len(records), 8)
        aliases = [record for record in records if 'equivalent_reuse' in record]
        self.assertEqual(len(aliases), 4)
        for record in aliases:
            current = dict(record['parameters'][1])['ensemble__random_state']
            source = dict(record['equivalent_reuse']['source_parameters'][1])['ensemble__random_state']
            self.assertEqual(current, source)
        for candidate in seed_results['receipt_parameters']:
            for seed, folds in zip((7, 13), candidate):
                for parameters in folds:
                    self.assertEqual(dict(parameters[1])['ensemble__random_state'], ['int', seed])
        final = result['model'].named_steps['ensemble']
        ratio = result['selection']['parameters']['ensemble__sv_selection_ratio']
        self.assertEqual(final.random_state, 97)
        self.assertEqual(final.sv_selection_ratio, ratio)
        self.assertEqual(len(final.selected_indices_), int(len(self.y) * ratio))
        self.assertNotEqual(len(final.selected_indices_), int(len(self.folds[0][0]) * ratio))
        np.testing.assert_array_equal(result['model'].named_steps['scale'].mean_, self.X.mean(axis=0))
        expected = clone(self.pipeline).set_params(**result['selection']['parameters'],
                                                  ensemble__random_state=97)
        with threadpool_limits(limits=1):
            expected.fit(self.X, self.y)
        np.testing.assert_array_equal(expected.named_steps['ensemble'].selected_indices_, final.selected_indices_)
        np.testing.assert_array_equal(expected.decision_function(self.X_test),
                                      result['model'].decision_function(self.X_test))
        self.assertNotIn('ensemble__random_state', result['selection']['parameters'])
        self.assertEqual(self.pipeline.named_steps['ensemble'].random_state, 42)

    def test_final_refit_default_keeps_estimator_seed_instead_of_a_cv_seed(self):
        result = self.execute(new_run_directory(self.root / 'default-final'), refit_seed=None)
        self.assertEqual(result['model'].named_steps['ensemble'].random_state, 42)

    def test_interruption_resumes_only_missing_fold_seed_transactions(self):
        folder = new_run_directory(self.root / 'interrupt')
        original = cv_checkpoint._fit_fold
        attempts = 0

        def interrupt_third(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 3:
                raise KeyboardInterrupt('controlled fold-seed interruption')
            return original(*args, **kwargs)

        with patch.object(cv_checkpoint, '_fit_fold', interrupt_third), self.assertRaises(KeyboardInterrupt):
            self.execute(folder, reuse=False)
        complete = {path.name: path.read_bytes() for path in (folder / 'cv-checkpoints/folds').glob('*.json')}
        self.assertEqual(len(complete), 2)
        with patch.object(cv_checkpoint, '_fit_fold', wraps=original) as remaining:
            result = self.execute(folder, resume=True, reuse=False)
        self.assertEqual(remaining.call_count, 6)
        self.assertEqual(result['coarse_search'].checkpoint_summary_['reused_folds'], 2)
        for name, content in complete.items():
            self.assertEqual((folder / 'cv-checkpoints/folds' / name).read_bytes(), content)
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('completed CV must not fit')), \
                patch.object(adaptive_search, 'clone', side_effect=AssertionError('completed final fit must not clone')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('completed run must not score')):
            resumed = self.execute(folder, resume=True, reuse=False)
        self.assertTrue(resumed['resumed_completed'])
        self.assertIsNone(resumed['model'])
        self.assertEqual(result['completed'], resumed['completed'])

    def test_changed_cv_seeds_order_or_refit_seed_cannot_resume(self):
        folder = new_run_directory(self.root / 'identity')
        self.execute(folder)
        status = (folder / 'status.json').read_bytes()
        for options in ({'seeds': (7, 19)}, {'seeds': (13, 7)}, {'refit_seed': 98}):
            with self.subTest(options=options), \
                    patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('identity mismatch must not fit')), \
                    self.assertRaises(cv_checkpoint.CheckpointError):
                self.execute(folder, resume=True, **options)
            self.assertEqual((folder / 'status.json').read_bytes(), status)

    def test_search_cannot_tune_seed_or_replace_its_estimator(self):
        for addition in ({'ensemble__random_state': [7]}, {'ensemble': [EnsembleKernelSVM()]}):
            with self.subTest(parameter=next(iter(addition))), \
                    patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('invalid grid must not fit')), \
                    self.assertRaises(ValueError):
                self.execute(new_run_directory(self.root / next(iter(addition))),
                             grid={**self.grid, **addition})


class MultiSeedCommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='multiseed-cli-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def loader_values(module):
        if module is tune_mnist_multiclass:
            y = np.repeat(np.arange(10), 2)
            X = np.column_stack((y, np.arange(20))).astype(float)
            return 'load_local_multiclass_data', (X, X.copy(), y, y.copy(), {})
        y = np.tile([0, 1], 10)
        X = np.column_stack((y, np.arange(20))).astype(float)
        return 'load_local_data', (X, X[:4], y, y[:4])

    def test_all_command_lines_forward_explicit_and_default_seed_options(self):
        for module in (tune_mnist, tune_mnist_svm, tune_mnist_multiclass):
            for explicit in (False, True):
                with self.subTest(module=module.__name__, explicit=explicit):
                    arguments = ['--data', 'unused.npz', '--run', '--no-cache', '--output-dir',
                                 str(self.root / (module.__name__ + str(explicit)))]
                    if explicit:
                        arguments += ['--cv-seeds', '7', '13', '--refit-seed', '97']
                    if module is tune_mnist_multiclass:
                        arguments += ['--folds', '2', '--train-size', '20']
                    loader, values = self.loader_values(module)
                    held = {'model': None, 'review': {'reasons': ['controlled hold']}}
                    with patch.object(module, loader, return_value=values), \
                            patch.object(adaptive_search, 'run_adaptive_search', return_value=held) as run, \
                            contextlib.redirect_stdout(io.StringIO()):
                        module.main(arguments)
                    kwargs = run.call_args.kwargs
                    if explicit:
                        self.assertEqual(list(kwargs['cv_seeds']), [7, 13])
                        self.assertEqual(kwargs['refit_seed'], 97)
                    else:
                        self.assertIsNone(kwargs.get('cv_seeds'))
                        self.assertIsNone(kwargs.get('refit_seed'))

    def test_preview_multiplies_fit_counts_by_seeds_but_not_candidates(self):
        for module in (tune_mnist, tune_mnist_svm, tune_mnist_multiclass):
            loader, _ = self.loader_values(module)
            for options, expected in (([], 625), (['--cv-seeds', '7', '13'], 1250)):
                with self.subTest(module=module.__name__, options=options):
                    stream = io.StringIO()
                    with patch.object(module, loader) as load, \
                            patch.object(adaptive_search, 'run_adaptive_search') as run, \
                            contextlib.redirect_stdout(stream):
                        module.main(['--data', 'missing.npz', *options])
                    load.assert_not_called()
                    run.assert_not_called()
                    output = stream.getvalue().replace(',', '')
                    self.assertIn('125 candidates', output)
                    self.assertRegex(output, rf'\b{expected}\s+(?:ensemble|OVO) CV fits\b')
                    self.assertIn('Prepared only', output)

    def test_invalid_seed_options_fail_before_loading_data_or_starting_work(self):
        invalid = [ ['--cv-seeds'], ['--cv-seeds', '7', '7'], ['--cv-seeds', '-1'],
                    ['--cv-seeds', '4294967296'], ['--cv-seeds', '1.5'],
                    ['--refit-seed', '7'], ['--cv-seeds', '7', '--refit-seed', '-1'] ]
        for module in (tune_mnist, tune_mnist_svm, tune_mnist_multiclass):
            loader, _ = self.loader_values(module)
            for options in invalid:
                with self.subTest(module=module.__name__, options=options), \
                        patch.object(module, loader) as load, \
                        patch.object(adaptive_search, 'run_adaptive_search') as run, \
                        contextlib.redirect_stderr(io.StringIO()), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        module.main(['--data', 'missing.npz', '--run', *options])
                    load.assert_not_called()
                    run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
