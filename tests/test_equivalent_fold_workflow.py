"""Equivalent-fold reuse remains opt-in and never replaces the final refit."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_dwd import EnsembleKernelSVM

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import cv_checkpoint
import tune_mnist
import tune_mnist_svm
import tune_mnist_multiclass
from search_support import new_run_directory


class EquivalentFoldWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='equivalent-workflow-')
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
        # The ratios retain ten rows on each training fold, but twenty versus
        # twenty-one rows on the complete training set.
        self.grid = {'ensemble__C': [1.], 'ensemble__gamma': [.5],
                     'ensemble__sv_selection_ratio': [.51, .525]}
        self.config = adaptive_search.AdaptiveSearchConfig(
            regularization='C', prefix='ensemble__', refine=False, max_candidates=2)
        self.runtime = patch.object(cv_checkpoint, 'runtime_fingerprint',
                                    return_value={'test_runtime': 'equivalent-workflow-v1'})
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def execute(self, folder, enabled, *, resume=False):
        # Exercise the actual folds, candidate selection and final refit. Only
        # parameter-review acceptance is controlled to keep this lifecycle test
        # independent of the separately tested expansion/refinement policy.
        proposals = [
            {'phase': 'coarse', 'candidates': list(ParameterGrid(self.grid)), 'state': {}, 'review': None},
            {'phase': 'done', 'candidates': [], 'state': {},
             'review': {'state': 'accepted', 'reasons': [], 'stopping_reason': 'controlled lifecycle review'}}]
        with patch.object(adaptive_search, 'next_search_stage', side_effect=proposals), \
                contextlib.redirect_stdout(io.StringIO()):
            return adaptive_search.run_adaptive_search(
                self.pipeline, self.grid, self.X, self.y, self.X_test, self.y_test,
                self.folds, folder, config=self.config, jobs=1, resume=resume,
                reuse_equivalent_folds=enabled)

    def test_equivalent_cv_counts_do_not_replace_full_training_refit(self):
        fits = []
        original = cv_checkpoint._fit_fold

        def recorded(*args, **kwargs):
            fits.append((len(args[9]), args[6]['ensemble__sv_selection_ratio']))
            return original(*args, **kwargs)

        with patch.object(cv_checkpoint, '_fit_fold', recorded), \
                patch.object(adaptive_search, 'clone', wraps=adaptive_search.clone) as final_clone:
            ordinary = self.execute(new_run_directory(self.root / 'ordinary'), False)
            self.assertEqual([size for size, _ in fits], [20, 20, 20, 20])
            self.assertEqual(final_clone.call_count, 1)
            fits.clear()
            folder = new_run_directory(self.root / 'reuse')
            reused = self.execute(folder, True)
            self.assertEqual([size for size, _ in fits], [20, 20])
            self.assertEqual(final_clone.call_count, 2)
            final_ratio = reused['selection']['parameters']['ensemble__sv_selection_ratio']
            final = reused['model'].named_steps['ensemble']
            self.assertEqual(len(final.selected_indices_), int(40 * final_ratio))
            self.assertNotEqual(len(final.selected_indices_), int(20 * final_ratio))
            self.assertEqual(final.sv_selection_ratio, final_ratio)
            self.assertEqual(ordinary['selection'], reused['selection'])
            for key in ('params', 'mean_test_score', 'std_test_score', 'split0_test_score', 'split1_test_score'):
                np.testing.assert_equal(ordinary['cv_results'][key], reused['cv_results'][key])
            np.testing.assert_array_equal(ordinary['model'].predict(self.X_test), reused['model'].predict(self.X_test))
            self.assertEqual(ordinary['completed']['held_out_accuracy'], reused['completed']['held_out_accuracy'])
            self.assertEqual(len(reused['cv_results']['params']), 2)
            self.assertEqual(reused['stages'][0]['new_candidates'], 2)
            facts = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            self.assertIs(facts['settings']['reuse_equivalent_folds'], True)
            fits.clear()
            resumed = self.execute(folder, True, resume=True)
            self.assertTrue(resumed['resumed_completed'])
            self.assertIsNone(resumed['model'])
            self.assertEqual(fits, [])
            self.assertEqual(final_clone.call_count, 2)
            saved_run = (folder / 'run.json').read_bytes()
            with self.assertRaises(cv_checkpoint.CheckpointError):
                self.execute(folder, False, resume=True)
            self.assertEqual((folder / 'run.json').read_bytes(), saved_run)

    def test_reuse_setting_is_a_boolean(self):
        for value in (1, 'yes', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be boolean'):
                self.execute(self.root / 'unused', value)

    def test_three_command_lines_forward_only_explicit_opt_in(self):
        for module in (tune_mnist, tune_mnist_svm, tune_mnist_multiclass):
            for enabled in (False, True):
                with self.subTest(module=module.__name__, enabled=enabled):
                    folder = self.root / (module.__name__ + str(enabled))
                    arguments = ['--data', 'unused.npz', '--run', '--no-cache', '--output-dir', str(folder)]
                    if enabled:
                        arguments.append('--reuse-equivalent-folds')
                    if module is tune_mnist_multiclass:
                        arguments.extend(['--folds', '2', '--train-size', '20'])
                        y = np.repeat(np.arange(10), 2)
                        X = np.column_stack((y, np.arange(20))).astype(float)
                        loader = 'load_local_multiclass_data'
                        values = (X, X.copy(), y, y.copy(), {})
                    else:
                        loader = 'load_local_data'
                        values = (self.X, self.X_test, self.y, self.y_test)
                    held = {'model': None, 'review': {'reasons': ['controlled hold']}}
                    with patch.object(module, loader, return_value=values), \
                            patch.object(adaptive_search, 'run_adaptive_search', return_value=held) as run, \
                            contextlib.redirect_stdout(io.StringIO()):
                        module.main(arguments)
                    self.assertIs(run.call_args.kwargs['reuse_equivalent_folds'], enabled)


if __name__ == '__main__':
    unittest.main()
