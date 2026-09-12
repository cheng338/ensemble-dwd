"""The SVM example previews its actual grid and uses C-aware boundaries."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from ensemble_dwd import mnist_svm_param_grid


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tune_mnist_svm.py"
SPEC = importlib.util.spec_from_file_location("svm_tuning_example", EXAMPLE)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)
import adaptive_search


class SVMExampleTests(unittest.TestCase):
    def test_preview_neither_loads_data_nor_starts_fits(self):
        result = subprocess.run([sys.executable, str(EXAMPLE), "--data", "missing.npz"],
                                text=True, capture_output=True, timeout=30, check=True)
        self.assertIn("125 candidates", result.stdout)
        self.assertIn("625 ensemble CV fits", result.stdout)
        self.assertIn("Prepared only", result.stdout)

    def test_boundary_detection_uses_C_not_lambda(self):
        grid = mnist_svm_param_grid("ensemble__")
        middle = {name: values[len(values) // 2] for name, values in grid.items()}
        self.assertEqual(example.boundary_parameters(middle, grid), [])
        for end in (0, -1):
            params = {**middle, "ensemble__C": grid["ensemble__C"][end]}
            self.assertEqual(example.boundary_parameters(params, grid), ["ensemble__C"])
            params["ensemble__gamma"] = grid["ensemble__gamma"][end]
            self.assertEqual(example.boundary_parameters(params, grid),
                             ["ensemble__C", "ensemble__gamma"])

    def test_fraction_endpoints_do_not_hold_an_interior_C_gamma_winner(self):
        grid = mnist_svm_param_grid("ensemble__")
        middle = {name: values[len(values) // 2] for name, values in grid.items()}
        for end in (0, -1):
            parameters = {**middle, "ensemble__theta": grid["ensemble__theta"][end],
                          "ensemble__sv_selection_ratio": grid["ensemble__sv_selection_ratio"][end]}
            self.assertEqual(example.boundary_parameters(parameters, grid), [])

    def test_cli_wires_adaptive_policy_and_accepts_held_result_without_model(self):
        X = np.tile([[-1., 2.], [1., -2.]], (5, 1))
        y = np.tile([0, 1], 5)
        calls = []

        def workflow(pipeline, grid, *args, **kwargs):
            parameters = {name: values[len(values) // 2] for name, values in grid.items()}
            parameters['ensemble__C'] = max(grid['ensemble__C'])
            self.assertEqual((pipeline.named_steps['ensemble'].M, pipeline.named_steps['ensemble'].theta), (5, .1))
            config = kwargs['config']
            self.assertEqual((config.regularization, config.prefix), ('C', 'ensemble__'))
            self.assertEqual((config.max_expansions, config.max_candidates), (3, 500))
            self.assertEqual((config.top_k, config.plateau_rounds), (3, 2))
            self.assertEqual((config.improvement_tol, config.tie_tol), (0.0, 1e-12))
            self.assertTrue(config.refine)
            self.assertEqual(kwargs['error_policy'], 'nan')
            reasons = ['candidate_budget']
            calls.append(parameters)
            return {'model': None, 'completed': None, 'review': {'reasons': reasons}}

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            stdout = io.StringIO()
            argv = [str(EXAMPLE), '--data', 'unused.npz', '--run', '--output-dir', str(run_dir)]
            with (patch.object(sys, 'argv', argv), patch.object(example, 'load_local_data',
                    return_value=(X, X[:4], y, y[:4])), patch.object(
                    adaptive_search, 'run_adaptive_search', side_effect=workflow), patch.object(
                    example, 'write_json_atomic', side_effect=AssertionError('Held model has no diagnostics.')),
                    contextlib.redirect_stdout(stdout)):
                example.main()
            self.assertEqual(len(calls), 1)
            self.assertIn('training-only review', stdout.getvalue())
            self.assertIn('test set was not scored', stdout.getvalue())
            self.assertFalse((run_dir / 'svm-diagnostics.json').exists())

    def test_broad_preview_and_explicit_limits_do_not_load_data(self):
        output = io.StringIO()
        with patch.object(example, 'load_local_data') as load, contextlib.redirect_stdout(output):
            example.main(['--broad-grid', '--max-expansions', '0', '--max-candidates', '405',
                          '--no-refine', '--data', 'missing.npz'])
        load.assert_not_called()
        self.assertIn('405 candidates', output.getvalue())
        self.assertIn('405 unique candidates total', output.getvalue())
        self.assertIn('refinement=False', output.getvalue())

    def test_invalid_adaptive_controls_fail_before_data_loading(self):
        for options in [['--max-expansions', '-1'], ['--max-candidates', '0'],
                        ['--refine', '--no-refine']]:
            with self.subTest(options=options), patch.object(example, 'load_local_data') as load:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    example.main(options)
                load.assert_not_called()

    def test_accepted_plateau_preserves_boundary_diagnostics(self):
        X = np.tile([[-1., 2.], [1., -2.]], (5, 1))
        y = np.tile([0, 1], 5)
        estimator = SimpleNamespace(n_iter_=[1], fit_status_=[0], converged_=True,
            support_counts_=[2], n_support_=[1, 1], gamma_=.01,
            subset_indices_=[np.array([0, 1])] * 5,
            selected_indices_=np.array([0, 1]), cache_hit_=False)
        result = {'model': SimpleNamespace(named_steps={'ensemble': estimator}),
                  'review': {'state': 'accepted', 'reasons': [],
                             'boundary_directions': [('ensemble__C', 'high')],
                             'stopping_reason': 'plateau'}}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            with (patch.object(example, 'load_local_data', return_value=(X, X[:4], y, y[:4])),
                  patch.object(adaptive_search, 'run_adaptive_search', return_value=result),
                  contextlib.redirect_stdout(io.StringIO())):
                example.main(['--data', 'unused.npz', '--run', '--output-dir', str(run_dir)])
            diagnostics = json.loads((run_dir / 'svm-diagnostics.json').read_text())
            self.assertEqual(diagnostics['boundary_parameters'], ['ensemble__C'])
            self.assertEqual(diagnostics['boundary_directions'], [['ensemble__C', 'high']])
            self.assertEqual(diagnostics['search_stopping_reason'], 'plateau')


if __name__ == "__main__":
    unittest.main()
