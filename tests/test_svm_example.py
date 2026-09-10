"""The SVM example previews its actual grid and uses C-aware boundaries."""
import importlib.util
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ensemble_dwd import mnist_svm_param_grid


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tune_mnist_svm.py"
SPEC = importlib.util.spec_from_file_location("svm_tuning_example", EXAMPLE)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class SVMExampleTests(unittest.TestCase):
    def test_preview_neither_loads_data_nor_starts_fits(self):
        result = subprocess.run([sys.executable, str(EXAMPLE), "--data", "missing.npz"],
                                text=True, capture_output=True, timeout=30, check=True)
        self.assertIn("810 candidates", result.stdout)
        self.assertIn("4050 ensemble CV fits", result.stdout)
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

    def test_cli_wires_training_review_and_accepts_held_result_without_model(self):
        X = np.tile([[-1., 2.], [1., -2.]], (5, 1))
        y = np.tile([0, 1], 5)
        calls = []

        def workflow(pipeline, grid, *args, **kwargs):
            parameters = {name: values[len(values) // 2] for name, values in grid.items()}
            parameters['ensemble__C'] = max(grid['ensemble__C'])
            reasons = kwargs['selection_review'](parameters)
            self.assertEqual(reasons, ['ensemble__C'])
            self.assertFalse(kwargs['refine'])
            self.assertEqual(kwargs['error_policy'], 'nan')
            calls.append(parameters)
            return {'model': None, 'completed': None, 'review': {'reasons': reasons}}

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            stdout = io.StringIO()
            argv = [str(EXAMPLE), '--data', 'unused.npz', '--run', '--output-dir', str(run_dir)]
            with (patch.object(sys, 'argv', argv), patch.object(example, 'load_local_data',
                    return_value=(X, X[:4], y, y[:4])), patch.object(
                    example, 'run_search_workflow', side_effect=workflow), patch.object(
                    example, 'write_json_atomic', side_effect=AssertionError('Held model has no diagnostics.')),
                    contextlib.redirect_stdout(stdout)):
                example.main()
            self.assertEqual(len(calls), 1)
            self.assertIn('training-only CV', stdout.getvalue())
            self.assertIn('test set was not scored', stdout.getvalue())
            self.assertFalse((run_dir / 'svm-diagnostics.json').exists())


if __name__ == "__main__":
    unittest.main()
