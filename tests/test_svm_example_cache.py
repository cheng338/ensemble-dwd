"""The SVM example exposes libsvm cache memory without changing its grid."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(EXAMPLES)) if str(EXAMPLES) not in sys.path else None
import tune_mnist_svm as example
import adaptive_search


class SVMCacheOptionTests(unittest.TestCase):
    def test_preview_discloses_cache_concurrency_and_other_allocations(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(example, 'load_local_data', side_effect=AssertionError('preview loaded data')):
            example.main(['--cache-size', '1024', '--jobs', '4'])
        self.assertIn('1024 MiB per active fit', output.getvalue())
        self.assertIn('4096 MiB across 4 CV workers', output.getvalue())
        self.assertIn('additional memory', output.getvalue())

    def test_invalid_cli_values_are_rejected(self):
        for value in ('0', '-1', 'nan', 'inf', 'bad'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                example.main(['--cache-size', value])

    def test_cache_reaches_estimator_and_frozen_settings_without_grid_changes(self):
        class BeforeFit(Exception):
            pass
        grids = []
        with tempfile.TemporaryDirectory() as directory:
            X = np.arange(40.).reshape(20, 2)
            y = np.tile([0, 1], 10)
            for position, args in enumerate(([], ['--cache-size', '1024'])):
                with patch.object(example, 'load_local_data', return_value=(X, X, y, y)), \
                     patch.object(adaptive_search, 'run_adaptive_search', side_effect=BeforeFit) as run, \
                     redirect_stdout(io.StringIO()), self.assertRaises(BeforeFit):
                    example.main(['--data', 'fixture.npz', '--run', '--no-cache',
                                  '--output-dir', str(Path(directory) / str(position))] + args)
                value = 200. if position == 0 else 1024.
                self.assertEqual(run.call_args.args[0].get_params()['ensemble__cache_size'], value)
                self.assertEqual(run.call_args.kwargs['settings']['svc_cache_size_mib'], value)
                grids.append(run.call_args.args[1])
        self.assertEqual(grids[0], grids[1])


if __name__ == '__main__':
    unittest.main()
