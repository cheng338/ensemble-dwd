"""MNIST preprocessing options reach each CLI pipeline and saved settings."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.preprocessing import StandardScaler

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import tune_mnist
import tune_mnist_svm
import tune_mnist_multiclass
from mnist_preprocessing import UnitPixelScaler, preprocessing_metadata


MODULES = (tune_mnist, tune_mnist_svm, tune_mnist_multiclass)


class MNISTPreprocessingCLITests(unittest.TestCase):
    def loader_name(self, module):
        return 'load_local_multiclass_data' if module is tune_mnist_multiclass else 'load_local_data'

    def test_default_and_unit_previews_do_not_load_data_or_search(self):
        for module in MODULES:
            for options, expected in (([], 'unit [0, 1], declared input range 0-255'),
                                      (['--preprocessing', 'unit'], 'unit [0, 1], declared input range 0-255'),
                                      (['--pixel-input-range', '0-1'], 'unit [0, 1], declared input range 0-1'),
                                      (['--preprocessing', 'standard'], 'standard'),
                                      (['--preprocessing', 'unit', '--pixel-input-range', '0-255'], 'unit [0, 1]'),
                                      (['--preprocessing', 'unit', '--pixel-input-range', '0-1'], 'unit [0, 1]')):
                with self.subTest(module=module.__name__, options=options), ExitStack() as stack:
                    output = stack.enter_context(redirect_stdout(io.StringIO()))
                    loader = stack.enter_context(patch.object(module, self.loader_name(module)))
                    search = stack.enter_context(patch.object(adaptive_search, 'run_adaptive_search'))
                    module.main(options)
                    loader.assert_not_called()
                    search.assert_not_called()
                    self.assertIn('MNIST preprocessing: ' + expected, output.getvalue())

    def test_invalid_declarations_fail_before_loading_or_creating_a_run(self):
        invalid = (['--preprocessing', 'standard', '--pixel-input-range', '0-255'],
                   ['--preprocessing', 'standard', '--pixel-input-range', '0-1'])
        for module in MODULES:
            for options in invalid:
                with self.subTest(module=module.__name__, options=options), ExitStack() as stack:
                    stack.enter_context(redirect_stderr(io.StringIO()))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    loader = stack.enter_context(patch.object(module, self.loader_name(module)))
                    directory = stack.enter_context(patch.object(module, 'new_run_directory'))
                    search = stack.enter_context(patch.object(adaptive_search, 'run_adaptive_search'))
                    with self.assertRaises(SystemExit) as error:
                        module.main(['--data', 'fixture.npz', '--run'] + options)
                    self.assertEqual(error.exception.code, 2)
                    loader.assert_not_called()
                    directory.assert_not_called()
                    search.assert_not_called()

    def test_pipeline_and_metadata_preserve_declared_mode_and_grid(self):
        class BeforeSearch(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            for module in MODULES:
                grids, configs = [], []
                policies = (([], 'unit', '0-255'),
                            (['--preprocessing', 'unit'], 'unit', '0-255'),
                            (['--preprocessing', 'unit', '--pixel-input-range', '0-255'], 'unit', '0-255'),
                            (['--pixel-input-range', '0-1'], 'unit', '0-1'),
                            (['--preprocessing', 'standard'], 'standard', None))
                for index, (options, mode, pixel_range) in enumerate(policies):
                    with self.subTest(module=module.__name__, mode=mode, pixel_range=pixel_range), ExitStack() as stack:
                        X = np.tile([[0., 255.], [255., 0.]], (50, 1))
                        if pixel_range == '0-1':
                            X /= 255.
                        y = np.tile(np.arange(10), 10) if module is tune_mnist_multiclass else np.tile([7, 9], 50)
                        loaded = (X, X.copy(), y, y.copy())
                        if module is tune_mnist_multiclass:
                            loaded += ({'data_path': 'fixture.npz'},)
                        run_dir = Path(directory) / module.__name__ / str(index)
                        stack.enter_context(patch.object(module, self.loader_name(module), return_value=loaded))
                        search = stack.enter_context(patch.object(adaptive_search, 'run_adaptive_search', side_effect=BeforeSearch))
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        args = ['--data', 'fixture.npz', '--run', '--no-cache', '--output-dir', str(run_dir)]
                        args += options
                        with self.assertRaises(BeforeSearch):
                            module.main(args)
                        pipeline = search.call_args.args[0]
                        self.assertEqual(pipeline.steps[0][0], 'scale')
                        scaler = pipeline.named_steps['scale']
                        if mode == 'standard':
                            self.assertIs(type(scaler), StandardScaler)
                            self.assertEqual(scaler.get_params(), StandardScaler().get_params())
                        else:
                            self.assertIs(type(scaler), UnitPixelScaler)
                            self.assertEqual(scaler.input_range, pixel_range)
                            expected = X / 255. if pixel_range == '0-255' else X
                            assert_array_equal(scaler.fit_transform(X), expected)
                        settings = search.call_args.kwargs['settings']
                        self.assertEqual(settings['preprocessing'], preprocessing_metadata(mode, pixel_range))
                        grids.append(search.call_args.args[1])
                        configs.append(search.call_args.kwargs['config'])
                        if module is tune_mnist_multiclass:
                            saved = json.loads((run_dir / 'data-selection.json').read_text(encoding='utf-8'))
                            self.assertEqual(saved['preprocessing'], settings['preprocessing'])
                self.assertTrue(all(grid == grids[0] for grid in grids))
                self.assertTrue(all(config == configs[0] for config in configs))


if __name__ == '__main__':
    unittest.main()
