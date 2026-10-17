"""Explicit pixel scaling preserves input data and preprocessing identities."""
import argparse
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
from mnist_preprocessing import (UnitPixelScaler, add_preprocessing_arguments,
                                 make_preprocessor, preprocessing_metadata)


class UnitPixelScalerTests(unittest.TestCase):
    def test_raw_uint8_and_float_inputs_have_identical_output(self):
        pixels = np.array([[0, 1, 255], [128, 64, 0]], dtype=np.uint8)
        expected = pixels.astype(np.float64) / 255.0
        for dtype in (np.uint8, np.int16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                value = pixels.astype(dtype)
                assert_array_equal(UnitPixelScaler(input_range='0-255').fit_transform(value), expected)

    def test_unit_input_is_not_divided_twice(self):
        pixels = np.array([[0., 1., .5], [1., 0., .25]])
        scaler = UnitPixelScaler(input_range='0-1').fit(pixels)
        output = scaler.transform(pixels)
        assert_array_equal(output, pixels)
        self.assertFalse(np.shares_memory(output, pixels))

    def test_low_raw_values_do_not_trigger_inference(self):
        pixels = np.array([[0., 1.], [1., 0.]])
        raw = UnitPixelScaler(input_range='0-255').fit(pixels)
        unit = UnitPixelScaler(input_range='0-1').fit(pixels)
        assert_array_equal(raw.transform(pixels), pixels / 255.)
        assert_array_equal(unit.transform(pixels), pixels)
        self.assertNotEqual(raw.get_params(), unit.get_params())

    def test_rare_and_constant_training_pixels_stay_bounded_at_test(self):
        train = np.zeros((101, 3))
        train[:, 0] = np.arange(101) % 2 * 255
        train[-1, 1] = 1.
        test = np.full((1, 3), 255.)
        unit = UnitPixelScaler(input_range='0-255').fit(train)
        assert_array_equal(unit.transform(test), np.ones((1, 3)))
        standardized = StandardScaler().fit_transform(train)
        scaled_then_standardized = StandardScaler().fit_transform(train / 255.)
        assert_allclose(standardized, scaled_then_standardized, rtol=1e-13, atol=1e-13)
        self.assertGreater(StandardScaler().fit(train).transform(test)[0, 1], 2000.)
        self.assertEqual(StandardScaler().fit(train).transform(test)[0, 2], 255.)

    def test_training_fold_does_not_estimate_a_range(self):
        first = np.zeros((4, 3))
        second = np.full((7, 3), 255.)
        query = np.array([[0., 128., 255.]])
        prototype = UnitPixelScaler(input_range='0-255')
        left, right = clone(prototype).fit(first), clone(prototype).fit(second)
        assert_array_equal(left.transform(query), right.transform(query))
        self.assertEqual(vars(left), vars(right))
        self.assertEqual(set(vars(left)), {'input_range', 'input_range_', 'divisor_', 'n_features_in_'})

    def test_input_policy_is_required_and_invalid_policy_rejected(self):
        with self.assertRaises(TypeError):
            UnitPixelScaler()
        for policy in ('auto', None, 255, 'standard', 'uint8', np.str_('0-255')):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                UnitPixelScaler(input_range=policy).fit(np.zeros((2, 2)))

    def test_option_validation_is_side_effect_free(self):
        for policy in ('0-255', '0-1'):
            scaler = UnitPixelScaler(input_range=policy)
            before = vars(scaler).copy()
            self.assertIsNone(scaler._validate_options())
            self.assertEqual(vars(scaler), before)
        invalid = UnitPixelScaler(input_range='auto')
        with self.assertRaises(ValueError):
            invalid._validate_options()
        self.assertEqual(vars(invalid), {'input_range': 'auto'})

    def test_invalid_pixels_are_rejected_during_fit_and_transform(self):
        values = (
            np.array([[np.nan, 0.]]), np.array([[np.inf, 0.]]),
            np.array([[-1., 0.]]), np.array([[256., 0.]]),
            np.array([[1j, 0j]]), np.array([['0', '1']]),
            np.array([[True, False]]), np.array([0., 1.]),
            np.zeros((0, 2)), np.zeros((2, 0)), np.zeros((2, 1, 2)),
            sparse.csr_matrix(np.zeros((2, 2))))
        for value in values:
            with self.subTest(value=repr(value)[:80]):
                with self.assertRaises((ValueError, TypeError)):
                    UnitPixelScaler(input_range='0-255').fit(value)
                scaler = UnitPixelScaler(input_range='0-255').fit(np.zeros((2, 2)))
                with self.assertRaises((ValueError, TypeError)):
                    scaler.transform(value)
        with self.assertRaises(ValueError):
            UnitPixelScaler(input_range='0-1').fit([[0., 1.01]])

    def test_feature_count_and_fitted_state_are_validated(self):
        scaler = UnitPixelScaler(input_range='0-255')
        with self.assertRaises(NotFittedError):
            scaler.transform(np.zeros((2, 2)))
        scaler.fit(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            scaler.transform(np.zeros((2, 3)))

    def test_failed_refit_clears_previous_fitted_state(self):
        scaler = UnitPixelScaler(input_range='0-255').fit(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            scaler.fit(np.full((2, 2), 256.))
        with self.assertRaises(NotFittedError):
            scaler.transform(np.zeros((2, 2)))
        self.assertEqual(vars(scaler), {'input_range': '0-255'})

    def test_postfit_policy_change_requires_refit(self):
        scaler = UnitPixelScaler(input_range='0-255').fit(np.zeros((2, 2)))
        scaler.set_params(input_range='0-1')
        with self.assertRaises(ValueError):
            scaler.transform(np.ones((2, 2)))

    def test_fit_does_not_retain_training_values(self):
        pixels = np.arange(12.).reshape(4, 3)
        scaler = UnitPixelScaler(input_range='0-255').fit(pixels)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in vars(scaler).values()))
        pixels[:] = 255.
        assert_array_equal(scaler.transform([[0., 1., 255.]]), [[0., 1./255., 1.]])

    def test_readonly_and_strided_inputs_are_not_modified_or_aliased(self):
        pixels = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
        original = pixels.copy()
        pixels.setflags(write=False)
        output = UnitPixelScaler(input_range='0-255').fit_transform(pixels)
        assert_array_equal(pixels, original)
        self.assertFalse(np.shares_memory(pixels, output))
        self.assertEqual(output.dtype, np.float64)
        self.assertTrue(output.flags.c_contiguous)
        self.assertEqual(output.nbytes, pixels.size * 8)

    def test_clone_and_pickle_preserve_explicit_policy_and_outputs(self):
        pixels = np.array([[0., 255.], [128., 16.]])
        scaler = UnitPixelScaler(input_range='0-255').fit(pixels)
        fresh = clone(scaler)
        self.assertEqual(fresh.get_params(), scaler.get_params())
        with self.assertRaises(NotFittedError):
            fresh.transform(pixels)
        restored = pickle.loads(pickle.dumps(scaler))
        self.assertEqual(restored.get_params(), scaler.get_params())
        assert_array_equal(restored.transform(pixels), scaler.transform(pixels))

    def test_pipeline_and_feature_names_are_supported(self):
        pipe = Pipeline([('scale', UnitPixelScaler(input_range='0-255'))])
        result = pipe.fit_transform([[0., 255.], [128., 64.]])
        assert_array_equal(result, np.array([[0., 1.], [128./255., 64./255.]]))
        assert_array_equal(pipe.get_feature_names_out(), np.array(['x0', 'x1'], dtype=object))
        self.assertEqual(pipe.get_params()['scale__input_range'], '0-255')


class PreprocessingPolicyTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        add_preprocessing_arguments(parser)
        return parser

    def test_defaults_divide_raw_pixels_by_255_and_record_effective_range(self):
        args = self.parser().parse_args([])
        self.assertEqual(vars(args), {'preprocessing': 'unit', 'pixel_input_range': None})
        scaler = make_preprocessor()
        self.assertIs(type(scaler), UnitPixelScaler)
        self.assertEqual(scaler.get_params(), {'input_range': '0-255'})
        X = np.array([[0., 1., 0.], [255., 0., 0.], [32., 0., 0.]])
        assert_array_equal(scaler.fit_transform(X), X / 255.)
        self.assertEqual(preprocessing_metadata(args.preprocessing, args.pixel_input_range),
                         preprocessing_metadata('unit', '0-255'))
        explicit = make_preprocessor('unit', '0-255')
        self.assertEqual(scaler.get_params(), explicit.get_params())

    def test_explicit_standard_preserves_exact_standard_scaler(self):
        args = self.parser().parse_args(['--preprocessing', 'standard'])
        scaler = make_preprocessor(args.preprocessing, args.pixel_input_range)
        self.assertIs(type(scaler), StandardScaler)
        self.assertEqual(scaler.get_params(), StandardScaler().get_params())
        X = np.array([[0., 1., 0.], [255., 0., 0.], [32., 0., 0.]])
        assert_array_equal(scaler.fit_transform(X), StandardScaler().fit_transform(X))

    def test_unit_modes_preserve_explicit_and_visible_input_range(self):
        for declaration, divisor in (('0-255', 255.), ('0-1', 1.)):
            with self.subTest(declaration=declaration):
                args = self.parser().parse_args([
                    '--preprocessing', 'unit', '--pixel-input-range', declaration])
                scaler = make_preprocessor(args.preprocessing, args.pixel_input_range)
                self.assertIs(type(scaler), UnitPixelScaler)
                self.assertEqual(scaler.get_params(), {'input_range': declaration})
                assert_array_equal(scaler.fit_transform([[0., 1.]]), [[0., 1./divisor]])
                metadata = preprocessing_metadata(args.preprocessing, args.pixel_input_range)
                self.assertEqual(metadata['preprocessing'], 'unit')
                self.assertEqual(metadata['pixel_input_range'], declaration)
                self.assertIn('declared', metadata['learned_statistics'])
                self.assertEqual(json.loads(json.dumps(metadata)), metadata)

    def test_factory_and_metadata_reject_invalid_or_unused_declarations(self):
        policies = (
            ('standard', '0-255'), ('standard', '0-1'),
            ('unit', 'auto'), ('unit', 255), ('unknown', None), (None, None))
        for policy in policies:
            for function in (make_preprocessor, preprocessing_metadata):
                with self.subTest(policy=policy, function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(*policy)

    def test_parser_rejects_unknown_choices(self):
        for arguments in (['--preprocessing', 'auto'], ['--pixel-input-range', 'auto']):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parser().parse_args(arguments)

    def test_metadata_keeps_three_policies_distinct_without_inspecting_data(self):
        policies = [('standard', None), ('unit', '0-255'), ('unit', '0-1')]
        records = [preprocessing_metadata(*policy) for policy in policies]
        self.assertEqual(len({json.dumps(record, sort_keys=True) for record in records}), 3)
        self.assertIn('training mean', records[0]['transform'])
        self.assertEqual(records[1]['transform'], 'X / 255')
        self.assertEqual(records[2]['transform'], 'X (already in [0, 1])')


if __name__ == '__main__':
    unittest.main()
