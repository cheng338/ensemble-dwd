"""Pure helper and archive-schema tests for the explicit tuning example."""

import argparse
import contextlib
import io
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "tune_mnist.py"
SPEC = importlib.util.spec_from_file_location("ensemble_dwd_tuning_example", EXAMPLE_PATH)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def candidate(lambd, gamma, theta=0.1, ratio=0.2, prefix="ensemble__"):
    return {prefix + "lambd": lambd, prefix + "gamma": gamma,
            prefix + "theta": theta, prefix + "sv_selection_ratio": ratio}


def results(params, means):
    means = np.asarray(means, dtype=float)
    return {"params": params, "mean_test_score": means,
            "split0_test_score": means.copy(), "split1_test_score": means.copy()}


class CandidateSelectionTests(unittest.TestCase):
    def test_any_failed_fold_excludes_candidate_and_ties_remain_stable(self):
        cv = results([candidate(1, 1)] * 4, [0.85, 0.99, 0.86, 0.85])
        cv["split1_test_score"][1] = np.nan
        assert_array_equal(example.fully_valid_candidate_mask(cv), [True, False, True, True])
        assert_array_equal(example.ranked_valid_indices(cv), [2, 0, 3])
        self.assertEqual(example.best_valid_candidate(cv), 2)

    def test_mean_and_each_split_must_be_finite(self):
        cv = results([candidate(1, 1)] * 4, [0.7, 0.8, 0.9, 1.0])
        cv["mean_test_score"][0] = np.nan
        cv["split0_test_score"][1] = np.inf
        cv["split1_test_score"][2] = -np.inf
        assert_array_equal(example.fully_valid_candidate_mask(cv), [False, False, False, True])
        self.assertEqual(example.best_valid_candidate(cv), 3)
        cv["split1_test_score"][3] = np.nan
        with self.assertRaises(ValueError):
            example.best_valid_candidate(cv)

    def test_inconsistent_cv_score_shapes_fail(self):
        cv = results([candidate(1, 1)] * 2, [0.7, 0.8])
        cv["split0_test_score"] = np.array([0.7])
        with self.assertRaises(ValueError):
            example.fully_valid_candidate_mask(cv)

    def test_refinement_uses_valid_centers_and_preserves_geometry(self):
        params = [candidate(1.0, 1.0, 0.1, 0.2),
                  candidate(100.0, 100.0, 0.2, 0.3)]
        cv = results(params, [0.8, 0.99])
        cv["split0_test_score"][1] = np.nan
        grid = example.local_refinement_grid(cv, top_k=3)
        self.assertEqual(len(grid), 9)
        for row in grid:
            self.assertEqual(row["ensemble__theta"], [0.1])
            self.assertEqual(row["ensemble__sv_selection_ratio"], [0.2])
            self.assertTrue(all(isinstance(values, list) and len(values) == 1
                                for values in row.values()))
        self.assertEqual({row["ensemble__lambd"][0] for row in grid}, {0.25, 1.0, 4.0})
        assert_allclose(sorted({row["ensemble__gamma"][0] for row in grid}),
                        [10 ** -0.5, 1.0, 10 ** 0.5], rtol=1e-14, atol=0)

    def test_refinement_can_cross_both_original_grid_endpoints(self):
        for lambd, gamma in [(2.0 ** -30, 1e-8), (1.0, 1.0)]:
            cv = results([candidate(lambd, gamma)], [0.8])
            grid = example.local_refinement_grid(cv, top_k=1)
            self.assertLess(min(row["ensemble__lambd"][0] for row in grid), lambd)
            self.assertGreater(max(row["ensemble__lambd"][0] for row in grid), lambd)
            self.assertLess(min(row["ensemble__gamma"][0] for row in grid), gamma)
            self.assertGreater(max(row["ensemble__gamma"][0] for row in grid), gamma)

    def test_overlapping_neighborhoods_deduplicate_only_same_geometry(self):
        params = [candidate(1.0, 1.0), candidate(4.0, 10 ** 0.5),
                  candidate(1.0, 1.0, theta=0.2)]
        grid = example.local_refinement_grid(results(params, [0.9, 0.8, 0.7]))
        # Two neighboring 3-by-3 grids share four candidates. Changing theta
        # creates a separate nine-candidate neighborhood.
        self.assertEqual(len(grid), 23)
        self.assertEqual(sum(row["ensemble__theta"] == [0.2] for row in grid), 9)

    def test_refinement_supports_unprefixed_estimator_parameters(self):
        cv = results([candidate(0.1, 0.2, prefix="")], [0.8])
        grid = example.local_refinement_grid(cv, prefix="", top_k=1)
        self.assertEqual(set(grid[0]), {"lambd", "gamma", "theta", "sv_selection_ratio"})

    def test_refinement_retains_distinct_adjacent_lambda_parents(self):
        nearby = float(np.nextafter(1.0, 2.0))
        parents = [candidate(1.0, 1.0), candidate(nearby, 1.0)]
        grid = example.local_refinement_grid(results(parents, [0.9, 0.8]), top_k=2)
        coordinates = {(row['ensemble__lambd'][0], row['ensemble__gamma'][0])
                       for row in grid}
        self.assertEqual(len(grid), 18)
        self.assertIn((1.0, 1.0), coordinates)
        self.assertIn((nearby, 1.0), coordinates)
        self.assertEqual(sum(row['ensemble__lambd'][0] == nearby for row in grid), 3)

    def test_refinement_retains_adjacent_gamma_and_extreme_scale_parents(self):
        for scale in (1e-200, 1.0, 1e200):
            nearby = float(np.nextafter(scale, np.inf))
            parents = [candidate(scale, scale), candidate(scale, nearby)]
            with self.subTest(scale=scale):
                grid = example.local_refinement_grid(results(parents, [0.9, 0.8]), top_k=2)
                coordinates = {(row['ensemble__lambd'][0], row['ensemble__gamma'][0])
                               for row in grid}
                self.assertIn((scale, scale), coordinates)
                self.assertIn((scale, nearby), coordinates)
                self.assertEqual(len(coordinates), len(grid))

    def test_refinement_still_collapses_exact_duplicate_parents(self):
        parents = [candidate(1.0, 1.0), candidate(1.0, 1.0)]
        first = example.local_refinement_grid(results(parents[:1], [0.9]), top_k=1)
        both = example.local_refinement_grid(results(parents, [0.9, 0.8]), top_k=2)
        self.assertEqual(both, first)


class LocalArchiveTests(unittest.TestCase):
    def test_uppercase_and_keras_archive_splits_remain_separate(self):
        X_train = np.arange(48, dtype=np.uint8).reshape(12, 2, 2)
        X_test = np.arange(100, 148, dtype=np.uint8).reshape(12, 2, 2)
        y_train = np.tile([2, 3, 9], 4)
        y_test = np.tile([9, 3, 2], 4)
        with tempfile.TemporaryDirectory() as folder:
            for train_name, test_name in [("X_train", "X_test"), ("x_train", "x_test")]:
                path = Path(folder) / (train_name + ".npz")
                np.savez(path, **{train_name: X_train, "y_train": y_train,
                                  test_name: X_test, "y_test": y_test})
                train, test, train_y, test_y = example.load_local_data(path, 2, 3)
                train_mask = np.isin(y_train, [2, 3])
                test_mask = np.isin(y_test, [2, 3])
                assert_array_equal(train, X_train[train_mask].reshape(8, 4))
                assert_array_equal(test, X_test[test_mask].reshape(8, 4))
                assert_array_equal(train_y, y_train[train_mask])
                assert_array_equal(test_y, y_test[test_mask])
                self.assertEqual(train.dtype, np.float64)
                self.assertEqual(test.dtype, np.float64)

    def test_unsplit_archive_gets_reproducible_disjoint_holdout(self):
        X = np.arange(96, dtype=float).reshape(24, 2, 2)
        y = np.tile([2, 3], 12)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unsplit.npz"
            np.savez(path, X=X, y=y)
            first = example.load_local_data(path, 2, 3)
            second = example.load_local_data(path, 2, 3)
        for a, b in zip(first, second):
            assert_array_equal(a, b)
        train, test, train_y, test_y = first
        self.assertEqual((len(train), len(test)), (18, 6))
        self.assertFalse(set(train[:, 0]) & set(test[:, 0]))
        self.assertEqual(set(train[:, 0]) | set(test[:, 0]), set(X[:, 0, 0]))
        self.assertEqual(set(train_y), {2, 3})
        self.assertEqual(set(test_y), {2, 3})

    def test_missing_schema_and_split_feature_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.npz"
            np.savez(path, unrelated=np.arange(3))
            with self.assertRaises(ValueError):
                example.load_local_data(path, 2, 3)
            np.savez(path, X_train=np.ones((4, 2)), y_train=[2, 3, 2, 3],
                     X_test=np.ones((4, 3)), y_test=[2, 3, 2, 3])
            with self.assertRaises(ValueError):
                example.load_local_data(path, 2, 3)

    def test_invalid_or_single_class_digit_labels_fail(self):
        for y in [[2, 2, 2, 2], [2, 3, np.nan, 3], [2, 3, 2.5, 3]]:
            with self.subTest(labels=y), self.assertRaises(ValueError):
                example.select_pair(np.ones((4, 2)), y, 2, 3)

    def test_worker_count_accepts_any_positive_integer(self):
        for value in ["1", "3", "8", "16"]:
            self.assertEqual(example.positive_int(value), int(value))
        for value in ["0", "-1", "1.5", "many"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                example.positive_int(value)


class AdaptiveDWDPreviewTests(unittest.TestCase):
    def test_coarse_and_broad_previews_load_no_data(self):
        for options, count in (([], 125), (['--broad-grid'], 405)):
            with self.subTest(options=options), patch.object(example, 'load_local_data') as load:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    example.main(options + ['--data', 'missing.npz'])
                load.assert_not_called()
                self.assertIn(f'{count} candidates', output.getvalue())
                self.assertIn('500 unique candidates total', output.getvalue())
                self.assertIn('refinement=True', output.getvalue())

    def test_explicit_budget_and_no_refine_preview(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            example.main(['--max-expansions', '0', '--max-candidates', '125', '--no-refine'])
        self.assertIn('125 unique candidates total', output.getvalue())
        self.assertIn('up to 0 complete expansion rounds', output.getvalue())
        self.assertIn('refinement=False', output.getvalue())


if __name__ == "__main__":
    unittest.main()
