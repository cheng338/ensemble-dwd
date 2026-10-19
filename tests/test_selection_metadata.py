"""Saved composition describes the existing selection without changing it."""

import json
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError

from ensemble_dwd import EnsembleKernelDWD, IneligibleSelectionError, OvOClassifier


def fixture():
    X = np.array([[-2.], [-1.], [-1.], [-.3], [.3], [1.], [1.], [2.], [2.]])
    y = np.array([-1, -1, 1, -1, -1, 1, 1, -1, -1])
    return X, y


def estimator(**options):
    params = dict(M=1, theta=1., sv_selection_ratio=.67, lambd=.04, gamma=.7,
                  stopping="fixed", max_iter=10, random_state=7)
    params.update(options)
    return EnsembleKernelDWD(**params)


class SelectionMetadataTests(unittest.TestCase):
    def test_counts_labels_and_pickle_preserve_selection_in_both_storage_modes(self):
        X, y = fixture()
        for labels in (np.where(y < 0, -7, 11), np.where(y < 0, "negative", "positive")):
            previous_scores = None
            for store in (False, True):
                with self.subTest(labels=labels.dtype, store=store):
                    model = estimator(store_base_estimators=store).fit(X, labels)
                    # Duplicate rows can acquire slightly different floating-point
                    # scores across numerical libraries. Selection uses the actual
                    # unrounded scores, with input order breaking exact ties.
                    expected = sorted(range(len(X)), key=lambda index:
                                      (abs(model.consensus_scores_[index]), index))[:6]
                    assert_array_equal(model.selected_indices_, expected)
                    assert_array_equal(model.final_estimator_._Xfit, X[expected])
                    assert_array_equal(model.training_class_counts_, [6, 3])
                    assert_array_equal(model.selected_class_counts_, [3, 3])
                    report = model.selection_composition_
                    self.assertEqual(report["training_size"], 9)
                    self.assertEqual(report["selected_size"], 6)
                    self.assertEqual(report["selection_ratio_realized"], 6 / 9)
                    self.assertEqual([row["label"] for row in report["classes"]], list(model.classes_))
                    self.assertEqual([row["training_fraction"] for row in report["classes"]], [2 / 3, 1 / 3])
                    self.assertEqual([row["selected_fraction"] for row in report["classes"]], [.5, .5])
                    self.assertEqual([row["class_retention_fraction"] for row in report["classes"]], [.5, 1.])
                    json.dumps(report, allow_nan=False)
                    scores = model.decision_function(X)
                    if previous_scores is not None:
                        assert_array_equal(scores, previous_scores)
                    previous_scores = scores
                    restored = pickle.loads(pickle.dumps(model))
                    self.assertEqual(restored.selection_composition_, report)
                    assert_array_equal(restored.selected_indices_, model.selected_indices_)
                    assert_array_equal(restored.consensus_scores_, model.consensus_scores_)
                    assert_array_equal(restored.training_class_counts_, model.training_class_counts_)
                    assert_array_equal(restored.selected_class_counts_, model.selected_class_counts_)
                    assert_array_equal(restored.predict(X), model.predict(X))

    def test_exact_consensus_ties_keep_input_order_at_selection_boundary(self):
        X, y = fixture()
        # Exact opposite-sign magnitudes exercise absolute-score ordering and
        # a tie crossing the sixth-row cutoff, independent of BLAS reductions.
        consensus = np.array([-4., 1., -1., 2., 0., -2., 3., -3., 4.])
        expected = [4, 1, 2, 3, 5, 6]
        for store in (False, True):
            with self.subTest(store=store):
                with patch("ensemble_dwd.estimator._binary_scores",
                           return_value=consensus.copy()) as score:
                    model = estimator(store_base_estimators=store).fit(X, y)
                self.assertEqual(score.call_count, 1)
                assert_array_equal(model.consensus_scores_, consensus)
                assert_array_equal(model.selected_indices_, expected)
                assert_array_equal(model.final_estimator_._Xfit, X[expected])
                assert_array_equal(model.training_class_counts_, [6, 3])
                assert_array_equal(model.selected_class_counts_, [3, 3])
                restored = pickle.loads(pickle.dumps(model))
                assert_array_equal(restored.selected_indices_, expected)
                assert_array_equal(restored.consensus_scores_, consensus)
                self.assertEqual(restored.selection_composition_, model.selection_composition_)
                assert_array_equal(restored.predict(X), model.predict(X))

    def test_failed_refit_clears_metadata_but_invalid_prediction_preserves_it(self):
        X, y = fixture()
        model = estimator().fit(X, y)
        snapshot = pickle.dumps(model)
        with self.assertRaises(ValueError):
            model.predict(np.full_like(X, np.nan))
        self.assertEqual(pickle.dumps(model), snapshot)
        for failure in ("ineligible", "learner"):
            with self.subTest(failure=failure):
                model = estimator().fit(X, y)
                if failure == "ineligible":
                    model.sv_selection_ratio = .01
                    with self.assertRaises(IneligibleSelectionError):
                        model.fit(X, y)
                else:
                    with patch.object(model, "_make_estimator", side_effect=FloatingPointError("failed learner")):
                        with self.assertRaisesRegex(FloatingPointError, "failed learner"):
                            model.fit(X, y)
                for name in ("training_class_counts_", "selected_class_counts_",
                             "selection_composition_", "final_estimator_"):
                    self.assertFalse(hasattr(model, name), name)
                with self.assertRaises(NotFittedError):
                    model.predict(X)

    def test_ovo_pair_local_counts_have_explicit_original_label_mapping(self):
        X = np.arange(18, dtype=float).reshape(-1, 1) / 5
        y = np.array(["A"] * 7 + ["B"] * 6 + ["C"] * 5)
        model = OvOClassifier(estimator(sv_selection_ratio=1.), n_jobs=1).fit(X, y)
        for pair, rows, child in zip(model.pairs_, model.pair_indices_, model.estimators_):
            original_labels = {1: pair[0], -1: pair[1]}
            self.assertEqual(child.selection_composition_["training_size"], len(rows))
            for item in child.selection_composition_["classes"]:
                expected = int(np.count_nonzero(y[rows] == original_labels[item["label"]]))
                self.assertEqual(item["training_count"], expected)
                self.assertEqual(item["selected_count"], expected)
        restored = pickle.loads(pickle.dumps(model))
        assert_array_equal(restored.predict(X), model.predict(X))


if __name__ == "__main__":
    unittest.main()
