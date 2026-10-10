"""Fit-free counts and best-effort proposal reporting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import warnings

from joblib import hash as parameter_hash
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier
EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import stage_reporting as reporting
from stage_reporting import (COUNT_FIELDS, candidate_budget, encode, explicit_proposal,
                             grid_proposal, publish_report, safe_report, workload_plan)


class CustomEstimator(BaseEstimator):
    def __sklearn_clone__(self):
        raise AssertionError("custom cloning must not run")


class ChildDWD(KernGDWD):
    def __sklearn_clone__(self):
        raise AssertionError("subclass cloning must not run")


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        # Three classes in fold 0, two in fold 1: never assume global K.
        self.folds = [(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
                      (np.array([0, 1, 3, 4]), np.array([2, 5, 6, 7]))]

    def plan(self, model, candidates=None, **kwargs):
        return workload_plan(model, [{}] if candidates is None else candidates,
                             self.y, self.folds, **kwargs)

    def test_ovo_uses_fold_classes_effective_M_and_seed_count(self):
        pipeline = Pipeline([("scale", StandardScaler()),
                             ("ovo", OvOClassifier(EnsembleKernelDWD(M=5)))])
        candidates = [{"ovo__estimator__M": 2}, {"ovo__estimator__M": np.int64(4)}]
        result = self.plan(pipeline, candidates, seeds=[7, 13])
        self.assertEqual(result["requested_pipeline_evaluations"], 8)
        self.assertEqual(result["totals"], {
            "binary_models": 16, "base_learner_fits": 48,
            "final_learner_fits": 16, "full_learner_fits": 0, "total_learner_fits": 64})
        self.assertEqual([row["counts"]["binary_models"] for row in result["candidates"][0]["folds"]], [6, 2])
        self.assertEqual(pipeline.named_steps["ovo"].estimator.M, 5)
        json.dumps(result, allow_nan=False)

    def test_full_dwd_and_svm_and_ensemble_svm_are_separate_counts(self):
        for model in (KernGDWD(), SVC()):
            result = self.plan(OvOClassifier(model))
            self.assertEqual(result["totals"]["full_learner_fits"], 4)
            self.assertEqual(result["totals"]["base_learner_fits"], 0)
            self.assertEqual(result["totals"]["final_learner_fits"], 0)
        result = self.plan(OvOClassifier(EnsembleKernelSVM(M=3)))
        self.assertEqual(result["totals"]["base_learner_fits"], 12)
        self.assertEqual(result["totals"]["final_learner_fits"], 4)
        self.assertEqual(result["totals"]["total_learner_fits"], 16)

    def test_no_candidates_has_zero_counts(self):
        result = self.plan(OvOClassifier(EnsembleKernelDWD()), [])
        self.assertEqual(result["totals"], dict.fromkeys(COUNT_FIELDS, 0))
        self.assertEqual(result["requested_pipeline_evaluations"], 0)

    def test_reuse_never_subtracts_speculative_work(self):
        model = OvOClassifier(EnsembleKernelDWD(M=5, cache_dir="unused"))
        ordinary = self.plan(model)
        for flags in ({"equivalent_folds": True}, {"pair_recovery": True},
                      {"base_cache": True, "checkpoint_replay": True}):
            result = self.plan(model, reuse=flags)
            self.assertEqual(result["totals"], ordinary["totals"])
            self.assertFalse(result["reuse_deductions_applied"])

    def test_unknown_and_subclass_do_not_clone_or_guess(self):
        for model in (CustomEstimator(), ChildDWD(), OvOClassifier(CustomEstimator()),
                      Pipeline([("scale", MinMaxScaler()), ("dwd", KernGDWD())])):
            result = self.plan(model)
            self.assertEqual(result["unknown_candidate_count"], 1)
            self.assertTrue(all(value is None for value in result["totals"].values()))

    def test_candidate_replacement_and_invalid_M_remain_visible(self):
        model = OvOClassifier(EnsembleKernelDWD())
        candidates = [{"estimator": CustomEstimator()}, {"estimator__M": 0},
                      {"estimator__M": True}, {"estimator__M": 1.5}, {"estimator__M": 2}]
        result = self.plan(model, candidates)
        self.assertEqual(result["unknown_candidate_count"], 4)
        self.assertEqual(result["known_candidate_count"], 1)
        self.assertIsNone(result["totals"]["total_learner_fits"])
        self.assertEqual(result["known_candidate_subtotal"]["total_learner_fits"], 12)
        self.assertEqual(len(result["candidates"]), 5)
        self.assertEqual(result["candidates"][0]["parameters"]["estimator"]["python_type"],
                         __name__ + ".CustomEstimator")
        json.dumps(result, allow_nan=False)

    def test_supported_replacement_uses_effective_learner(self):
        result = self.plan(OvOClassifier(EnsembleKernelDWD()), [{"estimator": KernGDWD()}])
        self.assertEqual(result["totals"]["full_learner_fits"], 4)
        self.assertEqual(result["totals"]["base_learner_fits"], 0)
        result = self.plan(OvOClassifier(EnsembleKernelDWD(M=0)), [{"estimator__M": 2}])
        self.assertEqual(result["totals"]["base_learner_fits"], 8)

    def test_one_class_and_direct_multiclass_are_unknown(self):
        result = self.plan(KernGDWD())
        self.assertEqual(result["unknown_candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["folds"][1]["counts"]["full_learner_fits"], 1)
        labels = np.zeros(len(self.y), dtype=int)
        result = workload_plan(OvOClassifier(KernGDWD()), [{}], labels, self.folds)
        self.assertEqual(result["unknown_candidate_count"], 1)

    def test_indices_never_truncate_wrap_or_overlap(self):
        invalid = ([0., 1.], [True, False], [-1, 0], [0, 8], [0, 0], [], [[0, 1]])
        for indices in invalid:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                workload_plan(KernGDWD(), [{}], self.y, [(indices, [4, 5])])
        with self.assertRaises(ValueError):
            workload_plan(KernGDWD(), [{}], self.y, [([0, 1], [1, 2])])

    def test_invalid_seeds_never_change_count_silently(self):
        for seeds in ([], [1, 1], [True], [1.5], [-1], [2**32], {1, 2}, np.array([1.])):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                self.plan(OvOClassifier(KernGDWD()), seeds=seeds)
        self.assertEqual(self.plan(OvOClassifier(KernGDWD()), seeds=np.array([0, 1]))["seed_count"], 2)

    def test_continuous_or_complex_targets_are_not_class_counts(self):
        for labels in (np.linspace(.1, .8, len(self.y)), self.y.astype(complex),
                       np.array([0, 1, 2, 0, 1, 2, 0, np.nan])):
            with self.subTest(dtype=labels.dtype), self.assertRaises(ValueError):
                workload_plan(OvOClassifier(KernGDWD()), [{}], labels, self.folds)

    def test_no_fit_transform_predict_or_input_mutation(self):
        model = Pipeline([("scale", StandardScaler()), ("ovo", OvOClassifier(EnsembleKernelDWD()))])
        before = self.y.copy()
        with patch.object(StandardScaler, "fit", side_effect=AssertionError("fit")), \
                patch.object(StandardScaler, "transform", side_effect=AssertionError("transform")), \
                patch.object(EnsembleKernelDWD, "fit", side_effect=AssertionError("fit")), \
                patch.object(OvOClassifier, "fit", side_effect=AssertionError("fit")), \
                patch.object(OvOClassifier, "predict", side_effect=AssertionError("predict")):
            result = self.plan(model)
        np.testing.assert_array_equal(self.y, before)
        self.assertEqual(result["totals"]["total_learner_fits"], 24)
        self.assertFalse(hasattr(model.named_steps["scale"], "mean_"))


class ProposalTests(unittest.TestCase):
    def test_complete_refused_proposal_is_exact(self):
        result = grid_proposal({"x": [1., 2., 3.]}, set(), 2, None)
        self.assertEqual(result["new_candidate_count"], 3)
        self.assertTrue(result["candidate_cap_refused"])
        self.assertEqual(result["count_status"], "exact")

    def test_huge_integer_product_and_raw_duplicate_scans_are_bounded(self):
        huge = {f"p{i}": [0, 1] for i in range(80)}
        result = grid_proposal(huge, set(), 3, None, scan_limit=4)
        self.assertEqual(result["raw_cartesian_entries"], 2 ** 80)
        self.assertEqual(result["scanned_entries"], 4)
        self.assertEqual(result["count_status"], "lower_bound")
        self.assertIsNone(result["new_candidate_count"])
        self.assertFalse(result["candidate_list_complete"])
        result = grid_proposal({"x": [1.] * 5000, "y": [2.] * 5000}, set(),
                               2, [{"x": 1., "y": 2.}], scan_limit=12)
        self.assertEqual(result["scanned_entries"], 12)
        self.assertEqual(result["raw_cartesian_entries"], 25_000_000)
        self.assertEqual(result["new_candidate_count"], 1)
        self.assertTrue(result["candidate_list_complete"])

    def test_policy_evidence_can_strengthen_an_incomplete_scan(self):
        result = grid_proposal({"x": list(range(20))}, set(), 10, None, scan_limit=2)
        self.assertEqual(result["new_candidate_count_lower_bound"], 11)
        self.assertEqual(len(result["candidate_parameters"]), 2)
        self.assertIsNone(result["new_candidate_count"])
        admitted = [{"x": item} for item in range(20)]
        result = grid_proposal({"x": list(range(20))}, set(), 20, admitted, scan_limit=2)
        self.assertEqual(result["new_candidate_count"], 20)
        self.assertEqual(result["candidate_parameters"], encode(admitted))
        self.assertFalse(result["candidate_cap_refused"])

    def test_seen_duplicate_and_nearby_float_identities(self):
        a, b = 4.033432887412458e-7, 4.0334328874124586e-7
        seen = {parameter_hash({"x": 0.})}
        grids = [{"x": [0., a, a]}, {"x": [b, a]}]
        before = parameter_hash((grids, seen))
        result = grid_proposal(grids, seen, 1, None)
        self.assertEqual(result["new_candidate_count"], 2)
        self.assertNotEqual(*result["candidate_identities"])
        self.assertEqual([float.fromhex(item["x"]["float_hex"])
                          for item in result["candidate_parameters"]], [a, b])
        self.assertEqual(result["generating_grids"], encode(grids))
        self.assertEqual(before, parameter_hash((grids, seen)))
        json.dumps(result, allow_nan=False)

    def test_explicit_ratio_or_expansion_proposal_is_exact(self):
        a, b = .3, np.nextafter(.3, 1.)
        candidates = [{"fraction": .1}, {"fraction": a}, {"fraction": a}, {"fraction": b}]
        seen = {parameter_hash({"fraction": .1})}
        before = deepcopy(candidates)
        for kind in ("ratio_neighborhood", "edge_expansion"):
            result = explicit_proposal(candidates, seen, 1, kind)
            self.assertEqual(result["new_candidate_count"], 2)
            self.assertEqual(result["proposal_kind"], kind)
            self.assertEqual(result["count_status"], "exact")
            self.assertTrue(result["candidate_cap_refused"])
        self.assertEqual(candidates, before)

    def test_numpy_literals_keep_dtype_and_bytes(self):
        literal = np.float32(.3)
        self.assertEqual(encode(literal), {"numpy_dtype": "float32",
                                         "bytes_hex": literal.tobytes().hex()})

    def test_invalid_scan_limit_is_not_silently_replaced(self):
        for limit in (0, -1, True, 1.5, np.int64(2)):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                grid_proposal({"x": [1]}, set(), 1, [{"x": 1}], scan_limit=limit)

    def test_budget_uses_completed_unique_count_and_charges_only_admission(self):
        refused = candidate_budget(50, 2, 49, False)
        self.assertEqual(refused["remaining_before"], 48)
        self.assertEqual(refused["remaining_after_admission"], 48)
        self.assertFalse(refused["optimality_claim"])
        admitted = candidate_budget(50, 2, 48, True)
        self.assertEqual(admitted["remaining_after_admission"], 0)
        self.assertEqual(candidate_budget(2, 3, 0, False)["remaining_before"], -1)
        unknown = candidate_budget(50, 2, None, False)
        self.assertIsNone(unknown["proposed_count"])
        self.assertEqual(unknown["remaining_after_admission"], 48)

    def test_invalid_count_is_not_silently_truncated(self):
        for bad in (True, np.bool_(False), -1, 2.5, None):
            for position in range(3):
                values = [50, 2, 3, False]
                values[position] = bad
                if bad is None and position == 2:
                    values[3] = True
                with self.subTest(bad=bad, position=position), self.assertRaises(ValueError):
                    candidate_budget(*values)
        with self.assertRaises(ValueError):
            candidate_budget(50, 2, 3, 1)

    def test_numpy_integral_counts_match_valid_policy_configuration(self):
        report = candidate_budget(np.int64(50), np.int32(2), np.uint64(3), True)
        self.assertEqual(report, candidate_budget(50, 2, 3, True))
        for name in ("maximum_unique_candidates", "completed_unique_candidates", "proposed_count",
                     "remaining_before", "remaining_after_admission"):
            self.assertIs(type(report[name]), int)
        json.dumps(report, allow_nan=False)


class BestEffortReportTests(unittest.TestCase):
    @staticmethod
    def fail():
        raise OSError("report destination is unavailable")

    def test_safe_factory_failure_leaves_authoritative_input_unchanged(self):
        answer = {"candidates": [{"gamma": .3}], "state": {"pending": True}}
        before = deepcopy(answer)
        with patch.object(reporting, "parameter_hash", side_effect=RuntimeError("hash failure")):
            report = safe_report(lambda: grid_proposal({"gamma": [.3]}, set(), 1,
                                                       answer["candidates"]))
        self.assertEqual(report["reporting_status"], "unavailable")
        self.assertEqual(answer, before)

    def test_exception_with_broken_string_can_be_reported(self):
        class BrokenError(Exception):
            def __str__(self):
                raise ValueError("unprintable")
        def fail():
            raise BrokenError()
        self.assertEqual(safe_report(fail)["error_type"], "BrokenError")

    def test_warning_as_error_and_failed_warning_sink_do_not_escape(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reporting.warn_reporting_failure(OSError("fail"))
        with patch.object(reporting.warnings, "warn", side_effect=OSError("warn fail")):
            reporting.warn_reporting_failure(OSError("fail"))

    def test_publish_build_and_write_failures_are_non_authoritative(self):
        with patch.object(reporting, "write_json_atomic") as writer, \
                warnings.catch_warnings():
            warnings.simplefilter("error")
            result = publish_report("unused.json", self.fail)
        self.assertEqual(result["reporting_status"], "unavailable")
        self.assertTrue(writer.call_args.kwargs["replace"])
        with patch.object(reporting, "write_json_atomic", side_effect=OSError("disk fail")), \
                warnings.catch_warnings():
            warnings.simplefilter("error")
            result = publish_report("unused.json", lambda: {"candidates": 3})
        self.assertEqual(result["reporting_status"], "unavailable")
        self.assertEqual(result["message"], "disk fail")

    def test_failed_announcement_keeps_written_report(self):
        expected = {"budget": candidate_budget(50, 2, 3, True)}
        with patch.object(reporting, "write_json_atomic") as writer, \
                patch("builtins.print", side_effect=BrokenPipeError("console closed")), \
                warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = publish_report("unused.json", lambda: expected, announce=True)
        self.assertEqual(actual, expected)
        writer.assert_called_once_with("unused.json", expected, replace=True)

    def test_announcement_labels_nominal_breakdown_and_remaining_allowance(self):
        expected = {
            "workload_plan": {"candidate_count": 2, "fold_count": 3, "seed_count": 1,
                              "totals": {"total_learner_fits": 72, "base_learner_fits": 60,
                                         "final_learner_fits": 12, "full_learner_fits": 0}},
            "budget": candidate_budget(20, 4, 2, True),
        }
        with patch.object(reporting, "write_json_atomic"), patch("builtins.print") as printer:
            publish_report("unused.json", lambda: expected, announce=True)
        printed = printer.call_args.args[0]
        for phrase in ("nominal cold learner fits: 72", "base/final/full learner fits: 60/12/0",
                       "proposed candidates: 2", "before stage: 16", "after admission: 14"):
            self.assertIn(phrase, printed)

    def test_refusal_announcement_shows_unknown_count_and_stopping_reason(self):
        expected = {"budget": candidate_budget(20, 4, None, False),
                    "stopping_reason": "candidate_budget: complete stage exceeds allowance"}
        with patch.object(reporting, "write_json_atomic"), patch("builtins.print") as printer:
            publish_report("unused.json", lambda: expected, announce=True)
        printed = printer.call_args.args[0]
        self.assertIn("proposed candidates: unknown", printed)
        self.assertIn("before stage: 16; after admission: 16", printed)
        self.assertIn("stopping reason: candidate_budget:", printed)
        self.assertNotIn("learner fits:", printed)

    def test_actual_atomic_report_is_strict_json_and_can_be_replaced(self):
        with tempfile.TemporaryDirectory(prefix="stage-report-") as directory:
            path = Path(directory) / "stage.json"
            first = {"proposal": grid_proposal({"x": [.3]}, set(), 1, [{"x": .3}])}
            publish_report(path, lambda: first)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), first)
            second = {"budget": candidate_budget(50, 3, 4, True)}
            publish_report(path, lambda: second)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), second)

    def test_interruptions_are_never_suppressed(self):
        def interrupt():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            safe_report(interrupt)
        with self.assertRaises(KeyboardInterrupt):
            publish_report("unused.json", interrupt)
        with patch.object(reporting, "write_json_atomic", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                publish_report("unused.json", lambda: {})


if __name__ == "__main__":
    unittest.main()
