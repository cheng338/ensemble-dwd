"""Opt-in stage timing must not change fits, cached work, or exception behavior."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.multiclass import OneVsOneClassifier
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelDWD
from ensemble_dwd._failure_context import failure_context
from ensemble_dwd.profiling import fit_profile, _stage


class SmallDWD:
    """Deterministic call fixture; numerical parity is separately tested below."""
    def __init__(self, **options):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_iter_, self.converged_ = 2, False
        return self

    def decision_function(self, X):
        return X[:, 0]


class FitProfilingTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack((np.linspace(-2., 2., 24), np.cos(np.arange(24))))
        self.y = np.tile([0, 1], 12)

    def model(self, **options):
        arguments = dict(M=2, theta=.5, sv_selection_ratio=1., lambd=.05,
                         gamma=.2, max_iter=3, stopping="fixed", random_state=17)
        arguments.update(options)
        return EnsembleKernelDWD(**arguments)

    def fake_fit(self, **options):
        return self.model(**options).fit(self.X, self.y)

    def test_disabled_profiling_has_no_clock_calls_or_estimator_artifact(self):
        with patch("ensemble_dwd.estimator.KernGDWD", SmallDWD), \
                patch("ensemble_dwd.profiling.perf_counter", side_effect=AssertionError("clock")):
            model = self.fake_fit()
            model.predict(self.X)
        self.assertFalse(any("profile" in key for key in vars(model)))
        self.assertFalse(any("profile" in key for key in model.get_params()))

    def test_real_fits_have_exact_state_and_training_predictions(self):
        with threadpool_limits(limits=1):
            original = self.fake_fit()
            with fit_profile() as profile:
                measured = self.fake_fit()
            assert_array_equal(original.decision_function(self.X), measured.decision_function(self.X))
            assert_array_equal(original.predict(self.X), measured.predict(self.X))
        for name in ("classes_", "selected_indices_", "consensus_scores_", "n_iter_", "converged_"):
            assert_array_equal(getattr(original, name), getattr(measured, name))
        for left, right in zip(original.subset_indices_, measured.subset_indices_):
            assert_array_equal(left, right)
        for left, right in zip(original.estimators_ + [original.final_estimator_],
                               measured.estimators_ + [measured.final_estimator_]):
            for name in ("dual_coef_", "intercept_", "objective_history_"):
                assert_array_equal(getattr(left, name), getattr(right, name))
            self.assertEqual(left.final_objective_, right.final_objective_)
        self.assertEqual(profile.as_dict()["counts"], {
            "fit_completed": 1, "base_fit_completed": 2, "base_scoring_completed": 2,
            "final_fit_completed": 1, "reused_base_learners": 0})
        self.assertFalse(any("profile" in key for key in vars(measured)))

    def test_inclusive_child_and_self_times_have_distinct_meanings(self):
        with patch("ensemble_dwd.profiling.perf_counter", side_effect=[0., 1., 2., 5., 7., 9.]):
            with fit_profile() as profile:
                with _stage("fit"):
                    with _stage("base_fit"):
                        pass
        report = profile.as_dict()
        self.assertEqual(report["context_wall_seconds"], 9.)
        self.assertEqual(report["stages"]["fit"]["inclusive_seconds"], 6.)
        self.assertEqual(report["stages"]["fit"]["child_seconds"], 3.)
        self.assertEqual(report["stages"]["fit"]["self_seconds"], 3.)
        self.assertEqual(report["stages"]["base_fit"]["self_seconds"], 3.)

    def test_profile_report_is_detached_and_json_compatible(self):
        with patch("ensemble_dwd.estimator.KernGDWD", SmallDWD):
            with fit_profile() as profile:
                self.fake_fit()
        report = profile.as_dict()
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["state"], "completed")
        self.assertEqual(report["cache_outcomes"], {"disabled": 1})
        self.assertAlmostEqual(sum(item["self_seconds"] for item in report["stages"].values()),
                               report["stages"]["fit"]["inclusive_seconds"], places=10)
        report["stages"]["fit"]["calls"] = 99
        self.assertEqual(profile.as_dict()["stages"]["fit"]["calls"], 1)

    def test_real_cache_hit_records_reuse_without_base_fit_time(self):
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1), \
                patch("ensemble_dwd.estimator._runtime_provenance", return_value={"fixture": "profile"}):
            with fit_profile() as cold:
                first = self.fake_fit(cache_dir=directory, store_base_estimators=False)
            with fit_profile() as warm:
                second = self.fake_fit(cache_dir=directory, store_base_estimators=False)
            assert_array_equal(first.predict(self.X), second.predict(self.X))
            assert_array_equal(first.final_estimator_.dual_coef_, second.final_estimator_.dual_coef_)
            assert_array_equal(first.consensus_scores_, second.consensus_scores_)
            assert_array_equal(first.selected_indices_, second.selected_indices_)
            self.assertEqual(first.cache_key_, second.cache_key_)
        self.assertFalse(first.cache_hit_)
        self.assertTrue(second.cache_hit_)
        self.assertEqual(cold.as_dict()["counts"]["base_fit_completed"], 2)
        self.assertEqual(cold.as_dict()["cache_outcomes"], {"miss": 1})
        report = warm.as_dict()
        self.assertEqual(report["counts"], {"fit_completed": 1, "base_fit_completed": 0,
            "base_scoring_completed": 0, "final_fit_completed": 1, "reused_base_learners": 2})
        self.assertEqual(report["cache_outcomes"], {"hit": 1})
        self.assertNotIn("base_fit", report["stages"])
        self.assertNotIn("base_scoring", report["stages"])
        self.assertNotIn("base_stage", report["stages"])

    def test_cache_created_without_profiling_is_reused_with_profiling(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("ensemble_dwd.estimator.KernGDWD", SmallDWD), \
                patch("ensemble_dwd.estimator._runtime_provenance", return_value={"fixture": "profile"}):
            first = self.fake_fit(cache_dir=directory)
            with fit_profile() as profile:
                second = self.fake_fit(cache_dir=directory)
        self.assertTrue(second.cache_hit_)
        self.assertEqual(first.cache_key_, second.cache_key_)
        self.assertEqual(profile.as_dict()["counts"]["reused_base_learners"], 2)

    def test_nested_contexts_are_independent_and_restore_outer(self):
        with patch("ensemble_dwd.estimator.KernGDWD", SmallDWD):
            with fit_profile() as outer:
                self.fake_fit()
                with fit_profile() as inner:
                    self.fake_fit()
                self.fake_fit()
        self.assertEqual(outer.as_dict()["counts"]["fit_completed"], 2)
        self.assertEqual(inner.as_dict()["counts"]["fit_completed"], 1)

    def test_context_objects_are_single_use(self):
        profile = fit_profile()
        self.assertEqual(profile.as_dict()["state"], "not_entered")
        with profile:
            with self.assertRaisesRegex(RuntimeError, "only be entered once"):
                with profile:
                    pass
        with self.assertRaisesRegex(RuntimeError, "only be entered once"):
            with profile:
                pass

    def test_exited_contexts_release_owner_and_prior_context_on_success_or_failure(self):
        with fit_profile() as outer:
            with fit_profile() as inner:
                pass
            self.assertIsNone(inner._owner)
            self.assertIsNone(inner._token)
            with self.assertRaises(KeyboardInterrupt):
                with fit_profile() as aborted:
                    raise KeyboardInterrupt("cancel")
            self.assertIsNone(aborted._owner)
            self.assertIsNone(aborted._token)
            with _stage("fit"):
                pass
        self.assertIsNone(outer._owner)
        self.assertIsNone(outer._token)
        self.assertEqual(outer.as_dict()["counts"]["fit_completed"], 1)

    def test_copied_context_cannot_extend_an_already_completed_profile(self):
        with fit_profile() as profile:
            captured = copy_context()
        with patch("ensemble_dwd.estimator.KernGDWD", SmallDWD), \
                patch("ensemble_dwd.profiling.perf_counter", side_effect=AssertionError("clock")):
            captured.run(self.fake_fit)
        self.assertEqual(profile.as_dict()["counts"]["fit_completed"], 0)

    def test_context_does_not_leak_to_threads_even_when_context_copied(self):
        def child():
            self.fake_fit()
            with fit_profile() as own:
                self.fake_fit()
            return own.as_dict()
        with patch("ensemble_dwd.estimator.KernGDWD", SmallDWD):
            with fit_profile() as parent, ThreadPoolExecutor(max_workers=1) as executor:
                child_report = executor.submit(copy_context().run, child).result()
                self.fake_fit()
        self.assertEqual(parent.as_dict()["counts"]["fit_completed"], 1)
        self.assertEqual(child_report["counts"]["fit_completed"], 1)

    def test_context_does_not_leak_to_new_asyncio_tasks(self):
        async def child():
            self.fake_fit()
            with fit_profile() as own:
                self.fake_fit()
            return own.as_dict()
        async def main():
            with fit_profile() as parent:
                child_report = await asyncio.create_task(child())
                self.fake_fit()
            return parent.as_dict(), child_report
        with patch("ensemble_dwd.estimator.KernGDWD", SmallDWD):
            parent, child_report = asyncio.run(main())
        self.assertEqual(parent["counts"]["fit_completed"], 1)
        self.assertEqual(child_report["counts"]["fit_completed"], 1)

    def test_sequential_one_vs_one_collects_each_binary_fit(self):
        X = np.column_stack((np.arange(60, dtype=float), np.cos(np.arange(60))))
        y = np.tile([0, 1, 2], 20)
        with patch("ensemble_dwd.estimator.KernGDWD", SmallDWD):
            with fit_profile() as profile:
                OneVsOneClassifier(self.model(M=1, theta=1.), n_jobs=1).fit(X, y)
        report = profile.as_dict()
        self.assertEqual(report["counts"]["fit_completed"], 3)
        self.assertEqual(report["counts"]["base_fit_completed"], 3)
        self.assertEqual(report["counts"]["final_fit_completed"], 3)

    def test_failed_base_call_keeps_exception_and_failure_context(self):
        for exception_type in (ValueError, KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception_type):
                error = exception_type("original failure")
                model = self.model()
                with patch("ensemble_dwd.estimator.KernGDWD.fit", side_effect=error):
                    with self.assertRaises(exception_type) as raised:
                        with fit_profile() as profile:
                            model.fit(self.X, self.y)
                self.assertIs(raised.exception, error)
                self.assertFalse(model.__sklearn_is_fitted__())
                self.assertEqual(failure_context(error)["stage"], "base_fit")
                report = profile.as_dict()
                self.assertEqual(report["state"], "aborted")
                self.assertEqual(report["counts"]["fit_completed"], 0)
                self.assertEqual(report["counts"]["base_fit_completed"], 0)
                self.assertEqual(report["stages"]["base_fit"]["aborted_calls"], 1)
                self.assertEqual(report["stages"]["fit"]["aborted_calls"], 1)
                self.assertEqual(report["stages"]["base_fit"]["exception_types"],
                                 {exception_type.__name__: 1})

    def test_caught_fit_exception_keeps_aborted_stage_without_aborting_context(self):
        with fit_profile() as profile:
            with self.assertRaises(ValueError):
                self.model(M=0).fit(self.X, self.y)
        self.assertEqual(profile.as_dict()["state"], "completed")
        self.assertEqual(profile.as_dict()["stages"]["options"]["aborted_calls"], 1)
        self.assertEqual(profile.as_dict()["stages"]["fit"]["aborted_calls"], 1)

    def test_failed_final_fit_reports_completed_base_work(self):
        class FailingFinal(SmallDWD):
            calls = 0
            def fit(inner, X, y):
                FailingFinal.calls += 1
                if FailingFinal.calls == 3:
                    raise KeyboardInterrupt("cancel final fit")
                return super().fit(X, y)
        with patch("ensemble_dwd.estimator.KernGDWD", FailingFinal):
            with self.assertRaises(KeyboardInterrupt):
                with fit_profile() as profile:
                    self.fake_fit()
        report = profile.as_dict()
        self.assertEqual(report["counts"]["base_fit_completed"], 2)
        self.assertEqual(report["counts"]["base_scoring_completed"], 2)
        self.assertEqual(report["counts"]["final_fit_completed"], 0)
        self.assertEqual(report["stages"]["final_fit"]["aborted_calls"], 1)

    def test_prediction_is_not_counted_as_fit_stage(self):
        with patch("ensemble_dwd.estimator.KernGDWD", SmallDWD):
            model = self.fake_fit()
            with fit_profile() as profile:
                model.predict(self.X)
        self.assertEqual(profile.as_dict()["stages"], {})

    def test_consensus_overflow_recovery_is_a_nested_stage(self):
        class LargeScores(SmallDWD):
            def decision_function(inner, X):
                return np.full(len(X), np.finfo(float).max * .75)
        with patch("ensemble_dwd.estimator.KernGDWD", LargeScores):
            original = self.fake_fit()
            with fit_profile() as profile:
                measured = self.fake_fit()
        assert_array_equal(original.consensus_scores_, measured.consensus_scores_)
        self.assertTrue(np.isfinite(measured.consensus_scores_).all())
        stages = profile.as_dict()["stages"]
        self.assertEqual(stages["base_consensus_recovery"]["completed_calls"], 1)
        self.assertEqual(stages["base_scoring"]["completed_calls"], 2)
        self.assertAlmostEqual(stages["base_consensus"]["child_seconds"],
                               stages["base_consensus_recovery"]["inclusive_seconds"])

    def test_cache_hit_then_final_cancellation_keeps_reuse_and_no_fresh_base_work(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("ensemble_dwd.estimator.KernGDWD", SmallDWD), \
                patch("ensemble_dwd.estimator._runtime_provenance", return_value={"fixture": "profile"}):
            self.fake_fit(cache_dir=directory)
            with patch.object(SmallDWD, "fit", side_effect=KeyboardInterrupt("cancel final")):
                with self.assertRaises(KeyboardInterrupt):
                    with fit_profile() as profile:
                        self.fake_fit(cache_dir=directory)
        report = profile.as_dict()
        self.assertEqual(report["counts"]["reused_base_learners"], 2)
        self.assertEqual(report["counts"]["base_fit_completed"], 0)
        self.assertEqual(report["counts"]["final_fit_completed"], 0)
        self.assertEqual(report["cache_outcomes"], {"hit": 1})
        self.assertNotIn("base_fit", report["stages"])
        self.assertEqual(report["stages"]["final_fit"]["aborted_calls"], 1)


if __name__ == "__main__":
    unittest.main()
