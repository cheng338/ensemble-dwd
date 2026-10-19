"""Cache identity follows the effective base execution policy."""

import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelDWD
from ensemble_dwd._cache import _cached_base_stage
from ensemble_dwd._parallel_base import eligibility


def _background(call):
    values, errors = [], []

    def run():
        try:
            values.append(call())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=60)
    if worker.is_alive():
        raise AssertionError("Background cache check did not finish.")
    if errors:
        raise errors[0]
    return values[0]


def _constant_kernel(X, Y):
    return np.ones((len(X), len(Y)))


class ParallelCacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.limiter = threadpool_limits(1)
        self.addCleanup(self.limiter.restore_original_limits)
        rng = np.random.RandomState(317)
        self.X = rng.normal(size=(120, 6))
        self.y = np.tile([0, 1], 60)
        self.subsets = [np.arange(start, start + 24) for start in (0, 24, 48, 72)]

    def model(self, **options):
        settings = dict(M=4, theta=.2, sv_selection_ratio=1., lambd=.05,
                        gamma=.1, random_state=17, max_iter=8, stopping="fixed",
                        base_jobs=4, cache_dir=self.directory.name)
        settings.update(options)
        return EnsembleKernelDWD(**settings)

    def background_fit(self, model):
        def fit():
            # AOCL foreign-thread numerical work is qualified only at native
            # one here. Native-N identity is tested separately without fits.
            with threadpool_limits(1):
                return model.fit(self.X, self.y)
        return _background(fit)

    def test_same_requested_jobs_separate_main_and_background_entries(self):
        main = self.model().fit(self.X, self.y)
        background = self.background_fit(self.model())
        self.assertFalse(main.cache_hit_)
        self.assertFalse(background.cache_hit_)
        self.assertNotEqual(main.cache_key_, background.cache_key_)
        self.assertEqual(main.base_execution_["effective_jobs"], 4)
        self.assertEqual(background.base_execution_["effective_jobs"], 1)
        main_again = self.model().fit(self.X, self.y)
        background_again = self.background_fit(self.model())
        self.assertTrue(main_again.cache_hit_)
        self.assertTrue(background_again.cache_hit_)
        self.assertEqual(main.cache_key_, main_again.cache_key_)
        self.assertEqual(background.cache_key_, background_again.cache_key_)
        for other in (background, main_again, background_again):
            np.testing.assert_array_equal(main.selected_indices_, other.selected_indices_)
            np.testing.assert_array_equal(main.predict(self.X), other.predict(self.X))
            np.testing.assert_allclose(main.consensus_scores_, other.consensus_scores_,
                                       rtol=1e-10, atol=1e-10)
        self.assertIsNot(main.final_estimator_, main_again.final_estimator_)

    def _base_result(self, model, kernel_kws=None):
        return model._base_result(self.X, self.y, np.array([0, 1]), self.subsets,
                                  {"gamma": .1} if kernel_kws is None else kernel_kws)

    def test_mocked_native_many_provenance_cannot_hide_execution_policy(self):
        observed, computed = [], []

        def capture(directory, key_data, compute, **kwargs):
            observed.append(key_data["parameters"].copy())
            return _cached_base_stage(directory, key_data, compute, **kwargs)

        def compute(model, *unused):
            origin = "main" if threading.current_thread() is threading.main_thread() else "background"
            computed.append(origin)
            return {"origin": origin}

        # Both callers report the same parent runtime. Only main-thread base
        # dispatch would reduce it to native one; no native-N child fit runs.
        provenance = {"pools": [{"num_threads": 8, "internal_api": "fixture"}]}
        with patch("ensemble_dwd.estimator._runtime_provenance", return_value=provenance), \
                patch("ensemble_dwd.estimator._cached_base_stage", side_effect=capture), \
                patch.object(EnsembleKernelDWD, "_base_stage", compute):
            main = self._base_result(self.model())
            background = _background(lambda: self._base_result(self.model()))
            main_again = self._base_result(self.model())
            background_again = _background(lambda: self._base_result(self.model()))
        self.assertEqual(computed, ["main", "background"])
        self.assertFalse(main[1])
        self.assertFalse(background[1])
        self.assertTrue(main_again[1])
        self.assertTrue(background_again[1])
        self.assertNotEqual(main[2], background[2])
        self.assertEqual(main[0], {"origin": "main"})
        self.assertEqual(background[0], {"origin": "background"})
        self.assertEqual(observed[0]["base_execution_resolved"], dict(
            policy="ordered-native-one-v1", effective_jobs=4, native_threads=1))
        self.assertEqual(observed[1]["base_execution_resolved"], dict(
            policy="sequential-caller-native-v1", effective_jobs=1, native_threads="caller"))

    def test_default_sequential_key_semantics_do_not_depend_on_python_thread(self):
        observed = []

        def capture(directory, key_data, compute, **kwargs):
            observed.append(key_data["parameters"].copy())
            return _cached_base_stage(directory, key_data, compute, **kwargs)

        with patch("ensemble_dwd.estimator._runtime_provenance", return_value={"native": 1}), \
                patch("ensemble_dwd.estimator._cached_base_stage", side_effect=capture), \
                patch.object(EnsembleKernelDWD, "_base_stage", return_value={"same": True}) as base:
            main = self._base_result(self.model(base_jobs=1))
            background = _background(lambda: self._base_result(self.model(base_jobs=1)))
        self.assertEqual(base.call_count, 1)
        self.assertFalse(main[1])
        self.assertTrue(background[1])
        self.assertEqual(main[2], background[2])
        self.assertTrue(all("base_execution_resolved" not in params for params in observed))

    def test_instance_custom_factory_bypasses_cache(self):
        model = self.model()
        model._make_estimator = lambda kws: None
        self.assertEqual(eligibility(model, {}, EnsembleKernelDWD._make_estimator),
                         (False, "custom learner factory"))
        with patch.object(EnsembleKernelDWD, "_base_stage", return_value={"custom": True}) as base, \
                patch("ensemble_dwd.estimator._runtime_provenance") as provenance:
            for unused in range(2):
                result, hit, key, status = self._base_result(model)
                self.assertEqual(result, {"custom": True})
                self.assertFalse(hit)
                self.assertIsNone(key)
                self.assertEqual(status, "bypassed: custom learner factory")
        self.assertEqual(base.call_count, 2)
        provenance.assert_not_called()

    def test_subclass_and_callable_fallbacks_still_bypass_cache(self):
        class Custom(EnsembleKernelDWD):
            pass

        cases = [
            (Custom(**self.model().get_params()), {}, "bypassed: custom estimator subclass"),
            (self.model(kernel=_constant_kernel), {}, "bypassed: callable kernel or keyword"),
            (self.model(), {"callback": _constant_kernel}, "bypassed: callable kernel or keyword"),
        ]
        with patch.object(EnsembleKernelDWD, "_base_stage", return_value={"fallback": True}), \
                patch("ensemble_dwd.estimator._runtime_provenance") as provenance:
            for model, kernel_kws, expected in cases:
                with self.subTest(reason=expected):
                    result, hit, key, status = self._base_result(model, kernel_kws)
                    self.assertEqual(result, {"fallback": True})
                    self.assertFalse(hit)
                    self.assertIsNone(key)
                    self.assertEqual(status, expected)
        provenance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
