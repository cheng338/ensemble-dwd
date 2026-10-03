"""External-native cache identity; fixture bytes are never executed as code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, _cache
from ensemble_dwd import _external_identity as external
from ensemble_dwd import estimator, svm
from ensemble_dwd._cache_identity import _CacheIdentityError


def fixture(folder, *, create=True):
    folder = Path(folder)
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    modules = {}
    for name in ("numpy", "scipy", "sklearn"):
        path = folder / (name + "_native.pyd")
        if create:
            path.write_bytes((name + "-same-version-build-one").encode())
        module = types.ModuleType(name + "._native_fixture")
        module.__file__ = str(path)
        modules[module.__name__] = module
    blas = folder / "numerical_backend.dll"
    if create:
        blas.write_bytes(b"same-version-blas-build-one")
    pools = [{"filepath": str(blas), "user_api": "blas", "internal_api": "fixture",
              "prefix": "fixture", "version": "same-version", "num_threads": 1}]
    return modules, pools


def external_process_worker(folder, cache):
    modules, pools = fixture(folder, create=False)
    process = external._ExternalNativeIdentity()
    with patch.object(external, "sys", types.SimpleNamespace(modules=modules)):
        native = process.get(pools)
    key_data = {"versions": "unchanged", "pool_metadata": {k: v for k, v in pools[0].items()
                if k != "filepath"}, "external_native": native}

    def compute():
        with (Path(cache) / "computations.txt").open("a") as stream:
            stream.write("computed\n")
        return {"answer": 37}

    value, hit, key = _cache._cached_base_stage(cache, key_data, compute)
    print(json.dumps({"hit": hit, "key": key, "value": value, "identity": key_data}))


class ExternalIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.modules, self.pools = fixture(self.folder / "artifacts")
        self.module_patch = patch.object(external, "sys", types.SimpleNamespace(modules=self.modules))
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.process = external._ExternalNativeIdentity()

    def test_loaded_pool_and_extension_files_hash_once_and_return_independent_records(self):
        with patch.object(external, "_file_digest", wraps=external._file_digest) as reads:
            expected = self.process.get(self.pools)
            self.assertEqual(len(expected["artifacts"]), 4)
            self.assertEqual(reads.call_count, 4)
            for _ in range(3):
                self.assertEqual(self.process.get(self.pools), expected)
            self.assertEqual(reads.call_count, 4)
            changed = self.process.get(self.pools)
            changed["artifacts"][0]["sha256"] = "caller mutation"
            self.assertEqual(self.process.get(self.pools), expected)

    def test_same_version_hot_change_disables_identity_even_after_restoration(self):
        self.process.get(self.pools)
        path = Path(self.pools[0]["filepath"])
        original = path.read_bytes()
        path.write_bytes(b"observable-rebuild-with-the-same-version")
        with self.assertRaisesRegex(_CacheIdentityError, "changed"):
            self.process.get(self.pools)
        path.write_bytes(original)
        with self.assertRaisesRegex(_CacheIdentityError, "changed"):
            self.process.get(self.pools)

    def test_unknown_pool_or_missing_native_module_cannot_be_an_empty_identity(self):
        for pools in ([], [{"user_api": "blas"}], [{"filepath": "missing.dll", "user_api": "blas"}]):
            with self.subTest(pools=pools), self.assertRaises(_CacheIdentityError):
                external._ExternalNativeIdentity().get(pools)
        del self.modules["sklearn._native_fixture"]
        with self.assertRaisesRegex(_CacheIdentityError, "sklearn"):
            self.process.get(self.pools)

    def test_unreadable_artifact_is_sticky_bypass(self):
        with patch.object(external, "_file_digest", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(_CacheIdentityError, "unreadable"):
                self.process.get(self.pools)
        with self.assertRaisesRegex(_CacheIdentityError, "unreadable"):
            self.process.get(self.pools)

    def test_loaded_native_addition_and_object_replacement_are_detected(self):
        self.process.get(self.pools)
        previous = self.modules["numpy._native_fixture"]
        replacement = types.ModuleType(previous.__name__)
        replacement.__file__ = previous.__file__
        self.modules[previous.__name__] = replacement
        with self.assertRaisesRegex(_CacheIdentityError, "inventory changed"):
            self.process.get(self.pools)
        process = external._ExternalNativeIdentity()
        process.get(self.pools)
        self.modules["numpy.new_native"] = previous
        with self.assertRaisesRegex(_CacheIdentityError, "inventory changed"):
            process.get(self.pools)

    def test_change_during_initial_hashing_is_rejected(self):
        read = external._file_digest
        path = Path(self.pools[0]["filepath"])

        def mutate(artifact):
            digest = read(artifact)
            if artifact == path:
                path.write_bytes(b"changed-during-hash")
            return digest

        with patch.object(external, "_file_digest", side_effect=mutate):
            with self.assertRaisesRegex(_CacheIdentityError, "during identification"):
                self.process.get(self.pools)

    def test_already_loaded_bridge_source_manifest_and_dll_are_included(self):
        root = self.folder / "aocl_runtime"
        root.mkdir()
        files = {"__init__.py": b"bridge source", "runtime_manifest.json": b'{"version":"same"}',
                 "runtime.dll": b"runtime-one"}
        for name, value in files.items():
            (root / name).write_bytes(value)
        bridge = types.ModuleType("aocl_runtime")
        bridge.__file__ = str(root / "__init__.py")
        bridge._STATE = {"address": object()}
        self.modules["aocl_runtime"] = bridge
        record = self.process.get(self.pools)
        self.assertEqual(len(record["artifacts"]), 7)
        self.assertNotIn("address", json.dumps(record))
        (root / "runtime_manifest.json").write_text('{"version":"changed"}')
        with self.assertRaises(_CacheIdentityError):
            self.process.get(self.pools)

    def test_absent_bridge_is_not_imported_and_pure_python_edits_are_outside_scope(self):
        source = self.folder / "helper.py"
        source.write_text("original Python")
        module = types.ModuleType("numpy.helper")
        module.__file__ = str(source)
        self.modules[module.__name__] = module
        record = self.process.get(self.pools)
        source.write_text("different same-version pure Python helper")
        self.assertEqual(record, self.process.get(self.pools))
        self.assertNotIn("aocl_runtime", self.modules)

    def run_child(self, cache):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(sys.path)
        for name in ("OMP_NUM_THREADS", "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[name] = "1"
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--external-worker",
                                    str(self.folder / "artifacts"), str(cache)],
                                   env=environment, capture_output=True, text=True, timeout=30,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(completed.stdout)

    def test_fresh_process_same_metadata_new_native_bytes_miss_unchanged_hit(self):
        cache = self.folder / "cache"
        cache.mkdir()
        cold = self.run_child(cache)
        warm = self.run_child(cache)
        self.assertFalse(cold["hit"])
        self.assertTrue(warm["hit"])
        Path(self.pools[0]["filepath"]).write_bytes(b"different-blas-build-same-version")
        rebuilt = self.run_child(cache)
        self.assertFalse(rebuilt["hit"])
        self.assertNotEqual(rebuilt["key"], cold["key"])
        self.assertEqual(rebuilt["identity"]["pool_metadata"], cold["identity"]["pool_metadata"])
        self.assertTrue(self.run_child(cache)["hit"])
        Path(self.modules["sklearn._native_fixture"].__file__).write_bytes(b"libsvm-new-same-version")
        sklearn_rebuilt = self.run_child(cache)
        self.assertFalse(sklearn_rebuilt["hit"])
        self.assertNotEqual(sklearn_rebuilt["key"], rebuilt["key"])
        self.assertEqual((cache / "computations.txt").read_text().splitlines(), ["computed"] * 3)


class CachePublicationGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.calls = 0

    def compute(self):
        self.calls += 1
        return {"generation": self.calls}

    def test_changed_runtime_after_compute_does_not_repeat_or_publish_fit(self):
        value, hit, key = _cache._cached_base_stage(self.folder, {"fixed": 1}, self.compute,
                                                 verify=iter([True, False]).__next__)
        self.assertEqual(value, {"generation": 1})
        self.assertEqual(self.calls, 1)
        self.assertFalse(hit)
        self.assertIsNone(key)
        self.assertEqual(list((self.folder / "base-stage-v2").glob("*.joblib")), [])

    def test_change_after_serialization_cleans_temporary_payloads(self):
        _, hit, key = _cache._cached_base_stage(self.folder, {"fixed": 1}, self.compute,
                                              verify=iter([True, True, False]).__next__)
        self.assertFalse(hit)
        self.assertIsNone(key)
        self.assertEqual(self.calls, 1)
        self.assertEqual([p for p in (self.folder / "base-stage-v2").iterdir() if p.suffix != '.lock'], [])

    def test_changed_runtime_after_load_discards_hit_and_computes_uncached(self):
        _cache._cached_base_stage(self.folder, {"fixed": 1}, self.compute)
        value, hit, key = _cache._cached_base_stage(self.folder, {"fixed": 1}, self.compute,
                                                 verify=iter([True, False]).__next__)
        self.assertEqual(value, {"generation": 2})
        self.assertFalse(hit)
        self.assertIsNone(key)

    def test_runtime_verifier_retains_explicit_failure_and_does_not_retry_identity(self):
        calls = []
        def unavailable():
            calls.append(1)
            raise _CacheIdentityError("unreadable loaded BLAS")
        verify = _cache._RuntimeVerifier(unavailable, {"original": True})
        self.assertFalse(verify())
        self.assertFalse(verify())
        self.assertEqual(verify.failure, "unreadable loaded BLAS")
        self.assertEqual(calls, [1])

    def test_both_estimator_base_stages_report_postcompute_bypass_without_repeating(self):
        X, y, classes = np.array([[-1.], [1.]]), np.array([0, 1]), np.array([0, 1])
        for cls, module, kernel in ((EnsembleKernelDWD, estimator, {"gamma": .1}),
                                    (EnsembleKernelSVM, svm, .1)):
            with self.subTest(estimator=cls.__name__):
                model = cls(M=1, theta=1., sv_selection_ratio=1., cache_dir=self.folder / cls.__name__)
                expected = {"fit_status": np.array([0]), "marker": "completed base work"}
                with patch.object(module, "_runtime_provenance", side_effect=[{"v": 1}, {"v": 1}, {"v": 2}]), \
                        patch.object(model, "_base_stage", return_value=expected) as compute:
                    result, hit, key, status = model._base_result(X, y, classes, [np.array([0, 1])], kernel)
                self.assertIs(result, expected)
                self.assertEqual(compute.call_count, 1)
                self.assertFalse(hit)
                self.assertIsNone(key)
                self.assertIn("bypassed: runtime provenance changed", status)

    def test_initial_library_discovery_failure_is_explicit_sticky_bypass(self):
        X, y, classes = np.array([[-1.], [1.]]), np.array([0, 1]), np.array([0, 1])
        for error in (OSError('module is inaccessible'), ValueError('unknown library path')):
            for cls, kernel in ((EnsembleKernelDWD, {"gamma": .1}), (EnsembleKernelSVM, .1)):
                with self.subTest(estimator=cls.__name__, error=type(error).__name__):
                    process = external._ExternalNativeIdentity()
                    model = cls(M=1, theta=1., sv_selection_ratio=1., cache_dir=self.folder)
                    expected = {"marker": "uncached base result"}
                    with patch.object(external, '_EXTERNAL_IDENTITY', process), \
                            patch.object(_cache, '_initialize_numerical_runtime'), \
                            patch.object(_cache, 'threadpool_info', side_effect=error), \
                            patch.object(model, '_base_stage', return_value=expected) as compute:
                        result, hit, key, status = model._base_result(X, y, classes, [np.array([0, 1])], kernel)
                    self.assertIs(result, expected)
                    self.assertFalse(hit)
                    self.assertIsNone(key)
                    self.assertEqual(compute.call_count, 1)
                    self.assertIn('bypassed: loaded numerical library discovery failed', status)
                    with self.assertRaisesRegex(_CacheIdentityError, 'discovery failed'):
                        process.get([])

    def test_postcompute_library_discovery_failure_does_not_repeat_or_publish(self):
        X, y, classes = np.array([[-1.], [1.]]), np.array([0, 1]), np.array([0, 1])
        for cls, kernel in ((EnsembleKernelDWD, {"gamma": .1}), (EnsembleKernelSVM, .1)):
            with self.subTest(estimator=cls.__name__):
                modules, pools = fixture(self.folder / cls.__name__ / 'native')
                process = external._ExternalNativeIdentity()
                directory = self.folder / cls.__name__ / 'cache'
                model = cls(M=1, theta=1., sv_selection_ratio=1., cache_dir=directory)
                expected = {"marker": "completed base result"}
                with patch.object(external, 'sys', types.SimpleNamespace(modules=modules)), \
                        patch.object(external, '_EXTERNAL_IDENTITY', process), \
                        patch.object(_cache, '_initialize_numerical_runtime'), \
                        patch.object(_cache, '_package_identity', return_value={'fixed': True}), \
                        patch.object(_cache, 'threadpool_info', side_effect=[pools, pools, OSError('observer failed')]), \
                        patch.object(model, '_base_stage', return_value=expected) as compute:
                    result, hit, key, status = model._base_result(X, y, classes, [np.array([0, 1])], kernel)
                self.assertIs(result, expected)
                self.assertEqual(compute.call_count, 1)
                self.assertFalse(hit)
                self.assertIsNone(key)
                self.assertIn('bypassed: loaded numerical library discovery failed', status)
                self.assertEqual(list((directory / 'base-stage-v2').glob('*.joblib')), [])
                self.assertEqual(list((directory / 'base-stage-v2').glob('*.json')), [])

    def test_numerical_warmup_errors_are_not_reclassified_as_observer_failures(self):
        error = ValueError('numerical warmup failure')
        with patch.object(_cache, '_initialize_numerical_runtime', side_effect=error):
            with self.assertRaises(ValueError) as caught:
                _cache._runtime_provenance(EnsembleKernelDWD)
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--external-worker":
        external_process_worker(*sys.argv[2:])
    else:
        unittest.main()
