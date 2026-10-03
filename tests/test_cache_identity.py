"""Native artifact identity without modifying or executing installed binaries."""

import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from numpy.testing import assert_array_equal

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM
from ensemble_dwd import _cache, _cache_identity as identity
from test_cache import CacheProbeDWD


def fixture_roots(folder):
    roots = {name: Path(folder) / name for name in ("ensemble_dwd", "dwd")}
    for root in roots.values():
        root.mkdir(parents=True)
        (root / "__init__.py").write_text('__version__ = "same-version"\n')
    (roots["dwd"] / "_residual_accel.pyd").write_bytes(b"native-build-one")
    return roots


def artifact_process_worker(folder, cache, available):
    """Fresh-process disk-identity test; fixture bytes are not a loadable DLL."""
    roots = {name: Path(folder) / name for name in ("ensemble_dwd", "dwd")}
    snapshot = identity._ArtifactSnapshot(roots)
    snapshot.check()
    key_data = {"version": "same-version", "digest": snapshot.digest,
                "native": snapshot.native_digests(), "accelerator_available": available == "1"}

    def compute():
        with (Path(cache) / "computations.txt").open("a") as stream:
            stream.write("computed\n")
        return {"answer": 37}

    value, hit, key = _cache._cached_base_stage(cache, key_data, compute)
    print(json.dumps({"value": value, "hit": hit, "key": key, "identity": key_data}))


class ArtifactSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.roots = fixture_roots(self.folder / "one")

    def test_native_helpers_source_and_extension_suffixes_are_fingerprinted(self):
        names = ("helper.dll", "helper.dylib", "_helper.cpython-312-x86_64-linux-gnu.so",
                 "libhelper.so.2.0", "residual_core.c", "residual_core.h")
        for name in names:
            (self.roots["dwd"] / name).write_bytes(b"helper")
        snapshot = identity._ArtifactSnapshot(self.roots)
        snapshot.check()
        self.assertEqual(len(snapshot.digests), 9)
        self.assertEqual(len(snapshot.native_digests()), 5)
        self.assertIn("dwd/_residual_accel.pyd", snapshot.native_digests())

    def test_repeated_checks_do_not_read_binary_or_source_bytes(self):
        snapshot = identity._ArtifactSnapshot(self.roots)
        with patch.object(identity, "_file_digest", wraps=identity._file_digest) as read:
            first = snapshot.check()
            reads = read.call_count
            for _ in range(5):
                self.assertEqual(snapshot.check(), first)
                snapshot.native_digests()
            self.assertEqual(reads, 3)
            self.assertEqual(read.call_count, reads)

    def test_relocated_identical_artifacts_keep_content_identity(self):
        other = self.folder / "two"
        shutil.copytree(self.folder / "one", other)
        a = identity._ArtifactSnapshot(self.roots)
        b = identity._ArtifactSnapshot({name: other / name for name in self.roots})
        self.assertEqual(a.check(), b.check())
        self.assertEqual(a.native_digests(), b.native_digests())

    def test_same_version_new_native_bytes_change_new_snapshot_identity(self):
        first = identity._ArtifactSnapshot(self.roots)
        original = first.check()
        (self.roots["dwd"] / "_residual_accel.pyd").write_bytes(b"native-build-two")
        second = identity._ArtifactSnapshot(self.roots)
        self.assertNotEqual(original, second.check())
        self.assertNotEqual(first.native_digests(), second.native_digests())
        self.assertEqual((self.roots["dwd"] / "__init__.py").read_text(), '__version__ = "same-version"\n')

    def test_observable_edit_bypasses_for_remainder_of_process_snapshot(self):
        snapshot = identity._ArtifactSnapshot(self.roots)
        snapshot.check()
        artifact = self.roots["dwd"] / "_residual_accel.pyd"
        artifact.write_bytes(b"changed-and-longer")
        with self.assertRaisesRegex(identity._CacheIdentityError, "changed.*restart"):
            snapshot.check()
        artifact.write_bytes(b"native-build-one")
        with self.assertRaisesRegex(identity._CacheIdentityError, "changed.*restart"):
            snapshot.check()

    def test_native_addition_and_removal_are_detected(self):
        snapshot = identity._ArtifactSnapshot(self.roots)
        snapshot.check()
        extra = self.roots["dwd"] / "new-helper.dll"
        extra.write_bytes(b"new")
        with self.assertRaises(identity._CacheIdentityError):
            snapshot.check()
        snapshot = identity._ArtifactSnapshot(self.roots)
        snapshot.check()
        extra.unlink()
        with self.assertRaises(identity._CacheIdentityError):
            snapshot.check()

    def test_source_mutation_is_also_detected(self):
        snapshot = identity._ArtifactSnapshot(self.roots)
        snapshot.check()
        (self.roots["dwd"] / "__init__.py").write_text("changed source")
        with self.assertRaises(identity._CacheIdentityError):
            snapshot.check()

    def test_unreadable_artifact_never_produces_an_unknown_shared_identity(self):
        snapshot = identity._ArtifactSnapshot(self.roots)
        with patch.object(identity, "_file_digest", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(identity._CacheIdentityError, "unreadable"):
                snapshot.check()
        with self.assertRaisesRegex(identity._CacheIdentityError, "unreadable"):
            snapshot.check()
        self.assertIsNone(snapshot.digest)

    def test_missing_source_or_directory_is_not_a_valid_identity(self):
        (self.roots["dwd"] / "__init__.py").unlink()
        with self.assertRaisesRegex(identity._CacheIdentityError, "source identity"):
            identity._ArtifactSnapshot(self.roots).check()
        with self.assertRaisesRegex(identity._CacheIdentityError, "directory"):
            identity._ArtifactSnapshot({"dwd": self.folder / "missing"}).check()

    def test_change_during_hashing_is_rejected(self):
        real = identity._file_digest
        artifact = self.roots["dwd"] / "_residual_accel.pyd"

        def read(path):
            value = real(path)
            if path == artifact:
                artifact.write_bytes(b"changed-during-read")
            return value

        with patch.object(identity, "_file_digest", side_effect=read):
            with self.assertRaisesRegex(identity._CacheIdentityError, "during identification"):
                identity._ArtifactSnapshot(self.roots).check()

    def test_loaded_accelerator_and_known_unavailable_state_are_distinct(self):
        snapshot = identity._ArtifactSnapshot(self.roots)
        snapshot.check()
        module = types.ModuleType("dwd._residual_accel")
        module.__file__ = str(self.roots["dwd"] / "_residual_accel.pyd")
        module.evaluate = lambda: None
        compiled = types.SimpleNamespace(_ACCEL=module)
        native = types.SimpleNamespace(_native_supported=lambda: True)
        modules = {"dwd._compiled_residual": compiled, "dwd._native_residual": native}
        with patch.object(identity.importlib, "import_module", side_effect=modules.__getitem__):
            loaded, _ = identity._accelerator_state(snapshot)
            compiled._ACCEL = None  # File still exists, but the import was unavailable.
            unavailable, _ = identity._accelerator_state(snapshot)
        self.assertTrue(loaded["available"])
        self.assertEqual(loaded["sha256"], snapshot.native_digests()["dwd/_residual_accel.pyd"])
        self.assertFalse(unavailable["available"])
        self.assertIsNone(unavailable["sha256"])
        self.assertNotEqual(loaded, unavailable)

    def test_unknown_or_external_loaded_accelerator_is_rejected(self):
        snapshot = identity._ArtifactSnapshot(self.roots)
        snapshot.check()
        module = types.ModuleType("dwd._residual_accel")
        module.evaluate = lambda: None
        compiled = types.SimpleNamespace(_ACCEL=module)
        native = types.SimpleNamespace(_native_supported=lambda: True)
        modules = {"dwd._compiled_residual": compiled, "dwd._native_residual": native}
        with patch.object(identity.importlib, "import_module", side_effect=modules.__getitem__):
            with self.assertRaisesRegex(identity._CacheIdentityError, "origin"):
                identity._accelerator_state(snapshot)
            external = self.folder / "external.pyd"
            external.write_bytes(b"external")
            module.__file__ = str(external)
            with self.assertRaisesRegex(identity._CacheIdentityError, "outside"):
                identity._accelerator_state(snapshot)

    def test_runtime_availability_change_disables_current_process_reuse(self):
        process = identity._ProcessIdentity()
        first = ({"available": False}, 1)
        second = ({"available": True}, 2)
        with patch.object(identity, "_package_roots", return_value=self.roots), patch.object(
                identity, "_accelerator_state", side_effect=[first, second]):
            self.assertFalse(process.get()["accelerator"]["available"])
            with self.assertRaisesRegex(identity._CacheIdentityError, "accelerator changed"):
                process.get()
        with self.assertRaisesRegex(identity._CacheIdentityError, "accelerator changed"):
            process.get()

    def test_runtime_import_failure_is_explicit_bypass(self):
        with patch.object(identity.importlib, "import_module", side_effect=ImportError("broken")):
            with self.assertRaisesRegex(identity._CacheIdentityError, "identity is unavailable"):
                identity._ProcessIdentity().get()

    def test_cache_module_import_anchors_metadata_before_first_cached_fit(self):
        # Load the real identity code under an isolated alias and fixture DWD
        # package. Its global anchor executes exactly as on normal import.
        source = self.roots["ensemble_dwd"] / "_cache_identity.py"
        source.write_bytes(Path(identity.__file__).read_bytes())
        fake_dwd = types.ModuleType("dwd")
        fake_dwd.__file__ = str(self.roots["dwd"] / "__init__.py")
        modules = {"dwd": fake_dwd,
                   "dwd._compiled_residual": types.SimpleNamespace(_ACCEL=None),
                   "dwd._native_residual": types.SimpleNamespace(_native_supported=lambda: False)}
        spec = importlib.util.spec_from_file_location("isolated_cache_identity", source)
        probe = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(probe)
            self.assertIsNotNone(probe._PROCESS_IDENTITY.snapshot.anchor_files)
            self.assertIsNone(probe._PROCESS_IDENTITY.snapshot.digests)
            (self.roots["dwd"] / "_residual_accel.pyd").write_bytes(b"changed-before-first-fit")
            with self.assertRaisesRegex(probe._CacheIdentityError, "since cache module import"):
                probe._package_identity()

    def test_unreadable_import_anchor_does_not_prevent_uncached_package_import(self):
        with patch.object(identity, "_package_roots", side_effect=identity._CacheIdentityError("unknown origin")):
            process = identity._ProcessIdentity(anchor=True)
        with self.assertRaisesRegex(identity._CacheIdentityError, "unknown origin"):
            process.get()


class CacheFormatAndProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def test_v1_entry_is_never_promoted_or_reused(self):
        key_data = {"same": "key"}
        key = joblib.hash(key_data, hash_name="sha1")
        old = self.folder / "base-stage-v1"
        old.mkdir()
        payload = old / (key + ".joblib")
        joblib.dump({"old": True}, payload)
        metadata = old / (key + ".json")
        metadata.write_text(json.dumps({"key": key, "format": 1, "sha256": _cache._sha256(payload)}))
        before = {p.name: p.read_bytes() for p in old.iterdir()}
        value, hit, _ = _cache._cached_base_stage(self.folder, key_data, lambda: {"new": True})
        self.assertFalse(hit)
        self.assertEqual(value, {"new": True})
        self.assertEqual(before, {p.name: p.read_bytes() for p in old.iterdir()})
        self.assertEqual(json.loads((self.folder / "base-stage-v2" / (key + ".json")).read_text())["format"], 2)

    def test_v1_metadata_in_v2_namespace_is_invalid(self):
        calls = []

        def compute():
            calls.append(1)
            return {"answer": 37}

        _, _, key = _cache._cached_base_stage(self.folder, {"x": 1}, compute)
        metadata = self.folder / "base-stage-v2" / (key + ".json")
        record = json.loads(metadata.read_text())
        record["format"] = 1
        metadata.write_text(json.dumps(record))
        _, hit, _ = _cache._cached_base_stage(self.folder, {"x": 1}, compute)
        self.assertFalse(hit)
        self.assertEqual(len(calls), 2)

    def test_runtime_pool_metadata_is_refreshed_without_rehashing_artifacts(self):
        with patch.object(_cache, "_initialize_numerical_runtime"), patch.object(
                _cache, "_package_identity", return_value={"artifacts_sha256": "same"}), patch.object(
                _cache, "_external_native_identity", return_value={"artifacts": "same"}), patch.object(
                _cache, "threadpool_info", side_effect=[[{"num_threads": 1}], [{"num_threads": 2}]]):
            first = _cache._runtime_provenance(CacheProbeDWD)
            second = _cache._runtime_provenance(CacheProbeDWD)
        self.assertEqual(first["format"], 2)
        self.assertEqual(first["package_identity"], second["package_identity"])
        self.assertNotEqual(first["pools"], second["pools"])

    def run_child(self, folder, cache, available=True):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(sys.path)
        for name in ("OMP_NUM_THREADS", "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[name] = "1"
        child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--artifact-worker",
                                str(folder), str(cache), "1" if available else "0"],
                               env=environment, text=True, capture_output=True, timeout=30,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        return json.loads(child.stdout)

    def test_fresh_process_same_version_rebuild_invalidates_and_unchanged_hits(self):
        roots = fixture_roots(self.folder / "artifacts")
        cache = self.folder / "cache"
        cache.mkdir()
        cold = self.run_child(self.folder / "artifacts", cache)
        warm = self.run_child(self.folder / "artifacts", cache)
        self.assertFalse(cold["hit"])
        self.assertTrue(warm["hit"])
        self.assertEqual(cold["key"], warm["key"])
        (roots["dwd"] / "_residual_accel.pyd").write_bytes(b"same-version-rebuilt")
        rebuilt = self.run_child(self.folder / "artifacts", cache)
        self.assertFalse(rebuilt["hit"])
        self.assertNotEqual(cold["key"], rebuilt["key"])
        self.assertTrue(self.run_child(self.folder / "artifacts", cache)["hit"])
        unavailable = self.run_child(self.folder / "artifacts", cache, available=False)
        self.assertFalse(unavailable["hit"])
        self.assertNotEqual(rebuilt["key"], unavailable["key"])
        self.assertTrue(self.run_child(self.folder / "artifacts", cache, available=False)["hit"])
        self.assertEqual((cache / "computations.txt").read_text().splitlines(), ["computed"] * 3)


class DiagnosticProbeDWD(CacheProbeDWD):
    def fit(self, X, y):
        super().fit(X, y)
        self.converged_ = False
        self.criterion_reached_ = True
        self.optimality_met_ = False
        self.objective_tolerance_met_ = True
        self.termination_reason_ = "objective_tolerance"
        self.backend_ = "cholesky"
        self.returned_iteration_ = np.int64(6)
        self.gradient_inf_norm_ = .03
        self.rkhs_gradient_norm_ = .04
        self.dual_gap_ = None
        self.dual_equality_residual_ = .02
        return self


class IdentityBypassAndDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.X = np.column_stack((np.linspace(-2., 2., 20), np.cos(np.arange(20))))
        self.y = np.tile([0, 1], 10)

    def model(self, **changes):
        settings = dict(M=3, theta=.3, sv_selection_ratio=.3, gamma=.02,
                        random_state=17, store_base_estimators=False, cache_dir=str(self.folder))
        settings.update(changes)
        return EnsembleKernelDWD(**settings)

    def test_unknown_identity_fits_uncached_with_explicit_status(self):
        DiagnosticProbeDWD.fits = []
        with patch("ensemble_dwd.estimator.KernGDWD", DiagnosticProbeDWD), patch(
                "ensemble_dwd.estimator._runtime_provenance",
                side_effect=identity._CacheIdentityError("unreadable native artifact")):
            model = self.model().fit(self.X, self.y)
        self.assertFalse(model.cache_hit_)
        self.assertIsNone(model.cache_key_)
        self.assertEqual(model.cache_status_, "bypassed: unreadable native artifact")
        self.assertEqual(len(DiagnosticProbeDWD.fits), 4)
        self.assertFalse(list(self.folder.iterdir()))

    def test_svm_unknown_identity_also_bypasses_without_hiding_fit_errors(self):
        sentinel = RuntimeError("base-fit-error")
        model = EnsembleKernelSVM(M=1, theta=1., sv_selection_ratio=1., cache_dir=str(self.folder))
        with patch("ensemble_dwd.svm._runtime_provenance",
                   side_effect=identity._CacheIdentityError("unknown native state")), patch.object(
                model, "_base_stage", side_effect=sentinel):
            with self.assertRaisesRegex(RuntimeError, "base-fit-error"):
                model.fit(self.X, self.y)

    def test_cold_and_warm_lean_base_diagnostics_preserve_distinct_stopping_flags(self):
        DiagnosticProbeDWD.fits = []
        with patch("ensemble_dwd.estimator.KernGDWD", DiagnosticProbeDWD):
            first = self.model().fit(self.X, self.y)
            second = self.model(sv_selection_ratio=.5).fit(self.X, self.y)
        self.assertFalse(first.cache_hit_)
        self.assertTrue(second.cache_hit_)
        self.assertEqual(len(DiagnosticProbeDWD.fits), 5)
        self.assertIsNone(first.estimators_)
        self.assertEqual(first.base_diagnostics_, second.base_diagnostics_)
        for record in second.base_diagnostics_:
            self.assertFalse(record["converged"])
            self.assertFalse(record["optimality_met"])
            self.assertTrue(record["criterion_reached"])
            self.assertTrue(record["objective_tolerance_met"])
            self.assertEqual(record["termination_reason"], "objective_tolerance")
            self.assertEqual(record["n_iter"], 7)
            self.assertEqual(record["returned_iteration"], 6)
            self.assertIsNone(record["dual_gap"])
            self.assertEqual(record["rkhs_gradient_norm"], .04)
        assert_array_equal(first.consensus_scores_, second.consensus_scores_)
        assert_array_equal(first.predict(self.X), second.predict(self.X))
        first.base_diagnostics_[0]["converged"] = True
        self.assertFalse(second.base_diagnostics_[0]["converged"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--artifact-worker":
        artifact_process_worker(*sys.argv[2:])
    else:
        unittest.main()
