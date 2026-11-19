"""Native cache inventory growth: real import evidence and finite fault controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import types
from contextlib import contextmanager

import pytest

from ensemble_dwd import _cache
from ensemble_dwd import _external_identity as external
from ensemble_dwd._cache_identity import _CacheIdentityError


def _probe(tmp_path, *, family="dwd", kernel="rbf", backend="cholesky", preimport=False,
           cache=None, label="probe"):
    cache = tmp_path / "cache" if cache is None else cache
    result_path = tmp_path / (label + ".json")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(item) for item in sys.path if item)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
        environment[name] = "1"
    command = [sys.executable, "-B", str(Path(__file__).with_name("_native_import_probe.py")),
               "--cache", str(cache), "--result", str(result_path), "--family", family,
               "--kernel", kernel, "--backend", backend]
    if preimport:
        command.append("--preimport")
    completed = subprocess.run(command, env=environment, capture_output=True, text=True,
                               timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(result_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("family,kernel,backend", [
    ("dwd", "rbf", "cholesky"), ("dwd", "rbf", "spectral"),
    ("dwd", "rbf", "lbfgs"), ("dwd", "linear", "cholesky"),
    ("svm", "rbf", "unused"), ("svm", "linear", "unused"),
])
def test_real_routes_admit_import_growth_then_reuse(tmp_path, record_property, family, kernel, backend):
    result = _probe(tmp_path, family=family, kernel=kernel, backend=backend)
    record_property("fresh_native_probe", json.dumps(result, sort_keys=True))
    initial, warm, post, stable = result["records"]
    assert not initial["hit"] and warm["hit"] and initial["key"] == warm["key"]
    assert result["identity_failure"] is None
    assert stable["hit"] and stable["key"] == post["key"]
    assert all(row["final_is_fresh"] for row in result["records"])
    if result["added_modules"]:
        assert not post["hit"] and post["key"] != initial["key"]
        assert [row["base_stages"] for row in result["records"]] == [1, 0, 1, 0]
        assert result["cache_entries"] == 2
    else:
        assert post["hit"] and post["key"] == initial["key"]
        assert [row["base_stages"] for row in result["records"]] == [1, 0, 0, 0]
        assert result["cache_entries"] == 1


@pytest.mark.parametrize("family", ["dwd", "svm"])
def test_fresh_import_order_reuses_same_full_identity(tmp_path, record_property, family):
    late = _probe(tmp_path, family=family, label="late")
    early = _probe(tmp_path, family=family, preimport=True, label="early")
    record_property("late_probe", json.dumps(late, sort_keys=True))
    record_property("early_probe", json.dumps(early, sort_keys=True))
    assert late["identity_failure"] is None and early["identity_failure"] is None
    assert early["records"][0]["hit"]
    assert early["records"][0]["key"] == late["records"][-1]["key"]
    assert all(row["hit"] and row["base_fits"] == 0 for row in early["records"])
    for name in ("predictions", "consensus", "selected", "subsets", "diagnostics"):
        assert early["records"][0][name] == late["records"][-1][name]


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    """Native-looking fixture bytes are hashed only, never loaded or executed."""
    folder = _cache._cache_path(tmp_path / "native")
    folder.mkdir()
    modules = {}

    def add(name, *, path=None):
        if path is None:
            path = folder / (name.replace(".", "_") + ".pyd")
            path.write_bytes((name + "-fixture-version-one").encode())
        module = types.ModuleType(name)
        module.__file__ = str(path)
        modules[name] = module
        return module

    for name in ("numpy.fixture", "scipy.fixture", "sklearn.fixture"):
        add(name)
    pool = folder / "pool.dll"
    pool.write_bytes(b"fixture-blas-version-one")
    pools = [dict(filepath=str(pool), user_api="blas", internal_api="fixture", prefix="fixture",
                  version="one", num_threads=1, threading_layer="fixture", architecture="fixture")]
    monkeypatch.setattr(external, "sys", types.SimpleNamespace(modules=modules))
    return types.SimpleNamespace(folder=folder, modules=modules, add=add, pools=pools, pool=pool,
                                 process=external._ExternalNativeIdentity())


def test_additions_hash_only_new_paths_keep_old_records_and_repeat(native_fixture, monkeypatch):
    fixture = native_fixture
    reads = []
    original = external._file_digest

    def counted(path):
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(external, "_file_digest", counted)
    first = fixture.process.get(fixture.pools)
    frozen_first = json.loads(json.dumps(first))
    assert len(reads) == 4
    added = fixture.add("scipy.io.fixture")
    second = fixture.process.get(fixture.pools)
    assert reads[4:] == [added.__file__]
    assert first == frozen_first and first != second
    first_paths = {item["path"]: item for item in first["artifacts"]}
    second_paths = {item["path"]: item for item in second["artifacts"]}
    assert all(second_paths[path] == value for path, value in first_paths.items())
    another = fixture.add("sklearn.another_native")
    third = fixture.process.get(fixture.pools)
    assert reads[5:] == [another.__file__]
    assert third != second and fixture.process.failure is None
    assert fixture.process.get(fixture.pools) == third and len(reads) == 6
    second["artifacts"][0]["roles"].append("caller-tamper")
    assert fixture.process.get(fixture.pools) == third


def test_same_path_alias_adds_role_without_rehashing(native_fixture, monkeypatch):
    fixture = native_fixture
    first = fixture.process.get(fixture.pools)
    old = fixture.modules["scipy.fixture"]
    fixture.modules["scipy.alias_fixture"] = old

    def forbidden(*args):
        raise AssertionError("Same-file alias must retain the verified file digest")

    monkeypatch.setattr(external, "_file_digest", forbidden)
    second = fixture.process.get(fixture.pools)
    assert len(second["artifacts"]) == len(first["artifacts"])
    assert first != second and fixture.process.failure is None
    item = next(item for item in second["artifacts"] if item["path"] == old.__file__)
    assert item["roles"] == ["module:scipy.alias_fixture", "module:scipy.fixture"]
    assert fixture.process.get(fixture.pools) == second


def test_stable_get_does_not_hash_or_run_extension_predicate(native_fixture, monkeypatch):
    fixture = native_fixture
    first = fixture.process.get(fixture.pools)
    observations = []
    observe = external._observed_artifacts

    def counted(pools):
        observations.append("observed")
        return observe(pools)

    def forbidden(*args):
        raise AssertionError("Healthy get must keep its existing single-observation path")

    monkeypatch.setattr(external, "_observed_artifacts", counted)
    monkeypatch.setattr(external, "_file_digest", forbidden)
    monkeypatch.setattr(external, "_module_only_extension", forbidden)
    for _ in range(3):
        assert fixture.process.get(fixture.pools) == first
    assert observations == ["observed"] * 3


def test_import_order_produces_identical_full_identity(native_fixture):
    fixture = native_fixture
    fixture.process.get(fixture.pools)
    fixture.add("scipy.io.fixture")
    late = fixture.process.get(fixture.pools)
    early = external._ExternalNativeIdentity().get(fixture.pools)
    assert early == late


@pytest.mark.parametrize("change", ["bytes", "removed", "object", "origin", "pool_role", "pool_path", "pool_added", "bridge_added"])
def test_addition_does_not_mask_changed_prior_or_nonmodule_binding(native_fixture, change):
    fixture = native_fixture
    first = fixture.process.get(fixture.pools)
    old_snapshot = (fixture.process.artifacts, fixture.process.roles, fixture.process.modules, fixture.process.identity)
    fixture.add("scipy.io.fixture")
    old = fixture.modules["numpy.fixture"]
    if change == "bytes":
        Path(old.__file__).write_bytes(b"different-existing-native-bytes-with-different-size")
    elif change == "removed":
        del fixture.modules["numpy.fixture"]
    elif change == "object":
        fixture.add("numpy.fixture", path=Path(old.__file__))
    elif change == "origin":
        replacement = fixture.folder / "replacement.pyd"
        replacement.write_bytes(b"new-origin")
        old.__file__ = str(replacement)
    elif change == "pool_role":
        fixture.pools[0]["prefix"] = "changed"
    elif change == "pool_path":
        replacement = fixture.folder / "replacement.dll"
        replacement.write_bytes(b"another-pool")
        fixture.pools[0]["filepath"] = str(replacement)
    elif change == "pool_added":
        fixture.pools.append(dict(fixture.pools[0], prefix="other"))
    else:
        bridge = fixture.folder / "bridge"
        bridge.mkdir()
        source = bridge / "__init__.py"
        source.write_bytes(b"bridge")
        fixture.add("aocl_runtime", path=source)
    with pytest.raises(_CacheIdentityError):
        fixture.process.get(fixture.pools)
    assert fixture.process.failure is not None
    assert (fixture.process.artifacts, fixture.process.roles, fixture.process.modules, fixture.process.identity) == old_snapshot
    assert first["artifacts"]
    failure = fixture.process.failure
    with pytest.raises(_CacheIdentityError, match="restart required|unavailable"):
        fixture.process.get(fixture.pools)
    assert fixture.process.failure == failure


@pytest.mark.parametrize("change", ["bridge_file_added", "bridge_bytes", "bridge_object"])
def test_existing_bridge_remains_strict_during_module_addition(native_fixture, change):
    fixture = native_fixture
    root = fixture.folder / "bridge"
    root.mkdir()
    source = root / "__init__.py"
    source.write_bytes(b"bridge-original")
    fixture.add("aocl_runtime", path=source)
    fixture.process.get(fixture.pools)
    fixture.add("scipy.io.fixture")
    if change == "bridge_file_added":
        (root / "new.json").write_bytes(b"{}")
    elif change == "bridge_bytes":
        source.write_bytes(b"changed-bridge-source")
    else:
        fixture.add("aocl_runtime", path=source)
    with pytest.raises(_CacheIdentityError):
        fixture.process.get(fixture.pools)
    assert fixture.process.failure is not None


@pytest.mark.parametrize("problem", ["missing", "directory", "unreadable"])
def test_unknown_new_artifact_cannot_extend_identity(native_fixture, monkeypatch, problem):
    fixture = native_fixture
    fixture.process.get(fixture.pools)
    old_identity = fixture.process.identity
    path = fixture.folder / "unknown.pyd"
    if problem == "directory":
        path.mkdir()
    elif problem == "unreadable":
        path.write_bytes(b"cannot-hash")
        original = external._file_digest

        def fail_new(artifact):
            if artifact == path:
                raise PermissionError("new fixture denied")
            return original(artifact)

        monkeypatch.setattr(external, "_file_digest", fail_new)
    fixture.add("scipy.io.unknown", path=path)
    with pytest.raises(_CacheIdentityError):
        fixture.process.get(fixture.pools)
    assert fixture.process.identity == old_identity and fixture.process.failure is not None


@pytest.mark.parametrize("change", ["another_addition", "old_replaced", "new_changed", "new_removed"])
def test_change_while_hashing_addition_cannot_commit_partial_identity(native_fixture, monkeypatch, change):
    fixture = native_fixture
    fixture.process.get(fixture.pools)
    old_snapshot = (fixture.process.artifacts, fixture.process.roles, fixture.process.modules, fixture.process.identity)
    new = fixture.add("scipy.io.fixture")
    original = external._file_digest

    def racing(path):
        digest = original(path)
        assert str(path) == new.__file__
        if change == "another_addition":
            fixture.add("scipy.io.concurrent")
        elif change == "old_replaced":
            old = fixture.modules["numpy.fixture"]
            fixture.add("numpy.fixture", path=Path(old.__file__))
        elif change == "new_changed":
            path.write_bytes(b"changed-during-new-artifact-hash")
        else:
            del fixture.modules["scipy.io.fixture"]
        return digest

    monkeypatch.setattr(external, "_file_digest", racing)
    with pytest.raises(_CacheIdentityError, match="during identification"):
        fixture.process.get(fixture.pools)
    assert (fixture.process.artifacts, fixture.process.roles, fixture.process.modules, fixture.process.identity) == old_snapshot
    assert fixture.process.failure is not None


def test_alias_growth_still_requires_second_observation(native_fixture, monkeypatch):
    fixture = native_fixture
    fixture.process.get(fixture.pools)
    old_identity = fixture.process.identity
    fixture.modules["scipy.alias"] = fixture.modules["scipy.fixture"]
    original = external._observed_artifacts
    calls = []

    def changed(pools):
        observed = original(pools)
        calls.append("observed")
        if len(calls) == 1:
            fixture.modules["scipy.another_alias"] = fixture.modules["scipy.fixture"]
        return observed

    monkeypatch.setattr(external, "_observed_artifacts", changed)
    with pytest.raises(_CacheIdentityError, match="during identification"):
        fixture.process.get(fixture.pools)
    assert calls == ["observed", "observed"]
    assert fixture.process.identity == old_identity


def _start_thread(call, results, errors):
    def run():
        try:
            results.append(call())
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    return thread


@pytest.mark.parametrize("second", ["get", "invalidate", "mutating_import"])
def test_concurrent_growth_is_serialized_and_invalidation_is_sticky(native_fixture, monkeypatch, second):
    fixture = native_fixture
    fixture.process.get(fixture.pools)
    fixture.add("scipy.io.fixture")
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    original = external._file_digest
    reads = []

    def blocked(path):
        reads.append(str(path))
        entered.set()
        assert release.wait(10), "test owner did not release the deterministic hash barrier"
        return original(path)

    monkeypatch.setattr(external, "_file_digest", blocked)
    results, errors = [], []
    first = _start_thread(lambda: fixture.process.get(fixture.pools), results, errors)
    second_thread = None
    try:
        assert entered.wait(10), "first observer did not reach hash barrier"

        def other():
            second_started.set()
            if second == "invalidate":
                fixture.process.invalidate("explicit invalidation; restart required")
                return "invalidated"
            return fixture.process.get(fixture.pools)

        if second == "mutating_import":
            fixture.add("scipy.io.concurrent")
        second_thread = _start_thread(other, results, errors)
        assert second_started.wait(10)
    finally:
        release.set()
        first.join(10)
        if second_thread is not None:
            second_thread.join(10)
    assert not first.is_alive() and second_thread is not None and not second_thread.is_alive()
    assert len(reads) == 1
    if second == "get":
        assert not errors and len(results) == 2 and results[0] == results[1]
        assert fixture.process.failure is None
    elif second == "invalidate":
        assert not errors and len(results) == 2 and "invalidated" in results
        assert fixture.process.failure == "explicit invalidation; restart required"
        with pytest.raises(_CacheIdentityError, match="explicit invalidation"):
            fixture.process.get(fixture.pools)
    else:
        assert not results and len(errors) == 2
        assert all(isinstance(error, _CacheIdentityError) for error in errors)
        assert fixture.process.failure is not None


@pytest.mark.parametrize("field,value", [("num_threads", 2), ("version", "two"),
                                         ("architecture", "other"), ("threading_layer", "other")])
def test_pool_metadata_still_changes_full_provenance(native_fixture, monkeypatch, field, value):
    fixture = native_fixture
    monkeypatch.setattr(external, "_EXTERNAL_IDENTITY", fixture.process)
    monkeypatch.setattr(_cache, "_initialize_numerical_runtime", lambda: None)
    monkeypatch.setattr(_cache, "_package_identity", lambda: {"fixture": "same"})
    monkeypatch.setattr(_cache, "threadpool_info", lambda: fixture.pools)
    before = _cache._runtime_provenance(_probe)
    fixture.pools[0][field] = value
    after = _cache._runtime_provenance(_probe)
    # These fields already live in the enclosing provenance, not artifact roles.
    assert before["external_native"] == after["external_native"]
    assert before != after and before["pools"] != after["pools"]
    assert fixture.process.failure is None


@pytest.mark.parametrize("boundary", ["under_lock", "after_load", "after_compute", "before_publication"])
def test_inflight_old_identity_refuses_reuse_or_publication_without_repeat(native_fixture, monkeypatch, tmp_path, boundary):
    fixture = native_fixture
    cache = _cache._cache_path(tmp_path / "cache")
    observe = lambda: {"native": fixture.process.get(fixture.pools)}
    original_provenance = observe()
    old_key_data = {"provenance": original_provenance}
    if boundary in ("under_lock", "after_load"):
        value, hit, old_key = _cache._cached_base_stage(cache, old_key_data, lambda: {"value": "old"},
                            verify=_cache._RuntimeVerifier(observe, original_provenance))
        assert value == {"value": "old"} and not hit and old_key is not None
    computations = []

    def add():
        fixture.add("scipy.io.fixture")

    def compute():
        computations.append("computed")
        if boundary == "after_compute":
            add()
        return {"value": "new"}

    with monkeypatch.context() as scoped:
        if boundary == "under_lock":
            original = _cache._key_lock

            @contextmanager
            def changed_lock(path):
                with original(path):
                    add()
                    yield

            scoped.setattr(_cache, "_key_lock", changed_lock)
        elif boundary == "after_load":
            original = _cache.joblib.load

            def changed_load(*args, **kwargs):
                value = original(*args, **kwargs)
                add()
                return value

            scoped.setattr(_cache.joblib, "load", changed_load)
        elif boundary == "before_publication":
            original = _cache.joblib.dump

            def changed_dump(*args, **kwargs):
                value = original(*args, **kwargs)
                add()
                return value

            scoped.setattr(_cache.joblib, "dump", changed_dump)
        verifier = _cache._RuntimeVerifier(observe, original_provenance)
        value, hit, key = _cache._cached_base_stage(cache, old_key_data, compute, verify=verifier)
    assert value == {"value": "new"} and computations == ["computed"]
    assert not hit and key is None and verifier.failure is not None
    assert fixture.process.failure is None  # Old operation is stale; new calls are admissible.
    assert not list(cache.rglob("*.tmp"))
    if boundary in ("after_compute", "before_publication"):
        assert not list(cache.rglob("*.joblib")) and not list(cache.rglob("*.json"))
    current = observe()
    assert current != original_provenance
    next_computations = []

    def next_call():
        next_computations.append("new invocation")
        return {"value": "next"}

    new_key_data = {"provenance": current}
    value, hit, key = _cache._cached_base_stage(cache, new_key_data, next_call,
                        verify=_cache._RuntimeVerifier(observe, current))
    assert value == {"value": "next"} and not hit and key is not None
    value, hit, repeated_key = _cache._cached_base_stage(cache, new_key_data, next_call,
                        verify=_cache._RuntimeVerifier(observe, current))
    assert hit and key == repeated_key and next_computations == ["new invocation"]
