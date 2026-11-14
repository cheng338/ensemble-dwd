"""Native cache observations; all artifact bytes are synthetic and never loaded.

Installations are required to remain immutable. These tests cover observable
changes between provenance boundaries, not every concurrent filesystem race.
"""

from contextlib import contextmanager
import errno
import hashlib
import importlib.machinery
import os
from pathlib import Path
import types

import pytest

from ensemble_dwd import _cache
from ensemble_dwd import _external_identity as external
from ensemble_dwd._cache_identity import _CacheIdentityError


@pytest.fixture
def native_files(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    modules = {}
    for package in ("numpy", "scipy", "sklearn"):
        path = root / (package + "_fixture.pyd")
        path.write_bytes((package + " initial fixture bytes").encode())
        module = types.ModuleType(package + ".fixture")
        module.__file__ = str(path)
        modules[module.__name__] = module
    pool_path = root / "libbackend.so.1"
    pool_path.write_bytes(b"initial native pool fixture")
    pools = [{"filepath": str(pool_path), "user_api": "blas", "internal_api": "fixture",
              "prefix": "fixture", "version": "constant", "num_threads": 1}]
    monkeypatch.setattr(external, "sys", types.SimpleNamespace(modules=modules))
    return types.SimpleNamespace(root=root, modules=modules, pools=pools, pool=pool_path)


def stamp(path):
    info = os.stat(path)
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def replace_artifact(path, suffix):
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_bytes(("replacement-" + suffix).encode())
    os.replace(replacement, path)


def add_module(native_files, name, origin, *, loader=None):
    module = types.ModuleType(name)
    module.__file__ = origin
    if loader is not None:
        module.__spec__ = types.SimpleNamespace(loader=loader)
    native_files.modules[name] = module
    return module


def test_records_have_exact_paths_stamps_roles_modules_and_content(native_files):
    pool = native_files.pool.absolute()
    expected_files = {str(pool): (pool, stamp(pool))}
    expected_roles = {str(pool): ("pool:blas:fixture:fixture",)}
    expected_modules = {}
    for name, module in native_files.modules.items():
        path = Path(module.__file__).absolute()
        key = str(path)
        expected_files[key] = (path, stamp(path))
        expected_roles[key] = ("module:" + name,)
        expected_modules[name] = (module, key)
    assert external._observed_artifacts(native_files.pools) == (
        expected_files, expected_roles, expected_modules)
    expected_identity = {"artifacts": [
        {"path": path, "sha256": hashlib.sha256(value[0].read_bytes()).hexdigest(),
         "roles": list(expected_roles[path])}
        for path, value in sorted(expected_files.items())
    ]}
    process = external._ExternalNativeIdentity()
    assert process.get(native_files.pools) == expected_identity
    assert process.get(native_files.pools) == expected_identity


@pytest.mark.parametrize("filename,native", [
    ("plain.py", False), ("plain.pyc", False), ("plain.PY", False),
    ("plain.PYC", False), (".py", False), (".so", False),
    ("extension.pyd", True), ("extension.DLL", True), ("libextension.dylib", True),
    ("extension.cpython-312-x86_64-linux-gnu.so", True), ("libextension.so.3", True),
    ("strange.so.fixture.py", True), ("strange.SO.fixture.PYC", True),
    ("notes.software.py", False),
])
def test_suffix_and_versioned_elf_recognition(native_files, filename, native):
    path = native_files.root / filename
    path.write_bytes(b"synthetic classifier fixture")
    module = add_module(native_files, "numpy.suffix_fixture", str(path))
    artifacts, roles, modules = external._observed_artifacts(native_files.pools)
    key = str(path.absolute())
    assert (module.__name__ in modules) is native
    assert (key in artifacts) is native
    if native:
        assert artifacts[key] == (path.absolute(), stamp(path))
        assert roles[key] == ("module:numpy.suffix_fixture",)
        assert modules[module.__name__] == (module, key)


def test_dot_so_in_directory_does_not_make_python_file_native(native_files):
    directory = native_files.root / "package.so.directory"
    directory.mkdir()
    path = directory / "pure.py"
    path.write_bytes(b"source fixture")
    module = add_module(native_files, "numpy.directory_fixture", str(path))
    artifacts, _, modules = external._observed_artifacts(native_files.pools)
    assert module.__name__ not in modules
    assert str(path.absolute()) not in artifacts


@pytest.mark.parametrize("filename", ("loader.py", "loader.pyc", "loader_without_suffix"))
def test_extension_loader_overrides_non_native_suffix(native_files, filename):
    path = native_files.root / filename
    path.write_bytes(b"not executed extension-loader fixture")
    name = "numpy.loader_fixture"
    loader = importlib.machinery.ExtensionFileLoader(name, str(path))
    module = add_module(native_files, name, str(path), loader=loader)
    artifacts, roles, modules = external._observed_artifacts(native_files.pools)
    key = str(path.absolute())
    assert artifacts[key] == (path.absolute(), stamp(path))
    assert roles[key] == ("module:" + name,)
    assert modules[name] == (module, key)


class PathOrigin:
    def __init__(self, path):
        self.path = str(path)

    def __fspath__(self):
        return self.path


class StringOrigin(str):
    def lower(self):
        raise AssertionError("A string subclass must retain the normal path conversion")


@pytest.mark.parametrize("filename,native", (("pathlike.py", False), ("pathlike.so.1", True)))
def test_custom_pathlike_origin_is_converted_normally(native_files, filename, native):
    path = native_files.root / filename
    path.write_bytes(b"pathlike fixture")
    module = add_module(native_files, "numpy.pathlike_fixture", PathOrigin(path))
    artifacts, _, modules = external._observed_artifacts(native_files.pools)
    key = str(path.absolute())
    assert (module.__name__ in modules) is native
    assert (key in artifacts) is native
    if native:
        assert artifacts[key] == (path.absolute(), stamp(path))
        assert modules[module.__name__] == (module, key)


def test_string_subclass_keeps_normal_path_conversion(native_files):
    path = native_files.root / "subclass.py"
    path.write_bytes(b"source fixture")
    module = add_module(native_files, "numpy.subclass_fixture", StringOrigin(str(path)))
    artifacts, _, modules = external._observed_artifacts(native_files.pools)
    assert module.__name__ not in modules
    assert str(path.absolute()) not in artifacts


@pytest.mark.parametrize("failure", ("bytes", "value", "runtime"))
def test_malformed_pathlike_is_not_shortcut_or_silenced(native_files, failure):
    class BrokenPath:
        def __fspath__(self):
            if failure == "bytes":
                return b"not-an-accepted-path.py"
            if failure == "value":
                raise ValueError("pathlike failure")
            raise RuntimeError("custom pathlike failure")
    add_module(native_files, "numpy.broken_path_fixture", BrokenPath())
    process = external._ExternalNativeIdentity()
    if failure == "runtime":
        with pytest.raises(RuntimeError, match="custom pathlike failure"):
            process.get(native_files.pools)
    else:
        with pytest.raises(_CacheIdentityError, match="unreadable or unknown"):
            process.get(native_files.pools)


@pytest.mark.parametrize("bad_value", (None, 42, b"bytes.py"))
def test_extension_loader_invalid_origin_refuses(native_files, bad_value):
    name = "numpy.invalid_loader_fixture"
    loader = importlib.machinery.ExtensionFileLoader(name, "unused")
    add_module(native_files, name, bad_value, loader=loader)
    with pytest.raises(_CacheIdentityError, match="origin is unavailable"):
        external._ExternalNativeIdentity().get(native_files.pools)


@pytest.mark.parametrize("kind", ("missing", "directory", "permission", "loop", "invalid_value"))
def test_unavailable_or_nonregular_artifacts_refuse(native_files, monkeypatch, kind):
    target = native_files.pool
    if kind == "missing":
        target.unlink()
    elif kind == "directory":
        target.unlink()
        target.mkdir()
    else:
        original_stat = Path.stat

        def failed_stat(self, *args, **kwargs):
            if self == target:
                if kind == "permission":
                    raise PermissionError(errno.EACCES, "denied")
                if kind == "loop":
                    raise OSError(errno.ELOOP, "loop")
                raise ValueError("invalid synthetic path")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", failed_stat)
    with pytest.raises(_CacheIdentityError):
        external._ExternalNativeIdentity().get(native_files.pools)


def test_transient_first_stat_failure_cannot_enable_reuse(native_files, monkeypatch):
    original_stat = Path.stat
    failed = False

    def missing_then_present(self, *args, **kwargs):
        nonlocal failed
        if self == native_files.pool and not failed:
            failed = True
            raise FileNotFoundError(errno.ENOENT, "transient first observation")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", missing_then_present)
    process = external._ExternalNativeIdentity()
    with pytest.raises(_CacheIdentityError):
        process.get(native_files.pools)
    # A later successful observation does not retroactively qualify the failure.
    assert original_stat(native_files.pool).st_size > 0
    with pytest.raises(_CacheIdentityError):
        process.get(native_files.pools)


def test_replacement_between_observations_is_sticky(native_files):
    process = external._ExternalNativeIdentity()
    before = native_files.pool.read_bytes()
    process.get(native_files.pools)
    replace_artifact(native_files.pool, "between-observations")
    with pytest.raises(_CacheIdentityError, match="changed"):
        process.get(native_files.pools)
    native_files.pool.write_bytes(before)
    with pytest.raises(_CacheIdentityError, match="changed"):
        process.get(native_files.pools)


def test_replacement_during_initial_hashing_refuses(native_files, monkeypatch):
    read = external._file_digest

    def mutate(path):
        value = read(path)
        if path == native_files.pool:
            replace_artifact(path, "during-initial-hashing")
        return value

    monkeypatch.setattr(external, "_file_digest", mutate)
    with pytest.raises(_CacheIdentityError, match="during identification"):
        external._ExternalNativeIdentity().get(native_files.pools)


@pytest.mark.parametrize("change", ("object", "addition", "origin_to_python", "pool_role"))
def test_native_inventory_and_object_changes_remain_visible(native_files, change):
    process = external._ExternalNativeIdentity()
    process.get(native_files.pools)
    old = native_files.modules["numpy.fixture"]
    if change == "object":
        add_module(native_files, old.__name__, old.__file__)
    elif change == "addition":
        native_files.modules["numpy.added_native"] = old
    elif change == "pool_role":
        native_files.pools[0]["prefix"] = "changed_role"
    else:
        old.__file__ = str(native_files.root / "now_python.py")
    with pytest.raises(_CacheIdentityError):
        process.get(native_files.pools)


@pytest.mark.parametrize("boundary", ("under_lock", "after_load", "after_compute", "before_publication"))
def test_replacement_at_cache_boundaries_never_reuses_or_repeats(
        native_files, monkeypatch, tmp_path, boundary):
    process = external._ExternalNativeIdentity()
    provenance = process.get(native_files.pools)

    def observe():
        return process.get(native_files.pools)

    cache_dir = tmp_path / "cache"
    key_data = {"synthetic_work": "identity-boundary", "provenance": provenance}
    if boundary in ("under_lock", "after_load"):
        _, hit, key = _cache._cached_base_stage(
            cache_dir, key_data, lambda: {"value": "old"},
            verify=_cache._RuntimeVerifier(observe, provenance))
        assert not hit and key is not None
    if boundary == "under_lock":
        old_lock = _cache._key_lock

        @contextmanager
        def changed_lock(path):
            with old_lock(path):
                replace_artifact(native_files.pool, boundary)
                yield

        monkeypatch.setattr(_cache, "_key_lock", changed_lock)
    elif boundary == "after_load":
        old_load = _cache.joblib.load

        def changed_load(*args, **kwargs):
            value = old_load(*args, **kwargs)
            replace_artifact(native_files.pool, boundary)
            return value

        monkeypatch.setattr(_cache.joblib, "load", changed_load)
    elif boundary == "before_publication":
        old_dump = _cache.joblib.dump

        def changed_dump(*args, **kwargs):
            value = old_dump(*args, **kwargs)
            replace_artifact(native_files.pool, boundary)
            return value

        monkeypatch.setattr(_cache.joblib, "dump", changed_dump)
    computations = []

    def compute():
        computations.append("computed")
        if boundary == "after_compute":
            replace_artifact(native_files.pool, boundary)
        return {"value": "new"}

    verifier = _cache._RuntimeVerifier(observe, provenance)
    value, hit, key = _cache._cached_base_stage(cache_dir, key_data, compute, verify=verifier)
    assert value == {"value": "new"}
    # This is an exactly-once work contract, not a timing/call-count optimization.
    assert computations == ["computed"]
    assert hit is False and key is None and verifier.failure is not None
    if boundary in ("after_compute", "before_publication"):
        assert not list(cache_dir.rglob("*.joblib"))
        assert not list(cache_dir.rglob("*.json"))
    assert not list(cache_dir.rglob("*.tmp"))
