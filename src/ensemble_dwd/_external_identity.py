"""Bounded disk identity of observed native numerical dependencies.

The first cached fit initializes known BLAS backends before this snapshot.
Installations must be immutable from import: this identifies disk artifacts,
not previously mapped machine code or arbitrary same-version Python edits.
"""
import importlib.machinery
from numbers import Integral
import os
from pathlib import Path
import sys
import threading

from ._cache_identity import _CacheIdentityError, _file_digest, _is_native, _stamp


_NUMERICAL_PACKAGES = ("numpy", "scipy", "sklearn")


def _observed_artifacts(pools):
    """Inspect loaded native modules and a small optional bridge package only."""
    artifacts, roles, modules = {}, {}, {}

    def add(origin, role):
        if not isinstance(origin, (str, os.PathLike)) or not origin:
            raise _CacheIdentityError("loaded native artifact origin is unavailable")
        path = Path(origin).absolute()
        if not path.is_file():
            raise _CacheIdentityError("loaded native artifact is unavailable: " + str(path))
        key = str(path)
        artifacts[key] = (path, _stamp(path))
        roles.setdefault(key, set()).add(role)
        return key

    if not isinstance(pools, (list, tuple)) or not any(
            isinstance(pool, dict) and pool.get("user_api") == "blas" for pool in pools):
        raise _CacheIdentityError("loaded BLAS identity is unavailable")
    for pool in pools:
        if not isinstance(pool, dict):
            raise _CacheIdentityError("loaded numerical pool identity is unavailable")
        threads = pool.get("num_threads")
        if (pool.get("user_api") not in ("blas", "openmp") or isinstance(threads, bool)
                or not isinstance(threads, Integral) or threads < 1):
            raise _CacheIdentityError("loaded numerical pool thread identity is unavailable")
        role = "pool:" + ":".join(str(pool.get(name)) for name in ("user_api", "internal_api", "prefix"))
        add(pool.get("filepath"), role)

    observed = {name: 0 for name in _NUMERICAL_PACKAGES}
    for name, module in tuple(sys.modules.items()):
        package = name.partition(".")[0]
        if package not in observed or module is None:
            continue
        origin = getattr(module, "__file__", None)
        loader = getattr(getattr(module, "__spec__", None), "loader", None)
        native = (isinstance(origin, (str, os.PathLike)) and _is_native(Path(origin)))
        if native or isinstance(loader, importlib.machinery.ExtensionFileLoader):
            key = add(origin, "module:" + name)
            modules[name] = (module, key)  # Private identity guard, never serialized.
            observed[package] += 1
    if not all(observed.values()):
        missing = ", ".join(name for name, count in observed.items() if not count)
        raise _CacheIdentityError("loaded native module identity is unavailable for " + missing)

    bridge = sys.modules.get("aocl_runtime")
    if bridge is not None:
        origin = getattr(bridge, "__file__", None)
        key = add(origin, "bridge:aocl_runtime")
        modules["aocl_runtime"] = (bridge, key)
        root = Path(origin).absolute().parent

        def failed(error):
            raise error

        for folder, directories, filenames in os.walk(root, onerror=failed):
            directories[:] = sorted(name for name in directories if name != "__pycache__")
            for filename in sorted(filenames):
                path = Path(folder) / filename
                if path.suffix.lower() in (".py", ".json") or _is_native(path):
                    add(path, "bridge:aocl_runtime/" + path.relative_to(root).as_posix())
    return artifacts, {path: tuple(sorted(values)) for path, values in roles.items()}, modules


class _ExternalNativeIdentity:
    """Hash observed files once; observable changes disable reuse until restart."""
    def __init__(self):
        self.artifacts = self.roles = self.modules = self.identity = None
        self.failure = None
        self.lock = threading.RLock()

    def invalidate(self, reason):
        with self.lock:
            if self.failure is None:
                self.failure = reason

    def get(self, pools):
        with self.lock:
            if self.failure is not None:
                raise _CacheIdentityError(self.failure)
            try:
                artifacts, roles, modules = _observed_artifacts(pools)
                if self.artifacts is None:
                    digests = {path: _file_digest(value[0]) for path, value in artifacts.items()}
                    if _observed_artifacts(pools) != (artifacts, roles, modules):
                        raise _CacheIdentityError("external native artifacts changed during identification; restart required")
                    self.artifacts, self.roles, self.modules = artifacts, roles, modules
                    self.identity = tuple((path, digests[path], roles[path]) for path in sorted(artifacts))
                elif (artifacts, roles, modules) != (self.artifacts, self.roles, self.modules):
                    raise _CacheIdentityError("external native artifacts or loaded inventory changed; restart required")
                # Fresh plain containers prevent consumers mutating the saved snapshot.
                return {"artifacts": [{"path": path, "sha256": digest, "roles": list(roles)}
                                      for path, digest, roles in self.identity]}
            except (OSError, ValueError, TypeError, AttributeError, _CacheIdentityError) as error:
                self.failure = (str(error) if isinstance(error, _CacheIdentityError)
                                else "external native identity is unreadable or unknown; restart required")
                raise _CacheIdentityError(self.failure) from error


_EXTERNAL_IDENTITY = _ExternalNativeIdentity()


def _external_native_identity(pools):
    return _EXTERNAL_IDENTITY.get(pools)


def _invalidate_external_identity(reason):
    _EXTERNAL_IDENTITY.invalidate(reason)
