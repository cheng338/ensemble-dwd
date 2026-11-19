"""Bounded disk identity of observed native numerical dependencies.

The first cached fit initializes known BLAS backends before this snapshot.
Installations must be immutable from import: this identifies disk artifacts,
not previously mapped machine code or arbitrary same-version Python edits.
"""
import importlib.machinery
from numbers import Integral
import os
import stat
from pathlib import Path
import sys
import threading

from ._cache_identity import _CacheIdentityError, _file_digest, _is_native


_NUMERICAL_PACKAGES = ("numpy", "scipy", "sklearn")


def _regular_artifact_stamp(path):
    """Reuse one live successful stat for classification and its identity stamp.

    No observation is retained across provenance boundaries. Installations must
    remain immutable: this removes the adjacent second successful stat and does
    not preserve every possible concurrent-mutation detection cutpoint.
    """
    try:
        info = path.stat()
    except (OSError, ValueError):
        # Keep the interpreter/platform's public Path.is_file error policy on
        # the exceptional path, rather than copying private errno tables.
        if not path.is_file():
            raise _CacheIdentityError("loaded native artifact is unavailable: " + str(path))
        raise
    if not stat.S_ISREG(info.st_mode):
        raise _CacheIdentityError("loaded native artifact is unavailable: " + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _observed_artifacts(pools):
    """Inspect loaded native modules and a small optional bridge package only."""
    artifacts, roles, modules = {}, {}, {}

    def add(origin, role):
        if not isinstance(origin, (str, os.PathLike)) or not origin:
            raise _CacheIdentityError("loaded native artifact origin is unavailable")
        path = Path(origin).absolute()
        stamp = _regular_artifact_stamp(path)
        key = str(path)
        artifacts[key] = (path, stamp)
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
        # Ordinary source/bytecode origins cannot be native. Preserve the full
        # path route for custom PathLike/string subclasses, extension loaders,
        # and the existing versioned-ELF '.so.' recognition (even before .py).
        if type(origin) is str and not isinstance(loader, importlib.machinery.ExtensionFileLoader):
            spelling = origin.lower()
            if spelling.endswith((".py", ".pyc")) and ".so." not in spelling:
                continue
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


def _module_only_extension(previous, current):
    """Admit new numerical modules without forgetting any earlier binding."""
    old_artifacts, old_roles, old_modules = previous
    artifacts, roles, modules = current
    if not modules.keys() > old_modules.keys():
        return False
    if (any(artifacts.get(path) != value for path, value in old_artifacts.items())
            or any(modules.get(name) != value for name, value in old_modules.items())
            or any(not set(values).issubset(roles.get(path, ()))
                   for path, values in old_roles.items())):
        return False

    def other_roles(observed_roles):
        return {path: other for path, values in observed_roles.items()
                if (other := tuple(role for role in values if not role.startswith("module:")))}

    # Added pool/bridge paths or roles are not module-only changes. In
    # particular, first loading the optional bridge still requires a restart.
    return other_roles(roles) == other_roles(old_roles)


class _ExternalNativeIdentity:
    """Hash files once; allow verified module additions, never replace anchors."""
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
                    if not _module_only_extension(
                            (self.artifacts, self.roles, self.modules), (artifacts, roles, modules)):
                        raise _CacheIdentityError("external native artifacts or loaded inventory changed; restart required")
                    # A lazy import changes the full identity and therefore the
                    # cache key. Keep every previously verified digest; newly
                    # observed files must pass the same double observation as
                    # the initial inventory before any anchor is extended.
                    digests = {path: digest for path, digest, _ in self.identity}
                    for path in sorted(artifacts.keys() - self.artifacts.keys()):
                        digests[path] = _file_digest(artifacts[path][0])
                    if _observed_artifacts(pools) != (artifacts, roles, modules):
                        raise _CacheIdentityError("external native artifacts changed during identification; restart required")
                    identity = tuple((path, digests[path], roles[path]) for path in sorted(artifacts))
                    self.artifacts, self.roles, self.modules, self.identity = artifacts, roles, modules, identity
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
