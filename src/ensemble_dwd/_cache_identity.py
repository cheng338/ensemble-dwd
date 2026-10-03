"""Bounded package-artifact identity for an immutable process installation.

Metadata is anchored when this ensemble cache module imports. Hashes describe
package files unchanged since that anchor, not machine code in memory or code
loaded before this module imported. Observable changes disable persistent
caching for the rest of this process. Fresh processes hash artifacts again. Stat
checks are not protection against deliberate timestamp-preserving edits.
Observed external native dependencies are identified separately at cache use.
"""

import hashlib
import importlib
import json
import os
from pathlib import Path
import threading
import types

import dwd


class _CacheIdentityError(RuntimeError):
    """Persistent reuse is unsafe; the estimator may still fit without it."""


def _is_native(path):
    name = path.name.lower()
    return (path.suffix.lower() in (".pyd", ".dll", ".dylib", ".so")
            or ".so." in name)


def _stamp(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _artifact_files(roots):
    def failed(error):
        raise error

    result = {}
    for name, root in sorted(roots.items()):
        if not root.is_dir():
            raise _CacheIdentityError("package artifact directory is unavailable")
        for folder, directories, filenames in os.walk(root, onerror=failed):
            directories[:] = sorted(item for item in directories if item != "__pycache__")
            for filename in sorted(filenames):
                path = Path(folder) / filename
                if path.suffix.lower() in (".py", ".c", ".h") or _is_native(path):
                    result[name + "/" + path.relative_to(root).as_posix()] = (path, _stamp(path))
        if name + "/__init__.py" not in result:
            raise _CacheIdentityError("package Python source identity is unavailable")
    return result


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _ArtifactSnapshot:
    """Hash once; detect observable replacement, edits, additions and removals."""

    def __init__(self, roots, *, anchor=False):
        self.roots = {name: Path(root).resolve() for name, root in roots.items()}
        self.files = None
        self.digests = None
        self.digest = None
        self.failure = None
        self.anchor_files = None
        if anchor:
            try:
                self.anchor_files = _artifact_files(self.roots)
            except (OSError, _CacheIdentityError) as error:
                self.failure = (str(error) if isinstance(error, _CacheIdentityError)
                                else "package artifacts are unreadable; restart required")

    def check(self):
        if self.failure is not None:
            raise _CacheIdentityError(self.failure)
        try:
            files = _artifact_files(self.roots)
            if self.files is None:
                if self.anchor_files is not None and files != self.anchor_files:
                    raise _CacheIdentityError("package artifacts changed since cache module import; restart required")
                digests = {name: _file_digest(path) for name, (path, _) in files.items()}
                if _artifact_files(self.roots) != files:
                    raise _CacheIdentityError("package artifacts changed during identification; restart required")
                self.files, self.digests = files, digests
                self.digest = hashlib.sha256(json.dumps(
                    digests, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            elif files != self.files:
                raise _CacheIdentityError("package artifacts changed in this process; restart required")
            return self.digest
        except (OSError, _CacheIdentityError) as error:
            self.failure = (str(error) if isinstance(error, _CacheIdentityError)
                            else "package artifacts are unreadable; restart required")
            raise _CacheIdentityError(self.failure) from error

    def native_digests(self):
        return {name: self.digests[name] for name, (path, _) in self.files.items()
                if _is_native(path)}


def _accelerator_state(snapshot):
    compiled = importlib.import_module("dwd._compiled_residual")
    native = importlib.import_module("dwd._native_residual")
    accelerator = compiled._ACCEL
    identity = {"available": accelerator is not None,
                "native_screen_supported": bool(native._native_supported()),
                "artifact": None, "sha256": None}
    if accelerator is not None:
        if not isinstance(accelerator, types.ModuleType) or not callable(getattr(accelerator, "evaluate", None)):
            raise _CacheIdentityError("loaded accelerator identity is unavailable")
        origin = getattr(accelerator, "__file__", None)
        if not isinstance(origin, (str, os.PathLike)) or not origin:
            raise _CacheIdentityError("loaded accelerator origin is unavailable")
        origin = Path(origin).resolve(strict=True)
        matches = [name for name, (path, _) in snapshot.files.items()
                   if _is_native(path) and path.resolve() == origin]
        if len(matches) != 1:
            raise _CacheIdentityError("loaded accelerator is outside the identified package artifacts")
        identity.update(artifact=matches[0], sha256=snapshot.digests[matches[0]])
    # Object identity is only an in-process guard; never put it in a cache key.
    return identity, id(accelerator)


def _package_roots():
    if not isinstance(getattr(dwd, "__file__", None), (str, os.PathLike)):
        raise _CacheIdentityError("DWD package origin is unavailable")
    return {"ensemble_dwd": Path(__file__).resolve().parent,
            "dwd": Path(dwd.__file__).resolve().parent}


class _ProcessIdentity:
    def __init__(self, *, anchor=False):
        self.snapshot = None
        self.accelerator = None
        self.failure = None
        self.lock = threading.RLock()
        if anchor:
            try:
                self.snapshot = _ArtifactSnapshot(_package_roots(), anchor=True)
            except (OSError, _CacheIdentityError) as error:
                self.failure = (str(error) if isinstance(error, _CacheIdentityError)
                                else "package artifacts are unreadable; restart required")

    def get(self):
        with self.lock:
            if self.failure is not None:
                raise _CacheIdentityError(self.failure)
            try:
                # Import the optional implementation before capturing its files.
                # None is a known portable fallback, not an unknown identity.
                importlib.import_module("dwd._compiled_residual")
                roots = _package_roots()
                if self.snapshot is None:
                    self.snapshot = _ArtifactSnapshot(roots)
                elif self.snapshot.roots != roots:
                    raise _CacheIdentityError("package origins changed in this process; restart required")
                digest = self.snapshot.check()
                accelerator = _accelerator_state(self.snapshot)
                if self.accelerator is None:
                    self.accelerator = accelerator
                elif accelerator != self.accelerator:
                    raise _CacheIdentityError("accelerator changed in this process; restart required")
                return {"artifacts_sha256": digest,
                        "native_artifacts": self.snapshot.native_digests(),
                        "accelerator": dict(accelerator[0])}
            except (OSError, ImportError, AttributeError, _CacheIdentityError) as error:
                self.failure = (str(error) if isinstance(error, _CacheIdentityError)
                                else "native runtime identity is unavailable; restart required")
                raise _CacheIdentityError(self.failure) from error


# Cheap metadata only: importing the cache does not load the optional native
# helper or read binary contents for users who leave caching disabled.
_PROCESS_IDENTITY = _ProcessIdentity(anchor=True)


def _package_identity():
    return _PROCESS_IDENTITY.get()
