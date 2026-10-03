"""Private, process-safe storage for deterministic base-stage results."""

from contextlib import contextmanager
import errno
from functools import lru_cache
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import joblib
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info

import dwd

from ._cache_identity import _CacheIdentityError, _package_identity
from ._external_identity import _external_native_identity, _invalidate_external_identity


_CACHE_FORMAT = 2
_CACHE_DIRECTORY = "base-stage-v2"


def _cache_path(path):
    """Normalize cache I/O without imposing the legacy Windows MAX_PATH limit."""
    absolute = os.path.abspath(os.fspath(Path(path).expanduser()))
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = ("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\")
                    else "\\\\?\\" + absolute)
    return Path(absolute)


@contextmanager
def _key_lock(path):
    # Operating-system locks are released even if a worker exits unexpectedly.
    with open(path, "a+b") as stream:
        # Both Windows byte-range locking and POSIX flock support an empty
        # file. Never initialize its contents before acquiring the lock: a
        # concurrent owner may already hold byte zero and deny that write.
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _factory_fingerprint(factory):
    try:
        factory_source = inspect.getsource(factory)
    except (OSError, TypeError):
        factory_source = repr(factory)
    return hashlib.sha256(factory_source.encode()).hexdigest()


@lru_cache(maxsize=1)
def _initialize_numerical_runtime():
    # Some BLAS builds load their OpenMP runtime on first use. Initialize the
    # NumPy and SciPy BLAS libraries before recording the pool inventory so a
    # fresh worker and an already used worker describe the same configuration.
    from scipy.linalg.blas import dgemm
    # Optimized DWD imports scipy.optimize lazily, even for MM. Identify its
    # native modules before a cold base stage can add them to the inventory.
    importlib.import_module("scipy.optimize")
    identity = np.eye(2, dtype=np.float64)
    np.dot(identity, identity)
    dgemm(1.0, identity, identity)


def _runtime_provenance(factory):
    _initialize_numerical_runtime()
    try:
        observed_pools = threadpool_info()
    except (OSError, ValueError) as error:
        reason = "loaded numerical library discovery failed; restart required"
        _invalidate_external_identity(reason)
        raise _CacheIdentityError(reason) from error
    external_native = _external_native_identity(observed_pools)
    pools = [{key: item.get(key) for key in (
        "user_api", "internal_api", "prefix", "version", "num_threads",
        "threading_layer", "architecture")} for item in observed_pools]
    pools.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return dict(format=_CACHE_FORMAT, source=_factory_fingerprint(factory),
                package_identity=_package_identity(), external_native=external_native,
                python=platform.python_version(), platform=platform.platform(),
                python_implementation=platform.python_implementation(),
                machine=platform.machine(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__,
                dwd=dwd.__version__, pools=pools)


class _RuntimeVerifier:
    """Carry an explicit cache-bypass reason without repeating completed fits."""
    def __init__(self, observe, provenance):
        self.observe, self.provenance, self.failure = observe, provenance, None

    def __call__(self):
        if self.failure is not None:
            return False
        try:
            if self.observe() != self.provenance:
                raise _CacheIdentityError("runtime provenance changed before cache publication")
        except _CacheIdentityError as error:
            self.failure = str(error)
            return False
        return True


def _cached_base_stage(cache_dir, key_data, compute, *, verify=None):
    """Return (result, cache_hit, key), computing once per concurrently used key.

    Files are pickle based and must be kept in a trusted, private directory.
    The checksum detects accidental truncation/corruption, not malicious edits.
    Neither failed computations nor partially written files become cache hits.
    """
    directory = _cache_path(cache_dir) / _CACHE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    key = joblib.hash(key_data, hash_name="sha1")
    payload = directory / (key + ".joblib")
    metadata = directory / (key + ".json")
    with _key_lock(directory / (key + ".lock")):
        if verify is not None and not verify():
            return compute(), False, None
        if payload.is_file() and metadata.is_file():
            try:
                record = json.loads(metadata.read_text(encoding="utf-8"))
                valid = (isinstance(record, dict) and record.get("key") == key and record.get("format") == _CACHE_FORMAT
                         and record.get("sha256") == _sha256(payload))
            except (ValueError, UnicodeError):
                valid = False
            if valid:
                # A matching checksum makes deserialization failures visible;
                # incompatible runtime/code has a different key already.
                result = joblib.load(payload)
                if verify is not None and not verify():
                    return compute(), False, None
                return result, True, key
        result = compute()
        if verify is not None and not verify():
            return result, False, None
        temporary = []
        try:
            for suffix in (".joblib.tmp", ".json.tmp"):
                handle, name = tempfile.mkstemp(prefix=key + ".", suffix=suffix, dir=directory)
                os.close(handle)
                temporary.append(Path(name))
            joblib.dump(result, temporary[0], compress=0)
            record = dict(key=key, format=_CACHE_FORMAT, sha256=_sha256(temporary[0]))
            temporary[1].write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            if verify is not None and not verify():
                return result, False, None
            os.replace(temporary[0], payload)
            os.replace(temporary[1], metadata)
        finally:
            for path in temporary:
                if path.exists():
                    path.unlink()
        return result, False, key
