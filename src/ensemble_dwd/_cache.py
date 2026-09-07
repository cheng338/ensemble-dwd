"""Private, process-safe storage for deterministic base-stage results."""

from contextlib import contextmanager
import errno
from functools import lru_cache
import hashlib
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


@contextmanager
def _key_lock(path):
    # Operating-system locks are released even if a worker exits unexpectedly.
    with open(path, "a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
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
def _source_fingerprint(factory):
    digest = hashlib.sha256()
    for root in (Path(__file__).parent, Path(dwd.__file__).parent):
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    try:
        factory_source = inspect.getsource(factory)
    except (OSError, TypeError):
        factory_source = repr(factory)
    digest.update(factory_source.encode())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _initialize_numerical_runtime():
    # Some BLAS builds load their OpenMP runtime on first use. Initialize the
    # NumPy and SciPy BLAS libraries before recording the pool inventory so a
    # fresh worker and an already used worker describe the same configuration.
    from scipy.linalg.blas import dgemm
    identity = np.eye(2, dtype=np.float64)
    np.dot(identity, identity)
    dgemm(1.0, identity, identity)


def _runtime_provenance(factory):
    _initialize_numerical_runtime()
    pools = [{key: item.get(key) for key in (
        "user_api", "internal_api", "prefix", "version", "num_threads",
        "threading_layer", "architecture")} for item in threadpool_info()]
    pools.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return dict(format=1, source=_source_fingerprint(factory),
                python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__,
                dwd=dwd.__version__, pools=pools)


def _cached_base_stage(cache_dir, key_data, compute):
    """Return (result, cache_hit, key), computing once per concurrently used key.

    Files are pickle based and must be kept in a trusted, private directory.
    The checksum detects accidental truncation/corruption, not malicious edits.
    Neither failed computations nor partially written files become cache hits.
    """
    directory = Path(cache_dir).expanduser() / "base-stage-v1"
    directory.mkdir(parents=True, exist_ok=True)
    key = joblib.hash(key_data, hash_name="sha1")
    payload = directory / (key + ".joblib")
    metadata = directory / (key + ".json")
    with _key_lock(directory / (key + ".lock")):
        if payload.is_file() and metadata.is_file():
            try:
                record = json.loads(metadata.read_text(encoding="utf-8"))
                valid = (isinstance(record, dict) and record.get("key") == key and record.get("format") == 1
                         and record.get("sha256") == _sha256(payload))
            except (ValueError, UnicodeError):
                valid = False
            if valid:
                # A matching checksum makes deserialization failures visible;
                # incompatible runtime/code has a different key already.
                return joblib.load(payload), True, key
        result = compute()
        temporary = []
        try:
            for suffix in (".joblib.tmp", ".json.tmp"):
                handle, name = tempfile.mkstemp(prefix=key + ".", suffix=suffix, dir=directory)
                os.close(handle)
                temporary.append(Path(name))
            joblib.dump(result, temporary[0], compress=0)
            record = dict(key=key, format=1, sha256=_sha256(temporary[0]))
            temporary[1].write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            os.replace(temporary[0], payload)
            os.replace(temporary[1], metadata)
        finally:
            for path in temporary:
                if path.exists():
                    path.unlink()
        return result, False, key
