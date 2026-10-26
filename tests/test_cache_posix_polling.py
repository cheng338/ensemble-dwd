"""Retry lock contention without weakening exclusion or unavailable-lock errors."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from ensemble_dwd import _cache as cache


@pytest.fixture
def posix_lock(monkeypatch):
    # Exercise the POSIX control flow on Windows too; real OS exclusion is
    # covered by the independent-process tests on the actual target filesystem.
    backend = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8)
    monkeypatch.setitem(sys.modules, "fcntl", backend)
    monkeypatch.setattr(cache, "os", SimpleNamespace(name="posix", fspath=os.fspath))
    return backend


def test_busy_lock_is_polled_until_actually_acquired(tmp_path, monkeypatch, posix_lock):
    operations, sleeps, entered = [], [], []
    failures = iter([errno.EAGAIN, errno.EACCES, None])

    def flock(fd, operation):
        operations.append(operation)
        if operation == posix_lock.LOCK_UN:
            assert entered == [True]
            return
        assert not entered
        code = next(failures)
        if code is not None:
            raise OSError(code, "held by another process")

    posix_lock.flock = flock
    monkeypatch.setattr(cache.time, "sleep", sleeps.append)
    with cache._key_lock(tmp_path / "key.lock"):
        entered.append(True)
    assert operations == [6, 6, 6, 8]
    assert sleeps == [.05, .05]
    assert (tmp_path / "key.lock").read_bytes() == b""


@pytest.mark.parametrize("code", sorted({errno.ENOLCK, errno.ENOSYS, errno.ENOTSUP}))
def test_unavailable_lock_after_contention_is_not_retried(tmp_path, monkeypatch, posix_lock, code):
    failure = OSError(code, "persistent unavailable lock")
    operations, sleeps = [], []

    def flock(fd, operation):
        operations.append(operation)
        if len(operations) == 1:
            raise OSError(errno.EAGAIN, "busy")
        raise failure

    posix_lock.flock = flock
    monkeypatch.setattr(cache.time, "sleep", sleeps.append)
    with pytest.raises(OSError, match="Set cache_dir") as caught:
        with cache._key_lock(tmp_path / "key.lock"):
            pytest.fail("entered without an OS lock")
    assert caught.value.errno == code
    assert caught.value.__cause__ is failure
    assert operations == [6, 6]
    assert sleeps == [.05]


@pytest.mark.parametrize("code", [errno.EIO, errno.EDEADLK, errno.EINTR])
def test_other_lock_errors_are_not_retried(tmp_path, monkeypatch, posix_lock, code):
    failure = OSError(code, "non-contention failure")
    operations = []

    def flock(fd, operation):
        operations.append(operation)
        raise failure

    posix_lock.flock = flock
    monkeypatch.setattr(cache.time, "sleep", lambda delay: pytest.fail("unexpected retry"))
    with pytest.raises(OSError) as caught:
        with cache._key_lock(tmp_path / "key.lock"):
            pytest.fail("entered without an OS lock")
    assert caught.value is failure
    assert operations == [6]


@pytest.mark.parametrize("failure", [RuntimeError("body failed"), KeyboardInterrupt(), SystemExit(3)])
def test_acquired_lock_is_released_on_exception_or_cancellation(tmp_path, posix_lock, failure):
    operations, descriptors = [], []

    def flock(fd, operation):
        descriptors.append(fd)
        operations.append(operation)

    posix_lock.flock = flock
    with pytest.raises(type(failure)) as caught:
        with cache._key_lock(tmp_path / "key.lock"):
            raise failure
    assert caught.value is failure
    assert operations == [6, 8]
    assert descriptors[0] == descriptors[1]
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_cancellation_while_busy_closes_without_unlocking_an_unowned_lock(tmp_path, monkeypatch, posix_lock):
    failure = KeyboardInterrupt()
    operations, descriptors = [], []

    def flock(fd, operation):
        descriptors.append(fd)
        operations.append(operation)
        raise OSError(errno.EAGAIN, "busy")

    def interrupt(delay):
        assert delay == .05
        raise failure

    posix_lock.flock = flock
    monkeypatch.setattr(cache.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        with cache._key_lock(tmp_path / "key.lock"):
            pytest.fail("entered without an OS lock")
    assert caught.value is failure
    assert operations == [6]
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


_OWNER_WORKER = r'''
import json, pathlib, psutil, sys
from ensemble_dwd._cache import _key_lock
root, role = pathlib.Path(sys.argv[1]), sys.argv[2]
me = psutil.Process()
identity = dict(pid=me.pid, created=me.create_time())
def marker(suffix):
    temporary = root / (role + '.tmp')
    temporary.write_text(json.dumps(identity))
    temporary.replace(root / (role + suffix))
marker('.ready')
with _key_lock(root / 'key.lock'):
    marker('.entered')
    if role == 'owner':
        sys.stdin.read()
print(json.dumps(identity))
'''


@pytest.mark.skipif(os.name == "nt", reason="Real POSIX lock owner-exit behavior")
def test_real_posix_lock_releases_after_owned_process_is_killed(tmp_path):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    children = []

    def spawn(role):
        child = psutil.Popen([sys.executable, "-c", _OWNER_WORKER, str(tmp_path), role],
                             env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        children.append(child)
        return child, dict(pid=child.pid, created=child.create_time())

    def await_marker(name):
        deadline = time.monotonic() + 20
        while not (tmp_path / name).exists():
            assert all(child.poll() is None for child in children), "child exited before marker"
            assert time.monotonic() < deadline, "lock worker readiness timed out"
            time.sleep(.01)
        return json.loads((tmp_path / name).read_text())

    try:
        owner, owner_identity = spawn("owner")
        assert await_marker("owner.entered") == owner_identity
        waiter, waiter_identity = spawn("waiter")
        assert await_marker("waiter.ready") == waiter_identity
        with pytest.raises(subprocess.TimeoutExpired):
            waiter.communicate(timeout=.3)
        assert not (tmp_path / "waiter.entered").exists()
        owner.kill()
        owner.communicate(timeout=20)
        assert owner.returncode != 0
        stdout, stderr = waiter.communicate(timeout=20)
        assert waiter.returncode == 0, stderr
        assert json.loads(stdout) == waiter_identity
        assert json.loads((tmp_path / "waiter.entered").read_text()) == waiter_identity
        assert all(not child.is_running() for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=20)
