"""Lock failures stay closed; real independent processes compute one shared key."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from ensemble_dwd._cache import _cached_base_stage, _CACHE_DIRECTORY


@pytest.mark.parametrize("code", sorted({errno.ENOLCK, errno.ENOSYS, errno.ENOTSUP}))
def test_unavailable_lock_explains_explicit_configuration_without_computing(tmp_path, monkeypatch, code):
    failure = OSError(code, "injected unavailable filesystem lock")
    def refuse(*args):
        raise failure
    if os.name == "nt":
        import msvcrt
        monkeypatch.setattr(msvcrt, "locking", refuse)
    else:
        import fcntl
        monkeypatch.setattr(fcntl, "flock", refuse)
    calls = []
    with pytest.raises(OSError, match="Set cache_dir") as caught:
        _cached_base_stage(tmp_path, {"key": 1}, lambda: calls.append(True))
    assert caught.value.errno == code
    assert caught.value.__cause__ is failure
    assert Path(caught.value.filename).parent == tmp_path / _CACHE_DIRECTORY
    assert not calls
    assert sorted(p.suffix for p in (tmp_path / _CACHE_DIRECTORY).iterdir()) == [".lock"]


def test_unrelated_lock_error_is_preserved(tmp_path, monkeypatch):
    failure = OSError(errno.EIO, "injected device failure")
    def refuse(*args):
        raise failure
    if os.name == "nt":
        import msvcrt
        monkeypatch.setattr(msvcrt, "locking", refuse)
    else:
        import fcntl
        monkeypatch.setattr(fcntl, "flock", refuse)
    with pytest.raises(OSError) as caught:
        _cached_base_stage(tmp_path, {}, lambda: pytest.fail("compute under failed lock"))
    assert caught.value is failure


_WORKER = r'''
import json, pathlib, sys, time, psutil
from ensemble_dwd._cache import _cached_base_stage
root = pathlib.Path(sys.argv[1])
me = psutil.Process()
(root / (str(me.pid) + '.ready')).write_text(str(me.create_time()))
deadline = time.monotonic() + 20
while not (root / 'start').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('start barrier timeout')
    time.sleep(.01)
def compute():
    with (root / 'computations.txt').open('a') as out:
        out.write('computed\n')
    time.sleep(.1)
    return {'answer': 37}
value, hit, key = _cached_base_stage(root / 'cache', {'shared': 1}, compute)
print(json.dumps(dict(value=value, hit=hit, key=key, pid=me.pid,
    created=me.create_time(), rss=me.memory_info().rss,
    cpu=me.cpu_times().user + me.cpu_times().system)))
'''


def test_four_processes_compute_once_with_psutil_ownership(tmp_path):
    # Explicit caller-selected scratch; never substitutes the production cache.
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    children = []
    identities = []
    try:
        for _ in range(4):
            child = psutil.Popen([sys.executable, "-c", _WORKER, str(tmp_path)],
                                 env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
            children.append(child)
            identities.append((child.pid, child.create_time()))
        deadline = time.monotonic() + 20
        while len(list(tmp_path.glob("*.ready"))) != 4:
            assert all(child.poll() is None for child in children), "worker exited before barrier"
            assert time.monotonic() < deadline, "ready barrier timeout"
            time.sleep(.02)
        for pid, created in identities:
            assert float((tmp_path / f"{pid}.ready").read_text()) == created
        (tmp_path / "start").write_text("go")
        rows = []
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, stderr
            rows.append(json.loads(stdout))
        assert (tmp_path / "computations.txt").read_text().splitlines() == ["computed"]
        assert sum(not row["hit"] for row in rows) == 1
        assert len({row["key"] for row in rows}) == 1
        assert all(row["value"] == {"answer": 37} for row in rows)
        assert [(r["pid"], r["created"]) for r in rows] == identities
        assert all(row["rss"] > 0 and row["cpu"] >= 0 for row in rows)
        assert all(not child.is_running() for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
