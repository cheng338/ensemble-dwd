"""Temporary Windows CI diagnostic. Observe only the deep-temp test's child."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def record(path, data):
    with path.open("a", encoding="utf-8") as stream:
        print(json.dumps(data), file=stream)


def mark_script(original):
    prefix = """
import sys
trace = open(sys.argv[3], 'a', encoding='utf-8', buffering=1)
print('entered Python', file=trace)
import faulthandler
faulthandler.dump_traceback_later(5, repeat=True, file=trace)
"""
    result = prefix
    for line in original.splitlines(keepends=True):
        if line.strip():
            indent = line[:len(line) - len(line.lstrip())]
            result += indent + "print(" + repr("before " + line.strip()) + ", file=trace)\n"
        result += line
    return result + "\nprint('all assertions completed', file=trace)\n"


def observe_process(process, log, stop):
    import psutil

    try:
        launcher = psutil.Process(process.pid)
        for delay in (0.15, 1, 3, 5, 7):
            if stop.wait(delay):
                return
            children = launcher.children(recursive=True)
            rows = []
            for entry in [launcher] + children:
                try:
                    rows.append([entry.pid, entry.ppid(), entry.exe(), entry.status(),
                                 entry.cpu_times()[:2], len(entry.threads())])
                except psutil.Error as exc:
                    rows.append([entry.pid, type(exc).__name__])
            record(log, {"tree": rows})
    except psutil.Error as exc:
        record(log, {"observer": type(exc).__name__})


def run_case(folder, mode, argv, original_kwargs):
    import psutil
    from temp_probe_native import sample_waits

    log = folder / (mode + ".jsonl")
    trace = folder / (mode + ".trace")
    argv = list(argv)
    kwargs = {**original_kwargs, "env": original_kwargs["env"].copy()}
    if mode != "original":
        idx = argv.index("-c") + 1
        argv[idx] = mark_script(argv[idx])
        argv.append(str(trace))
    if mode == "base-deep":
        argv[0] = sys._base_executable
        kwargs["env"]["__PYVENV_LAUNCHER__"] = sys.executable
    if mode == "venv-shallow":
        root = folder / "shallow"
        root.mkdir(exist_ok=True)
        for key in ("TMP", "TEMP", "TMPDIR"):
            kwargs["env"][key] = str(root)
        argv[-2] = str(root)
    if mode == "venv-extended":
        for key in ("TMP", "TEMP", "TMPDIR"):
            kwargs["env"][key] = argv[-2]
    if mode == "venv-stdio":
        kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if mode == "deep-after-start":
        for key in ("TMP", "TEMP", "TMPDIR"):
            kwargs["env"][key] = str(folder)
        idx = argv.index("-c") + 1
        original_line = "\nbefore = dict(os.environ)\n"
        assert argv[idx].count(original_line) == 1
        argv[idx] = argv[idx].replace(original_line,
            "\nos.environ.update({key: sys.argv[4] for key in ('TEMP', 'TMP', 'TMPDIR')})" + original_line)
        argv.append(original_kwargs["env"]["TEMP"])
    started = time.monotonic()
    stop = threading.Event()
    threads = []
    record(log, {"python": sys.version, "launcher": argv[0],
                 "base": sys._base_executable, "temp_length": len(kwargs["env"]["TEMP"])})
    class ObservedPopen(subprocess.Popen):
        def __init__(self, *args, **popen_kwargs):
            super().__init__(*args, **popen_kwargs)
            if self.args != argv:
                return
            record(log, {"handle_pid": self.pid})
            t = threading.Thread(target=observe_process, args=(self, log, stop), daemon=True)
            threads.append(t)
            t.start()
            if mode == "native-wait":
                owned = psutil.Process(self.pid)
                def inspect_wait():
                    if not stop.wait(3) and owned.is_running():
                        sample_waits(owned.pid, folder / "native-waits.log")
                t = threading.Thread(target=inspect_wait, daemon=True)
                threads.append(t)
                t.start()
    try:
        with patch.object(subprocess, "Popen", ObservedPopen):
            result = subprocess.run(argv, **kwargs)
        record(log, {"returned": result.returncode})
        return result
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        record(log, {"error": type(exc).__name__, "status": getattr(exc, "returncode", None)})
        raise
    finally:
        stop.set()
        for t in threads:
            t.join(2)
        record(log, {"elapsed_seconds": round(time.monotonic() - started, 4)})


@pytest.fixture(autouse=True)
def observe_deep_temp(request, monkeypatch):
    if request.node.name != "test_fresh_process_keeps_requested_deep_temp_and_restores_environment":
        return
    folder = Path(os.environ["TEMP_DIAG_OUT"])
    folder.mkdir(parents=True, exist_ok=True)
    def run(argv, **kwargs):
        original_error = None
        result = None
        for mode in ("original", "venv-deep", "base-deep", "venv-shallow", "venv-extended",
                     "deep-after-start", "venv-stdio", "native-wait"):
            try:
                case_result = run_case(folder, mode, argv, kwargs)
                if mode == "original":
                    result = case_result
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                if mode == "original":
                    original_error = exc
        if original_error is not None:
            raise original_error
        return result
    monkeypatch.setattr(request.module, "subprocess", SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL))
