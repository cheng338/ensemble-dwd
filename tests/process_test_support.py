"""Launch test workers whose identity is checked by the parent process."""
import os
import sys

import psutil


def popen_python(args, *, env, **kwargs):
    """Return the Python worker itself, using the active interpreter environment."""
    executable = sys.executable
    if os.name == "nt":
        base = getattr(sys, "_base_executable", None)
        if base and os.path.normcase(base) != os.path.normcase(executable):
            # CPython's Windows venv redirector would give Popen a different PID.
            # Use the same child environment setup as multiprocessing's spawn.
            env = dict(env, __PYVENV_LAUNCHER__=executable)
            executable = base
    return psutil.Popen([executable, *args], env=env, **kwargs)
