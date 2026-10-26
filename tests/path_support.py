"""Filesystem spelling for temporary test trees; no system configuration changes."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


def extended_path(path):
    """Retain the requested location and enable deep Windows I/O and cleanup."""
    absolute = os.path.abspath(os.fspath(Path(path).expanduser()))
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = ("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\")
                    else "\\\\?\\" + absolute)
    return absolute


@contextmanager
def extended_temporary_environment():
    """Normalize candidates before tempfile probes them; restore caller state."""
    if os.name != "nt":
        yield
        return
    previous_tempdir = tempfile.tempdir
    previous_env = {name: os.environ.get(name) for name in ("TEMP", "TMP", "TMPDIR")}
    try:
        for name, value in previous_env.items():
            if value:
                os.environ[name] = extended_path(value)
        # Reevaluate the same candidate order with usable Windows spellings;
        # an earlier ordinary-path probe may have fallen back from deep TEMP.
        tempfile.tempdir = None
        root = extended_path(tempfile.gettempdir())
        tempfile.tempdir = root
        os.environ.update({name: root for name in previous_env})
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
