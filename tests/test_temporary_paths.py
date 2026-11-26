"""Deep helper reads, replacement and cleanup use the original requested tree."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from path_support import extended_path, ordinary_path


def test_deep_temporary_reads_and_cleanup(tmp_path):
    root = Path(extended_path(tmp_path))
    for index in range(6):
        root /= (str(index) + "-long-test-directory-" * 3)
    root.mkdir(parents=True)
    assert len(str(root)) > 260
    with tempfile.TemporaryDirectory(dir=root) as directory:
        folder = Path(directory)
        record = folder / "checkpoint.json"
        record.write_text('{"complete": false}', encoding="utf-8")
        replacement = folder / "replacement.json"
        replacement.write_bytes(b'{"complete": true}')
        os.replace(replacement, record)
        assert record.read_text(encoding="utf-8") == '{"complete": true}'
        assert record.read_bytes() == b'{"complete": true}'
    assert not folder.exists()
    assert root.exists()


def test_normalization_preserves_location_and_is_idempotent(tmp_path):
    normalized = extended_path(tmp_path / "parent" / ".." / "requested")
    assert extended_path(normalized) == normalized
    Path(normalized).mkdir()
    assert (tmp_path / "requested").is_dir()
    if os.name == "nt":
        assert normalized.startswith("\\\\?\\")


def test_default_temp_name_and_cleanup_are_extended_on_windows():
    with tempfile.TemporaryDirectory() as name:
        if os.name == "nt":
            assert name.startswith("\\\\?\\")
        Path(name, "helper.txt").write_text("readable", encoding="utf-8")
        assert Path(name, "helper.txt").read_text(encoding="utf-8") == "readable"
    assert not Path(name).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows drive and network-share spelling")
@pytest.mark.parametrize("root", [r"C:\test directory", r"\\server\share\test directory"])
def test_fresh_process_receives_ordinary_temp_path(root):
    requested = Path(root).joinpath(*["long directory " * 4 + "child"] * 5)
    deep = Path(extended_path(requested))
    with patch.object(subprocess, "run") as launch:
        _run_deep_temp_probe(deep)
    launch.assert_called_once()
    assert launch.call_args.args[0][-1] == str(deep)
    for name in ("TEMP", "TMP", "TMPDIR"):
        assert launch.call_args.kwargs["env"][name] == str(requested)


@pytest.mark.skipif(os.name != "nt", reason="Windows extended temporary-path spelling")
def test_fresh_process_keeps_requested_deep_temp_and_restores_environment(tmp_path):
    deep = Path(extended_path(tmp_path))
    for _ in range(6):
        deep /= "requested-deep-temp-" * 3
    deep.mkdir(parents=True)
    _run_deep_temp_probe(deep)


def _run_deep_temp_probe(deep):
    ordinary = str(ordinary_path(deep))
    assert len(ordinary) > 260
    env = os.environ.copy()
    env.update(TEMP=ordinary, TMP=ordinary, TMPDIR=ordinary)
    code = '''
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from path_support import extended_temporary_environment
before = dict(os.environ)
cached = tempfile.tempdir
with extended_temporary_environment():
    root = Path(tempfile.gettempdir())
    assert os.path.samefile(root, sys.argv[2])
    with tempfile.TemporaryDirectory() as name:
        child = Path(name)
        assert child.parent == root
        (child / 'receipt').write_bytes(b'passed')
        assert (child / 'receipt').read_bytes() == b'passed'
    assert not child.exists()
assert dict(os.environ) == before
assert tempfile.tempdir is cached
'''
    executable = sys.executable
    if os.name == "nt":
        base = getattr(sys, "_base_executable", None)
        if base and os.path.normcase(base) != os.path.normcase(executable):
            # Bypass the Windows venv launcher with a deep TEMP, keeping the venv identity.
            env["__PYVENV_LAUNCHER__"] = executable
            executable = base
    subprocess.run(
        [executable, "-I", "-S", "-B", "-c", code, str(Path(__file__).parent), str(deep)],
        env=env, stdin=subprocess.DEVNULL, check=True, timeout=20,
    )
