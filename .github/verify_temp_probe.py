"""Verify the direct-interpreter control on disposable copies of the bundled test."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from xml.etree import ElementTree


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Expected exactly one copy of " + old)
    return source.replace(old, new)


def run_fixture(parent, bundled, source, name, expected_passes, expected_failure_line=None):
    dest = parent / name
    dest.mkdir()
    for helper in ("conftest.py", "path_support.py"):
        shutil.copyfile(bundled / helper, dest / helper)
    module = dest / "test_temporary_paths.py"
    module.write_text(source, encoding="utf-8")
    receipt = dest / "result.xml"
    target = str(module)
    if expected_failure_line:
        target += "::test_fresh_process_keeps_requested_deep_temp_and_restores_environment"
    proc = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           target, "--junitxml", str(receipt)], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, errors="replace", timeout=85)
    cases = ElementTree.parse(receipt).findall(".//testcase")
    failures = [case.find("failure") for case in cases if case.find("failure") is not None]
    assert not any(case.find("skipped") is not None or case.find("error") is not None for case in cases)
    combined = proc.stdout + proc.stderr
    if expected_failure_line:
        assert proc.returncode == 1 and len(failures) == 1, combined
        assert "CalledProcessError" in failures[0].get("message", "") and "AssertionError" in combined, combined
        assert expected_failure_line in combined, combined
    else:
        assert proc.returncode == 0 and not failures, combined
    assert len(cases) - len(failures) == expected_passes
    print(json.dumps({"control": name, "passed": expected_passes, "detected_faults": len(failures),
                      "cases": [c.get("name") for c in cases]}), flush=True)


def main():
    bundled = Path.cwd() / "tests"
    original = (bundled / "test_temporary_paths.py").read_text(encoding="utf-8")
    assert os.name == "nt" and sys.executable != sys._base_executable
    direct = replace_once(original, "    # This probe needs only", (
        "    env['__PYVENV_LAUNCHER__'] = sys.executable\n    # This probe needs only"))
    direct = replace_once(direct, "[sys.executable,", "[sys._base_executable,")
    with tempfile.TemporaryDirectory(dir=os.environ["RUNNER_TEMP"]) as directory:
        parent = Path(directory)
        run_fixture(parent, bundled, direct, "unmodified-assertions", 6)
        wrong = replace_once(direct, "assert os.path.samefile(root, sys.argv[2])",
                             "assert os.path.samefile(root, root.parent)")
        run_fixture(parent, bundled, wrong, "deliberately-wrong-root", 0, "os.path.samefile(root, root.parent)")
        corrupt = replace_once(direct, "write_bytes(b'passed')", "write_bytes(b'different')")
        run_fixture(parent, bundled, corrupt, "deliberately-wrong-contents", 0, "read_bytes() == b'passed'")
        identity = "import json,sys;print(json.dumps([sys.executable,sys._base_executable]))"
        args = ["-I", "-S", "-B", "-c", identity]
        ordinary = subprocess.check_output([sys.executable, *args], text=True)
        # The same identity must be visible through both launch paths.
        env = dict(os.environ, __PYVENV_LAUNCHER__=sys.executable)
        observed = subprocess.check_output([sys._base_executable, *args], env=env, text=True)
        assert list(map(os.path.normcase, json.loads(ordinary))) == list(map(os.path.normcase, json.loads(observed)))
        print("Active virtual-environment and base interpreter paths matched on both routes.", flush=True)


if __name__ == "__main__":
    main()
