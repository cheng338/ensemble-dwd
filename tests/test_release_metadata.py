"""Verify the package pair and its release installation instructions."""
from pathlib import Path
import re
import tomllib
import unittest

import dwd
import ensemble_dwd


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_rights_notice_is_in_package_metadata_and_source_archive(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["license"], {"file": "LICENSE"})
        self.assertIn("LICENSE", metadata["tool"]["flit"]["sdist"]["include"])
        notice = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Copyright 2026 Chang Cheng", notice)
        self.assertIn("All Rights Reserved.", notice)
        self.assertIn("has not yet been selected or granted", notice)
        self.assertFalse(any("OSI Approved" in value
                             for value in metadata["project"]["classifiers"]))

    def test_development_pair_matches_dependency_and_installation_guide(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = [value for value in metadata["project"]["dependencies"]
                        if value.startswith("dwd")]
        self.assertEqual(dependencies, [f"dwd=={dwd.__version__}"])
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"Ensemble DWD `{ensemble_dwd.__version__}` requires DWD `{dwd.__version__}`", readme)

    def test_readme_released_wheel_example(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        example_wheels = re.findall(r"dist/(ensemble_dwd-[^\s]+\.whl)", readme)
        self.assertEqual(example_wheels,
                         ["ensemble_dwd-0.4.0-py3-none-any.whl"])

    def test_published_dwd_version_and_both_wheel_hashes_are_locked(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock = (ROOT / "requirements-dwd.txt").read_text(encoding="utf-8")
        lines = [line.strip() for line in lock.splitlines()
                 if line.strip() and not line.startswith("#")]
        self.assertEqual(lines[0].removesuffix("\\").strip(), "dwd==2.0.1")
        hashes = re.findall(r"--hash=sha256:([a-f0-9]{64})", lock)
        self.assertEqual(len(hashes), 2)
        self.assertEqual(len(set(hashes)), 2)
        self.assertIn("requirements-dwd.txt", metadata["tool"]["flit"]["sdist"]["include"])
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("--no-index --find-links wheels --no-deps --force-reinstall --require-hashes", readme)
        self.assertIn("-r requirements-dwd.txt", readme)
        self.assertIn("releases/expanded_assets/v2.0.1", readme)


if __name__ == "__main__":
    unittest.main()
