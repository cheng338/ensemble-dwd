"""Keep installation examples consistent with the released package metadata."""
from pathlib import Path
import re
import tomllib
import unittest

import ensemble_dwd


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_readme_local_wheel_matches_package_version(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        example_wheels = re.findall(r"dist/(ensemble_dwd-[^\s]+\.whl)", readme)
        self.assertEqual(example_wheels,
                         [f"ensemble_dwd-{ensemble_dwd.__version__}-py3-none-any.whl"])

    def test_dwd_version_and_both_wheel_hashes_are_locked(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = [value for value in metadata["project"]["dependencies"]
                        if value.startswith("dwd")]
        self.assertEqual(dependencies, ["dwd==1.3.13"])
        lock = (ROOT / "requirements-dwd.txt").read_text(encoding="utf-8")
        lines = [line.strip() for line in lock.splitlines()
                 if line.strip() and not line.startswith("#")]
        self.assertEqual(lines[0].removesuffix("\\").strip(), dependencies[0])
        hashes = re.findall(r"--hash=sha256:([a-f0-9]{64})", lock)
        self.assertEqual(len(hashes), 2)
        self.assertEqual(len(set(hashes)), 2)
        self.assertIn("requirements-dwd.txt", metadata["tool"]["flit"]["sdist"]["include"])
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("--no-index --find-links wheels --no-deps --force-reinstall --require-hashes", readme)
        self.assertIn("-r requirements-dwd.txt", readme)
        self.assertIn(dependencies[0].split("==")[1], readme)


if __name__ == "__main__":
    unittest.main()
