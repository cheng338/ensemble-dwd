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

    def test_both_dwd_platform_pins_are_hashed_and_documented(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = [value for value in metadata["project"]["dependencies"]
                        if value.startswith("dwd @ ")]
        self.assertEqual(len(dependencies), 2)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        versions, kinds = set(), set()
        for value in dependencies:
            match = re.match(
                r"dwd @ https://github\.com/cheng338/dwd/releases/download/v([^/]+)/"
                r"dwd-([^/]+)-(cp311-abi3-win_amd64|py3-none-any)\.whl"
                r"#sha256=([a-f0-9]{64}) ; .+", value)
            self.assertIsNotNone(match, value)
            tag_version, wheel_version, kind, digest = match.groups()
            self.assertEqual(tag_version, wheel_version)
            versions.add(tag_version)
            kinds.add(kind)
            if kind == "py3-none-any":
                # The documented free-threaded workaround must use the same pin.
                self.assertIn(value.split(" ; ")[0], readme)
        self.assertEqual(len(versions), 1)
        self.assertEqual(kinds, {"cp311-abi3-win_amd64", "py3-none-any"})
        version = next(iter(versions))
        self.assertIn(f"https://github.com/cheng338/dwd/releases/tag/v{version}", readme)


if __name__ == "__main__":
    unittest.main()
