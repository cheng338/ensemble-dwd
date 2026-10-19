"""CI artifact checks reject stale imports, payload drift and silent fallback."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

_SPEC = importlib.util.spec_from_file_location(
    '_ci_artifact_checks', Path(__file__).with_name('ci_artifacts.py'))
checks = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checks)


class InstalledArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.site = self.root / 'site'
        self.site.mkdir()
        self.payload = {'fixture/__init__.py': b'__version__ = "1.2"\n',
                        'fixture-1.2.dist-info/METADATA': b'Name: fixture\nVersion: 1.2\n',
                        'fixture-1.2.dist-info/RECORD': b'wheel record'}
        self.wheel = self.root / 'fixture-1.2-py3-none-any.whl'
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            for name, data in self.payload.items():
                archive.writestr(name, data)
                installed = self.site / name
                installed.parent.mkdir(parents=True, exist_ok=True)
                installed.write_bytes(data)
        self.module = SimpleNamespace(__file__=str(self.site / 'fixture/__init__.py'),
                                      __version__='1.2')
        self.distribution = SimpleNamespace(version='1.2', locate_file=lambda name: self.site / name)

    def verify(self):
        with patch.object(checks.importlib, 'import_module', return_value=self.module), \
                patch.object(checks.importlib.metadata, 'distribution', return_value=self.distribution):
            return checks.verify_artifact(self.wheel, 'fixture', allowed_roots=[self.site])

    def test_exact_payload_and_origin_pass_allowing_installer_record_changes(self):
        (self.site / 'fixture-1.2.dist-info/RECORD').write_bytes(b'installer updated record')
        record = self.verify()
        self.assertEqual(record['verified_members'], 2)
        self.assertEqual(record['version'], '1.2')
        self.assertFalse(record['declares_accelerator'])

    def test_same_version_source_import_is_rejected(self):
        self.module.__file__ = str(self.root / 'checkout/fixture/__init__.py')
        with self.assertRaisesRegex(RuntimeError, 'outside the selected installation'):
            self.verify()

    def test_module_and_distribution_versions_must_match_wheel(self):
        for changed in (self.module, self.distribution):
            name = '__version__' if changed is self.module else 'version'
            with self.subTest(field=name), patch.object(changed, name, '1.1'):
                with self.assertRaisesRegex(RuntimeError, 'versions differ'):
                    self.verify()

    def test_payload_drift_is_rejected_even_with_same_version(self):
        (self.site / 'fixture/__init__.py').write_bytes(b'__version__ = "1.2"\n# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'differs from the exact wheel'):
            self.verify()

    def test_missing_payload_is_rejected(self):
        (self.site / 'fixture-1.2.dist-info/METADATA').unlink()
        with self.assertRaisesRegex(RuntimeError, 'differs from the exact wheel'):
            self.verify()

    def test_wheel_directory_must_not_be_ambiguous(self):
        self.assertEqual(checks.single_wheel(self.root), self.wheel)
        (self.root / 'other.whl').touch()
        with self.assertRaisesRegex(RuntimeError, 'exactly one wheel'):
            checks.single_wheel(self.root)

    def test_installer_selects_exact_wheel_and_requested_extra(self):
        with patch.object(sys, 'argv', ['ci_artifacts.py', 'install', '--wheel-dir',
                                      str(self.root), '--extras', 'test-base']), \
                patch.object(checks.subprocess, 'run') as run:
            checks.main()
        self.assertEqual(run.call_args.args[0],
                         [sys.executable, '-m', 'pip', 'install', str(self.wheel) + '[test-base]'])
        self.assertTrue(run.call_args.kwargs['check'])


class SourceFreeBundleTests(unittest.TestCase):
    def test_bundle_keeps_tests_metadata_and_examples_without_package_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / 'source', root / 'detached'
            fixtures = ['tests/test_example.py', 'tests/fixtures/data.json', 'examples/run.py',
                        'doc/example_notebooks/Basic functionality.ipynb', 'pyproject.toml',
                        'setup.py', 'MANIFEST.in', 'README.md', 'LICENSE', 'requirements-dwd.txt']
            excluded = ['dwd/__init__.py', 'src/ensemble_dwd/__init__.py',
                        'ensemble_dwd/__init__.py', 'tests/__pycache__/stale.pyc']
            for name in fixtures + excluded:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            checks.prepare_bundle(source, destination)
            for name in fixtures:
                self.assertEqual((destination / name).read_text(), 'fixture')
            for name in excluded:
                self.assertFalse((destination / name).exists())
            with self.assertRaises(FileExistsError):
                checks.prepare_bundle(source, destination)
            with self.assertRaises(ValueError):
                checks.prepare_bundle(source, source / 'nested')


class InstalledBackendTests(unittest.TestCase):
    def test_declared_native_wheel_must_not_silently_fall_back(self):
        from dwd import _compiled_residual as compiled
        with patch.object(compiled, '_ACCEL', None), \
                patch.object(compiled, 'compiled_values', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'silently fell back'):
                checks.verify_backend({'declares_accelerator': True}, 'native')

    def test_portable_wheel_must_not_use_another_installed_accelerator(self):
        from dwd import _compiled_residual as compiled
        with patch.object(compiled, '_ACCEL', object()), \
                patch.object(compiled, 'compiled_values', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'unexpectedly uses'):
                checks.verify_backend({'declares_accelerator': False}, 'portable')

    def test_backend_must_match_expected_ci_leg(self):
        with self.assertRaisesRegex(RuntimeError, 'Expected native wheel'):
            checks.verify_backend({'declares_accelerator': False}, 'native')


if __name__ == '__main__':
    unittest.main()
