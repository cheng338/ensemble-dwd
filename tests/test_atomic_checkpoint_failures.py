"""Atomic receipts preserve primary failures without weakening required writes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
import search_support as support


class AtomicCheckpointFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'result.json'
        self.path.write_bytes(b'{"previous": true}\n')
        self.previous = self.path.read_bytes()

    def test_serialization_precedes_temp_creation_and_cleanup(self):
        primary = ValueError('serialization failed')
        with patch.object(support.json, 'dumps', side_effect=primary), \
                patch.object(support.tempfile, 'NamedTemporaryFile') as temporary, \
                patch.object(Path, 'unlink') as unlinked:
            with self.assertRaises(ValueError) as caught:
                support.write_json_atomic(self.path, {'next': True}, replace=True)
        self.assertIs(caught.exception, primary)
        temporary.assert_not_called()
        unlinked.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.previous)

    def test_replace_and_cleanup_failure_preserves_original_base_exception(self):
        for primary in (OSError('replace failed'), KeyboardInterrupt(), SystemExit(17)):
            with self.subTest(primary=type(primary).__name__), \
                    patch.object(support.os, 'replace', side_effect=primary), \
                    patch.object(Path, 'unlink', side_effect=PermissionError('cleanup failed')):
                with self.assertRaises(type(primary)) as caught:
                    support.write_json_atomic(self.path, {'next': True}, replace=True)
            self.assertIs(caught.exception, primary)
            self.assertEqual(self.path.read_bytes(), self.previous)
        self.assertEqual(len(list(self.root.glob('.record-*'))), 3)

    def test_fsync_and_cleanup_failure_preserves_write_failure(self):
        primary = OSError('sync failed')
        with patch.object(support.os, 'fsync', side_effect=primary), \
                patch.object(Path, 'unlink', side_effect=PermissionError('cleanup failed')):
            with self.assertRaises(OSError) as caught:
                support.write_json_atomic(self.path, {'next': True}, replace=True)
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.path.read_bytes(), self.previous)

    def test_failed_temp_creation_does_not_clean_unowned_paths(self):
        primary = PermissionError('cannot create temporary file')
        with patch.object(support.tempfile, 'NamedTemporaryFile', side_effect=primary), \
                patch.object(Path, 'unlink') as unlinked:
            with self.assertRaises(PermissionError) as caught:
                support.write_json_atomic(self.path, {'next': True}, replace=True)
        self.assertIs(caught.exception, primary)
        unlinked.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.previous)

    def test_required_write_failure_is_strict_and_normal_cleanup_succeeds(self):
        primary = PermissionError('cannot publish')
        with patch.object(support.os, 'replace', side_effect=primary):
            with self.assertRaises(PermissionError) as caught:
                support.write_json_atomic(self.path, {'next': True}, replace=True)
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.path.read_bytes(), self.previous)
        self.assertEqual(list(self.root.glob('.record-*')), [])

    def test_existing_destination_rejected_without_temp_creation(self):
        with patch.object(support.tempfile, 'NamedTemporaryFile') as temporary:
            with self.assertRaises(FileExistsError):
                support.write_json_atomic(self.path, {'next': True})
        temporary.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.previous)

    def test_success_preserves_format_and_leaves_no_temporary_file(self):
        payload = {'b': [1, 'é'], 'a': True}
        support.write_json_atomic(self.path, payload, replace=True)
        self.assertEqual(self.path.read_text(encoding='utf-8'),
                         json.dumps(payload, sort_keys=True, allow_nan=False, indent=2) + '\n')
        self.assertEqual(list(self.root.glob('.record-*')), [])


if __name__ == '__main__':
    unittest.main()
