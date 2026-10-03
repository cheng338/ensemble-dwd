"""Workflow recovery must reuse completed folds without repeating test scoring."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
import adaptive_search
import cv_checkpoint
from cv_checkpoint import CheckpointError
from search_support import data_fingerprint, new_run_directory


class InterruptibleSurface(ClassifierMixin, BaseEstimator):
    attempts = []
    predictions = []
    fail_at = None
    fail_final = False
    on_first_fit = None

    def __init__(self, lambd=4., gamma=4.):
        self.lambd = lambd
        self.gamma = gamma

    def fit(self, X, y):
        row_ids = tuple(np.asarray(X)[:, 0].astype(int))
        type(self).attempts.append((row_ids, self.lambd, self.gamma))
        callback = type(self).on_first_fit
        if callback is not None:
            type(self).on_first_fit = None
            callback()
        if (len(type(self).attempts) == type(self).fail_at
                or (type(self).fail_final and len(y) == 20)):
            raise KeyboardInterrupt('controlled interruption')
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        type(self).predictions.append(tuple(X[:, 0].astype(int)))
        y = X[:, 1].astype(int)
        return y if (self.lambd, self.gamma) == (4., 4.) else 1 - y


class AdaptiveResumeTests(unittest.TestCase):
    def setUp(self):
        # Exercise real fits, scoring, receipts and locks here. Runtime discovery
        # is covered separately by guard tests and real-estimator validation.
        runtime_patch = patch.object(cv_checkpoint, 'runtime_fingerprint',
                                     return_value={'synthetic_runtime': 'adaptive-resume-v1'})
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.run_dir = new_run_directory(Path(self.folder.name) / 'run')
        rows = np.arange(20)
        self.X = np.column_stack([rows, rows % 2])
        self.y = rows % 2
        heldout = np.arange(100, 108)
        self.test_X = np.column_stack([heldout, heldout % 2])
        self.test_y = heldout % 2
        self.folds = [(np.setdiff1d(rows, np.arange(i, i + 4)), np.arange(i, i + 4))
                      for i in range(0, 20, 4)]
        self.pipeline = Pipeline([('ensemble', InterruptibleSurface())])
        self.grid = {'ensemble__lambd': [1., 4., 16.], 'ensemble__gamma': [1., 4., 16.]}
        self.config = adaptive_search.AdaptiveSearchConfig(refine=False, max_candidates=9)
        InterruptibleSurface.attempts = []
        InterruptibleSurface.predictions = []
        InterruptibleSurface.fail_at = None
        InterruptibleSurface.fail_final = False
        InterruptibleSurface.on_first_fit = None

    def run_search(self, *, resume=False, X=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return adaptive_search.run_adaptive_search(
                self.pipeline, self.grid, self.X if X is None else X, self.y,
                self.test_X, self.test_y, self.folds, self.run_dir,
                config=self.config, resume=resume)

    def read(self, name):
        return json.loads((self.run_dir / name).read_text())

    def test_interrupt_after_four_completed_folds_resumes_only_missing_work(self):
        InterruptibleSurface.fail_at = 5
        with self.assertRaises(KeyboardInterrupt):
            self.run_search()
        self.assertEqual(self.read('status.json')['state'], 'interrupted')
        completed_attempts = InterruptibleSurface.attempts[:4]
        InterruptibleSurface.fail_at = None
        result = self.run_search(resume=True)
        self.assertEqual(result['completed']['held_out_accuracy'], 1.)
        for attempt in completed_attempts:
            self.assertEqual(InterruptibleSurface.attempts.count(attempt), 1)
        # 45 successful CV fits, one interrupted attempt and one final refit.
        self.assertEqual(len(InterruptibleSurface.attempts), 47)
        self.assertEqual(self.read('status.json')['state'], 'completed')

    def test_completed_resume_does_not_fit_or_score_again(self):
        first = self.run_search()
        counts = (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions))
        resumed = self.run_search(resume=True)
        self.assertTrue(resumed['resumed_completed'])
        self.assertIsNone(resumed['model'])
        self.assertEqual(first['completed'], resumed['completed'])
        self.assertEqual(counts, (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions)))

    def test_changed_data_refuses_resume_before_fit_or_status_replacement(self):
        InterruptibleSurface.fail_at = 5
        with self.assertRaises(KeyboardInterrupt):
            self.run_search()
        previous_status = (self.run_dir / 'status.json').read_bytes()
        previous_attempts = len(InterruptibleSurface.attempts)
        changed = self.X.copy()
        changed[0, 0] = -1
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_search(resume=True, X=changed)
        self.assertEqual(previous_attempts, len(InterruptibleSurface.attempts))
        self.assertEqual(previous_status, (self.run_dir / 'status.json').read_bytes())

    def test_interrupted_final_fit_reuses_cv_and_does_not_score_until_success(self):
        InterruptibleSurface.fail_final = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_search()
        self.assertEqual(self.read('status.json')['phase'], 'final_refit')
        self.assertFalse(any(min(ids) >= 100 for ids in InterruptibleSurface.predictions))
        cv_attempts = [row for row in InterruptibleSurface.attempts if len(row[0]) == 16]
        InterruptibleSurface.fail_final = False
        result = self.run_search(resume=True)
        self.assertEqual(result['completed']['held_out_accuracy'], 1.)
        self.assertEqual(cv_attempts, [row for row in InterruptibleSurface.attempts if len(row[0]) == 16])
        self.assertEqual(sum(min(ids) >= 100 for ids in InterruptibleSurface.predictions), 1)

    def assert_heldout_mutation_isolated(self, field):
        original_X, original_y = self.test_X.copy(), self.test_y.copy()
        expected_identity = data_fingerprint(original_X, original_y)

        def mutate_caller_alias():
            # The first CV fit runs after the workflow has established its identity.
            if field == 'X':
                self.test_X[:, 1] = 1 - self.test_X[:, 1]
            else:
                self.test_y[:] = 1 - self.test_y

        InterruptibleSurface.on_first_fit = mutate_caller_alias
        result = self.run_search()
        self.assertEqual(result['completed']['held_out_accuracy'], 1.)
        self.assertEqual(sum(min(ids) >= 100 for ids in InterruptibleSurface.predictions), 1)
        changed = self.test_X if field == 'X' else self.test_y
        original = original_X if field == 'X' else original_y
        self.assertFalse(np.array_equal(changed, original))
        self.assertEqual(self.read('run.json')['data']['held_out'], expected_identity)
        identity = self.read('cv-checkpoints/identity.json')['payload']['identity']
        self.assertEqual(identity['provenance']['data']['held_out'], expected_identity)

    def test_heldout_features_are_owned_before_cv_can_mutate_caller_alias(self):
        self.assert_heldout_mutation_isolated('X')

    def test_heldout_labels_are_owned_before_cv_can_mutate_caller_alias(self):
        self.assert_heldout_mutation_isolated('y')

    def test_committed_completion_repairs_failed_public_mirror_without_fit_or_score(self):
        original_write = adaptive_search.write_json_atomic

        def fail_completed_mirror(path, value, **kwargs):
            if Path(path).name == 'completed.json':
                raise OSError('controlled public mirror write failure')
            return original_write(path, value, **kwargs)

        with patch.object(adaptive_search, 'write_json_atomic', side_effect=fail_completed_mirror):
            with self.assertRaisesRegex(OSError, 'controlled public mirror'):
                self.run_search()
        receipt_path = self.run_dir / 'cv-checkpoints/artifacts/completed.json'
        committed_bytes = receipt_path.read_bytes()
        committed = json.loads(committed_bytes)['payload']['value']
        self.assertFalse((self.run_dir / 'completed.json').exists())
        self.assertEqual(self.read('status.json')['state'], 'failed')
        counts = (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions))

        resumed = self.run_search(resume=True)

        self.assertTrue(resumed['resumed_completed'])
        self.assertIsNone(resumed['model'])
        self.assertEqual(counts, (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions)))
        self.assertEqual(receipt_path.read_bytes(), committed_bytes)
        for name in ('selection', 'review', 'refit', 'completed'):
            self.assertEqual(self.read(name + '.json'), committed[name])
        self.assertEqual(self.read('status.json')['state'], 'completed')

    def test_corrupt_committed_receipt_refuses_resume_before_fit_score_or_status_write(self):
        self.run_search()
        path = self.run_dir / 'cv-checkpoints/artifacts/completed.json'
        record = json.loads(path.read_text())
        record['payload']['value']['completed']['held_out_accuracy'] = 0.
        path.write_text(json.dumps(record), encoding='utf-8')
        previous_status = (self.run_dir / 'status.json').read_bytes()
        counts = (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions))

        with self.assertRaisesRegex(CheckpointError, 'checksum or identity mismatch'):
            self.run_search(resume=True)

        self.assertEqual(counts, (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions)))
        self.assertEqual(previous_status, (self.run_dir / 'status.json').read_bytes())

    def test_changed_policy_refuses_completed_resume_without_fit_score_or_status_write(self):
        self.run_search()
        previous_status = (self.run_dir / 'status.json').read_bytes()
        counts = (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions))
        self.config = adaptive_search.AdaptiveSearchConfig(refine=False, max_candidates=10)

        with self.assertRaises(CheckpointError):
            self.run_search(resume=True)

        self.assertEqual(counts, (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions)))
        self.assertEqual(previous_status, (self.run_dir / 'status.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
