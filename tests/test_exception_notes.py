"""Failure-only metadata checks. No DWD or SVM numerical fits are performed."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import traceback
import unittest
from unittest.mock import patch
import warnings

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import numpy as np
import search_support
import cv_checkpoint as checkpoint
from sklearn.base import BaseEstimator, ClassifierMixin
from ensemble_dwd import OvOClassifier
from ensemble_dwd.estimator import IneligibleSelectionError
from search_support import exception_details


class FailureStub(ClassifierMixin, BaseEstimator):
    """Inspect pair labels encoded in the first feature; never fit a solver."""

    def __init__(self, failure='numerical'):
        self.failure = failure

    def fit(self, X, y):
        if self.failure in ('numerical', 'ineligible', 'interrupted') and tuple(np.unique(X[:, 0])) == (0.0, 2.0):
            if self.failure == 'ineligible':
                raise IneligibleSelectionError('injected one-class selected set')
            if self.failure == 'interrupted':
                error = KeyboardInterrupt('injected interruption')
                error.add_note('Injected interruption at a binary fit boundary.')
                raise error
            raise FloatingPointError('injected numerical failure')
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if self.failure == 'prediction':
            raise FloatingPointError('injected prediction failure')
        return np.ones(len(X), dtype=np.int64)


class NotedIneligibleStub(ClassifierMixin, BaseEstimator):
    calls = 0

    def fit(self, X, y):
        type(self).calls += 1
        error = IneligibleSelectionError('injected annotated selection failure')
        error.add_note('The selected rows contain only one class.')
        raise error

    def predict(self, X):
        raise AssertionError('An ineligible estimator must not predict.')


class ExceptionNoteTests(unittest.TestCase):
    def test_absent_empty_and_malformed_notes_preserve_legacy_shape(self):
        for notes in (None, [], (), "string-is-not-a-note-list", 3, {"note": "value"}, [1, None, {}]):
            error = FloatingPointError("same message")
            if notes is not None:
                error.__notes__ = notes
            with self.subTest(notes=notes):
                self.assertEqual(exception_details(error),
                                 {"category": "numerical_error", "exception_type": "builtins.FloatingPointError",
                                  "message": "same message"})

    def test_valid_strings_are_preserved_without_coercing_non_strings(self):
        error = RuntimeError("same message")
        error.__notes__ = ["While fitting OVO pair (0, 2).", 5, "", None, "second valid note"]
        self.assertEqual(exception_details(error)["exception_notes"],
                         ["While fitting OVO pair (0, 2).", "", "second valid note"])

    def test_count_and_character_bounds_are_explicit(self):
        error = RuntimeError("same message")
        error.__notes__ = ["a" * 1200] + [str(i) for i in range(20)]
        details = exception_details(error)
        self.assertEqual(len(details["exception_notes"]), 8)
        self.assertEqual(len(details["exception_notes"][0]), 1024)
        self.assertIs(details["exception_notes_truncated"], True)
        self.assertEqual(len(error.__notes__), 21)
        self.assertEqual(len(error.__notes__[0]), 1200)

    def test_malformed_accessor_and_user_defined_container_are_not_called(self):
        class BadNotesError(RuntimeError):
            @property
            def __notes__(self):
                raise ValueError("unreadable notes")
        class BadList(list):
            def __iter__(self):
                raise AssertionError("custom iterator must not run")
        error = RuntimeError("ordinary")
        error.__notes__ = BadList(["ignored"])
        self.assertNotIn("exception_notes", exception_details(error))
        self.assertNotIn("exception_notes", exception_details(BadNotesError("bad accessor")))

    def test_string_subclass_hooks_do_not_replace_original_error(self):
        class BadString(str):
            def __str__(self):
                raise AssertionError("custom str must not run")
            def __len__(self):
                raise AssertionError("custom len must not run")
            def __getitem__(self, key):
                raise AssertionError("custom slice must not run")
        error = RuntimeError("ordinary")
        error.__notes__ = [BadString("pair context")]
        self.assertEqual(exception_details(error)["exception_notes"], ["pair context"])

    def test_exception_type_cause_context_and_traceback_are_unchanged(self):
        try:
            try:
                raise ValueError("original cause")
            except ValueError as cause:
                error = FloatingPointError("original failure")
                error.add_note("While fitting OVO pair (0, 2).")
                raise error from cause
        except FloatingPointError as error:
            before = (type(error), error.args, error.__cause__, error.__context__, error.__traceback__,
                      dict(error.__dict__), "".join(traceback.format_exception(error)))
            details = exception_details(error)
            after = (type(error), error.args, error.__cause__, error.__context__, error.__traceback__,
                     dict(error.__dict__), "".join(traceback.format_exception(error)))
            self.assertEqual(before, after)
            self.assertEqual(details["message"], "original failure")
            self.assertIn("OVO pair", before[-1])

    def test_serial_and_joblib_failure_records_retain_pair_context(self):
        X = np.repeat(np.arange(3, dtype=float), 3)[:, None]
        y = X[:, 0].astype(int)
        for jobs, failure in ((1, "numerical"), (2, "numerical"), (1, "ineligible")):
            with self.subTest(jobs=jobs, failure=failure), tempfile.TemporaryDirectory() as directory:
                evaluator = search_support.CandidateEvaluator(OvOClassifier(FailureStub(failure), n_jobs=jobs),
                                                              error_policy="raise", failure_dir=directory)
                expected = FloatingPointError if failure == "numerical" else IneligibleSelectionError
                with self.assertRaises(expected) as caught:
                    evaluator.fit(X, y)
                files = list(Path(directory).glob("failure-*.json"))
                self.assertEqual(len(files), 1)
                receipt = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertIn("OVO pair", json.dumps(receipt))
                self.assertFalse(hasattr(evaluator, "estimator_"))
                if failure == "numerical":
                    self.assertEqual(receipt["message"], "injected numerical failure")
                    self.assertEqual(receipt["exception_notes"], ["While fitting OVO pair (0, 2)."])
                    if jobs == 2:
                        self.assertIn("OVO pair", str(caught.exception.__cause__))

    def test_logging_failure_with_warnings_as_errors_preserves_numerical_exception(self):
        X = np.array([[0.0], [2.0]])
        y = np.array([0, 2])
        with tempfile.TemporaryDirectory() as directory:
            evaluator = search_support.CandidateEvaluator(OvOClassifier(FailureStub()), error_policy="raise",
                                                          failure_dir=directory)
            with patch.object(search_support, "write_json_atomic", side_effect=OSError("injected full disk")), \
                 warnings.catch_warnings():
                warnings.simplefilter("error")
                with self.assertRaisesRegex(FloatingPointError, "injected numerical failure") as caught:
                    evaluator.fit(X, y)
            self.assertEqual(caught.exception.__notes__, ["While fitting OVO pair (0, 2)."])
            self.assertFalse(hasattr(evaluator, "estimator_"))


class CheckpointFailureNoteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.X = np.repeat(np.arange(3, dtype=float), 3)[:, None]
        self.y = self.X[:, 0].astype(int)
        train, validation = np.array([0, 1, 3, 4, 6, 7]), np.array([2, 5, 8])
        self.folds = [(train, validation), (validation, train)]
        self.runtime = {'python': 'failure-notes-test-runtime', 'files_sha256': 'c' * 64,
                        'native_pools': [{'num_threads': 1, 'version': 'test-build'}]}
        guard = patch.object(checkpoint, 'runtime_fingerprint', side_effect=lambda **kwargs: deepcopy(self.runtime))
        guard.start()
        self.addCleanup(guard.stop)

    def run_directory(self, name):
        directory = self.root / name
        directory.mkdir()
        (directory / 'failures').mkdir()
        return directory

    def session(self, directory, estimator, *, resume=False, error_policy='raise'):
        return checkpoint.CheckpointedCV(run_dir=directory, estimator=estimator, X=self.X, y=self.y,
                                         folds=self.folds, resume=resume, error_policy=error_policy,
                                         provenance={'policy': 'failure-notes-test'})

    def attempt(self, directory):
        files = list((directory / 'cv-checkpoints/attempts').glob('*.json'))
        self.assertEqual(len(files), 1)
        value = json.loads(files[0].read_text(encoding='utf-8'))
        self.assertEqual(value['sha256'], checkpoint._digest(value['payload']))
        return value['payload']

    def failures(self, directory):
        return [json.loads(path.read_text(encoding='utf-8')) for path in (directory / 'failures').glob('*.json')]

    def test_failed_attempt_and_failure_records_retain_serial_and_joblib_pair_notes(self):
        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                directory = self.run_directory('jobs-' + str(jobs))
                estimator = OvOClassifier(FailureStub(), n_jobs=jobs)
                with self.assertRaisesRegex(FloatingPointError, 'injected numerical failure'), self.session(directory, estimator) as store:
                    store.search({})
                attempt = self.attempt(directory)
                self.assertEqual(attempt['state'], 'failed')
                self.assertEqual(attempt['failure']['exception_notes'], ['While fitting OVO pair (0, 2).'])
                failures = self.failures(directory)
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]['exception_notes'], attempt['failure']['exception_notes'])
                self.assertEqual(failures[0]['message'], 'injected numerical failure')
                self.assertFalse(list((directory / 'cv-checkpoints/folds').glob('*.json')))
                self.assertFalse(hasattr(estimator, 'estimators_'))

    def test_interrupted_attempt_retains_existing_notes_without_a_complete_receipt(self):
        directory = self.run_directory('interrupted')
        estimator = OvOClassifier(FailureStub('interrupted'))
        with self.assertRaisesRegex(KeyboardInterrupt, 'injected interruption'), self.session(directory, estimator) as store:
            store.search({})
        attempt = self.attempt(directory)
        self.assertEqual(attempt['state'], 'interrupted')
        self.assertEqual(attempt['failure']['exception_type'], 'builtins.KeyboardInterrupt')
        self.assertEqual(attempt['failure']['exception_notes'], ['Injected interruption at a binary fit boundary.'])
        self.assertFalse(list((directory / 'cv-checkpoints/folds').glob('*.json')))
        self.assertFalse(hasattr(estimator, 'estimators_'))

    def test_prediction_failure_records_retain_existing_pair_notes(self):
        directory = self.run_directory('prediction')
        with self.assertRaisesRegex(FloatingPointError, 'injected prediction failure'), \
             self.session(directory, OvOClassifier(FailureStub('prediction'))) as store:
            store.search({})
        attempt = self.attempt(directory)
        self.assertEqual(attempt['state'], 'failed')
        self.assertEqual(attempt['failure']['exception_notes'], ['While predicting OVO pair (0, 1).'])
        failures = self.failures(directory)
        self.assertTrue(any(item['phase'] == 'cv_predict' for item in failures))
        self.assertTrue(all(item['exception_notes'] == attempt['failure']['exception_notes'] for item in failures))

    def test_noted_ineligible_receipt_is_reused_without_refitting(self):
        directory = self.run_directory('ineligible')
        estimator = NotedIneligibleStub()
        NotedIneligibleStub.calls = 0
        with self.session(directory, estimator, error_policy='nan') as store:
            store.search({})
        self.assertEqual(NotedIneligibleStub.calls, 2)
        files = list((directory / 'cv-checkpoints/folds').glob('*.json'))
        self.assertEqual(len(files), 2)
        original = {path.name: path.read_bytes() for path in files}
        for value in original.values():
            receipt = json.loads(value)['payload']
            self.assertEqual(receipt['state'], 'ineligible')
            self.assertEqual(receipt['failure']['exception_notes'], ['The selected rows contain only one class.'])
        with self.session(directory, estimator, resume=True, error_policy='nan') as store:
            repeated = store.search({})
        self.assertEqual(NotedIneligibleStub.calls, 2)
        self.assertEqual(repeated.checkpoint_summary_['reused_folds'], 2)
        self.assertEqual({path.name: path.read_bytes() for path in files}, original)

    def test_failed_attempt_logger_error_does_not_mask_exception_or_save_completion(self):
        directory = self.run_directory('failed-log')
        original_write = search_support.write_json_atomic

        def fail_failure_writes(path, payload, **kwargs):
            if Path(path).parent.name == 'failures' or payload.get('payload', {}).get('state') == 'failed':
                raise OSError('injected full disk')
            return original_write(path, payload, **kwargs)

        with self.session(directory, OvOClassifier(FailureStub())) as store, \
             patch.object(search_support, 'write_json_atomic', side_effect=fail_failure_writes), \
             warnings.catch_warnings():
            warnings.simplefilter('error')
            with self.assertRaisesRegex(FloatingPointError, 'injected numerical failure') as caught:
                store.search({})
        self.assertEqual(caught.exception.__notes__, ['While fitting OVO pair (0, 2).'])
        self.assertEqual(self.attempt(directory)['state'], 'running')
        self.assertFalse(list((directory / 'cv-checkpoints/folds').glob('*.json')))


if __name__ == "__main__":
    unittest.main()
