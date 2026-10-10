"""Structured failure evidence survives actual ensemble and CV boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.validation import check_is_fitted

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier
from ensemble_dwd.estimator import IneligibleSelectionError
from ensemble_dwd import estimator as dwd_module, svm as svm_module
from ensemble_dwd import _cache as cache_module
from ensemble_dwd import _failure_context as context_module
import cv_checkpoint
import search_support
from search_support import exception_details


class EvidenceLearner:
    """Small affine stand-in with failures at specified learner boundaries."""
    def __init__(self, family, position, failure):
        self.family, self.position, self.failure = family, position, failure

    def fit(self, X, y):
        if ((self.failure in ('base_fit', 'interrupt') and self.position == 2)
                or (self.failure == 'final_fit' and self.position == 3)):
            if self.failure == 'interrupt':
                raise KeyboardInterrupt('controlled ensemble interruption')
            raise FloatingPointError('controlled learner failure')
        self.classes_ = np.unique(y)
        if self.failure == 'final_validation' and self.position == 3:
            self.classes_ = self.classes_[::-1]
        self.n_iter_ = 1 if self.family == 'dwd' else np.array([1])
        self.converged_, self.fit_status_ = True, 0
        self.n_support_ = np.array([1, 1])
        return self

    def decision_function(self, X):
        if self.failure == 'base_scoring' and self.position == 2:
            raise FloatingPointError('controlled scoring failure')
        return -np.ones(len(X)) if self.failure == 'selection' else np.asarray(X)[:, 0]


class FailureDWD(EnsembleKernelDWD):
    def __init__(self, failure='base_fit'):
        super().__init__(M=2, theta=.25, sv_selection_ratio=.5, gamma=.1, random_state=0)
        self.failure = failure

    def _make_estimator(self, settings):
        self._probe_position_ = getattr(self, '_probe_position_', 0) + 1
        return EvidenceLearner('dwd', self._probe_position_, self.failure)


class FailureSVM(EnsembleKernelSVM):
    def __init__(self, failure='base_fit'):
        super().__init__(M=2, theta=.25, sv_selection_ratio=.5, gamma=.1, random_state=0)
        self.failure = failure

    def _make_estimator(self, settings):
        self._probe_position_ = getattr(self, '_probe_position_', 0) + 1
        return EvidenceLearner('svm', self._probe_position_, self.failure)


def data():
    X = np.r_[-np.arange(1, 21) / 10., np.arange(1, 21) / 10.][:, None]
    return X, np.repeat([-1, 1], 20)


class FailureContextWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = data()

    def assert_clean(self, model):
        self.assertFalse(any(name.endswith('_') for name in vars(model)))
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

    def test_base_fit_score_selection_and_final_boundaries(self):
        for cls in (FailureDWD, FailureSVM):
            for failure in ('base_fit', 'base_scoring', 'selection', 'final_fit', 'final_validation'):
                with self.subTest(family=cls.__name__, failure=failure):
                    model = cls(failure)
                    expected = IneligibleSelectionError if failure == 'selection' else (
                        ValueError if failure == 'final_validation' else FloatingPointError)
                    with self.assertRaises(expected) as caught:
                        model.fit(self.X, self.y)
                    evidence = exception_details(caught.exception)['ensemble_failure_context']
                    self.assertEqual(evidence['stage'], failure)
                    self.assertEqual(evidence['base_fit_calls_completed'], 1 if failure == 'base_fit' else 2)
                    self.assertEqual(evidence['base_score_calls_completed'],
                                     1 if failure in ('base_fit', 'base_scoring') else 2)
                    self.assertEqual(evidence['cached_base_learners'], 0)
                    self.assertEqual(evidence['final_fit_calls_completed'], int(failure == 'final_validation'))
                    self.assertEqual(evidence['training_class_counts'], [20, 20])
                    if failure in ('base_fit', 'base_scoring'):
                        self.assertEqual(evidence['base_position'], 2)
                        self.assertEqual(sum(evidence['subset_class_counts']), 10)
                    else:
                        self.assertEqual(sum(evidence['selected_class_counts']), 20)
                    self.assert_clean(model)

    def test_interrupt_context_preserves_original_type_and_cleanup(self):
        for cls in (FailureDWD, FailureSVM):
            with self.subTest(cls=cls.__name__):
                model = cls('interrupt')
                with self.assertRaisesRegex(KeyboardInterrupt, 'controlled ensemble interruption') as caught:
                    model.fit(self.X, self.y)
                context = exception_details(caught.exception)['ensemble_failure_context']
                self.assertEqual(context['stage'], 'base_fit')
                self.assertEqual(context['base_fit_calls_completed'], 1)
                self.assert_clean(model)

    def test_serial_and_joblib_ovo_durable_errors_preserve_context(self):
        for cls in (FailureDWD, FailureSVM):
            for jobs, failure in ((1, 'base_fit'), (2, 'final_fit'), (2, 'selection')):
                with self.subTest(cls=cls.__name__, jobs=jobs, failure=failure), tempfile.TemporaryDirectory() as folder:
                    model = OvOClassifier(cls(failure), n_jobs=jobs)
                    evaluator = search_support.CandidateEvaluator(model, error_policy='raise', failure_dir=folder)
                    expected = IneligibleSelectionError if failure == 'selection' else FloatingPointError
                    with self.assertRaises(expected) as caught:
                        evaluator.fit(self.X, self.y)
                    files = list(Path(folder).glob('failure-*.json'))
                    self.assertEqual(len(files), 1)
                    receipt = json.loads(files[0].read_text(encoding='utf-8'))
                    self.assertEqual(receipt['ensemble_failure_context']['stage'], failure)
                    self.assertEqual(receipt['ensemble_failure_context'],
                                     exception_details(caught.exception)['ensemble_failure_context'])
                    self.assertIn('OVO pair', json.dumps(receipt))
                    self.assertFalse(hasattr(evaluator, 'estimator_'))

    def test_cached_bases_are_distinct_from_current_work_after_final_failure(self):
        for cls, module, family in ((EnsembleKernelDWD, dwd_module, 'dwd'),
                                    (EnsembleKernelSVM, svm_module, 'svm')):
            with self.subTest(cls=cls.__name__), tempfile.TemporaryDirectory() as folder:
                sequence = []
                def make(owner, settings):
                    sequence.append(len(sequence) + 1)
                    return EvidenceLearner(family, len(sequence), 'final_fit')
                model = cls(M=2, theta=.25, sv_selection_ratio=.5, gamma=.1, random_state=0, cache_dir=folder)
                with patch.object(cls, '_make_estimator', make), \
                        patch.object(module, '_runtime_provenance', return_value={'fixed_test_runtime': family}):
                    with self.assertRaises(FloatingPointError) as first:
                        model.fit(self.X, self.y)
                    fresh = exception_details(first.exception)['ensemble_failure_context']
                    self.assertEqual(fresh['base_fit_calls_completed'], 2)
                    self.assertEqual(fresh['cached_base_learners'], 0)
                    self.assertTrue(list(Path(folder).rglob('*.joblib')))
                    def fail_final(owner, settings):
                        return EvidenceLearner(family, 3, 'final_fit')
                    with patch.object(cls, '_make_estimator', fail_final):
                        with self.assertRaises(FloatingPointError) as repeated:
                            model.fit(self.X, self.y)
                    cached = exception_details(repeated.exception)['ensemble_failure_context']
                    self.assertEqual(cached['base_fit_calls_completed'], 0)
                    self.assertEqual(cached['base_score_calls_completed'], 0)
                    self.assertEqual(cached['cached_base_learners'], 2)
                    self.assertEqual(cached['base_work_source'], 'cache')
                    self.assert_clean(model)

    def test_logging_failure_and_malformed_metadata_keep_original_error(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluator = search_support.CandidateEvaluator(FailureDWD(), error_policy='raise', failure_dir=folder)
            with patch.object(search_support, 'write_json_atomic', side_effect=OSError('disk full')), \
                    warnings.catch_warnings():
                warnings.simplefilter('error')
                with self.assertRaisesRegex(FloatingPointError, 'controlled learner failure') as caught:
                    evaluator.fit(self.X, self.y)
            self.assertEqual(exception_details(caught.exception)['ensemble_failure_context']['stage'], 'base_fit')
        error = FloatingPointError('original')
        with patch.object(search_support, 'failure_context', side_effect=RuntimeError('metadata failed')):
            self.assertEqual(exception_details(error)['message'], 'original')

    def test_cache_write_failure_keeps_completed_current_base_work(self):
        for cls, module, family in ((EnsembleKernelDWD, dwd_module, 'dwd'),
                                    (EnsembleKernelSVM, svm_module, 'svm')):
            with self.subTest(cls=cls.__name__), tempfile.TemporaryDirectory() as folder:
                model = cls(M=2, theta=.25, sv_selection_ratio=.5, gamma=.1, random_state=0, cache_dir=folder)
                original = OSError('controlled cache write failure')
                with patch.object(cls, '_make_estimator', side_effect=lambda settings: EvidenceLearner(family, 1, 'none')), \
                        patch.object(module, '_runtime_provenance', return_value={'fixed_test_runtime': family}), \
                        patch.object(cache_module.joblib, 'dump', side_effect=original):
                    with self.assertRaises(OSError) as caught:
                        model.fit(self.X, self.y)
                self.assertIs(caught.exception, original)
                context = exception_details(original)['ensemble_failure_context']
                self.assertEqual(context['stage'], 'base_cache')
                self.assertEqual(context['base_fit_calls_completed'], 2)
                self.assertEqual(context['base_score_calls_completed'], 2)
                self.assertEqual(context['cached_base_learners'], 0)
                self.assertEqual(context['final_fit_calls_completed'], 0)
                self.assert_clean(model)

    def test_cached_svc_warning_as_error_keeps_reused_work(self):
        class LimitedLearner(EvidenceLearner):
            def fit(self, X, y):
                super().fit(X, y)
                self.fit_status_ = 1
                return self
        # Module-level pickle construction is unnecessary for this warning test:
        # cache payloads in lean mode retain diagnostics and consensus only.
        with tempfile.TemporaryDirectory() as folder:
            model = EnsembleKernelSVM(M=2, theta=.25, sv_selection_ratio=.5, gamma=.1,
                                      random_state=0, cache_dir=folder, store_base_estimators=False)
            with patch.object(EnsembleKernelSVM, '_make_estimator',
                              side_effect=lambda settings: LimitedLearner('svm', 1, 'none')), \
                    patch.object(svm_module, '_runtime_provenance', return_value={'fixed_test_runtime': 'svm'}):
                model.fit(self.X, self.y)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    with self.assertRaises(ConvergenceWarning) as caught:
                        model.fit(self.X, self.y)
            context = exception_details(caught.exception)['ensemble_failure_context']
            self.assertEqual(context['stage'], 'base_cache')
            self.assertEqual(context['base_fit_calls_completed'], 0)
            self.assertEqual(context['base_score_calls_completed'], 0)
            self.assertEqual(context['cached_base_learners'], 2)
            self.assert_clean(model)

    def test_attachment_failure_never_replaces_original_fit_exception(self):
        for cls, module in ((EnsembleKernelDWD, dwd_module), (EnsembleKernelSVM, svm_module)):
            for kind in (FloatingPointError, KeyboardInterrupt):
                with self.subTest(cls=cls.__name__, kind=kind.__name__):
                    model = cls(M=2, theta=.25, sv_selection_ratio=.5, gamma=.1, random_state=0)
                    original = kind('retain the original exception')
                    with patch.object(model, '_make_estimator', side_effect=original), \
                            patch.object(module, 'attach_failure_context', side_effect=RuntimeError('reporter failed')):
                        with self.assertRaises(kind) as caught:
                            model.fit(self.X, self.y)
                    self.assertIs(caught.exception, original)
                    self.assert_clean(model)

    def test_reused_exception_object_gets_fresh_context_for_each_fit(self):
        for cls, family in ((EnsembleKernelDWD, 'dwd'), (EnsembleKernelSVM, 'svm')):
            with self.subTest(cls=cls.__name__):
                model = cls(M=2, theta=.25, sv_selection_ratio=.5, gamma=.1, random_state=0)
                original = FloatingPointError('reused custom exception')
                for fail_at, stage, completed in ((2, 'base_fit', 1), (3, 'final_fit', 2)):
                    calls = []
                    def make(settings):
                        calls.append(len(calls) + 1)
                        if len(calls) == fail_at:
                            raise original
                        return EvidenceLearner(family, len(calls), 'none')
                    with patch.object(model, '_make_estimator', side_effect=make):
                        with self.assertRaises(FloatingPointError) as caught:
                            model.fit(self.X, self.y)
                    self.assertIs(caught.exception, original)
                    context = exception_details(original)['ensemble_failure_context']
                    self.assertEqual(context['stage'], stage)
                    self.assertEqual(context['base_fit_calls_completed'], completed)
                    self.assert_clean(model)

    def test_overridden_base_results_do_not_invent_completed_counts(self):
        for cls in (EnsembleKernelDWD, EnsembleKernelSVM):
            with self.subTest(cls=cls.__name__):
                model = cls(M=2, theta=.25, sv_selection_ratio=.5, gamma=.1, random_state=0)
                fabricated = {'consensus': np.zeros(len(self.y)), 'order': np.arange(len(self.y))}
                with patch.object(model, '_base_result', return_value=(fabricated, False, None, 'custom')):
                    with self.assertRaises(IneligibleSelectionError) as caught:
                        model.fit(self.X, self.y)
                context = exception_details(caught.exception)['ensemble_failure_context']
                self.assertIsNone(context['base_fit_calls_completed'])
                self.assertIsNone(context['cached_base_learners'])
                self.assertEqual(context['stage'], 'selection')
                self.assert_clean(model)

    def test_custom_callable_method_hooks_cannot_break_successful_fits(self):
        class HiddenFunction:
            def __init__(self, function):
                self.function = function
            def __getattribute__(self, name):
                if name == '__func__':
                    raise AssertionError('A reporting check must not invoke this hook')
                return object.__getattribute__(self, name)
            def __call__(self, *args, **kwargs):
                return self.function(*args, **kwargs)
        for cls in (FailureDWD, FailureSVM):
            baseline = cls('none').fit(self.X, self.y)
            for method in ('_base_stage', '_base_result'):
                for failure in ('none', 'final_fit'):
                    with self.subTest(cls=cls.__name__, method=method, failure=failure):
                        model = cls(failure)
                        setattr(model, method, HiddenFunction(getattr(model, method)))
                        if failure == 'none':
                            model.fit(self.X, self.y)
                            np.testing.assert_array_equal(model.decision_function(self.X),
                                                          baseline.decision_function(self.X))
                            np.testing.assert_array_equal(model.selected_indices_, baseline.selected_indices_)
                        else:
                            with self.assertRaisesRegex(FloatingPointError, 'controlled learner failure') as caught:
                                model.fit(self.X, self.y)
                            context = exception_details(caught.exception)['ensemble_failure_context']
                            self.assertEqual(context['stage'], 'final_fit')
                            self.assertIsNone(context['base_fit_calls_completed'])
                            self.assert_clean(model)

    def test_user_interrupt_during_success_path_inspection_propagates(self):
        for cls in (FailureDWD, FailureSVM):
            for kind in (KeyboardInterrupt, SystemExit):
                with self.subTest(cls=cls.__name__, kind=kind.__name__):
                    model = cls('none')
                    original = kind('interrupt a successful-path reporting check')
                    with patch.object(context_module, 'getattr_static', side_effect=original):
                        with self.assertRaises(kind) as caught:
                            model.fit(self.X, self.y)
                    self.assertIs(caught.exception, original)
                    self.assert_clean(model)


class CheckpointContextTests(unittest.TestCase):
    def test_attempt_context_and_ineligible_replay_are_durable(self):
        X, y = data()
        folds = [(np.arange(0, 40, 2), np.arange(1, 40, 2)),
                 (np.arange(1, 40, 2), np.arange(0, 40, 2))]
        runtime = {'python': 'failure-context-test', 'files_sha256': 'e' * 64,
                   'native_pools': [{'num_threads': 1, 'version': 'test'}]}
        for failure in ('base_fit', 'selection', 'interrupt'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder, \
                    patch.object(cv_checkpoint, 'runtime_fingerprint', side_effect=lambda **kw: deepcopy(runtime)):
                root = Path(folder)
                (root / 'failures').mkdir()
                def session(resume=False):
                    return cv_checkpoint.CheckpointedCV(
                        run_dir=root, estimator=FailureDWD(failure), X=X, y=y, folds=folds,
                        error_policy='nan' if failure == 'selection' else 'raise', resume=resume)
                if failure == 'selection':
                    with session() as checkpoint:
                        checkpoint.search({})
                    saved = {path.name: path.read_bytes() for path in (root / 'cv-checkpoints/folds').glob('*.json')}
                    self.assertEqual(len(saved), 2)
                    for value in saved.values():
                        record = json.loads(value)['payload']
                        self.assertEqual(record['state'], 'ineligible')
                        self.assertEqual(record['failure']['ensemble_failure_context']['stage'], 'selection')
                    with patch.object(FailureDWD, '_make_estimator', side_effect=AssertionError('must reuse')):
                        with session(resume=True) as checkpoint:
                            replay = checkpoint.search({})
                    self.assertEqual(replay.checkpoint_summary_['reused_folds'], 2)
                    self.assertEqual(saved, {path.name: path.read_bytes() for path in (root / 'cv-checkpoints/folds').glob('*.json')})
                else:
                    expected = KeyboardInterrupt if failure == 'interrupt' else FloatingPointError
                    with self.assertRaises(expected), session() as checkpoint:
                        checkpoint.search({})
                    attempts = list((root / 'cv-checkpoints/attempts').glob('*.json'))
                    self.assertEqual(len(attempts), 1)
                    wrapper = json.loads(attempts[0].read_text(encoding='utf-8'))
                    self.assertEqual(wrapper['sha256'], cv_checkpoint._digest(wrapper['payload']))
                    self.assertEqual(wrapper['payload']['failure']['ensemble_failure_context']['stage'], 'base_fit')
                    self.assertEqual(wrapper['payload']['state'], 'interrupted' if failure == 'interrupt' else 'failed')
                    self.assertFalse(list((root / 'cv-checkpoints/folds').glob('*.json')))


if __name__ == '__main__':
    unittest.main()
