"""Work reports distinguish observed new work from durable saved evidence."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ensemble_dwd import (EnsembleKernelDWD, EnsembleKernelSVM,
                          IneligibleSelectionError, OvOClassifier)

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import cv_checkpoint
from numerical_confirmation import NumericalConfirmationConfig
from search_support import new_run_directory


def accepted_review():
    return {'state': 'accepted', 'reasons': [],
            'stopping_reason': 'controlled lifecycle acceptance'}


class WorkReportingFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='work-reporting-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        x = np.linspace(-3., 3., 40)
        self.X = np.column_stack((x, x * x))
        self.y = (x > 0).astype(int)
        self.test_X = np.array([[-2.1, 4.41], [-.3, .09], [.4, .16], [1.9, 3.61]])
        self.test_y = np.array([0, 0, 1, 1])
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.runtime_patch = patch.object(cv_checkpoint, 'runtime_fingerprint',
                                          return_value={'test_runtime': 'work-reporting-v1'})
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    @staticmethod
    def pipeline(kind='svm', **options):
        settings = dict(M=1, theta=.5, gamma=.5, sv_selection_ratio=.8,
                        random_state=7, store_base_estimators=False)
        settings.update(options)
        if kind == 'dwd':
            model = EnsembleKernelDWD(lambd=.05, max_iter=2, stopping='fixed', **settings)
        else:
            model = EnsembleKernelSVM(C=1., **settings)
        return Pipeline([('scale', StandardScaler()), ('ensemble', model)])

    def search(self, folder, pipeline, grid, *, resume=False, seeds=None,
               reuse=False, jobs=1, X=None, y=None, folds=None, seed_parameter=None):
        options = {} if seeds is None else {
            'cv_seeds': seeds, 'seed_parameter': seed_parameter or 'ensemble__random_state'}
        with cv_checkpoint.CheckpointedCV(folder, pipeline,
                self.X if X is None else X, self.y if y is None else y,
                self.folds if folds is None else folds, resume=resume, jobs=jobs,
                reuse_equivalent_folds=reuse, **options) as checkpoint:
            return checkpoint.search(grid)

    @staticmethod
    def receipts(folder):
        return [json.loads(path.read_text(encoding='utf-8'))['payload']
                for path in (folder / 'cv-checkpoints/folds').glob('*.json')]

    def assert_counts(self, summary, *, executed=0, aliases=0, replayed=0):
        counts = summary['evaluations']
        self.assertEqual(counts['executed'], executed)
        self.assertEqual(counts['equivalent_aliases_created'], aliases)
        self.assertEqual(counts['checkpoint_replayed'], replayed)
        self.assertEqual(counts['total'], executed + aliases + replayed)


class CheckpointWorkReportingTests(WorkReportingFixture):
    def test_real_cold_and_warm_cache_reports_preserve_independent_fold_scores(self):
        for kind in ('dwd', 'svm'):
            with self.subTest(kind=kind):
                pipeline = self.pipeline(kind, cache_dir=str(self.root / (kind + '-cache')))
                grid = {'ensemble__sv_selection_ratio': [.8]}
                cold = self.search(new_run_directory(self.root / (kind + '-cold')), pipeline, grid)
                warm = self.search(new_run_directory(self.root / (kind + '-warm')), pipeline, grid)
                for result, hits, misses in ((cold, 0, 2), (warm, 2, 0)):
                    self.assert_counts(result.work_summary_, executed=2)
                    self.assertEqual(result.work_summary_, result.checkpoint_summary_['work_summary'])
                    cache = result.work_summary_['base_cache']
                    self.assertEqual(cache['reported_hits'], hits)
                    self.assertEqual(cache['reported_misses'], misses)
                    self.assertEqual(cache['evaluations_with_complete_cache_flags'], 2)
                    self.assertGreater(result.work_summary_['current_worker_seconds']['fit'], 0.)
                    self.assertEqual(result.work_summary_['historical_receipt_seconds']['replayed_fit'], 0.)
                expected = []
                with threadpool_limits(limits=1):
                    for training, validation in self.folds:
                        model = clone(pipeline).set_params(ensemble__cache_dir=None)
                        model.fit(self.X[training], self.y[training])
                        expected.append(accuracy_score(self.y[validation], model.predict(self.X[validation])))
                for index, score in enumerate(expected):
                    np.testing.assert_array_equal(cold.cv_results_[f'split{index}_test_score'], [score])
                    np.testing.assert_array_equal(warm.cv_results_[f'split{index}_test_score'], [score])

    def test_created_aliases_and_replayed_receipts_are_exclusive_and_keep_legacy_times(self):
        folder = new_run_directory(self.root / 'aliases')
        pipeline = self.pipeline()
        grid = {'ensemble__sv_selection_ratio': [.51, .525]}
        first = self.search(folder, pipeline, grid, reuse=True, seeds=(7, 13))
        self.assert_counts(first.work_summary_, executed=4, aliases=4)
        self.assertEqual(first.work_summary_['base_cache']['reported_misses'], 0)
        self.assertEqual(first.work_summary_['base_cache']['reported_disabled'], 4)
        self.assertGreater(first.work_summary_['current_worker_seconds']['equivalent_alias'], 0.)
        records = self.receipts(folder)
        self.assertEqual(len(records), 8)
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('replayed fit')), \
                patch.object(cv_checkpoint, '_reuse_fold', side_effect=AssertionError('replayed alias')):
            replay = self.search(folder, pipeline, grid, reuse=True, seeds=(7, 13), resume=True)
        self.assert_counts(replay.work_summary_, replayed=8)
        self.assertEqual(replay.work_summary_['evaluations']['replayed_equivalent_aliases'], 4)
        self.assertTrue(all(value == 0 for value in replay.work_summary_['current_worker_seconds'].values()))
        self.assertTrue(all(value == 0 for value in replay.work_summary_['base_cache'].values()))
        historical = replay.work_summary_['historical_receipt_seconds']
        self.assertAlmostEqual(historical['replayed_fit'], sum(row['fit_time'] for row in records))
        self.assertAlmostEqual(historical['replayed_score'], sum(row['score_time'] for row in records))
        self.assertAlmostEqual(historical['replayed_alias'], sum(
            row.get('equivalent_reuse', {}).get('reuse_seconds', 0.) for row in records))
        for name in ('mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                     'mean_test_score', 'rank_test_score'):
            np.testing.assert_array_equal(first.cv_results_[name], replay.cv_results_[name])
        # These older fields intentionally have different meanings: the alias
        # total includes saved aliases, whereas executed folds is current work.
        self.assertEqual(replay.checkpoint_summary_['equivalent_reused_folds'], 4)
        self.assertEqual(replay.checkpoint_summary_['executed_folds'], 0)
        self.assertGreater(float(replay.cv_results_['mean_fit_time'][0]), 0.)

    def test_ineligible_fits_report_executed_work_and_unknown_cache_evidence(self):
        folder = new_run_directory(self.root / 'ineligible')
        pipeline = self.pipeline()
        grid = {'ensemble__sv_selection_ratio': [.8]}
        with patch.object(EnsembleKernelSVM, 'fit',
                          side_effect=IneligibleSelectionError('controlled missing cache evidence')):
            result = self.search(folder, pipeline, grid)
        self.assert_counts(result.work_summary_, executed=2)
        self.assertEqual(result.work_summary_['evaluations']['executed_ineligible'], 2)
        cache = result.work_summary_['base_cache']
        self.assertEqual(cache['reported_hits'], 0)
        self.assertEqual(cache['reported_misses'], 0)
        self.assertEqual(cache['evaluations_with_no_cache_flags'], 2)
        self.assertTrue(np.isnan(result.cv_results_['mean_test_score']).all())
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('ineligible replay fit')):
            replay = self.search(folder, pipeline, grid, resume=True)
        self.assert_counts(replay.work_summary_, replayed=2)
        self.assertEqual(replay.work_summary_['evaluations']['replayed_ineligible'], 2)
        self.assertTrue(all(value == 0 for value in replay.work_summary_['base_cache'].values()))

    def test_ovo_cache_flags_count_observed_pairs_without_counting_final_learners(self):
        X, y = make_classification(n_samples=60, n_features=4, n_informative=3,
            n_redundant=0, n_classes=3, n_clusters_per_class=1, class_sep=2., random_state=91)
        folds = list(StratifiedKFold(2, shuffle=True, random_state=29).split(X, y))
        binary = self.pipeline().named_steps['ensemble']
        binary.set_params(theta=.8, cache_dir=str(self.root / 'ovo-cache'))
        pipeline = Pipeline([('scale', StandardScaler()),
                             ('ensemble', OvOClassifier(binary, n_jobs=1))])
        grid = {'ensemble__estimator__sv_selection_ratio': [.8]}
        for mode, hits, misses in (('cold', 0, 6), ('warm', 6, 0)):
            with self.subTest(mode=mode):
                result = self.search(new_run_directory(self.root / ('ovo-' + mode)), pipeline,
                                     grid, X=X, y=y, folds=folds)
                self.assert_counts(result.work_summary_, executed=2)
                cache = result.work_summary_['base_cache']
                self.assertEqual(cache['reported_hits'], hits)
                self.assertEqual(cache['reported_misses'], misses)
                self.assertEqual(cache['evaluations_with_complete_cache_flags'], 2)
                self.assertEqual(cache['unknown_diagnostic_nodes'], 0)

    def test_parallel_multiseed_counts_each_fold_seed_once_and_matches_serial_scores(self):
        self.runtime_patch.stop()
        pipeline = self.pipeline()
        grid = {'ensemble__sv_selection_ratio': [.6, .8]}
        serial = self.search(new_run_directory(self.root / 'serial'), pipeline, grid, seeds=(7, 13))
        parallel = self.search(new_run_directory(self.root / 'parallel'), pipeline, grid,
                               seeds=(7, 13), jobs=2)
        for result in (serial, parallel):
            self.assert_counts(result.work_summary_, executed=8)
            self.assertEqual(result.work_summary_['base_cache']['reported_misses'], 0)
            self.assertEqual(result.work_summary_['base_cache']['reported_disabled'], 8)
            self.assertEqual(result.work_summary_['search_call_count'], 1)
            self.assertGreater(result.work_summary_['search_call_wall_seconds'], 0.)
            self.assertEqual(result.checkpoint_summary_['seed_count'], 2)
            self.assertEqual(result.checkpoint_summary_['evaluation_count'], 8)
        for name in ('params', 'split0_test_score', 'split1_test_score', 'mean_test_score', 'rank_test_score'):
            np.testing.assert_equal(serial.cv_results_[name], parallel.cv_results_[name])

    def test_checkpoint_summary_failure_preserves_scores_and_committed_receipts(self):
        folder = new_run_directory(self.root / 'checkpoint-report-failure')
        pipeline = self.pipeline()
        grid = {'ensemble__sv_selection_ratio': [.6, .8]}
        with patch.object(cv_checkpoint, 'summarize_records',
                          side_effect=RuntimeError('controlled checkpoint summary failure')), \
                patch.object(cv_checkpoint, '_fit_fold', wraps=cv_checkpoint._fit_fold) as fits:
            result = self.search(folder, pipeline, grid)
        self.assertEqual(fits.call_count, 4)
        self.assertEqual(len(self.receipts(folder)), 4)
        self.assertIsNone(result.work_summary_)
        self.assertIsNone(result.checkpoint_summary_['work_summary'])
        self.assertTrue(np.isfinite(result.cv_results_['mean_test_score']).all())
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('saved score replay')):
            replay = self.search(folder, pipeline, grid, resume=True)
        self.assert_counts(replay.work_summary_, replayed=4)
        for name in ('params', 'split0_test_score', 'split1_test_score', 'mean_test_score',
                     'mean_fit_time', 'mean_score_time', 'rank_test_score'):
            np.testing.assert_equal(result.cv_results_[name], replay.cv_results_[name])


class AdaptiveWorkReportingTests(WorkReportingFixture):
    def execute(self, folder, *, resume=False, confirmation=False, reuse=False,
                seeds=None, hold=False, grid=None):
        pipeline = self.pipeline('dwd')
        grid = ({'ensemble__lambd': [.05], 'ensemble__gamma': [.5],
                 'ensemble__sv_selection_ratio': [.51, .525]} if grid is None else grid)
        review = ({'state': 'needs_review', 'reasons': ['controlled coefficient hold'],
                   'stopping_reason': 'controlled coefficient hold'} if hold else accepted_review())
        proposals = [
            {'phase': 'coarse', 'candidates': list(ParameterGrid(grid)), 'state': {}, 'review': None},
            {'phase': 'done', 'candidates': [], 'state': {}, 'review': review}]
        config = adaptive_search.AdaptiveSearchConfig(
            regularization='lambd', prefix='ensemble__', refine=False, max_candidates=2)
        options = {} if seeds is None else {'cv_seeds': seeds, 'refit_seed': 97}
        with patch.object(adaptive_search, 'next_search_stage', side_effect=proposals), \
                contextlib.redirect_stdout(io.StringIO()):
            return adaptive_search.run_adaptive_search(
                pipeline, grid, self.X, self.y, self.test_X, self.test_y,
                self.folds, folder, config=config, resume=resume,
                numerical_confirmation=NumericalConfirmationConfig(max_iter=4, top_k=2)
                if confirmation else None, reuse_equivalent_folds=reuse, **options)

    @staticmethod
    def public_summary(folder):
        return json.loads((folder / 'work-summary.json').read_text(encoding='utf-8'))

    def test_confirmation_counts_only_child_work_and_retains_separate_phase_timings(self):
        folder = new_run_directory(self.root / 'confirmation')
        result = self.execute(folder, confirmation=True, reuse=True, seeds=(7, 13))
        summary = result['work_summary']
        self.assertEqual(self.public_summary(folder), summary)
        self.assertEqual(summary['scope'], 'current_invocation')
        self.assertEqual(summary['state'], 'completed')
        self.assertEqual(summary['coverage'], 'complete')
        self.assertIsNone(summary['incomplete_phase'])
        self.assert_counts(summary['original_search']['work_summary'], executed=4, aliases=4)
        self.assert_counts(summary['numerical_confirmation']['work_summary'], executed=4, aliases=4)
        self.assert_counts(summary['cv_totals'], executed=8, aliases=8)
        self.assertEqual(summary['cv_totals']['search_call_count'], 2)
        self.assertEqual(len(summary['original_search']['stages']), 1)
        self.assertEqual(summary['original_search']['stages'][0]['name'], 'coarse')
        confirmation = result['numerical_confirmation']
        self.assertEqual(summary['numerical_confirmation']['work_summary'],
                         confirmation['checkpoint_summary']['work_summary'])
        self.assertEqual(summary['numerical_confirmation']['state'], 'stable')
        # The confirmation report contains baseline receipts for comparison;
        # they must not be counted as a third CV stage or fitted a second time.
        self.assertIn('baseline_stage_summaries', confirmation)
        self.assertEqual(summary['final_refit']['state'], 'completed')
        self.assertTrue(summary['final_refit']['attempted'])
        self.assertAlmostEqual(summary['final_refit']['wall_seconds'], result['completed']['refit_seconds'])
        self.assertEqual(summary['held_out_scoring']['state'], 'completed')
        self.assertTrue(summary['held_out_scoring']['attempted'])
        self.assertGreater(summary['held_out_scoring']['wall_seconds'], 0.)
        self.assertGreater(summary['elapsed_seconds'], 0.)
        self.assertIn('completion_work_summary', result)
        self.assert_counts(result['completion_work_summary']['cv_totals'], executed=8, aliases=8)

    def test_completed_replay_zeroes_current_work_and_preserves_completion_history(self):
        folder = new_run_directory(self.root / 'completed-replay')
        first = self.execute(folder)
        saved_history = first['completion_work_summary']
        self.assertGreater(saved_history['cv_totals']['current_worker_seconds']['fit'], 0.)
        (folder / 'work-summary.json').unlink()
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('completed replay CV')), \
                patch.object(adaptive_search, 'clone', side_effect=AssertionError('completed replay refit')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('completed replay score')):
            replay = self.execute(folder, resume=True)
        summary = replay['work_summary']
        self.assertEqual(self.public_summary(folder), summary)
        self.assertEqual(summary['state'], 'completed_replay')
        self.assertNotEqual(summary['invocation_id'], first['work_summary']['invocation_id'])
        self.assert_counts(summary['cv_totals'])
        self.assertEqual(summary['cv_totals']['search_call_count'], 0)
        self.assertEqual(summary['original_search']['stages'], [])
        for field in ('current_worker_seconds', 'historical_receipt_seconds'):
            self.assertTrue(all(value == 0 for value in summary['cv_totals'][field].values()))
        for phase in ('final_refit', 'held_out_scoring'):
            self.assertEqual(summary[phase]['state'], 'not_performed_replay')
            self.assertFalse(summary[phase]['attempted'])
            self.assertEqual(summary[phase]['wall_seconds'], 0.)
        self.assertTrue(summary['historical_completion_summary_available'])
        self.assertEqual(replay['completion_work_summary'], saved_history)
        self.assertEqual(replay['completed'], first['completed'])
        self.assertEqual(replay['stages'], first['stages'])

    def test_interrupted_original_stage_reports_incomplete_coverage_then_resume_counts_receipts(self):
        folder = new_run_directory(self.root / 'coarse-interrupt')
        original = cv_checkpoint._fit_fold
        attempts = 0

        def interrupt_second(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise KeyboardInterrupt('controlled original-stage interruption')
            return original(*args, **kwargs)

        with patch.object(cv_checkpoint, '_fit_fold', interrupt_second), \
                self.assertRaisesRegex(KeyboardInterrupt, 'original-stage'):
            self.execute(folder)
        interrupted = self.public_summary(folder)
        self.assertEqual(interrupted['state'], 'interrupted')
        self.assertEqual(interrupted['coverage'], 'completed_stages_only')
        self.assertIsNotNone(interrupted['incomplete_phase'])
        self.assertIn('coarse', json.dumps(interrupted['incomplete_phase']))
        self.assert_counts(interrupted['cv_totals'])
        self.assertEqual(interrupted['original_search']['stages'], [])
        # A completed worker receipt exists, but the interrupted search did not
        # return an exact coordinator summary; zero is only a completed-stage subtotal.
        self.assertEqual(len(self.receipts(folder)), 1)
        with patch.object(cv_checkpoint, '_fit_fold', wraps=original) as fits:
            resumed = self.execute(folder, resume=True)
        self.assertEqual(fits.call_count, 3)
        current = resumed['work_summary']
        self.assertEqual(current['coverage'], 'complete')
        self.assert_counts(current['cv_totals'], executed=3, replayed=1)
        self.assertGreater(current['cv_totals']['historical_receipt_seconds']['replayed_fit'], 0.)
        self.assertEqual(len(self.receipts(folder)), 4)

    def test_interrupted_confirmation_keeps_original_subtotal_and_resumes_without_double_count(self):
        folder = new_run_directory(self.root / 'confirmation-interrupt')
        original = cv_checkpoint._fit_fold
        confirmation_attempts = 0

        def interrupt_second_confirmation(*args, **kwargs):
            nonlocal confirmation_attempts
            if 'numerical-confirmation' in Path(args[0]).parts:
                confirmation_attempts += 1
                if confirmation_attempts == 2:
                    raise KeyboardInterrupt('controlled child-stage interruption')
            return original(*args, **kwargs)

        with patch.object(cv_checkpoint, '_fit_fold', interrupt_second_confirmation), \
                self.assertRaisesRegex(KeyboardInterrupt, 'child-stage'):
            self.execute(folder, confirmation=True)
        interrupted = self.public_summary(folder)
        self.assertEqual(interrupted['coverage'], 'completed_stages_only')
        self.assertIn('numerical_confirmation', json.dumps(interrupted['incomplete_phase']))
        self.assert_counts(interrupted['cv_totals'], executed=4)
        self.assertIsNone(interrupted['numerical_confirmation']['work_summary'])
        self.assertEqual(len(self.receipts(folder / 'numerical-confirmation')), 1)
        with patch.object(cv_checkpoint, '_fit_fold', wraps=original) as fits:
            resumed = self.execute(folder, resume=True, confirmation=True)
        self.assertEqual(fits.call_count, 3)
        current = resumed['work_summary']
        self.assert_counts(current['original_search']['work_summary'], replayed=4)
        self.assert_counts(current['numerical_confirmation']['work_summary'], executed=3, replayed=1)
        self.assert_counts(current['cv_totals'], executed=3, replayed=5)

    def test_review_hold_reports_unstarted_confirmation_refit_and_scoring(self):
        folder = new_run_directory(self.root / 'hold')
        with patch.object(adaptive_search, 'clone', side_effect=AssertionError('held final fit')), \
                patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('held score')):
            result = self.execute(folder, confirmation=True, hold=True)
        summary = result['work_summary']
        self.assertEqual(summary['state'], 'needs_review')
        self.assert_counts(summary['cv_totals'], executed=4)
        self.assertEqual(summary['numerical_confirmation']['state'], 'not_started')
        self.assertIsNone(summary['numerical_confirmation']['work_summary'])
        for phase in ('final_refit', 'held_out_scoring'):
            self.assertEqual(summary[phase]['state'], 'not_started')
            self.assertFalse(summary[phase]['attempted'])
            self.assertEqual(summary[phase]['wall_seconds'], 0.)

    def test_identity_mismatch_preserves_published_summary_before_any_new_work(self):
        folder = new_run_directory(self.root / 'identity')
        self.execute(folder)
        names = ('work-summary.json', 'run.json', 'status.json')
        previous = {name: (folder / name).read_bytes() for name in names}
        with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('mismatched identity CV')), \
                self.assertRaises(cv_checkpoint.CheckpointError):
            self.execute(folder, resume=True, seeds=(7, 13))
        for name, data in previous.items():
            self.assertEqual((folder / name).read_bytes(), data)

    def test_summary_publication_failure_preserves_success_and_original_interruption(self):
        original_write = adaptive_search.write_json_atomic

        def fail_summary_only(path, *args, **kwargs):
            if Path(path).name == 'work-summary.json':
                raise OSError('controlled reporting write failure')
            return original_write(path, *args, **kwargs)

        folder = new_run_directory(self.root / 'publication-success')
        with patch.object(adaptive_search, 'write_json_atomic', fail_summary_only), \
                patch.object(cv_checkpoint, '_fit_fold', wraps=cv_checkpoint._fit_fold) as fits, \
                patch.object(adaptive_search, 'clone', wraps=adaptive_search.clone) as refits:
            result = self.execute(folder)
        self.assertIsNotNone(result['model'])
        self.assertIsNotNone(result['completed'])
        self.assertEqual(fits.call_count, 4)
        self.assertEqual(refits.call_count, 1)
        self.assertEqual(result['work_summary']['state'], 'completed')
        self.assertTrue((folder / 'completed.json').exists())
        failed_folder = new_run_directory(self.root / 'publication-interruption')
        with patch.object(adaptive_search, 'write_json_atomic', fail_summary_only), \
                patch.object(cv_checkpoint, '_fit_fold',
                             side_effect=KeyboardInterrupt('original worker interruption')), \
                self.assertRaisesRegex(KeyboardInterrupt, 'original worker interruption'):
            self.execute(failed_folder)
        self.assertEqual(json.loads((failed_folder / 'status.json').read_text())['state'], 'interrupted')

    def test_summary_calculation_failure_cannot_undo_successful_fits_or_completion(self):
        for helper in ('aggregate_work_summaries', 'summarize_cache_diagnostics'):
            folder = new_run_directory(self.root / helper)
            with self.subTest(helper=helper), \
                    patch.object(adaptive_search, helper,
                                 side_effect=ValueError('controlled reporting calculation failure')), \
                    patch.object(cv_checkpoint, '_fit_fold', wraps=cv_checkpoint._fit_fold) as fits, \
                    patch.object(adaptive_search, 'clone', wraps=adaptive_search.clone) as refits:
                result = self.execute(folder)
            self.assertIsNotNone(result['model'])
            self.assertIsNotNone(result['completed'])
            self.assertEqual(fits.call_count, 4)
            self.assertEqual(refits.call_count, 1)
            self.assertTrue((folder / 'completed.json').exists())
            self.assertTrue((folder / 'cv-checkpoints/artifacts/completed.json').exists())
            with patch.object(cv_checkpoint, '_fit_fold', side_effect=AssertionError('completed replay CV')), \
                    patch.object(adaptive_search, 'clone', side_effect=AssertionError('completed replay refit')):
                replay = self.execute(folder, resume=True)
            self.assertTrue(replay['resumed_completed'])
            self.assertEqual(replay['completed'], result['completed'])
            self.assert_counts(replay['work_summary']['cv_totals'])

    def test_refit_failure_and_interruption_report_attempt_without_scoring(self):
        for exception in (RuntimeError('controlled final-fit failure'),
                          KeyboardInterrupt('controlled final-fit interruption')):
            state = 'interrupted' if isinstance(exception, KeyboardInterrupt) else 'failed'
            folder = new_run_directory(self.root / ('refit-' + state))
            with self.subTest(state=state), \
                    patch.object(adaptive_search, 'clone', side_effect=exception), \
                    patch.object(adaptive_search, 'accuracy_score', side_effect=AssertionError('failed refit score')), \
                    self.assertRaises(type(exception)):
                self.execute(folder)
            summary = self.public_summary(folder)
            self.assertEqual(summary['state'], state)
            self.assert_counts(summary['cv_totals'], executed=4)
            self.assertEqual(summary['final_refit']['state'], state)
            self.assertTrue(summary['final_refit']['attempted'])
            self.assertGreaterEqual(summary['final_refit']['wall_seconds'], 0.)
            self.assertFalse(summary['held_out_scoring']['attempted'])


if __name__ == '__main__':
    unittest.main()
