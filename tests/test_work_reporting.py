"""Current work, receipt history and observed cache flags remain distinct."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.dummy import DummyClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint
from work_reporting import (aggregate_work_summaries, empty_work_summary,
                            summarize_cache_diagnostics, summarize_records)


def receipt(*, fit=2., score_time=.25, score=.75, diagnostics=None,
            state='complete', alias=None, task='task'):
    result = {'state': state, 'fit_time': fit, 'score_time': score_time,
              'score': score if state == 'complete' else None,
              'task_sha256': task, 'parameters': {'parameter': task}}
    if diagnostics is not None:
        result['diagnostics'] = diagnostics
    if alias is not None:
        result.update(fit_time=0., score_time=0.)
        result['equivalent_reuse'] = {
            'reuse_seconds': alias, 'source_fit_time': 9000.,
            'source_score_time': 3000., 'source_task_sha256': 'source'}
    return result


class WorkSummaryTests(unittest.TestCase):
    def assert_partition(self, summary):
        counts = summary['evaluations']
        self.assertEqual(counts['total'], sum(counts[name] for name in
                         ('executed', 'equivalent_aliases_created', 'checkpoint_replayed')))
        cache = summary['base_cache']
        self.assertEqual(counts['executed'], sum(cache[name] for name in
                         ('evaluations_with_complete_cache_flags',
                          'evaluations_with_partial_cache_flags',
                          'evaluations_with_no_cache_flags')))

    def test_empty_summaries_have_independent_nested_values(self):
        first, second = empty_work_summary(), empty_work_summary()
        first['evaluations']['executed'] = 19
        first['current_worker_seconds']['fit'] = 27.
        self.assertEqual(second, summarize_records({}))
        self.assertEqual(second, aggregate_work_summaries([]))
        self.assert_partition(second)

    def test_new_execution_alias_and_replay_partition_includes_ineligibility(self):
        records = {
            (0, 0): (receipt(fit=10., score_time=2., diagnostics={
                'cache_hit_': True, 'base_diagnostics_': [{'cache_hit_': False}],
                'final_estimator': {'cache_hit_': False}}), False),
            (0, 1): (receipt(fit=.5, score_time=.01, state='ineligible'), False),
            (1, 0): (receipt(alias=.25, diagnostics={'cache_hit_': False}), False),
            (1, 1): (receipt(fit=7., score_time=3., diagnostics={'cache_hit_': True}), True),
            (2, 0): (receipt(alias=.75, diagnostics={'cache_hit_': False}), True),
            (2, 1): (receipt(fit=5., score_time=1., state='ineligible'), True),
        }
        original = deepcopy(records)
        summary = summarize_records(records)
        self.assertEqual(summary['evaluations'], {
            'total': 6, 'executed': 2, 'equivalent_aliases_created': 1,
            'checkpoint_replayed': 3, 'executed_ineligible': 1,
            'replayed_ineligible': 1, 'replayed_equivalent_aliases': 1})
        self.assertEqual(summary['current_worker_seconds'],
                         {'fit': 10.5, 'score': 2.01, 'equivalent_alias': .25})
        self.assertEqual(summary['historical_receipt_seconds'],
                         {'replayed_fit': 12., 'replayed_score': 4., 'replayed_alias': .75})
        self.assertEqual(summary['base_cache'], {
            'reported_hits': 1, 'reported_misses': 0,
            'reported_disabled': 0, 'reported_bypassed': 0,
            'reported_non_hits_unknown_status': 0, 'unknown_diagnostic_nodes': 0,
            'evaluations_with_complete_cache_flags': 1,
            'evaluations_with_partial_cache_flags': 0,
            'evaluations_with_no_cache_flags': 1})
        self.assertEqual(records, original)
        self.assertEqual(summary, summarize_records(iter(records.values())))
        self.assert_partition(summary)

    def test_pairwise_cache_observations_ignore_base_and_final_diagnostics(self):
        facts = {'cache_hit_': True, 'pairwise_estimators': [
            {'pair': [0, 1], 'diagnostics': {'cache_hit_': True,
             'base_diagnostics_': [{'cache_hit_': False}],
             'final_estimator': {'cache_hit_': False}}},
            {'pair': [0, 2], 'diagnostics': {'cache_hit_': False, 'cache_status_': 'miss'}},
            {'pair': [0, 3], 'diagnostics': {}},
            {'pair': [1, 2]},
            {'pair': [1, 3], 'diagnostics': {'pairwise_estimators': [
                {'diagnostics': {'cache_hit_': True}}]}},
        ]}
        self.assertEqual(summarize_cache_diagnostics(facts), {
            'reported_hits': 2, 'reported_misses': 1,
            'reported_disabled': 0, 'reported_bypassed': 0,
            'reported_non_hits_unknown_status': 0, 'unknown_diagnostic_nodes': 1,
            'evaluations_with_complete_cache_flags': 0,
            'evaluations_with_partial_cache_flags': 1,
            'evaluations_with_no_cache_flags': 0})
        summary = summarize_records([(receipt(diagnostics=facts), False)])
        self.assertEqual(summary['base_cache'], summarize_cache_diagnostics(facts))
        self.assert_partition(summary)

    def test_missing_pair_information_does_not_invent_missing_pair_counts(self):
        for facts in (None, {'pairwise_estimators': []}, {'pairwise_estimators': None},
                      {'pairwise_estimators': [{}]}, {'pairwise_estimators': [None]}):
            with self.subTest(facts=facts):
                summary = summarize_cache_diagnostics(facts)
                self.assertEqual(summary['reported_hits'], 0)
                self.assertEqual(summary['reported_misses'], 0)
                self.assertEqual(summary['unknown_diagnostic_nodes'], 0)
                self.assertEqual(summary['evaluations_with_no_cache_flags'], 1)
                self.assertEqual(summary['evaluations_with_complete_cache_flags'], 0)
        partial = summarize_cache_diagnostics({'pairwise_estimators': [
            {'diagnostics': {'cache_hit_': False, 'cache_status_': 'miss'}}, {}]})
        self.assertEqual(partial['unknown_diagnostic_nodes'], 0)
        self.assertEqual(partial['reported_misses'], 1)
        self.assertEqual(partial['evaluations_with_partial_cache_flags'], 1)

    def test_only_boolean_cache_flags_are_reported_observations(self):
        for flag in (None, 0, 1, '', 'False', [], {}):
            with self.subTest(flag=flag):
                summary = summarize_cache_diagnostics({'cache_hit_': flag})
                self.assertEqual(summary['reported_hits'], 0)
                self.assertEqual(summary['reported_misses'], 0)
                self.assertEqual(summary['unknown_diagnostic_nodes'], 1)
                self.assertEqual(summary['evaluations_with_no_cache_flags'], 1)
        self.assertEqual(summarize_cache_diagnostics({})['unknown_diagnostic_nodes'], 1)
        for flag in (False, True):
            summary = summarize_cache_diagnostics(
                {'cache_hit_': flag, 'cache_status_': 'hit' if flag else 'miss'})
            self.assertEqual(summary['reported_hits'], int(flag))
            self.assertEqual(summary['reported_misses'], int(not flag))
            self.assertEqual(summary['evaluations_with_complete_cache_flags'], 1)

    def test_nonhits_distinguish_miss_disabled_bypassed_and_unknown_reason(self):
        cases = [('miss', 'reported_misses'), ('disabled', 'reported_disabled'),
                 ('bypassed: random_state=None', 'reported_bypassed'),
                 ('bypassed: custom estimator subclass', 'reported_bypassed'),
                 (None, 'reported_non_hits_unknown_status'),
                 ('unrecognized', 'reported_non_hits_unknown_status'),
                 ('hit', 'reported_non_hits_unknown_status')]
        for status, category in cases:
            with self.subTest(status=status):
                facts = {'cache_hit_': False}
                if status is not None:
                    facts['cache_status_'] = status
                summary = summarize_cache_diagnostics(facts)
                for name in ('reported_hits', 'reported_misses', 'reported_disabled',
                             'reported_bypassed', 'reported_non_hits_unknown_status'):
                    self.assertEqual(summary[name], int(name == category))
                self.assertEqual(summary['unknown_diagnostic_nodes'], 0)
                self.assertEqual(summary['evaluations_with_complete_cache_flags'], 1)
        inconsistent = summarize_cache_diagnostics({'cache_hit_': True, 'cache_status_': 'miss'})
        self.assertEqual(inconsistent['reported_hits'], 0)
        self.assertEqual(inconsistent['reported_misses'], 0)
        self.assertEqual(inconsistent['unknown_diagnostic_nodes'], 1)
        self.assertEqual(inconsistent['evaluations_with_no_cache_flags'], 1)

    def test_aggregate_adds_each_completed_call_once_and_preserves_input(self):
        first = summarize_records([(receipt(fit=2.), False), (receipt(alias=.2), False)])
        second = summarize_records([(receipt(fit=2.), True), (receipt(alias=.2), True),
                                    (receipt(fit=3.), False)])
        first.update(search_call_count=1, search_call_wall_seconds=.4)
        second.update(search_call_count=1, search_call_wall_seconds=.6)
        originals = deepcopy([first, second])
        summary = aggregate_work_summaries([first, second])
        self.assertEqual(summary['evaluations']['total'], 5)
        self.assertEqual(summary['evaluations']['executed'], 2)
        self.assertEqual(summary['evaluations']['checkpoint_replayed'], 2)
        self.assertEqual(summary['current_worker_seconds']['fit'], 5.)
        self.assertEqual(summary['historical_receipt_seconds']['replayed_fit'], 2.)
        self.assertEqual(summary['current_worker_seconds']['equivalent_alias'], .2)
        self.assertEqual(summary['historical_receipt_seconds']['replayed_alias'], .2)
        self.assertEqual(summary['search_call_count'], 2)
        self.assertEqual(summary['search_call_wall_seconds'], 1.)
        self.assertEqual([first, second], originals)
        self.assert_partition(summary)
        wrong = empty_work_summary()
        wrong['schema'] = 999
        with self.assertRaises(ValueError):
            aggregate_work_summaries([wrong])


class SearchSummaryAttachmentTests(unittest.TestCase):
    def test_public_single_seed_timer_includes_all_private_search_work(self):
        store = object.__new__(checkpoint.CheckpointedCV)
        store.cv_seeds, store.identity = None, None
        work = summarize_records([(receipt(), False)])
        expected = SimpleNamespace(work_summary_=work,
                                   checkpoint_summary_={'wall_seconds': 3., 'work_summary': work})
        store._search_single_seed = Mock(return_value=expected)
        with patch.object(checkpoint, 'perf_counter', side_effect=[100., 109.]):
            result = store.search({'offset': [0.]}, stage='example')
        self.assertIs(result, expected)
        self.assertIs(result.work_summary_, result.checkpoint_summary_['work_summary'])
        self.assertEqual(result.work_summary_['search_call_wall_seconds'], 9.)
        self.assertEqual(result.work_summary_['search_call_count'], 1)
        self.assertEqual(result.checkpoint_summary_['wall_seconds'], 3.)
        store._search_single_seed.assert_called_once_with({'offset': [0.]}, stage='example')

    def test_missing_auxiliary_summary_does_not_block_single_seed_result(self):
        store = object.__new__(checkpoint.CheckpointedCV)
        store.cv_seeds, store.identity = None, None
        expected = SimpleNamespace(work_summary_=None, cv_results_={'mean_test_score': [1.]},
                                   checkpoint_summary_={'work_summary': None})
        store._search_single_seed = Mock(return_value=expected)
        self.assertIs(store.search({'offset': [0.]}), expected)
        self.assertEqual(expected.cv_results_['mean_test_score'], [1.])

    def test_public_multiseed_timer_starts_before_checks_and_preserves_legacy_start(self):
        store = object.__new__(checkpoint.CheckpointedCV)
        store.cv_seeds, store.seed_parameter = (7, 13), 'random_state'
        store.identity = {'seed_policy': store._seed_policy()}
        store._active = Mock()
        store.estimator = DummyClassifier(random_state=42)
        work = summarize_records([(receipt(), False)])
        raw = SimpleNamespace(work_summary_=work,
                              checkpoint_summary_={'wall_seconds': 4., 'work_summary': work})
        store._search_single_seed = Mock(return_value=raw)
        store._aggregate_seeds = Mock(return_value=raw)
        with patch.object(checkpoint, 'perf_counter', side_effect=[10., 12., 19.]):
            result = store.search({'strategy': ['most_frequent']}, stage='multiseed')
        store._active.assert_called_once_with()
        store._search_single_seed.assert_called_once_with([
            {'strategy': ['most_frequent'], 'random_state': [7]},
            {'strategy': ['most_frequent'], 'random_state': [13]}],
            stage='multiseed', _include_records=True)
        store._aggregate_seeds.assert_called_once_with([{'strategy': 'most_frequent'}], raw, 12.)
        self.assertEqual(result.work_summary_['search_call_wall_seconds'], 9.)
        self.assertEqual(result.work_summary_['search_call_count'], 1)
        self.assertEqual(result.checkpoint_summary_['wall_seconds'], 4.)

    def test_multiseed_aggregation_keeps_raw_work_once_and_legacy_scores_times(self):
        store = object.__new__(checkpoint.CheckpointedCV)
        store.cv_seeds, store.seed_parameter = (7, 13), 'random_state'
        store.folds, store.reuse_equivalent_folds = [None, None], True
        candidates = [{'offset': -1.}, {'offset': 1.}]
        records = {}
        scores = np.empty((2, 2, 2))
        fits = np.empty_like(scores)
        for ci in range(2):
            for si in range(2):
                for fi in range(2):
                    index = ci * 4 + si * 2 + fi
                    record = receipt(fit=float(index + 1), score_time=.5,
                                     score=.5 + index * .05, task=str(index),
                                     alias=.2 if index in (1, 7) else None)
                    records[ci * 2 + si, fi] = record, index in (0, 1, 4)
                    scores[ci, si, fi], fits[ci, si, fi] = record['score'], record['fit_time']
        work = summarize_records(records)
        prior = deepcopy(work)
        raw = SimpleNamespace(_fold_records_=records, work_summary_=work,
                              checkpoint_summary_={'work_summary': work, 'wall_seconds': 2.,
                                                   'new_folds': 5, 'reused_folds': 3})
        with patch.object(checkpoint, 'perf_counter', return_value=10.):
            result = store._aggregate_seeds(candidates, raw, 6.)
        self.assertIs(result.work_summary_, work)
        self.assertIs(result.checkpoint_summary_['work_summary'], work)
        self.assertEqual(result.work_summary_, prior)
        self.assertEqual(result.work_summary_['evaluations']['total'], 8)
        self.assertEqual(result.work_summary_['evaluations']['executed'], 4)
        self.assertEqual(result.work_summary_['evaluations']['equivalent_aliases_created'], 1)
        self.assertEqual(result.work_summary_['evaluations']['checkpoint_replayed'], 3)
        self.assertEqual(result.work_summary_['evaluations']['replayed_equivalent_aliases'], 1)
        self.assertEqual(result.checkpoint_summary_['equivalent_reuse'], result.seed_results_['equivalent_reuse'])
        self.assertEqual(result.checkpoint_summary_['wall_seconds'], 4.)
        self.assertEqual(result.checkpoint_summary_['new_folds'], 5)
        self.assertEqual(result.checkpoint_summary_['reused_folds'], 3)
        np.testing.assert_array_equal(result.cv_results_['mean_test_score'], scores.mean(axis=1).mean(axis=1))
        np.testing.assert_array_equal(result.cv_results_['mean_fit_time'], fits.sum(axis=1).mean(axis=1))
        np.testing.assert_array_equal(result.cv_results_['std_fit_time'], fits.sum(axis=1).std(axis=1))
        np.testing.assert_array_equal(result.seed_results_['scores'], scores)
        np.testing.assert_array_equal(result.seed_results_['fit_times'], fits)
        self.assertEqual(result.best_index_, 1)


if __name__ == '__main__':
    unittest.main()
