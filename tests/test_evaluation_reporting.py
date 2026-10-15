"""Selection and numerical reports preserve evidence, not an inferred ideal fit."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import balanced_accuracy_score

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
from evaluation_reporting import (classification_report, numerical_check_summary,
                                  selection_composition)
from work_reporting import summarize_records, aggregate_work_summaries


def learner(checks, refinement=.3, compensated=.7, native=.6, wrapper='solver_diagnostics'):
    return {wrapper: {'linear_system_diagnostics': {
        'native_refinement_trials': checks, 'native_refinement_discarded': checks,
        'native_refinement_acceptances': 0, 'native_refinement_seconds': refinement,
        'compensated_residual_checks': checks, 'compensated_residual_seconds': compensated,
        'native_residual_checks': checks, 'native_residual_seconds': native}}}


def ensemble(cache=False):
    return {'cache_hit_': cache, 'subset_sizes': [3] * 5,
            'base_diagnostics_': [learner(index) for index in range(1, 6)],
            'final_estimator': learner(10, wrapper='diagnostics')}


class EvaluationReportingTests(unittest.TestCase):
    def test_metrics_use_actual_support_and_do_not_confuse_accuracy_with_balance(self):
        actual, predicted = np.array(['A'] * 9 + ['B']), np.array(['A'] * 10)
        report = classification_report(actual, predicted)
        self.assertEqual(report['accuracy'], .9)
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertEqual(report['balanced_accuracy'], balanced_accuracy_score(actual, predicted))
        self.assertEqual([row['recall'] for row in report['per_class']], [1., 0.])
        json.dumps(report, allow_nan=False)

    def test_predicted_only_class_has_unknown_recall_not_zero_in_mean(self):
        report = classification_report([2, 2, 4, 4], [2, 6, 4, 6])
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertIsNone(report['per_class'][-1]['recall'])
        self.assertEqual(report['per_class'][-1]['support'], 0)

    def test_invalid_label_shapes_fail_without_touching_arrays(self):
        for actual, prediction in (([], []), ([1], [1, 2]), ([[1]], [[1]])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                classification_report(actual, prediction)

    def test_selection_distinguishes_mixture_from_class_retention(self):
        labels = np.array(['A'] * 8 + ['B'] * 2)
        facts = {'selected_rows': {'size': 3, 'class_counts': [
            {'label': 'A', 'count': 1}, {'label': 'B', 'count': 2}]}}
        before = deepcopy(facts)
        report = selection_composition(facts, labels)
        self.assertEqual(report['selection_ratio_realized'], .3)
        self.assertEqual(report['classes'][1]['training_fraction'], .2)
        self.assertEqual(report['classes'][1]['selected_fraction'], 2 / 3)
        self.assertEqual(report['classes'][1]['class_retention_fraction'], 1.)
        self.assertEqual(facts, before)

    def test_missing_class_is_zero_only_with_complete_selected_counts(self):
        facts = {'selected_rows': {'size': 2, 'class_counts': [{'label': 0, 'count': 2}]}}
        report = selection_composition(facts, [0, 0, 1, 1])
        self.assertEqual(report['classes'][1]['selected_count'], 0)
        facts['selected_rows']['size'] = 3
        self.assertFalse(selection_composition(facts, [0, 0, 1, 1])['available'])

    def test_unavailable_and_invalid_selection_counts_stay_unknown(self):
        cases = ({}, {'selected_rows': {'size': 1}},
                 {'selected_rows': {'size': 3, 'class_counts': [{'label': 0, 'count': 3}]}},
                 {'selected_rows': {'size': 1, 'class_counts': [{'label': 9, 'count': 1}]}},
                 {'selected_rows': {'size': 2, 'class_counts': [{'label': 0, 'count': 1},
                                                              {'label': 0, 'count': 1}]}})
        for facts in cases:
            with self.subTest(facts=facts):
                self.assertFalse(selection_composition(facts, [0, 0, 1, 1])['available'])

    def test_ovo_training_denominator_and_original_label_mapping(self):
        facts = {'pairwise_estimators': [{'pair': ['ant', 'zebra'], 'diagnostics': {
            'selected_rows': {'size': 2, 'class_counts': [{'label': -1, 'count': 2}]}}}]}
        report = selection_composition(facts, ['ant'] * 3 + ['moth'] * 10 + ['zebra'] * 2)
        pair = report['pairwise'][0]
        self.assertEqual(pair['training_size'], 5)
        self.assertEqual(pair['classes'][0]['label'], -1)
        self.assertEqual(pair['classes'][0]['training_count'], 2)
        self.assertEqual(pair['label_mapping'], [{'encoded': 1, 'original': 'ant'},
                                                 {'encoded': -1, 'original': 'zebra'}])

    def test_all_five_bases_and_final_are_counted_once_without_timer_overlap(self):
        facts = ensemble()
        before = deepcopy(facts)
        report = numerical_check_summary(facts)
        self.assertEqual(len(report['learners']), 6)
        fields = report['by_origin']['current_fit']['fields']
        self.assertEqual(fields['native_refinement_trials']['sum'], 25)
        self.assertEqual(fields['refinement_and_compensated_seconds']['sum'], 6.)
        self.assertAlmostEqual(fields['native_residual_seconds']['sum'], 3.6)
        self.assertEqual(fields['native_refinement_trials']['unknown_learners'], 0)
        self.assertEqual(facts, before)

    def test_cache_hit_bases_remain_historical_final_is_current(self):
        report = numerical_check_summary(ensemble(True))
        groups = report['by_origin']
        self.assertEqual(groups['historical_reference']['learner_records'], 5)
        self.assertEqual(groups['current_fit']['learner_records'], 1)
        self.assertEqual(groups['current_fit']['fields']['native_refinement_trials']['sum'], 10)

    def test_adaptive_counters_cover_all_bases_and_final_without_overlapping_timer(self):
        fields = ('adaptive_residual_probes', 'adaptive_residual_probe_errors',
                  'adaptive_residual_probe_invalidations', 'adaptive_residual_probe_acceptances',
                  'adaptive_native_trials_skipped', 'adaptive_residual_probe_fallbacks',
                  'adaptive_residual_probe_reuses', 'adaptive_residual_history_resets')
        facts = ensemble()
        nodes = [(node, 'solver_diagnostics') for node in facts['base_diagnostics_']]
        nodes.append((facts['final_estimator'], 'diagnostics'))
        for index, (node, wrapper) in enumerate(nodes, 1):
            linear = node[wrapper]['linear_system_diagnostics']
            linear.update({name: index * multiplier for multiplier, name in enumerate(fields, 1)})
            linear['adaptive_residual_probe_seconds'] = 100.
            node[wrapper]['residual_check_order'] = 'adaptive'
        before = deepcopy(facts)
        report = numerical_check_summary(facts)['by_origin']['current_fit']
        self.assertEqual(report['learner_records'], 6)
        for multiplier, name in enumerate(fields, 1):
            self.assertEqual(report['fields'][name], {
                'sum': 21 * multiplier, 'reported_learners': 6, 'unknown_learners': 0})
        self.assertNotIn('adaptive_residual_probe_seconds', report['fields'])
        self.assertNotIn('residual_check_order', report['fields'])
        self.assertEqual(report['fields']['refinement_and_compensated_seconds']['sum'], 6.)
        self.assertEqual(facts, before)

    def test_old_cached_adaptive_counts_remain_unknown(self):
        facts = ensemble(True)
        facts['final_estimator']['diagnostics']['linear_system_diagnostics']['adaptive_native_trials_skipped'] = 7
        groups = numerical_check_summary(facts)['by_origin']
        self.assertEqual(groups['current_fit']['fields']['adaptive_native_trials_skipped'], {
            'sum': 7, 'reported_learners': 1, 'unknown_learners': 0})
        self.assertEqual(groups['historical_reference']['fields']['adaptive_native_trials_skipped'], {
            'sum': None, 'reported_learners': 0, 'unknown_learners': 5})

    def test_older_numeric_summary_missing_counter_retains_unknown_learner_coverage(self):
        record = {'state': 'complete', 'fit_time': 1., 'score_time': .1,
                  'diagnostics': ensemble()}
        summary = summarize_records([(record, False)])
        for group in summary['numerical_checks']['by_origin'].values():
            for name in list(group['fields']):
                if name.startswith('adaptive_'):
                    del group['fields'][name]
        combined = aggregate_work_summaries([summary])['numerical_checks']['by_origin']
        self.assertEqual(combined['current_fit']['fields']['adaptive_native_trials_skipped'], {
            'sum': None, 'reported_learners': 0, 'unknown_learners': 6})
        self.assertEqual(combined['current_fit']['fields']['native_refinement_trials']['sum'], 25)

    def test_unknown_cache_origin_is_not_assumed_new_work(self):
        facts = ensemble()
        del facts['cache_hit_']
        report = numerical_check_summary(facts)
        self.assertEqual(report['by_origin']['unknown_origin']['learner_records'], 5)
        self.assertEqual(report['by_origin']['current_fit']['learner_records'], 1)
        facts.update(cache_hit_=True, cache_status_='miss')
        self.assertEqual(numerical_check_summary(facts)['by_origin']['unknown_origin']['learner_records'], 5)

    def test_recovered_pair_and_replayed_receipts_have_no_new_solver_work(self):
        facts = {'pairwise_estimators': [{'pair': [0, 1], 'work_origin': 'recovered',
                                         'diagnostics': ensemble()}]}
        report = numerical_check_summary(facts)
        self.assertEqual(report['by_origin']['current_fit']['learner_records'], 0)
        self.assertEqual(report['by_origin']['historical_reference']['learner_records'], 6)
        replay = numerical_check_summary(ensemble(), historical=True)
        self.assertEqual(replay['by_origin']['historical_reference']['learner_records'], 6)

    def test_missing_base_records_do_not_turn_missing_values_into_zero(self):
        facts = ensemble()
        facts['base_diagnostics_'] = facts['base_diagnostics_'][:2]
        facts['final_estimator'] = {}
        report = numerical_check_summary(facts)
        self.assertTrue(report['learner_inventory_incomplete'])
        field = report['by_origin']['current_fit']['fields']['native_refinement_trials']
        self.assertEqual(field, {'sum': 3, 'reported_learners': 2, 'unknown_learners': 4})
        unknown = numerical_check_summary({})['by_origin']['current_fit']['fields']['native_refinement_trials']
        self.assertIsNone(unknown['sum'])

    def test_invalid_numeric_facts_and_incomplete_pair_inventory_are_unknown(self):
        facts = learner(1)
        linear = facts['solver_diagnostics']['linear_system_diagnostics']
        for invalid in (True, -1, float('nan'), float('inf'), '1', 10 ** 1000):
            linear['compensated_residual_seconds'] = invalid
            report = numerical_check_summary(facts)
            self.assertIsNone(report['learners'][0]['values']['refinement_and_compensated_seconds'])
        self.assertTrue(numerical_check_summary({'pairwise_estimators': []})['learner_inventory_incomplete'])

    def test_numeric_reporting_error_does_not_erase_committed_work(self):
        record = {'state': 'complete', 'fit_time': 1., 'score_time': .1}
        with patch('work_reporting.numerical_check_summary', side_effect=ValueError('malformed evidence')):
            report = summarize_records([(record, False)])
        self.assertEqual(report['current_worker_seconds']['fit'], 1.)
        self.assertEqual(report['numerical_checks']['evaluations_without_summary'], 1)

    def test_work_summaries_exclude_aliases_and_separate_replay(self):
        record = {'state': 'complete', 'fit_time': 1., 'score_time': .1,
                  'diagnostics': ensemble()}
        alias = {**record, 'equivalent_reuse': {'reuse_seconds': .01}}
        summary = summarize_records([(record, False), (record, True), (alias, False)])
        evidence = summary['numerical_checks']
        self.assertEqual(evidence['aliases_excluded'], 1)
        self.assertEqual(evidence['by_origin']['current_fit']['learner_records'], 6)
        self.assertEqual(evidence['by_origin']['historical_reference']['learner_records'], 6)
        combined = aggregate_work_summaries([summary, summary])['numerical_checks']
        self.assertEqual(combined['by_origin']['current_fit']['fields']['native_refinement_trials']['sum'], 50)
        old = deepcopy(summary)
        del old['numerical_checks']
        self.assertEqual(aggregate_work_summaries([old])['numerical_checks']['evaluations_without_summary'], 3)


if __name__ == '__main__':
    unittest.main()
