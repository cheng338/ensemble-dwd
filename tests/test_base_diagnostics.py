"""Lean base records retain compact reported facts without extra numerical work."""

from copy import deepcopy
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelDWD
from ensemble_dwd.estimator import _base_diagnostics


class DiagnosticDWD:
    fits = []
    scores = 0

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, X, y):
        type(self).fits.append(self)
        self.classes_ = np.unique(y)
        self.n_iter_ = np.int64(7)
        self.returned_iteration_ = np.int64(6)
        self.total_iterations_ = np.int64(11)
        self.iterations_per_attempt_ = np.array([4, 7])
        self.converged_ = False
        self.criterion_reached_ = True
        self.objective_tolerance_met_ = True
        self.optimality_met_ = False
        self.stationarity_checked_ = True
        self.termination_reason_ = 'objective_tolerance'
        self.final_objective_ = np.float64(.125)
        self.rkhs_gradient_norm_ = .04
        self.dual_gap_ = None
        self.diagnostics_ = {
            'setup_seconds': .1, 'optimization_seconds': .2, 'total_seconds': .4,
            'auto_fallback': True,
            'linear_system_diagnostics': {
                'native_refinement_trials': 12, 'native_refinement_discarded': 8,
                'native_refinement_acceptances': 4, 'native_refinement_seconds': .3,
                'compensated_residual_checks': 8, 'compensated_residual_seconds': .2,
                'native_residual_checks': 4, 'native_residual_seconds': .1},
            'spectral_restart': {
                'attempts': 2, 'discarded_completed_updates': 4,
                'successful_updates': 7, 'total_completed_updates': 11,
                'total_attempted_updates': 12, 'history_scope': 'successful_spectral_attempt',
                'failed_attempt': {'completed_iterations': 4, 'failed_iteration': 5},
                'successful_attempt_timing': {'total_seconds': .3}},
            'dual_certificate': {
                'dual_objective_lower_bound': 0., 'dual_objective_is_lower_bound': True,
                'fallback_reason': 'nonzero_dual_bound_unavailable',
                'kernel_assumption': 'symmetric positive-semidefinite represented kernel',
                'primal_objective_and_gap_are_estimates': True},
            'mm_function_recovery': {
                'attempted': True, 'accepted_actions': 3, 'certificate_status': 'certified_exact_psd_rank',
                'kernel_approximation_used': False, 'positive_directions_discarded': 0,
                'resource_caps': {'max_rank': 32},
                'certificate': {'certified': True, 'rank': 2},
                'preparation': {'original_kernel_unchanged': True},
                'last_action': {'true_MM_function_checked': True},
                'large_internal_state': np.ones((10, 10))},
            'objective_history': np.arange(1000), 'validation_history': [1.] * 1000}
        return self

    def decision_function(self, X):
        type(self).scores += 1
        return np.asarray(X)[:, 0]


def without_timings(value):
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items()
                if not key.endswith('_seconds')}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


class BaseDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        DiagnosticDWD.fits = []
        DiagnosticDWD.scores = 0
        self.X = np.column_stack((np.linspace(-2., 2., 20), np.cos(np.arange(20))))
        self.y = np.tile([0, 1], 10)

    def model(self, **changes):
        options = dict(M=3, theta=.3, sv_selection_ratio=.5, gamma=.02, random_state=17)
        options.update(changes)
        return EnsembleKernelDWD(**options)

    def test_scalar_flags_objective_and_work_are_independent_reported_values(self):
        model = DiagnosticDWD().fit(self.X, self.y)
        record = _base_diagnostics(model)
        self.assertEqual(record['final_objective'], model.final_objective_)
        self.assertEqual(record['total_iterations'], 11)
        self.assertEqual(record['iterations_per_attempt'], [4, 7])
        self.assertEqual(record['n_iter'], 7)
        self.assertEqual(record['returned_iteration'], 6)
        self.assertTrue(record['criterion_reached'])
        self.assertTrue(record['objective_tolerance_met'])
        self.assertTrue(record['stationarity_checked'])
        self.assertFalse(record['converged'])
        self.assertFalse(record['optimality_met'])
        self.assertIsNone(record['dual_gap'])

    def test_recovery_certificate_and_timing_summaries_do_not_retain_state(self):
        model = DiagnosticDWD().fit(self.X, self.y)
        original = deepcopy(model.diagnostics_)
        record = _base_diagnostics(model)
        facts = record['solver_diagnostics']
        self.assertEqual(facts['spectral_restart'], original['spectral_restart'])
        self.assertEqual(facts['dual_certificate'], original['dual_certificate'])
        self.assertEqual(facts['linear_system_diagnostics'], original['linear_system_diagnostics'])
        # These reported categories overlap; copying must not sum them into total time.
        self.assertGreater(sum(value for key, value in facts['linear_system_diagnostics'].items()
                               if key.endswith('_seconds')), facts['total_seconds'])
        self.assertEqual(facts['total_seconds'], .4)
        self.assertEqual(facts['mm_function_recovery']['accepted_actions'], 3)
        self.assertFalse(facts['mm_function_recovery']['kernel_approximation_used'])
        self.assertNotIn('objective_history', facts)
        self.assertNotIn('validation_history', facts)
        self.assertNotIn('large_internal_state', facts['mm_function_recovery'])
        model.diagnostics_['spectral_restart']['failed_attempt']['failed_iteration'] = 999
        model.iterations_per_attempt_[0] = 999
        self.assertEqual(facts['spectral_restart']['failed_attempt']['failed_iteration'], 5)
        self.assertEqual(record['iterations_per_attempt'], [4, 7])
        assert_array_equal(model.diagnostics_['objective_history'], original['objective_history'])

    def test_absent_old_fields_remain_unknown(self):
        old = SimpleNamespace(n_iter_=3, converged_=False)
        self.assertEqual(_base_diagnostics(old), {'n_iter': 3, 'converged': False})
        self.assertEqual(_base_diagnostics(SimpleNamespace()), {})
        explicit_unknown = SimpleNamespace(final_objective_=None, diagnostics_={
            'dual_certificate': {'fallback_reason': None}})
        self.assertEqual(_base_diagnostics(explicit_unknown), {
            'final_objective': None, 'solver_diagnostics': {'dual_certificate': {'fallback_reason': None}}})

    def test_lean_adaptive_diagnostics_preserve_each_base_and_final_learner(self):
        counters = (
            'adaptive_residual_probes', 'adaptive_residual_probe_errors',
            'adaptive_residual_probe_invalidations', 'adaptive_residual_probe_acceptances',
            'adaptive_native_trials_skipped', 'adaptive_residual_probe_fallbacks',
            'adaptive_residual_probe_reuses', 'adaptive_residual_history_resets')

        class AdaptiveDiagnosticDWD(DiagnosticDWD):
            def fit(self, X, y):
                super().fit(X, y)
                self.diagnostics_.update(
                    residual_check_order='adaptive', requested_residual_check_order='adaptive')
                checks = self.diagnostics_['linear_system_diagnostics']
                checks.update({name: len(type(self).fits) for name in counters})
                checks.update(adaptive_residual_probe_seconds=.01,
                              last_adaptive_residual_probe_error='ArithmeticError: fixture')
                return self

        with patch('ensemble_dwd.estimator.KernGDWD', AdaptiveDiagnosticDWD):
            lean = self.model(store_base_estimators=False, acceleration='restart',
                              residual_check_order='adaptive').fit(self.X, self.y)
        self.assertEqual(len(DiagnosticDWD.fits), lean.M + 1)
        records = lean.base_diagnostics_ + [_base_diagnostics(lean.final_estimator_)]
        self.assertEqual(len(records), lean.M + 1)
        for index, (record, model) in enumerate(zip(records, DiagnosticDWD.fits), 1):
            facts = record['solver_diagnostics']
            self.assertEqual(facts['residual_check_order'], 'adaptive')
            self.assertEqual(facts['requested_residual_check_order'], 'adaptive')
            self.assertEqual(facts['linear_system_diagnostics'],
                             model.diagnostics_['linear_system_diagnostics'])
            self.assertTrue(all(facts['linear_system_diagnostics'][name] == index
                                for name in counters))
        DiagnosticDWD.fits[0].diagnostics_['linear_system_diagnostics']['adaptive_native_trials_skipped'] = 999
        self.assertEqual(records[0]['solver_diagnostics']['linear_system_diagnostics']
                         ['adaptive_native_trials_skipped'], 1)

    def test_summary_bounds_reject_arrays_traces_and_unbounded_text(self):
        model = SimpleNamespace(final_objective_=np.arange(1000),
            iterations_per_attempt_=list(range(1000)), diagnostics_={
                'fallback_reason': 'x' * 1025, 'objective_history': list(range(1000)),
                'linear_system_diagnostics': {'linear_solves': np.int64(10),
                                              'last_adaptive_residual_probe_error': 'x' * 1025,
                                              'factorization_attempts': [{}] * 1000},
                'dual_certificate': {'kernel_assumption': np.ones(10)}})
        self.assertEqual(_base_diagnostics(model), {'solver_diagnostics': {
            'linear_system_diagnostics': {'linear_solves': 10}}})

    def test_lean_and_retained_models_have_same_facts_and_numerical_calls(self):
        with patch('ensemble_dwd.estimator.KernGDWD', DiagnosticDWD):
            retained = self.model().fit(self.X, self.y)
            lean = self.model(store_base_estimators=False).fit(self.X, self.y)
        self.assertEqual(len(DiagnosticDWD.fits), 2 * (retained.M + 1))
        self.assertEqual(DiagnosticDWD.scores, 2 * retained.M)
        self.assertIsNone(lean.estimators_)
        self.assertEqual(retained.base_diagnostics_, lean.base_diagnostics_)
        for model, record in zip(retained.estimators_, retained.base_diagnostics_):
            self.assertEqual(record, _base_diagnostics(model))
        assert_array_equal(retained.consensus_scores_, lean.consensus_scores_)
        assert_array_equal(retained.selected_indices_, lean.selected_indices_)
        assert_array_equal(retained.predict(self.X), lean.predict(self.X))

    def test_cache_replays_original_facts_without_repeating_base_work(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('ensemble_dwd.estimator.KernGDWD', DiagnosticDWD), \
                patch('ensemble_dwd.estimator._runtime_provenance', return_value={'fixture': 'diagnostics'}):
            cold = self.model(store_base_estimators=False, cache_dir=directory).fit(self.X, self.y)
            warm = self.model(store_base_estimators=False, cache_dir=directory).fit(self.X, self.y)
        self.assertFalse(cold.cache_hit_)
        self.assertTrue(warm.cache_hit_)
        self.assertEqual(len(DiagnosticDWD.fits), cold.M + 2)
        self.assertEqual(DiagnosticDWD.scores, cold.M)
        self.assertEqual(cold.base_diagnostics_, warm.base_diagnostics_)
        assert_array_equal(cold.consensus_scores_, warm.consensus_scores_)
        assert_array_equal(cold.selected_indices_, warm.selected_indices_)
        assert_array_equal(cold.predict(self.X), warm.predict(self.X))
        cold.base_diagnostics_[0]['solver_diagnostics']['total_seconds'] = 999.
        self.assertEqual(warm.base_diagnostics_[0]['solver_diagnostics']['total_seconds'], .4)

    def test_old_cache_records_do_not_fabricate_new_diagnostics(self):
        old = {'n_iter': 7, 'converged': False}
        with tempfile.TemporaryDirectory() as directory, \
                patch('ensemble_dwd.estimator.KernGDWD', DiagnosticDWD), \
                patch('ensemble_dwd.estimator._runtime_provenance', return_value={'fixture': 'legacy-diagnostics'}):
            with patch('ensemble_dwd.estimator._base_diagnostics', return_value=old):
                self.model(store_base_estimators=False, cache_dir=directory).fit(self.X, self.y)
            warm = self.model(store_base_estimators=False, cache_dir=directory).fit(self.X, self.y)
        self.assertTrue(warm.cache_hit_)
        self.assertEqual(warm.base_diagnostics_, [old] * warm.M)
        self.assertEqual(len(DiagnosticDWD.fits), warm.M + 2)

    def test_real_fixed_fits_preserve_arrays_and_report_original_base_objectives(self):
        X = np.column_stack((np.arange(24, dtype=float) - 11.5, np.arange(24) % 3)) / 6.
        y = np.zeros(24, dtype=int)
        y[[10, 23]] = 1
        options = dict(M=2, theta=.5, sv_selection_ratio=1., lambd=.05,
                       gamma=.2, max_iter=3, stopping='fixed', random_state=17)
        with threadpool_limits(limits=1):
            retained = EnsembleKernelDWD(**options).fit(X, y)
            lean = EnsembleKernelDWD(**options, store_base_estimators=False).fit(X, y)
            assert_array_equal(retained.decision_function(X), lean.decision_function(X))
        for first, second in zip(retained.subset_indices_, lean.subset_indices_):
            assert_array_equal(first, second)
        assert_array_equal(retained.consensus_scores_, lean.consensus_scores_)
        assert_array_equal(retained.selected_indices_, lean.selected_indices_)
        assert_array_equal(retained.final_estimator_.dual_coef_, lean.final_estimator_.dual_coef_)
        assert_array_equal(retained.final_estimator_.intercept_, lean.final_estimator_.intercept_)
        self.assertEqual(without_timings(retained.base_diagnostics_), without_timings(lean.base_diagnostics_))
        for base, record in zip(retained.estimators_, retained.base_diagnostics_):
            self.assertEqual(record, _base_diagnostics(base))
            self.assertEqual(record['final_objective'], base.final_objective_)
            self.assertEqual(record['solver_diagnostics']['dual_certificate'], base.diagnostics_['dual_certificate'])


if __name__ == '__main__':
    unittest.main()
