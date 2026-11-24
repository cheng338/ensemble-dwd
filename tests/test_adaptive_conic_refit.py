"""Whole-pipeline resource controls during adaptive refit and final recovery."""
from copy import deepcopy
import json

import numpy as np
import pytest
from workflow_refit_support import faer_thread_limit
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_dwd import EnsembleKernelDWD
from test_final_thread_preflight import Workflow
from test_final_recovery import final_recovery, protect_cv


@pytest.mark.parametrize('recovery', [False, True], ids=['adaptive', 'fresh-recovery'])
@pytest.mark.parametrize('qualifies', [True, False], ids=['clarabel', 'solver-path'])
def test_conic_selector_retains_cv_and_applies_final_timing_policy(
        tmp_path, monkeypatch, qualifies, recovery):
    cp = pytest.importorskip('cvxpy')
    clarabel = pytest.importorskip('clarabel')
    from dwd.socp_dwd import DWD

    if not hasattr(clarabel.DefaultSettings(), 'max_threads'):
        pytest.skip('Requires a configurable Clarabel thread limit')
    solver_options = dict(max_threads=4, direct_solve_method='faer', canon_backend='SCIPY')
    if qualifies:
        options = dict(solver='CLARABEL', **solver_options)
    else:
        if not hasattr(cp.Problem, '_solve_solver_path'):
            pytest.skip('Requires CVXPY solver paths')
        options = {'solver_path': [('CLARABEL', solver_options)]}

    # Workflow shortens the acceptance policy after two candidates. Its
    # actual folds, selected fit, prediction, handoff and receipts still run.
    workflow = Workflow(tmp_path, monkeypatch)
    workflow.X = np.tile([[-2., 1.], [2., 3.], [-1., 0.], [1., 2.]], (5, 1))
    workflow.test_X = workflow.X[:8] + .025
    selector = SelectFromModel(DWD(solver_kws=deepcopy(options)), threshold=0.)
    worker_selector = SelectFromModel(LinearRegression(n_jobs=3), threshold=0.)
    # Both spellings are valid sklearn step names. Resource controls must
    # reach the actual estimator inside each selector on the final clone.
    preprocess = Pipeline([('worker_', worker_selector), ('', selector)])
    ensemble = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1.,
                                 max_iter=3, stopping='fixed', random_state=42)
    workflow.pipeline = Pipeline([('prep_', preprocess), ('ensemble', ensemble)])
    fits = []
    original_fit = DWD.fit
    predictions = []
    original_predict = EnsembleKernelDWD.predict

    def observed_fit(self, X, y):
        try:
            fitted = original_fit(self, X, y)
        except Exception as error:
            if not fits and 'Bad value for field "direct_solve_method"' in str(error):
                pytest.skip('Requires the optional FAER backend in the Clarabel build')
            raise
        solver = self.problem_._solver_cache['CLARABEL']
        fits.append((len(y), faer_thread_limit(solver)))
        return fitted

    def observed_predict(self, X):
        predictions.append(len(X))
        return original_predict(self, X)

    monkeypatch.setattr(DWD, 'fit', observed_fit)
    monkeypatch.setattr(EnsembleKernelDWD, 'predict', observed_predict)
    if recovery:
        workflow.interrupt_after_handoff()
        source = workflow.root / 'run'
        params = json.loads((source / 'selection.json').read_text())['parameters']
        selected = clone(workflow.pipeline).set_params(**params)
        protect_cv(monkeypatch)
        args = (selected, workflow.X, workflow.y, workflow.test_X, workflow.test_y,
                source, workflow.root / 'attempt')
        result = final_recovery.recover_final_fit(*args, final_native_threads=1)
        replay = final_recovery.recover_final_fit(*args, final_native_threads=1, resume=True)
        completed = result['completed']
        refit = completed['refit']
        saved = json.loads((workflow.root / 'attempt' / 'completed.json').read_text())
        assert saved['payload'] == completed
        assert completed['full_pipeline_fit_calls'] == completed['held_out_predict_calls'] == 1
        assert completed['cv_fits_performed'] == 0
        assert selected['prep_'][''].estimator.solver_kws == options
        assert selected['prep_']['worker_'].estimator.n_jobs == 3
    else:
        result = workflow.run()
        protect_cv(monkeypatch)
        replay = workflow.run(resume=True)
        completed = result['completed']
        refit = json.loads((workflow.root / 'run' / 'refit.json').read_text())

    assert replay['resumed_completed'] and replay['model'] is None
    assert replay['completed'] == completed
    fold_fits = [(len(train), 4) for train, validation in workflow.folds]
    candidate_count = len(workflow.grid['ensemble__lambd'])
    assert fits[:-1] == fold_fits * candidate_count
    assert fits[-1] == (len(workflow.y), 1 if qualifies else 4)
    fold_predictions = [len(validation) for train, validation in workflow.folds]
    assert predictions == fold_predictions * candidate_count + [len(workflow.test_y)]
    assert selector.estimator.solver_kws == options
    assert worker_selector.estimator.n_jobs == 3
    assert not hasattr(selector, 'estimator_') and not hasattr(worker_selector, 'estimator_')
    fitted_preprocess = result['model']['prep_']
    assert fitted_preprocess['worker_'].estimator_.n_jobs == 1
    fitted = fitted_preprocess[''].estimator_
    assert fitted.solver_kws == ({**options, 'max_threads': 1} if qualifies else options)
    assert completed['held_out_accuracy'] == 1.
    assert completed['held_out_prediction_seconds'] >= 0.
    runtime = refit['native_runtime']
    assert runtime['estimator_workers'] == runtime['requested_native_threads'] == 1
    workers = runtime['worker_parameters']
    assert workers['ensemble__base_jobs'] == 1
    assert list(workers.values()) == [1, 1]
    durations = [refit['wall_seconds'], refit['pipeline_fit_seconds'],
                 refit['fit_management_seconds'], completed['fit_seconds' if recovery else 'refit_seconds']]
    if qualifies:
        controls = runtime['conic_threads']
        backends = controls['canonicalization_backends']
        assert list(backends.values()) == ['SCIPY']
        assert controls['clarabel_max_threads'] == {path: 1 for path in backends}
        assert completed['final_native_threads'] == 1
        assert all(np.isfinite(duration) and duration >= 0 for duration in durations)
        assert refit['pipeline_fit_seconds'] > 0
    else:
        assert 'conic_threads' not in runtime and completed['final_native_threads'] is None
        assert all(duration is None for duration in durations)
        assert runtime['timing_unavailable'] == completed['timing_unavailable']


def returning_template(owner):
    class ReturningEnsemble(EnsembleKernelDWD):
        def __sklearn_clone__(self):
            return owner

    return Pipeline([('ensemble', ReturningEnsemble(**owner.get_params(deep=False)))])


@pytest.fixture
def ensemble_workflow(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, monkeypatch)
    workflow.X = np.tile([[-2., 1.], [2., 3.], [-1., 0.], [1., 2.]], (5, 1))
    workflow.test_X = workflow.X[:8] + .025
    owner = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1., base_jobs=2,
                              stopping='fixed', max_iter=3, random_state=42)
    fits, predictions = [], []
    fit, predict = EnsembleKernelDWD.fit, EnsembleKernelDWD.predict

    def observe_fit(self, X, y):
        fits.append((len(y), self.base_jobs))
        return fit(self, X, y)

    def observe_predict(self, X):
        predictions.append((len(X), self.base_jobs))
        return predict(self, X)

    monkeypatch.setattr(EnsembleKernelDWD, 'fit', observe_fit)
    monkeypatch.setattr(EnsembleKernelDWD, 'predict', observe_predict)
    return workflow, owner, fits, predictions


@pytest.mark.parametrize('route', ['adaptive', 'recover-template', 'recover-stock'])
def test_custom_clone_returning_stock_model_keeps_its_workers(ensemble_workflow, monkeypatch, route):
    workflow, owner, fits, predictions = ensemble_workflow
    workflow.pipeline = returning_template(owner)
    assert clone(workflow.pipeline)['ensemble'] is owner
    source = workflow.root / 'run'
    if route == 'adaptive':
        result = workflow.run()
        protect_cv(monkeypatch)
        replay = workflow.run(resume=True)
        refit = result['refit']
    else:
        workflow.interrupt_after_handoff()
        params = json.loads((source / 'selection.json').read_text())['parameters']
        selected = Pipeline([('ensemble', clone(owner))]).set_params(**params)
        recovery_owner = selected['ensemble']
        if route == 'recover-template':
            selected = returning_template(recovery_owner)
            assert clone(selected)['ensemble'] is recovery_owner
        protect_cv(monkeypatch)
        args = (selected, workflow.X, workflow.y, workflow.test_X, workflow.test_y,
                source, workflow.root / 'attempt')
        result = final_recovery.recover_final_fit(*args, final_native_threads=1)
        replay = final_recovery.recover_final_fit(*args, final_native_threads=1, resume=True)
        assert recovery_owner.base_jobs == selected['ensemble'].base_jobs == 2
        assert result['completed']['cv_fits_performed'] == 0
        refit = result['completed']['refit']

    handoff = json.loads((source / 'cv-checkpoints/finalization/handoff.json').read_text())['payload']
    assert handoff['selected_estimator'] == handoff['final_estimator']
    assert workflow.pipeline['ensemble'].base_jobs == owner.base_jobs == 2
    assert type(result['model']['ensemble']) is EnsembleKernelDWD
    n_candidates = len(workflow.grid['ensemble__lambd'])
    assert fits == [(len(train), 2) for train, _ in workflow.folds] * n_candidates + [(len(workflow.y), 2)]
    assert predictions == [(len(val), 2) for _, val in workflow.folds] * n_candidates + [(len(workflow.test_y), 2)]
    completed = result['completed']
    assert completed['held_out_accuracy'] == 1.
    assert replay['resumed_completed'] and replay['model'] is None and replay['completed'] == completed
    runtime = refit['native_runtime']
    assert runtime['worker_parameters'] == {} and runtime['estimator_workers'] is None
    for key in ('timing_unavailable', 'prediction_timing_unavailable'):
        assert runtime[key] and completed[key] == runtime[key]
    assert refit['wall_seconds'] is completed['pipeline_fit_seconds'] is None
    assert completed['final_native_threads'] is completed['held_out_prediction_seconds'] is None


@pytest.mark.parametrize('recovery', [False, True], ids=['adaptive', 'fresh-recovery'])
def test_selected_passthrough_replaces_frozen_scaler_and_restores_timing(
        ensemble_workflow, monkeypatch, recovery):
    FrozenEstimator = pytest.importorskip('sklearn.frozen').FrozenEstimator
    workflow, owner, fits, predictions = ensemble_workflow
    scaler = StandardScaler().fit(workflow.X)
    frozen = FrozenEstimator(scaler)
    workflow.pipeline = Pipeline([('prep', frozen), ('ensemble', owner)])
    assert clone(workflow.pipeline)['prep'] is frozen
    workflow.grid['prep'] = ['passthrough']
    source = workflow.root / 'run'
    if recovery:
        workflow.interrupt_after_handoff()
        params = json.loads((source / 'selection.json').read_text())['parameters']
        selected = clone(workflow.pipeline).set_params(**params)
        protect_cv(monkeypatch)
        args = (selected, workflow.X, workflow.y, workflow.test_X, workflow.test_y,
                source, workflow.root / 'attempt')
        result = final_recovery.recover_final_fit(*args, final_native_threads=1)
        replay = final_recovery.recover_final_fit(*args, final_native_threads=1, resume=True)
        refit = result['completed']['refit']
        assert selected['ensemble'].base_jobs == 2
        assert result['completed']['cv_fits_performed'] == 0
    else:
        result = workflow.run()
        protect_cv(monkeypatch)
        replay = workflow.run(resume=True)
        refit = result['refit']

    completed = result['completed']
    assert replay['resumed_completed'] and replay['model'] is None and replay['completed'] == completed
    handoff = json.loads((source / 'cv-checkpoints/finalization/handoff.json').read_text())['payload']
    assert handoff['selection']['parameters']['prep'] == result['model']['prep'] == 'passthrough'
    assert workflow.pipeline['prep'] is frozen and frozen.estimator is scaler
    assert owner.base_jobs == 2 and result['model']['ensemble'].base_jobs == 1
    n_candidates = len(workflow.grid['ensemble__lambd'])
    cv_fits = [(len(train), 2) for train, _ in workflow.folds] * n_candidates
    cv_predictions = [(len(val), 2) for _, val in workflow.folds] * n_candidates
    assert fits == cv_fits + [(len(workflow.y), 1)]
    assert predictions == cv_predictions + [(len(workflow.test_y), 1)]
    assert refit['native_runtime']['worker_parameters'] == {'ensemble__base_jobs': 1}
    assert completed['final_native_threads'] == 1 and completed['held_out_accuracy'] == 1.
    assert refit['wall_seconds'] >= completed['pipeline_fit_seconds'] > 0
    assert completed['held_out_prediction_seconds'] >= 0
