"""Final fitting must preserve nested search choices and frozen model ownership."""
import json
from pathlib import Path
import sys

from joblib import effective_n_jobs, parallel_config
import numpy as np
import pytest
from workflow_refit_support import faer_thread_limit
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_info, threadpool_limits

from ensemble_dwd import EnsembleKernelDWD


EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
from phase_threads import configure_final_model, timed_model_phase
from tune_mnist import new_run_directory, run_search_workflow


@pytest.fixture
def conic():
    cp = pytest.importorskip('cvxpy')
    clarabel = pytest.importorskip('clarabel')
    if 'CLARABEL' not in cp.installed_solvers() or not hasattr(clarabel.DefaultSettings(), 'max_threads'):
        pytest.skip('Requires Clarabel with a max_threads setting')
    from dwd.socp_dwd import DWD
    return DWD


def run(pipeline, grid, folder):
    X = np.array([[-2., 1.], [2., 3.], [-1., 0.], [1., 2.]] * 3)
    y = np.tile([0, 1], 6)
    folds = list(StratifiedKFold(2).split(X, y))
    with parallel_config(backend='sequential', n_jobs=1), threadpool_limits(limits=1):
        output = run_search_workflow(pipeline, grid, X, y, X[:4], y[:4], folds,
                                     new_run_directory(folder), jobs=1, verbose=0)
        np.testing.assert_array_equal(output['model'].predict(X), y)
    assert output['completed']['held_out_accuracy'] == 1.
    return output


@pytest.mark.parametrize('step', ['', 'learner_', 'learner'],
                         ids=['empty', 'trailing-underscore', 'ordinary'])
def test_named_pipeline_step_refits_and_scores_the_owned_clone(step, monkeypatch, tmp_path):
    X = np.array([[-2., 1.], [2., 3.]] * 6)
    y = np.tile([0, 1], 6)
    folds = list(StratifiedKFold(2).split(X, y))
    # One base learner keeps the CV fixture serial even with a wider request.
    template = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1., base_jobs=2,
                                 gamma=.2, kernel_kws={'n_jobs': 1}, stopping='fixed',
                                 max_iter=4)
    pipeline = Pipeline([(step, template)])
    cv_jobs = []
    final_scopes = []
    fit, predict = EnsembleKernelDWD.fit, EnsembleKernelDWD.predict

    def observe(stage, model):
        final_scopes.append((stage, model.base_jobs, effective_n_jobs(2),
                             {pool['num_threads'] for pool in threadpool_info()}))

    def observed_fit(self, X, y):
        if len(X) == 12:
            observe('fit', self)
        else:
            cv_jobs.append(self.base_jobs)
        return fit(self, X, y)

    def observed_predict(self, X):
        if len(X) == 4:
            observe('predict', self)
        return predict(self, X)

    monkeypatch.setattr(EnsembleKernelDWD, 'fit', observed_fit)
    monkeypatch.setattr(EnsembleKernelDWD, 'predict', observed_predict)
    folder = new_run_directory(tmp_path / 'named')
    # Tune a pipeline parameter that sklearn can address for every step name.
    with parallel_config(backend='threading', n_jobs=2), threadpool_limits(limits=2):
        output = run_search_workflow(pipeline, {'verbose': [True, False]}, X, y,
                                     X[:4], y[:4], folds, folder, jobs=1, verbose=0)
        assert effective_n_jobs(None) == 2

    assert cv_jobs == [2] * 4
    assert final_scopes == [('fit', 1, 1, {1}), ('predict', 1, 1, {1})]
    assert template.base_jobs == 2 and not hasattr(template, 'classes_')
    assert pipeline.verbose is False
    assert output['model'][step] is not template
    assert output['model'][step].base_execution_['effective_jobs'] == 1
    assert output['model'].verbose is True
    assert output['selection']['parameters'] == {'verbose': True}
    np.testing.assert_array_equal(output['coarse_search'].cv_results_['mean_test_score'], [1., 1.])
    assert output['completed']['held_out_accuracy'] == 1.
    refit = json.loads((folder / 'refit.json').read_text(encoding='utf-8'))
    assert refit['state'] == 'completed'
    assert refit['wall_seconds'] == output['completed']['refit_seconds'] > 0


def test_inner_search_overrides_the_conic_template_and_parent_omits_timing(conic, tmp_path):
    # The inner search chooses its own complete option dictionary. Its actual
    # CV and full-data fits in this test all use one solver thread.
    options = dict(solver='CLARABEL', canon_backend='SCIPY', max_threads=1)
    template = conic(solver_kws={**options, 'max_threads': 3})
    search = GridSearchCV(template, {'solver_kws': [options]}, cv=2, n_jobs=1,
                          error_score='raise')
    folder = tmp_path / 'nested'
    output = run(Pipeline([('inner', search)]), {'inner__estimator__C': [.1]}, folder)
    selected = output['model']['inner']
    assert selected.best_estimator_.C == .1
    assert selected.best_estimator_.solver_kws == selected.best_params_['solver_kws'] == options
    assert selected.n_splits_ == 2
    assert np.isfinite(selected.cv_results_['mean_test_score']).all()
    assert template.solver_kws == {**options, 'max_threads': 3}
    assert search.param_grid == {'solver_kws': [options]}
    record = json.loads((folder / 'refit.json').read_text(encoding='utf-8'))
    assert record['wall_seconds'] is None
    assert output['completed']['refit_seconds'] is None
    assert output['completed']['timing_unavailable']


class DelegatingAdapter:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __sklearn_clone__(self):
        return type(self)(clone(self.model))

    def set_params(self, **params):
        self.model.set_params(**params)
        return self


@pytest.mark.parametrize('replacement', [False, True], ids=['template', 'selected-grid-value'])
def test_clone_hook_returning_an_ordinary_shared_learner_preserves_its_workers(replacement, tmp_path):
    # The source hook exists only on this instance. Its result looks exactly
    # like an ordinary clone, but the caller still owns the returned learner.
    owner = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1., base_jobs=2,
                              gamma=.2, stopping='fixed', max_iter=4)
    source = clone(owner).set_params(base_jobs=3)
    source.__sklearn_clone__ = lambda: owner
    if replacement:
        pipeline = Pipeline([('learner', clone(owner))])
        grid = {'learner': [source]}
    else:
        pipeline = Pipeline([('learner', source)])
        grid = {}

    folder = tmp_path / 'shared'
    output = run(pipeline, grid, folder)
    if replacement:
        assert output['selection']['parameters']['learner'] is source
    assert output['model']['learner'] is owner
    assert source.base_jobs == 3
    assert owner.base_jobs == owner.base_execution_['requested_jobs'] == 2
    assert output['selection']['mean_cv_accuracy'] == 1.
    record = json.loads((folder / 'refit.json').read_text(encoding='utf-8'))
    assert record['state'] == 'completed'
    assert record['wall_seconds'] is output['completed']['refit_seconds'] is None
    assert record['native_runtime']['worker_parameters'] == {}
    for key in ('timing_unavailable', 'prediction_timing_unavailable'):
        assert output['completed'][key] and output['completed'][key] == record['native_runtime'][key]


@pytest.fixture
def single_base_learner():
    return EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1., base_jobs=2,
                             gamma=.2, stopping='fixed', max_iter=4)


@pytest.mark.parametrize('key', ['scale', 'steps', 'inner__scale'])
def test_replacing_a_frozen_step_restores_selected_clone_timing(key, single_base_learner, tmp_path):
    FrozenEstimator = pytest.importorskip('sklearn.frozen').FrozenEstimator
    frozen = FrozenEstimator(StandardScaler().fit([[-2., 1.], [2., 3.]]))
    inner = Pipeline([('scale', frozen), ('learner', single_base_learner)])
    pipeline = Pipeline([('inner', inner)]) if key == 'inner__scale' else inner
    replacement = [('scale', 'passthrough'), ('learner', single_base_learner)] if key == 'steps' else 'passthrough'
    folder = tmp_path / 'replaced'
    output = run(pipeline, {key: [replacement]}, folder)
    selected = output['model']['inner'] if key == 'inner__scale' else output['model']
    assert selected['scale'] == 'passthrough' and inner['scale'] is frozen
    assert selected['learner'] is not single_base_learner
    assert single_base_learner.base_jobs == 2
    assert selected['learner'].base_execution_['requested_jobs'] == selected['learner'].base_jobs == 1
    record = json.loads((folder / 'refit.json').read_text(encoding='utf-8'))
    assert record['wall_seconds'] == output['completed']['refit_seconds'] > 0
    assert 'prediction_timing_unavailable' not in output['completed']


@pytest.mark.parametrize('replacement', [False, True], ids=['template', 'selected-grid-value'])
def test_shared_steps_container_preserves_its_learners_workers(replacement, single_base_learner, tmp_path):
    class SharedSteps(list):
        def __deepcopy__(self, memo):
            return self

    shared = SharedSteps([('learner', single_base_learner)])
    if replacement:
        pipeline = Pipeline([('learner', clone(single_base_learner))])
        grid = {'steps': [shared]}
    else:
        pipeline = Pipeline(shared)
        grid = {}
    folder = tmp_path / 'shared-steps'
    output = run(pipeline, grid, folder)
    assert output['model']['learner'] is single_base_learner
    assert single_base_learner.base_execution_['requested_jobs'] == single_base_learner.base_jobs == 2
    record = json.loads((folder / 'refit.json').read_text(encoding='utf-8'))
    assert record['wall_seconds'] is output['completed']['refit_seconds'] is None
    assert record['native_runtime']['worker_parameters'] == {}
    assert output['completed']['timing_unavailable'] and output['completed']['prediction_timing_unavailable']


@pytest.mark.parametrize('nested', [False, True], ids=['root', 'pipeline-step'])
def test_delegated_estimator_preserves_worker_requests_and_omits_timing(nested, tmp_path):
    class WorkerIdentity(FunctionTransformer):
        def __init__(self, n_jobs=3):
            super().__init__()
            self.n_jobs = n_jobs

    # One base learner makes the requested two-worker budget harmless for CV.
    learner = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1., base_jobs=2,
                                gamma=.2, stopping='fixed', max_iter=4)
    template = DelegatingAdapter(Pipeline([('learner', learner)]))
    sibling = WorkerIdentity()
    model = Pipeline([('first', sibling), ('last', template)]) if nested else template
    folder = tmp_path / 'delegated'
    output = run(model, {}, folder)
    selected = output['model']
    if nested:
        assert selected['first'].n_jobs == sibling.n_jobs == 3
        selected = selected['last']
    assert type(selected) is DelegatingAdapter and selected is not template
    assert selected.model['learner'].base_jobs == learner.base_jobs == 2
    assert hasattr(selected.model['learner'], 'classes_') and not hasattr(learner, 'classes_')
    refit = json.loads((folder / 'refit.json').read_text(encoding='utf-8'))
    assert refit['wall_seconds'] is output['completed']['refit_seconds'] is None
    assert refit['native_runtime']['worker_parameters'] == {}
    for key in ('timing_unavailable', 'prediction_timing_unavailable'):
        assert output['completed'][key] and output['completed'][key] == refit['native_runtime'][key]


def test_opaque_estimator_option_with_delegated_parameters_omits_timing():
    class OpaqueEstimator(BaseEstimator):
        def __init__(self, child, n_jobs=3):
            self.child = child
            self.n_jobs = n_jobs

    model = OpaqueEstimator(DelegatingAdapter(EnsembleKernelDWD(base_jobs=2)))
    record = configure_final_model(model)
    assert record['timing_unavailable'] and record['prediction_timing_unavailable']
    assert record['worker_parameters'] == {}
    assert model.n_jobs == 3 and model.child.model.base_jobs == 2


def test_delegated_feature_selector_retains_private_solver_thread_limit(conic):
    options = dict(solver='CLARABEL', direct_solve_method='faer', max_threads=4, canon_backend='SCIPY')
    selector = SelectFromModel(conic(solver_kws=options), threshold=0.)
    template = Pipeline([('features', DelegatingAdapter(selector)),
                         ('classifier', DecisionTreeClassifier(random_state=42))])
    model = clone(template)
    record = configure_final_model(model)
    assert record['timing_unavailable'] and record['prediction_timing_unavailable']
    assert record['worker_parameters'] == {}
    X = np.array([[-2., 1.], [2., 3.], [-1., 0.], [1., 2.]])
    y = [0, 1, 0, 1]
    # Exercise the selected model without recording a duration for this solve.
    with timed_model_phase():
        try:
            model.fit(X, y)
        except Exception as error:
            if 'Bad value for field "direct_solve_method"' in str(error):
                pytest.skip('Requires the optional FAER backend in the Clarabel build')
            raise
        np.testing.assert_array_equal(model.predict(X), y)
    fitted = model['features'].model.estimator_
    assert faer_thread_limit(fitted.problem_._solver_cache['CLARABEL']) == 4
    assert fitted.solver_kws == selector.estimator.solver_kws == options


def test_frozen_conic_estimator_retains_caller_owned_solver_settings(conic, tmp_path):
    FrozenEstimator = pytest.importorskip('sklearn.frozen').FrozenEstimator
    options = dict(solver='CLARABEL', canon_backend='SCIPY', max_threads=1)
    X = np.array([[-2., 1.], [2., 3.], [-1., 0.], [1., 2.]])
    with threadpool_limits(limits=1):
        fitted = conic(solver_kws=options).fit(X, [0, 1, 0, 1])
    # The frozen wrapper's fit is a no-op. This setting is for its owner's
    # future fit and is never used to solve a problem in this test.
    owner_options = {**options, 'max_threads': 4}
    fitted.set_params(solver_kws=owner_options)
    frozen = FrozenEstimator(fitted)
    run(Pipeline([('model', frozen)]), {}, tmp_path / 'frozen')
    assert frozen.estimator is fitted
    assert fitted.solver_kws is owner_options
