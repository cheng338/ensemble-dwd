"""Solver-private resources in the legacy example's full-training refit."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from workflow_refit_support import faer_thread_limit
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
from phase_threads import configure_final_model
from tune_mnist import new_run_directory, run_search_workflow


X = np.array([[-2., 1.], [2., 3.], [-1., 0.], [1., 2.]] * 3)
Y = np.tile([0, 1], 6)


@pytest.fixture
def conic():
    cp = pytest.importorskip('cvxpy')
    from dwd.socp_dwd import DWD
    from dwd.svm import SVM
    return cp, {'DWD': DWD, 'SVM': SVM}


def run(model, tmp_path):
    pipeline = Pipeline([('nested', Pipeline([('model', model)]))])
    return run_search_workflow(
        pipeline, {'nested__model__C': [.1]}, X, Y, X[:4], Y[:4],
        list(StratifiedKFold(2).split(X, Y)), new_run_directory(tmp_path / 'run'),
        jobs=1, verbose=0, refine=False)


def receipt(tmp_path):
    return json.loads((tmp_path / 'run' / 'refit.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('family,backend,cpp_threads', [('DWD', 'SCIPY', 3), ('SVM', 'CPP', 1)])
def test_clarabel_thread_limit_applies_only_to_final_clone(
        conic, family, backend, cpp_threads, monkeypatch, tmp_path):
    cp, classes = conic
    cls = classes[family]
    clarabel = pytest.importorskip('clarabel')
    if not hasattr(clarabel.DefaultSettings(), 'max_threads') or not hasattr(cp, 'get_num_threads'):
        pytest.skip('Requires configurable Clarabel and CVXPY thread limits')
    options = dict(solver='CLARABEL', direct_solve_method='faer', max_threads=4,
                   max_iter=100, canon_backend=backend)
    model = cls(solver_kws=deepcopy(options))
    observed = []
    fit = cls.fit

    def observe(self, X, y):
        try:
            result = fit(self, X, y)
        except Exception as error:
            if len(y) == 6 and 'Bad value for field "direct_solve_method"' in str(error):
                pytest.skip('Requires the optional FAER backend in the Clarabel build')
            raise
        solver = self.problem_._solver_cache['CLARABEL']
        observed.append((len(y), cp.get_num_threads(), faer_thread_limit(solver)))
        return result

    monkeypatch.setattr(cls, 'fit', observe)
    original = cp.get_num_threads()
    cp.set_num_threads(cpp_threads)
    try:
        output = run(model, tmp_path)
        assert cp.get_num_threads() == cpp_threads
    finally:
        cp.set_num_threads(original)
    assert observed == [(6, cpp_threads, 4), (6, cpp_threads, 4), (12, cpp_threads, 1)]
    assert model.solver_kws == options
    assert output['model']['nested']['model'].solver_kws == {**options, 'max_threads': 1}
    assert output['completed']['refit_seconds'] == receipt(tmp_path)['wall_seconds'] > 0
    controls = receipt(tmp_path)['native_runtime']['conic_threads']
    assert controls['canonicalization_backends'] == {'nested__model': backend}
    assert controls.get('cvxpy_cpp_threads') == (1 if backend == 'CPP' else None)


@pytest.mark.parametrize('options', [
    None, {'solver': 'SCS'}, {'solver': 'HIGHS'},
    {'solver': 'SCIPY', 'scipy_options': {'method': 'highs-ds'}},
    {'solver_path': [('CLARABEL', {'max_iter': 100})]},
])
def test_other_solver_dispatch_keeps_the_model_and_omits_refit_timing(conic, options, tmp_path):
    cp, classes = conic
    if options and 'solver' in options and options['solver'] not in cp.installed_solvers():
        pytest.skip('Requires ' + options['solver'])
    if options and 'solver_path' in options and not hasattr(cp.Problem, '_solve_solver_path'):
        pytest.skip('Requires CVXPY solver paths')
    model = classes['SVM'](solver_kws=deepcopy(options))
    output = run(model, tmp_path)
    fitted = output['model']['nested']['model']
    assert fitted.solver_kws == model.solver_kws == options
    assert np.array_equal(fitted.predict(X[:4]), Y[:4])
    assert output['completed']['held_out_accuracy'] == 1
    assert output['completed']['refit_seconds'] is output['completed']['final_native_threads'] is None
    assert receipt(tmp_path)['wall_seconds'] is None
    assert output['completed']['timing_unavailable'] == receipt(tmp_path)['native_runtime']['timing_unavailable']


@pytest.mark.parametrize('options', [None, {'solver': 'CLARABEL', 'canon_backend': 'SCIPY'}])
def test_final_error_preserves_compiler_setting_and_failure_receipt(conic, options, monkeypatch, tmp_path):
    cp, classes = conic
    SVM = classes['SVM']
    if not hasattr(cp, 'get_num_threads') or 'CLARABEL' not in cp.installed_solvers():
        pytest.skip('Requires CVXPY thread controls and Clarabel')
    error = RuntimeError('selected conic fit failed')
    fit = SVM.fit

    def fail_selected(self, X, y):
        if len(y) == len(Y):
            raise error
        return fit(self, X, y)

    original_threads = cp.get_num_threads()
    monkeypatch.setattr(SVM, 'fit', fail_selected)
    with pytest.raises(RuntimeError) as caught:
        run(SVM(solver_kws=deepcopy(options)), tmp_path)
    assert caught.value is error
    assert cp.get_num_threads() == original_threads
    record = receipt(tmp_path)
    assert record['state'] == 'failed'
    import clarabel
    timing_available = options is not None and hasattr(clarabel.DefaultSettings(), 'max_threads')
    assert (record['wall_seconds'] is not None) == timing_available
    assert not (tmp_path / 'run' / 'completed.json').exists()


def test_custom_conic_subclass_retains_fit_and_held_out_scoring(conic, tmp_path):
    _, classes = conic

    class CustomSVM(classes['SVM']):
        def __init__(self, C=1.):
            self.C = C

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.training_rows_ = len(y)
            return self

        def predict(self, X):
            return self.classes_[(X[:, 0] > 0).astype(int)]

    output = run(CustomSVM(), tmp_path)
    fitted = output['model']['nested']['model']
    assert fitted.training_rows_ == len(Y)
    np.testing.assert_array_equal(fitted.predict(X), Y)
    assert output['completed']['held_out_accuracy'] == 1.
    assert output['completed']['refit_seconds'] is None
    assert 'timing_unavailable' in receipt(tmp_path)['native_runtime']


@pytest.mark.parametrize('custom_first', [False, True])
def test_custom_conic_prediction_is_unverified_in_either_pipeline_position(conic, custom_first):
    _, classes = conic

    class CustomSVM(classes['SVM']):
        def __init__(self, C=1.):
            self.C = C  # This model has no solver_kws option.

    standard = classes['SVM']()
    custom = CustomSVM()
    first, last = (custom, standard) if custom_first else (standard, custom)
    model = Pipeline([('select', SelectFromModel(first)), ('model', last)])
    for cap in (None, True):
        receipt = configure_final_model(model, accepted_cap=cap)
        assert receipt['timing_unavailable'] == receipt['prediction_timing_unavailable']
        assert standard.solver_kws is None
        assert not hasattr(custom, 'solver_kws')


def test_effective_compilation_backend_preserves_environment_and_explicit_choice(conic, monkeypatch):
    cp, classes = conic
    clarabel = pytest.importorskip('clarabel')
    if not hasattr(clarabel.DefaultSettings(), 'max_threads') or not hasattr(cp, 'get_num_threads'):
        pytest.skip('Requires configurable Clarabel and CVXPY thread limits')
    monkeypatch.setenv('CVXPY_DEFAULT_CANON_BACKEND', 'unverified')
    original = cp.get_num_threads()
    model = classes['SVM'](solver_kws={'solver': 'CLARABEL', 'max_threads': 4})
    record = configure_final_model(model)
    assert 'timing_unavailable' in record
    assert model.solver_kws['max_threads'] == 4
    monkeypatch.setenv('CVXPY_DEFAULT_CANON_BACKEND', 'CPP')
    cp.set_num_threads(3)
    try:
        record = configure_final_model(model)
        assert 'CPP compilation' in record['timing_unavailable']
        assert cp.get_num_threads() == 3
        assert model.solver_kws['max_threads'] == 4
    finally:
        cp.set_num_threads(original)
    model.set_params(solver_kws={**model.solver_kws, 'canon_backend': 'SCIPY'})
    record = configure_final_model(model)
    assert record['conic_threads']['canonicalization_backends'] == {'<root>': 'SCIPY'}
    assert model.solver_kws['max_threads'] == 1
    assert cp.get_num_threads() == original
