"""Concrete affine requests, public search and dependency/cache boundaries.

Existing affine tests retain real fitted-state, exact cancellation, pickle and
cache-reuse checks. These focused contracts cover removal of the auto alias.
"""
import inspect

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from grid_search_test_support import fit_with_serial_refit
from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier
import ensemble_dwd.estimator as estimator_module


X = np.column_stack((np.linspace(-1., 1., 12), np.arange(12) % 3 / 4.))
Y = np.tile([-3, 7], 6)


def model(**overrides):
    options = dict(M=1, theta=1., sv_selection_ratio=1., lambd=.1,
                   gamma=.25, max_iter=1, stopping='fixed', random_state=19,
                   prediction_batch_size=None)
    options.update(overrides)
    return EnsembleKernelDWD(**options)


def test_default_clone_and_nested_parameters_are_concrete_joint():
    default = EnsembleKernelDWD()
    signature = inspect.signature(EnsembleKernelDWD)
    assert signature.parameters['affine_computation'].default == 'joint'
    assert default.get_params()['affine_computation'] == 'joint'
    assert clone(default).affine_computation == 'joint'
    assert default.stopping == 'objective'
    assert (default.max_iter, default.obj_tol, default.tol) == (100, 1e-5, 1e-6)
    for name in ('backend', 'acceleration', 'residual_check_order'):
        assert default.get_params()[name] == 'auto'
    wrapper = clone(OvOClassifier(default, n_jobs=1))
    assert wrapper.get_params()['estimator__affine_computation'] == 'joint'
    assert wrapper.set_params(estimator__affine_computation='standard') is wrapper
    assert clone(wrapper).get_params()['estimator__affine_computation'] == 'standard'
    assert default.affine_computation == 'joint'


@pytest.mark.parametrize('requested', [None, 'joint', 'standard'])
def test_hidden_capable_constructor_receives_concrete_request(monkeypatch, requested):
    calls = []

    def hidden(**kwargs):
        calls.append(kwargs.copy())
        return KernGDWD(**kwargs)

    monkeypatch.setattr(estimator_module, 'KernGDWD', hidden)
    subject = model(**({} if requested is None else {'affine_computation': requested}))
    child = subject._make_estimator({'gamma': .25})
    expected = 'joint' if requested is None else requested
    assert len(calls) == 1
    assert calls[0]['affine_computation'] == expected
    assert child.affine_computation == expected
    assert type(child) is KernGDWD
    assert child.initialization == 'zero'
    assert child.stopping == 'fixed'


@pytest.mark.parametrize('policy', ['standard', 'joint'])
@pytest.mark.parametrize('reason', ['unsupported_keyword', 'internal_constructor'])
def test_dependency_typeerror_retains_cause_and_has_no_retry(monkeypatch, policy, reason):
    calls = []
    message = ("unexpected keyword argument 'affine_computation'"
               if reason == 'unsupported_keyword' else 'internal constructor failure')
    original = TypeError(message)

    def broken(**kwargs):
        calls.append(kwargs.copy())
        raise original

    monkeypatch.setattr(estimator_module, 'KernGDWD', broken)
    with pytest.raises(ImportError, match='requested numerical computation options.*pinned') as caught:
        model(affine_computation=policy)._make_estimator({'gamma': .25})
    assert caught.value.__cause__ is original
    assert len(calls) == 1
    assert calls[0]['affine_computation'] == policy


@pytest.mark.parametrize('assignment', ['constructor', 'set_params'])
def test_removed_auto_fails_before_dependency_construction(monkeypatch, assignment):
    def forbidden(**kwargs):
        pytest.fail('Removed affine alias reached dependency construction')

    monkeypatch.setattr(estimator_module, 'KernGDWD', forbidden)
    subject = model(**({'affine_computation': 'auto'} if assignment == 'constructor' else {}))
    if assignment == 'set_params':
        subject.set_params(affine_computation='auto')
    copied = clone(subject)
    with pytest.raises(ValueError, match='affine_computation.*standard.*joint'):
        copied.fit(X, Y)
    assert not any(name.endswith('_') for name in vars(copied))


@pytest.mark.parametrize('route', ['direct', 'pipeline', 'ovo'])
def test_removed_auto_grid_is_rejected_not_silently_resolved(monkeypatch, route):
    def forbidden(**kwargs):
        pytest.fail('Removed affine grid value reached dependency construction')

    monkeypatch.setattr(estimator_module, 'KernGDWD', forbidden)
    if route == 'direct':
        subject, parameter, labels = model(), 'affine_computation', Y
    elif route == 'pipeline':
        subject, parameter, labels = Pipeline([('ensemble', model())]), 'ensemble__affine_computation', Y
    else:
        subject = OvOClassifier(model(), n_jobs=1)
        parameter, labels = 'estimator__affine_computation', np.repeat(['a', 'b', 'c'], 4)
    with pytest.raises(ValueError, match='affine_computation.*standard.*joint'):
        GridSearchCV(subject, {parameter: ['auto']}, cv=2,
                     n_jobs=1, refit=False, error_score='raise').fit(X, labels)


def test_cache_identity_uses_concrete_requested_and_resolved_values(monkeypatch, tmp_path):
    # A cache-key unit contract: no base fit, cache file, or numerical result is
    # produced here. Existing tests separately exercise real cold/warm reuse.
    keys = []
    sentinel = object()

    def cached(directory, key_data, compute, *, verify):
        assert verify()
        keys.append(key_data['parameters'].copy())
        return sentinel, True, 'test-only-key'

    monkeypatch.setattr(estimator_module, '_runtime_provenance', lambda cls: {'fixed_test_identity': True})
    monkeypatch.setattr(estimator_module, '_cached_base_stage', cached)
    for requested in (None, 'joint', 'standard'):
        subject = model(cache_dir=tmp_path,
                        **({} if requested is None else {'affine_computation': requested}))
        result, hit, key, status = subject._base_result(
            X, Y, np.unique(Y), [np.arange(len(Y))], {'gamma': .25})
        assert result is sentinel
        assert (hit, key, status) == (True, 'test-only-key', 'hit')
        expected = 'joint' if requested is None else requested
        assert keys[-1]['affine_computation'] == expected
        assert keys[-1]['affine_computation_resolved'] == expected
    assert keys[0] == keys[1]
    changed = {name for name in keys[1] if keys[1][name] != keys[2][name]}
    assert changed == {'affine_computation', 'affine_computation_resolved'}


@pytest.mark.parametrize('route', ['direct', 'pipeline'])
def test_cv_keeps_both_concrete_values_and_refits_selected_policy(route):
    subject = model() if route == 'direct' else Pipeline([('ensemble', model())])
    parameter = 'affine_computation' if route == 'direct' else 'ensemble__affine_computation'
    search = GridSearchCV(subject, {parameter: ['joint', 'standard']}, cv=2,
                          n_jobs=1, error_score='raise')
    fit_with_serial_refit(search, X, Y)
    assert [row[parameter] for row in search.cv_results_['params']] == ['joint', 'standard']
    fitted = (search.best_estimator_ if route == 'direct'
              else search.best_estimator_.named_steps['ensemble'])
    expected = search.best_params_[parameter]
    assert type(fitted) is EnsembleKernelDWD
    assert fitted.affine_computation == fitted.affine_computation_ == expected
    for learner in [*fitted.estimators_, fitted.final_estimator_]:
        assert learner.affine_computation == learner.affine_computation_ == expected
