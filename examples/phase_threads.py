"""Scoped final-fit thread budgets for the search examples."""
import argparse
from collections.abc import Mapping, Sequence, Set
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np
from joblib import parallel_config
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from ensemble_dwd import OvOClassifier


def validate_final_native_threads(value, estimator=None):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('final_native_threads must be a positive integer.')
    value = int(value)
    if value != 1 and estimator is not None:
        for name, setting in estimator.get_params(deep=True).items():
            if name.split('__')[-1] == 'n_jobs' and (
                    isinstance(setting, bool) or not isinstance(setting, Integral)
                    or setting != 1):
                raise ValueError('Nondefault final_native_threads requires explicit '
                                 'n_jobs=1 for every nested estimator; found ' + name + '.')
    return value


def validate_timed_native_threads(value):
    if validate_final_native_threads(value) != 1:
        raise ValueError('Timed final fitting requires final_native_threads=1.')
    return 1


def _resource_key(path):
    """Use sklearn labels when they round-trip, and JSON arrays for ambiguous paths."""
    if not path:
        return '<root>'
    name = '__'.join(path)
    if all(path) and name != '<root>' and not name.startswith('[') and tuple(name.split('__')) == path:
        return name
    return json.dumps(path, ensure_ascii=True, separators=(',', ':'))


def _unavailable_model_timing(path, reason):
    message = f'{_resource_key(path)} {reason}'
    return {'estimator_workers': None, 'worker_parameters': {},
            'timing_unavailable': message, 'prediction_timing_unavailable': message}


def _custom_clone(component):
    hook = getattr(component, '__sklearn_clone__', None)
    return hook is not None and not (isinstance(hook, MethodType) and hook.__self__ is component
                                     and hook.__func__ is BaseEstimator.__sklearn_clone__)


def clone_risk_paths(source):
    """Locate clone inputs whose products could contain caller-owned estimators."""
    paths = []

    def visit_steps(value, path):
        if type(value) not in (list, tuple):
            paths.append(['steps', *path])
            return
        for index, pair in enumerate(value):
            if type(pair) in (list, tuple):
                visit(pair, [*path, index])
            else:
                paths.append(['step_pair', *path, index])

    def visit(value, path):
        if isinstance(value, type):
            return
        kind = type(value)
        if _custom_clone(value):
            paths.append(['value', *path])
        elif kind is dict:
            for key, item in value.items():
                if value is source and isinstance(key, str) and key.split('__')[-1] == 'steps':
                    visit_steps(item, [*path, key])
                else:
                    visit(item, [*path, key])
        elif kind in (list, tuple):
            for index, item in enumerate(value):
                visit(item, [*path, index])
        elif kind in (set, frozenset):
            if any(clone_risk_paths(item) for item in value):
                paths.append(['value', *path])
        elif callable(getattr(value, 'get_params', None)):
            for key, item in value.get_params(deep=False).items():
                if kind is Pipeline and key == 'steps':
                    visit_steps(item, [*path, key])
                else:
                    visit(item, [*path, key])
        elif (kind not in (str, bytes, bytearray, range, np.ndarray)
              and not (isinstance(value, np.generic) and kind is value.dtype.type)
              and (isinstance(value, (Mapping, Sequence, Set))
                   or callable(getattr(value, 'fit', None))
                   or callable(getattr(value, '__deepcopy__', None)))):
            # sklearn deep-copies container subclasses. Even an ordinary list
            # returned by their hook may contain the caller's original steps.
            paths.append(['value', *path])

    visit(source, [])
    return paths


def _reachable_model_objects(value):
    """Keep references to reachable estimators across sklearn's parameter updates."""
    objects = {}
    pending = [value]
    seen = set()
    while pending:
        item = pending.pop()
        if isinstance(item, type) or id(item) in seen:
            continue
        seen.add(id(item))
        if callable(getattr(item, 'get_params', None)):
            objects[id(item)] = item
            if not _custom_clone(item):
                pending.extend(item.get_params(deep=False).values())
                if type(item) is Pipeline:
                    pending.extend(step for _, step in item.steps)
        elif isinstance(item, Mapping):
            pending.extend(item.values())
        elif isinstance(item, (Sequence, Set)) and not isinstance(item, (str, bytes, bytearray, range)):
            pending.extend(item)
        elif callable(getattr(item, 'fit', None)):
            objects[id(item)] = item
    return objects


def _clone_products(model, paths):
    borrowed = {}
    for shape, *path in paths:
        value = model
        for key in path:
            if callable(getattr(value, 'get_params', None)):
                value = value.get_params(deep=False)[key]
            else:
                value = value[key]
        # Pipeline accepts iterable pairs. Their copying protocol can hide a
        # shared estimator in a container without any estimator interface.
        roots = [value]
        if shape != 'value':
            try:
                if shape == 'steps':
                    roots.extend(step for _, step in value)
                else:
                    _, step = value
                    roots.append(step)
            except (TypeError, ValueError, IndexError):
                # A non-pipeline estimator may also have an option named steps.
                pass
        borrowed.update(_reachable_model_objects(roots))
    return borrowed


def clone_final_model(clone, template, parameters=None, *, original_risks=()):
    """Apply sklearn's selected replacements, then check which clone risks survived."""
    template_paths = clone_risk_paths(template)
    parameter_paths = clone_risk_paths(parameters) if parameters is not None else ()
    model = clone(template)
    borrowed = _clone_products(model, [*original_risks, *template_paths])
    if parameters is not None:
        replacements = clone(parameters, safe=False)
        borrowed.update(_clone_products(replacements, parameter_paths))
        model = model.set_params(**replacements)
    may_share = bool(borrowed and not borrowed.keys().isdisjoint(_reachable_model_objects(model)))
    return model, may_share


def configure_final_model(model, *, accepted_cap=None, source_may_share=False):
    """Configure an owned full-training clone and return its timing evidence."""
    if source_may_share:
        return _unavailable_model_timing((), 'may contain caller-owned state from cloning the selected model.')
    pending = [((), model)]
    nodes = {}
    updates = []
    while pending:
        path, component = pending.pop()
        kind = type(component)
        if _custom_clone(component):
            return _unavailable_model_timing(path, 'has custom cloning and may share caller-owned state.')
        if isinstance(component, type) or not callable(getattr(kind, 'get_params', None)):
            return _unavailable_model_timing(path, 'has no verifiable estimator parameter interface.')
        shallow = component.get_params(deep=False)
        children = ()
        if kind is Pipeline:
            if (type(component.steps) not in (list, tuple)
                    or any(type(pair) not in (list, tuple) for pair in component.steps)):
                return _unavailable_model_timing(path, 'has a steps container with unverified cloning behavior.')
            children = [(name, step) for name, step in component.steps
                        if step is not None and not (isinstance(step, str) and step == 'passthrough')]
            other_options = shallow.values()
        elif kind in (OvOClassifier, SelectFromModel):
            children = [('estimator', component.estimator)]
            other_options = [value for option, value in shallow.items() if option != 'estimator']
        else:
            # Other compositions may expose children only in their deep parameters.
            other_options = (shallow.values() if kind.get_params is BaseEstimator.get_params
                             else component.get_params(deep=True).values())
        if ((isinstance(component, MetaEstimatorMixin) and kind not in (Pipeline, OvOClassifier, SelectFromModel))
                or any(not isinstance(value, type) and callable(getattr(value, 'get_params', None))
                       for value in other_options)):
            return _unavailable_model_timing(path, 'has unverified worker behavior for its nested estimators.')
        nodes[path] = component
        options = {name: 1 for name in shallow if name in ('n_jobs', 'base_jobs')}
        if options:
            updates.append((path, component, options))
        pending.extend((path + (name,), child) for name, child in reversed(children))

    # Check ownership throughout the composition before mutating any component.
    # Pipeline step names do not have to round-trip through sklearn's '__' syntax.
    workers = {}
    for path, component, options in updates:
        component.set_params(**options)
        workers.update((_resource_key(path + (option,)), 1) for option in options)
    receipt = {'estimator_workers': 1, 'worker_parameters': workers}
    _configure_conic_refit(nodes, receipt, accepted_cap)
    return receipt


def _configure_conic_refit(nodes, receipt, accepted_cap):
    """Bound the paired packages' CVXPY models on the owned final clone."""
    # CVXPY is optional. A caller using these classes has already loaded them.
    classes = tuple(cls for cls in (getattr(sys.modules.get('dwd.socp_dwd'), 'DWD', None),
                                    getattr(sys.modules.get('dwd.svm'), 'SVM', None))
                    if isinstance(cls, type))
    if not classes:
        return
    models = [(_resource_key(path), value) for path, value in nodes.items()
              if isinstance(value, classes)]
    if not models:
        return
    for name, value in models:
        if type(value) not in classes:
            reason = f'{name} has custom conic fitting and prediction behavior without a verified thread limit.'
            receipt.update(timing_unavailable=reason, prediction_timing_unavailable=reason)
            return

    # Recovery reapplies the cap authenticated by the handoff even if today's
    # compiler settings prevent comparing a new fit time. The resulting model
    # is checked against the handoff's exact typed identity before fitting.
    if accepted_cap is True:
        for _, value in models:
            value.set_params(solver_kws={**value.solver_kws, 'max_threads': 1})

    import cvxpy
    default_backend = os.environ.get('CVXPY_DEFAULT_CANON_BACKEND',
                                     getattr(cvxpy.settings, 'DEFAULT_CANON_BACKEND', None))
    backends = {}
    for name, value in models:
        if value.solver_kws is not None and type(value.solver_kws) is not dict:
            receipt['timing_unavailable'] = f'{name} has custom solver behavior without a verified thread limit.'
            return
        options = value.solver_kws or {}
        solver = options.get('solver')
        backends[name] = options.get('canon_backend') or default_backend
        # Keep automatic dispatch, fallback order, custom solve methods and
        # other libraries' private schedulers intact, including their failures.
        if (type(solver) is not str or solver != 'CLARABEL'
                or options.get('solver_path') is not None or options.get('method') is not None
                or backends[name] not in ('CPP', 'SCIPY')):
            receipt['timing_unavailable'] = (
                f'{name} uses a CVXPY solve or compilation route without a verified single-thread limit.')
            return

    import clarabel
    if not hasattr(clarabel.DefaultSettings(), 'max_threads'):
        receipt['timing_unavailable'] = 'Installed Clarabel lacks a max_threads control.'
        return
    # CPP's setting is global to this process. Preserve it during this fit
    # and any overlapping CVXPY solves. SciPy uses the enclosing pool scope.
    if 'CPP' in backends.values() and (
            not hasattr(cvxpy, 'get_num_threads') or cvxpy.get_num_threads() != 1):
        receipt['timing_unavailable'] = 'CVXPY CPP compilation requires an existing one-thread setting for timing.'
        return

    if accepted_cap is False and any(value.solver_kws.get('max_threads') != 1 for _, value in models):
        receipt['timing_unavailable'] = 'The accepted conic model has no single-thread solver cap.'
        return
    if accepted_cap is None:
        for _, value in models:
            value.set_params(solver_kws={**value.solver_kws, 'max_threads': 1})
    receipt['conic_threads'] = {'canonicalization_backends': backends,
                               'clarabel_max_threads': {name: 1 for name, _ in models}}
    if 'CPP' in backends.values():
        receipt['conic_threads']['cvxpy_cpp_threads'] = 1


@contextmanager
def timed_model_phase():
    """Limit ambient joblib work and all native pools loaded at phase entry."""
    with parallel_config(backend='sequential', n_jobs=1), threadpool_limits(limits=1):
        yield


def _cli_final_threads(value):
    try:
        return validate_timed_native_threads(int(value))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_final_thread_argument(parser):
    parser.add_argument('--final-native-threads', type=_cli_final_threads, default=1, metavar='1',
                        help='Timed final fitting and held-out prediction use one worker '
                             'and one native thread. The only supported value is 1.')


def report_final_threads(value):
    print(f'Native threads: CV 1 per worker; final pipeline fit {value}; held-out scoring 1.')


def final_thread_policy_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _one_thread_identity(runtime):
    result = deepcopy(runtime)
    for pool in result.get('native_pools', []):
        pool['num_threads'] = 1
    return result


@contextmanager
def _final_fit_threads(value, estimator, receipt, *, frozen_runtime=None, native_baseline=None):
    """Check the actual phase runtime and restore limits on every exit.

    The caller already holds a one-thread workflow scope. This context changes
    only the fresh final pipeline fit, and accepts no parallel nested estimators
    for a wider budget. It does not set CV workers or change estimator defaults.
    """
    value = validate_final_native_threads(value, estimator)
    receipt.update(requested_native_threads=value, state='preparing')
    if value == 1:
        # The existing workflow already holds this limit. Keep its original
        # observations and failure behavior; do not add inventory work here.
        receipt.update(scope_changed=False, observation='existing_workflow_scope',
                       observed_final_runtime=None)
        if frozen_runtime is not None:
            receipt['frozen_cv_runtime'] = frozen_runtime
        try:
            yield
        except BaseException as error:
            receipt.update(state='failed', fit_exception=type(error).__name__)
            raise
        receipt['state'] = 'completed'
        return

    import cv_checkpoint
    from search_support import exception_message

    receipt['scope_changed'] = True
    full_boundary_check = frozen_runtime is None
    if native_baseline is None:
        baseline = cv_checkpoint.runtime_fingerprint(check_inventory=full_boundary_check)
        if frozen_runtime is not None and cv_checkpoint._stable(baseline) != frozen_runtime:
            raise cv_checkpoint.CheckpointError('Final-fit runtime differs from frozen CV identity')
    else:
        if frozen_runtime is not None:
            raise ValueError('Native resource authority cannot replace a frozen checkpoint runtime')
        baseline = deepcopy(native_baseline)
        if _native_resource_runtime(1) != baseline:
            raise cv_checkpoint.CheckpointError('Final-fit native runtime changed since preflight')
        receipt.update(runtime_policy=_NATIVE_SCOPE_POLICY, source_installation_attestation=False)
    receipt['before_scope'] = baseline
    original_error = None
    entered = False

    def observe():
        runtime = (cv_checkpoint.runtime_fingerprint(check_inventory=False, native_threads=value)
                   if native_baseline is None else _native_resource_runtime(value))
        if _one_thread_identity(runtime) != baseline:
            raise cv_checkpoint.CheckpointError('Final-fit runtime changed beyond its thread budget')
        return runtime

    try:
        with threadpool_limits(limits=value):
            entered = True
            receipt['before_fit'] = observe()
            receipt['state'] = 'running'
            try:
                yield
            except BaseException as error:
                receipt['fit_exception'] = type(error).__name__
                try:
                    receipt['after_fit'] = observe()
                except BaseException as observation_error:
                    receipt['observation_error'] = exception_message(observation_error, catch=BaseException)
                raise
            receipt['after_fit'] = observe()
            receipt['state'] = 'completed'
    except BaseException as error:
        original_error = error
        receipt['state'] = 'failed'
        raise
    finally:
        if entered:
            try:
                restored = (cv_checkpoint.runtime_fingerprint(check_inventory=full_boundary_check)
                            if native_baseline is None else _native_resource_runtime(1))
                receipt['after_scope'] = restored
                if restored != baseline:
                    raise cv_checkpoint.CheckpointError('Final-fit scope did not restore the CV runtime')
                receipt['restored'] = True
            except BaseException as error:
                receipt['restored'] = False
                receipt['restoration_error'] = exception_message(error, catch=BaseException)
                if original_error is None:
                    receipt['state'] = 'failed'
                    raise


def _preflight_final_native_threads(value, estimator, *, frozen_runtime=None, native_only=False):
    """Check feasibility before CV; the actual final-fit scope stays independent."""
    value = validate_final_native_threads(value, estimator)
    receipt = {'requested_native_threads': value, 'fit_performed': False}
    if value == 1:
        # CV already checks its existing one-thread scope.
        return {**receipt, 'state': 'existing_cv_scope', 'runtime_probe': None}
    probe = {}
    with threadpool_limits(limits=1):
        # Preserve the original validation after entering the one-thread scope.
        value = validate_final_native_threads(value, estimator)
        probe.update(requested_native_threads=value, state='preparing')
        import cv_checkpoint
        from search_support import exception_message

        probe['scope_changed'] = True
        probe['observation'] = 'native_resource_only' if native_only else 'native_budget_only'
        full_boundary_check = frozen_runtime is None
        if native_only:
            if frozen_runtime is not None:
                raise ValueError('Native resource authority cannot replace a frozen checkpoint runtime')
            probe.update(runtime_policy=_NATIVE_SCOPE_POLICY, source_installation_attestation=False)
            baseline = _native_resource_runtime(1)
        else:
            baseline = cv_checkpoint.runtime_fingerprint(check_inventory=full_boundary_check)
        if frozen_runtime is not None and cv_checkpoint._stable(baseline) != frozen_runtime:
            raise cv_checkpoint.CheckpointError('Final-fit runtime differs from frozen CV identity')
        probe['before_scope'] = baseline
        original_error = None
        entered = False

        def observe():
            # This empty scope observes native fields. Strict callers also
            # retain complete runtime identity checks at both boundaries.
            pools = cv_checkpoint._native_pool_fingerprint({}, native_threads=value)
            runtime = {'native_pools': sorted(pools, key=lambda row: row['filepath'])}
            if _one_thread_identity(runtime)['native_pools'] != baseline['native_pools']:
                raise cv_checkpoint.CheckpointError('Final-fit runtime changed beyond its thread budget')
            return runtime

        try:
            with threadpool_limits(limits=value):
                entered = True
                probe['before_fit'] = observe()
                probe['state'] = 'running'
                probe['after_fit'] = observe()
                probe['state'] = 'completed'
        except BaseException as error:
            original_error = error
            probe['state'] = 'failed'
            raise
        finally:
            if entered:
                try:
                    restored = (_native_resource_runtime(1) if native_only else
                                cv_checkpoint.runtime_fingerprint(check_inventory=full_boundary_check))
                    probe['after_scope'] = restored
                    if restored != baseline:
                        raise cv_checkpoint.CheckpointError('Final-fit scope did not restore the CV runtime')
                    probe['restored'] = True
                except BaseException as error:
                    probe['restored'] = False
                    probe['restoration_error'] = exception_message(error, catch=BaseException)
                    if original_error is None:
                        probe['state'] = 'failed'
                        raise
    return {**receipt, 'state': 'feasible', 'runtime_probe': probe}


# Only run_search_workflow explicitly opts into this noncheckpoint resource scope.
# Strict public wrappers below retain full/frozen package authority for their callers.
_NATIVE_SCOPE_POLICY = 'noncheckpoint-native-scope-v1'


def _native_resource_runtime(native_threads):
    import cv_checkpoint
    pools = cv_checkpoint._native_pool_fingerprint({}, native_threads=native_threads)
    return {'runtime_policy': _NATIVE_SCOPE_POLICY,
            'source_installation_attestation': False,
            'native_pools': sorted(pools, key=lambda row: row['filepath'])}


@contextmanager
def final_fit_threads(value, estimator, receipt, *, frozen_runtime=None):
    """Apply the original full/frozen runtime guard to the actual final fit."""
    with _final_fit_threads(value, estimator, receipt, frozen_runtime=frozen_runtime):
        yield


def preflight_final_native_threads(value, estimator, *, frozen_runtime=None):
    """Preserve strict feasibility checks, including a not-yet-frozen checkpoint."""
    return _preflight_final_native_threads(value, estimator, frozen_runtime=frozen_runtime)


def _noncheckpoint_preflight_final_native_threads(value, estimator):
    """Check resources before CV; no package/source installation attestation."""
    return _preflight_final_native_threads(value, estimator, native_only=True)


@contextmanager
def _noncheckpoint_final_fit_threads(value, estimator, receipt, *, preflight):
    """Keep one native baseline across noncheckpoint CV and the final fit."""
    value = validate_final_native_threads(value)
    baseline = None
    if value != 1:
        if (not isinstance(preflight, dict) or preflight.get('state') != 'feasible'
                or preflight.get('requested_native_threads') != value):
            raise ValueError('A matching successful native preflight is required')
        probe = preflight.get('runtime_probe')
        if (not isinstance(probe, dict) or probe.get('state') != 'completed'
                or probe.get('restored') is not True
                or probe.get('before_scope') != probe.get('after_scope')):
            raise ValueError('A completed and restored native preflight is required')
        baseline = probe.get('before_scope')
        if (not isinstance(baseline, dict)
                or baseline.get('runtime_policy') != _NATIVE_SCOPE_POLICY
                or baseline.get('source_installation_attestation') is not False):
            raise ValueError('A native resource baseline is required')
    receipt.update(runtime_policy=_NATIVE_SCOPE_POLICY, source_installation_attestation=False)
    with _final_fit_threads(value, estimator, receipt, native_baseline=baseline):
        yield
