"""Scoped final-fit thread budgets for the search examples."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from numbers import Integral
from pathlib import Path

from threadpoolctl import threadpool_limits


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


def _cli_positive(value):
    try:
        return validate_final_native_threads(int(value))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_final_thread_argument(parser):
    parser.add_argument('--final-native-threads', type=_cli_positive, default=1,
                        help='Native threads for the selected full-training pipeline fit; '
                             'default 1. CV and held-out scoring remain at one thread; '
                             'nested pair fitting must remain sequential.')


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
                    receipt['observation_error'] = str(observation_error)
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
                receipt['restoration_error'] = str(error)
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
                    probe['restoration_error'] = str(error)
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
