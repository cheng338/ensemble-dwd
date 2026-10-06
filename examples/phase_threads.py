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
def final_fit_threads(value, estimator, receipt, *, frozen_runtime=None):
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
    baseline = cv_checkpoint.runtime_fingerprint(check_inventory=full_boundary_check)
    if frozen_runtime is not None and cv_checkpoint._stable(baseline) != frozen_runtime:
        raise cv_checkpoint.CheckpointError('Final-fit runtime differs from frozen CV identity')
    receipt['before_scope'] = baseline
    original_error = None
    entered = False

    def observe():
        runtime = cv_checkpoint.runtime_fingerprint(check_inventory=False, native_threads=value)
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
                restored = cv_checkpoint.runtime_fingerprint(check_inventory=full_boundary_check)
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
