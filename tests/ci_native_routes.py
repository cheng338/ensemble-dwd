"""Mandatory installed-wheel route witnesses for explicitly selected CI contracts.

This is test tooling, not a production dispatch policy. Expected routes are
independent of the production predicates whose regressions it must detect.
"""
from contextlib import contextmanager
import inspect
import math
import platform
import sys
import sysconfig
import types


_PREFIX = 'Native route preflight: '


def _require(condition, message):
    if not condition:
        raise RuntimeError(_PREFIX + message)


def expected_routes(backend, contract, expected_platform, expected_python):
    """Return the declared CI contract without consulting production guards."""
    _require(backend in ('native', 'portable'), 'unknown wheel backend')
    _require(contract in ('current', 'released-1.3.19'), 'unknown route contract')
    _require(expected_platform in ('windows', 'linux'), 'unknown expected platform')
    _require(expected_python in ('3.11', '3.12'), 'unknown expected Python')
    extension = backend == 'native'
    scalar = expected_python == '3.12'
    guarded = scalar or (extension and expected_platform == 'windows')
    return {
        'raw_compiled_residual': extension,
        'guarded_residual_n8': ('compiled_values' if extension and guarded else
                                'scalar_fallback' if scalar else 'declined'),
        'guarded_residual_n4': 'scalar_fallback' if scalar else 'declined',
        'ordinary_scores': extension and expected_platform == 'windows',
        'reporting_scores': extension,
        'reporting_affine': extension and contract == 'current',
        'ordinary_affine': False,
    }


def _runtime_identity(expected_platform, expected_python):
    actual = {
        'implementation': sys.implementation.name,
        'platform': sys.platform,
        'python': '.'.join(str(v) for v in sys.version_info[:2]),
        'machine': platform.machine().lower(),
        'pointer_bits': 64 if sys.maxsize > 2**32 else 32,
        'gil_disabled': sysconfig.get_config_var('Py_GIL_DISABLED'),
        'float_format': [sys.float_info.radix, sys.float_info.mant_dig,
                         sys.float_info.max_exp],
    }
    _require(actual['implementation'] == 'cpython', 'CI contract requires CPython')
    _require(actual['platform'] == {'windows': 'win32', 'linux': 'linux'}[expected_platform],
             'actual platform does not match expected platform')
    _require(actual['python'] == expected_python, 'actual Python does not match expected Python')
    _require(actual['machine'] in ('amd64', 'x86_64') and actual['pointer_bits'] == 64,
             'CI contract requires x86-64')
    _require(actual['gil_disabled'] in (None, 0), 'CI contract requires a GIL-enabled build')
    _require(actual['float_format'] == [2, 53, 1024], 'CI contract requires binary64')
    return actual


def _array_equal(np, value, expected, route):
    _require(type(value) is np.ndarray and value.dtype == np.dtype('float64')
             and value.shape == expected.shape and bool(np.isfinite(value).all())
             and bool(np.array_equal(value, expected)), route + ' returned incorrect values or shape')


def _native_entry(accelerator, name):
    entry = getattr(accelerator, name, None)
    _require(isinstance(entry, types.BuiltinFunctionType)
             and getattr(entry, '__self__', None) is accelerator,
             name + ' is missing or is not the real installed native entry')
    return entry


@contextmanager
def _entry_witness(accelerator, name, expected_native):
    """Forward the actual native callable, restoring it on every exit."""
    original = getattr(accelerator, name, None)
    if expected_native:
        _native_entry(accelerator, name)
    observed = {'observed_calls': 0, 'completed_calls': 0, 'statuses': []}
    def forward(*args, **kwargs):
        observed['observed_calls'] += 1
        status = original(*args, **kwargs)
        observed['completed_calls'] += 1
        observed['statuses'].append(status)
        return status
    if original is not None:
        setattr(accelerator, name, forward)
    try:
        yield observed
    finally:
        if original is not None:
            setattr(accelerator, name, original)


def _check_witness(observed, expected_native, route):
    if expected_native:
        _require(observed['observed_calls'] > 0, route + ' required native entry did not execute')
        _require(observed['completed_calls'] == observed['observed_calls']
                 and all(type(status) is int and status == 0 for status in observed['statuses']),
                 route + ' native entry did not complete successfully')
    else:
        _require(observed['observed_calls'] == 0, route + ' unexpectedly entered native code')


def _score_probe(np, accelerator, name, expected_native, function, expected, route):
    try:
        with _entry_witness(accelerator, name, expected_native) as observed:
            result = function()
    except Exception as error:
        raise RuntimeError(_PREFIX + route + ' raised during its route witness') from error
    _array_equal(np, result, expected, route)
    _check_witness(observed, expected_native, route)
    return {'expected_native': expected_native, **observed, 'output_verified': True}


def _residual_probe(np, accelerator, native, residual_profile, n, expected_route):
    K, x, rhs = np.eye(n), np.ones(n), np.zeros(n)
    route = 'guarded_residual_n' + str(n)
    try:
        with _entry_witness(accelerator, 'evaluate', expected_route == 'compiled_values') as entry_observed:
            with residual_profile() as profile:
                result = native.native_compensated_residual(K, .5, rhs, x, .25, 0.)
            observed = profile.as_dict()
    except Exception as error:
        raise RuntimeError(_PREFIX + route + ' raised') from error
    _check_witness(entry_observed, expected_route == 'compiled_values', route)
    counts = observed['counts']
    _require(counts['calls'] == 1 and counts['raised'] == 0, route + ' has invalid call evidence')
    if expected_route == 'declined':
        _require(result is None and counts['declined'] == 1 and counts['bounded_return'] == 0
                 and counts['scalar_sumprod_calls_started'] == 0
                 and observed['compiled_helper']['values_returned'] == 0,
                 route + ' unexpectedly admitted an unsupported route')
    else:
        _require(isinstance(result, tuple) and len(result) == 6,
                 route + ' silently declined')
        _require(counts['bounded_return'] == 1 and counts['declined'] == 0
                 and observed['routes'][expected_route]['bounded_return'] == 1,
                 route + ' did not use expected ' + expected_route)
        _array_equal(np, result[0], np.full(n, -1.75), route + ' residual')
        _array_equal(np, result[2], np.ones(n), route + ' scores')
        _require(type(result[1]) is float and result[1] == -float(n),
                 route + ' returned an incorrect constraint')
        for index in (3, 5):
            bound = result[index]
            _require(type(bound) is np.ndarray and bound.dtype == np.dtype('float64')
                     and bound.shape == (n,) and bool(np.isfinite(bound).all())
                     and bool((bound >= 0).all()), route + ' returned invalid bounds')
        _require(math.isfinite(result[4]) and result[4] >= 0,
                 route + ' returned an invalid constraint bound')
        expected_scalar_calls = 2 * n if expected_route == 'scalar_fallback' else 0
        _require(counts['scalar_sumprod_calls_started'] == expected_scalar_calls
                 and counts['scalar_sumprod_calls_completed'] == expected_scalar_calls,
                 route + ' has incorrect scalar-call evidence')
        _require(observed['compiled_helper']['values_returned'] ==
                 (1 if expected_route == 'compiled_values' else 0),
                 route + ' has incorrect compiled-dispatch evidence')
    _array_equal(np, K, np.eye(n), route + ' kernel immutability')
    _array_equal(np, x, np.ones(n), route + ' coefficient immutability')
    _array_equal(np, rhs, np.zeros(n), route + ' rhs immutability')
    return {'expected_route': expected_route, 'observed': observed,
            'entry_witness': entry_observed,
            'bounded_return': result is not None, 'output_verified': True}


def verify_routes(backend, contract, expected_platform, expected_python):
    """Require intended routes and exact tiny results; never skip from a guard."""
    expected = expected_routes(backend, contract, expected_platform, expected_python)
    actual_runtime = _runtime_identity(expected_platform, expected_python)
    import numpy as np
    from dwd import _compiled_residual as compiled
    from dwd import _native_residual as native
    from dwd import _kernel_scores as scores
    from dwd import _affine_scores as affine
    from dwd.profiling import residual_profile

    accelerator = compiled._ACCEL
    if contract == 'released-1.3.19':
        _require(getattr(accelerator, 'score_affine_rows', None) is None
                 and '_reporting' not in inspect.signature(affine.affine_scores).parameters,
                 'released-1.3.19 affine absence contract does not match installed artifact')
    if backend == 'native':
        _require(accelerator is not None, 'native wheel has no loaded accelerator')
        _native_entry(accelerator, 'evaluate')
    else:
        _require(accelerator is None, 'portable wheel unexpectedly loaded an accelerator')
    K, alpha, rhs = np.eye(8), np.ones(8), np.zeros(8)
    try:
        with _entry_witness(accelerator, 'evaluate', expected['raw_compiled_residual']) as raw_witness:
            raw = compiled.compiled_values(K, alpha, rhs, .5, .25)
    except Exception as error:
        raise RuntimeError(_PREFIX + 'raw_compiled_residual raised') from error
    _check_witness(raw_witness, expected['raw_compiled_residual'], 'raw_compiled_residual')
    if expected['raw_compiled_residual']:
        _require(isinstance(raw, tuple) and len(raw) == 3, 'raw_compiled_residual silently declined')
        for value, reference in zip(raw, (np.ones(8), np.full(8, -1.75), np.ones(8))):
            _array_equal(np, value, reference, 'raw_compiled_residual')
    else:
        _require(raw is None, 'portable wheel unexpectedly used raw compiled residual')
    routes = {'raw_compiled_residual': {
        'expected_native': expected['raw_compiled_residual'],
        **raw_witness, 'values_returned': raw is not None, 'output_verified': True}}
    for n in (8, 4):
        key = 'guarded_residual_n' + str(n)
        routes[key] = _residual_probe(np, accelerator, native, residual_profile, n, expected[key])
    routes['ordinary_scores'] = _score_probe(
        np, accelerator, 'score_rows', expected['ordinary_scores'],
        lambda: scores.compensated_kernel_matvec(K, alpha), np.ones(8), 'ordinary_scores')
    routes['reporting_scores'] = _score_probe(
        np, accelerator, 'score_rows', expected['reporting_scores'],
        lambda: scores._reporting_kernel_matvec(K, alpha), np.ones(8), 'reporting_scores')
    affine_report_call = (
        (lambda: affine.affine_scores(K, alpha, .25, policy='compensated', _reporting=True))
        if contract == 'current' else
        (lambda: affine.affine_scores(K, alpha, .25, policy='compensated')))
    routes['reporting_affine'] = _score_probe(
        np, accelerator, 'score_affine_rows', expected['reporting_affine'],
        affine_report_call, np.full(8, 1.25), 'reporting_affine')
    routes['reporting_affine']['api_scope'] = (
        'current reporting opt-in' if contract == 'current' else
        'released portable affine API; no reporting opt-in or affine native entry')
    routes['ordinary_affine'] = _score_probe(
        np, accelerator, 'score_affine_rows', False,
        lambda: affine.affine_scores(K, alpha, .25, policy='compensated'),
        np.full(8, 1.25), 'ordinary_affine')
    _array_equal(np, K, np.eye(8), 'fixture kernel immutability')
    _array_equal(np, alpha, np.ones(8), 'fixture coefficient immutability')
    _array_equal(np, rhs, np.zeros(8), 'fixture rhs immutability')
    return {'contract': contract, 'backend': backend,
            'expected_runtime': {'platform': expected_platform, 'python': expected_python},
            'actual_runtime': actual_runtime, 'expected_routes': expected, 'routes': routes}
