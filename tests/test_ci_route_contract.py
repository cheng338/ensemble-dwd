"""CI route contracts and command wiring, independent of the installed backend.

These unit controls do not qualify native execution. The live ``check`` command
performs that mandatory preflight on the actual wheel and interpreter in each
CI job. This module imports only the standard library, pytest and test tooling.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
checks = load_file('_route_contract_artifacts', HERE / 'ci_artifacts.py')
routes = load_file('_route_contract_expectations', HERE / 'ci_native_routes.py')


# Explicit requirements, not answers obtained from production eligibility guards.
# backend, platform, Python, raw C, guarded n8, guarded n4, ordinary score, report K*w
RUNTIME_CASES = [
    ('native', 'windows', '3.11', True, 'compiled_values', 'declined', True, True),
    ('native', 'windows', '3.12', True, 'compiled_values', 'scalar_fallback', True, True),
    ('native', 'linux', '3.11', True, 'declined', 'declined', False, True),
    ('native', 'linux', '3.12', True, 'compiled_values', 'scalar_fallback', False, True),
    ('portable', 'windows', '3.11', False, 'declined', 'declined', False, False),
    ('portable', 'windows', '3.12', False, 'scalar_fallback', 'scalar_fallback', False, False),
    ('portable', 'linux', '3.11', False, 'declined', 'declined', False, False),
    ('portable', 'linux', '3.12', False, 'scalar_fallback', 'scalar_fallback', False, False),
]


@pytest.mark.parametrize('contract', ['current', 'released-1.3.19'])
@pytest.mark.parametrize('case', RUNTIME_CASES, ids=['-'.join(row[:3]) for row in RUNTIME_CASES])
def test_declared_routes_do_not_depend_on_runtime_guards(contract, case):
    backend, system, version, raw, large, small, ordinary, reporting = case
    assert routes.expected_routes(backend, contract, system, version) == {
        'raw_compiled_residual': raw,
        'guarded_residual_n8': large,
        'guarded_residual_n4': small,
        'ordinary_scores': ordinary,
        'reporting_scores': reporting,
        'reporting_affine': backend == 'native' and contract == 'current',
        'ordinary_affine': False,
    }


@pytest.mark.parametrize('index,value,message', [
    (0, 'wheel', 'unknown wheel backend'),
    (1, 'future', 'unknown route contract'),
    (2, 'darwin', 'unknown expected platform'),
    (3, '3.13', 'unknown expected Python'),
])
def test_unknown_contract_arguments_fail_explicitly(index, value, message):
    args = ['native', 'current', 'windows', '3.12']
    args[index] = value
    with pytest.raises(RuntimeError, match='^Native route preflight: ' + message + '$'):
        routes.expected_routes(*args)


def test_missing_required_entry_is_reported_without_assuming_a_fallback():
    witness = {'observed_calls': 0, 'completed_calls': 0, 'statuses': []}
    with pytest.raises(RuntimeError, match=(
            '^Native route preflight: fixture required native entry did not execute$')):
        routes._check_witness(witness, True, 'fixture')
    assert witness == {'observed_calls': 0, 'completed_calls': 0, 'statuses': []}


@pytest.mark.parametrize('missing', ['--route-contract', '--expected-platform', '--expected-python'])
def test_check_requires_each_explicit_route_argument(missing, monkeypatch, capsys):
    argv = ['ci_artifacts.py', 'check', '--dwd-wheel-dir', 'unused', '--backend', 'wheel',
            '--route-contract', 'current', '--expected-platform', 'windows',
            '--expected-python', '3.12']
    start = argv.index(missing)
    del argv[start:start + 2]
    monkeypatch.setattr(sys, 'argv', argv)
    def unexpected(*args, **kwargs):
        raise AssertionError('Artifact work began before required-argument validation')
    monkeypatch.setattr(checks, 'single_wheel', unexpected)
    with pytest.raises(SystemExit) as raised:
        checks.main()
    assert raised.value.code == 2
    assert missing in capsys.readouterr().err


def arrange_cli(monkeypatch, backend, contract, system, version, *, companion=False):
    """CLI plumbing fixture; no wheel verification or numerical route is claimed."""
    events = []
    wheel = Path('selected-dwd.whl')
    ensemble_wheel = Path('selected-ensemble.whl')
    artifact = {'declares_accelerator': backend == 'native', 'origin': 'mock/dwd/__init__.py'}
    argv = ['ci_artifacts.py', 'check', '--dwd-wheel-dir', 'dwd-wheels', '--backend', 'wheel',
            '--route-contract', contract, '--expected-platform', system,
            '--expected-python', version]
    if companion:
        argv += ['--ensemble-wheel-dir', 'ensemble-wheels']
    monkeypatch.setattr(sys, 'argv', argv)
    def select(directory):
        events.append(('wheel', directory))
        assert directory in ('dwd-wheels', 'ensemble-wheels')
        return wheel if directory == 'dwd-wheels' else ensemble_wheel
    def verify(selected, module):
        events.append(('artifact', module))
        assert (selected, module) in ((wheel, 'dwd'), (ensemble_wheel, 'ensemble_dwd'))
        return artifact if module == 'dwd' else {'distribution': 'ensemble-dwd'}
    def verify_backend(record, requested):
        events.append(('backend', requested))
        assert record is artifact and requested == 'wheel'
        return {'backend': backend, 'compiled_helper_executed': backend == 'native'}
    monkeypatch.setattr(checks, 'single_wheel', select)
    monkeypatch.setattr(checks, 'verify_artifact', verify)
    monkeypatch.setattr(checks, 'verify_backend', verify_backend)
    return events


@pytest.mark.parametrize('backend,contract,system,version,companion', [
    ('native', 'current', 'windows', '3.11', False),
    ('portable', 'current', 'linux', '3.12', False),
    ('native', 'released-1.3.19', 'linux', '3.11', True),
    ('portable', 'released-1.3.19', 'windows', '3.12', True),
])
def test_check_passes_verified_backend_and_all_declared_fields(
        backend, contract, system, version, companion, monkeypatch, capsys):
    events = arrange_cli(monkeypatch, backend, contract, system, version, companion=companion)
    calls = []
    receipt = {'fixture': 'route checker returned evidence'}
    def checker(*args):
        events.append(('route', args))
        calls.append(args)
        return receipt
    monkeypatch.setattr(checks, '_route_checker', lambda: checker)
    checks.main()
    assert calls == [(backend, contract, system, version)]
    expected = [('wheel', 'dwd-wheels'), ('artifact', 'dwd'), ('backend', 'wheel'),
                ('route', (backend, contract, system, version))]
    if companion:
        expected += [('wheel', 'ensemble-wheels'), ('artifact', 'ensemble_dwd')]
    assert events == expected
    result = json.loads(capsys.readouterr().out)
    assert result['backend'] == backend and result['native_routes'] == receipt
    assert ('ensemble' in result) is companion


def test_check_cannot_report_success_after_route_checker_failure(monkeypatch, capsys):
    events = arrange_cli(monkeypatch, 'native', 'current', 'windows', '3.12', companion=True)
    error = RuntimeError('Native route preflight: deliberate controlled witness failure')
    def checker(*args):
        assert args == ('native', 'current', 'windows', '3.12')
        events.append(('route', args))
        raise error
    monkeypatch.setattr(checks, '_route_checker', lambda: checker)
    with pytest.raises(RuntimeError) as raised:
        checks.main()
    assert raised.value is error
    assert events == [('wheel', 'dwd-wheels'), ('artifact', 'dwd'), ('backend', 'wheel'),
                      ('route', ('native', 'current', 'windows', '3.12'))]
    assert capsys.readouterr().out == ''


def test_sibling_loader_needs_no_script_directory_on_import_path(tmp_path, monkeypatch):
    """Model -I's path exclusion; actual isolated launches are the live CI command."""
    isolated = tmp_path / 'isolated-helpers'
    isolated.mkdir()
    for filename in ('ci_artifacts.py', 'ci_native_routes.py'):
        shutil.copyfile(HERE / filename, isolated / filename)
    excluded = {HERE, isolated.resolve()}
    monkeypatch.setattr(sys, 'path', [entry for entry in sys.path
                                    if entry and Path(entry).resolve() not in excluded])
    def wrong_module(*args, **kwargs):
        raise AssertionError('Imported unrelated module by name instead of exact sibling')
    monkeypatch.setitem(sys.modules, 'ci_native_routes', SimpleNamespace(verify_routes=wrong_module))
    before = {name: module for name, module in sys.modules.items()
              if name == 'dwd' or name.startswith('dwd.')}
    isolated_artifacts = load_file('_isolated_route_contract_artifacts', isolated / 'ci_artifacts.py')
    checker = isolated_artifacts._route_checker()
    assert callable(checker) and checker is not wrong_module
    assert Path(checker.__globals__['__file__']).resolve() == (isolated / 'ci_native_routes.py').resolve()
    assert all(Path(entry).resolve() not in excluded for entry in sys.path)
    after = {name: module for name, module in sys.modules.items()
             if name == 'dwd' or name.startswith('dwd.')}
    assert after.keys() == before.keys() and all(after[name] is value for name, value in before.items())
