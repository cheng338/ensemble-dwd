"""Explicit native-resource contract; strict checkpoint authority stays separate."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

from test_preflight_native_contract import RuntimeModel, NeverFit
from test_runtime_artifact_identity import fresh_runtime, tiny_runtime  # noqa: F401
from test_final_thread_preflight import Workflow, reset_counting_estimators  # noqa: F401
from test_phase_threads import observed_counts, observed_runtime
import cv_checkpoint as cp
import phase_threads as phase


@pytest.fixture
def native(tmp_path, monkeypatch):
    return RuntimeModel(monkeypatch, tmp_path)


def preflight(native, requested=4):
    return phase._noncheckpoint_preflight_final_native_threads(requested, NeverFit())


@contextmanager
def fitting(native, pre, receipt, estimator=None):
    with native.limits(limits=1):
        with phase._noncheckpoint_final_fit_threads(
                pre['requested_native_threads'], NeverFit() if estimator is None else estimator,
                receipt, preflight=pre):
            yield


def test_success_is_fresh_native_only_and_keeps_one_baseline(native):
    pre = preflight(native)
    assert native.full == [] and len(native.reads) == 4
    assert native.active == 3 and pre['fit_performed'] is False
    baseline = deepcopy(pre['runtime_probe']['before_scope'])
    receipt = {}
    model = NeverFit()
    with fitting(native, pre, receipt, model):
        assert native.active == 4
    assert model.validations == [True]  # No duplicate get_params call in wrapper.
    assert len(native.reads) == 8 and native.full == [] and native.active == 3
    assert pre['runtime_probe']['before_scope'] == baseline == receipt['before_scope']
    assert receipt['after_scope'] == baseline and receipt['restored'] is True
    assert receipt['state'] == 'completed'
    for value in (pre['runtime_probe'], baseline, receipt):
        assert value['runtime_policy'] == 'noncheckpoint-native-scope-v1'
        assert value['source_installation_attestation'] is False
        assert 'files_sha256' not in value and 'packages' not in value
    for key in ('before_fit', 'after_fit'):
        assert all(row['num_threads'] == 4 for row in receipt[key]['native_pools'])


def test_default_one_never_initializes_native_or_full_authority(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Default1 must not acquire extra authority')
    monkeypatch.setattr(cp, 'runtime_fingerprint', forbidden)
    monkeypatch.setattr(cp, '_native_pool_fingerprint', forbidden)
    monkeypatch.setattr(phase, 'threadpool_limits', forbidden)
    pre = phase._noncheckpoint_preflight_final_native_threads(1, NeverFit())
    receipt = {}
    with phase._noncheckpoint_final_fit_threads(1, NeverFit(), receipt, preflight=pre):
        pass
    assert pre['runtime_probe'] is None
    assert receipt['scope_changed'] is False and receipt['state'] == 'completed'
    assert receipt['observation'] == 'existing_workflow_scope'
    assert cp._RUNTIME_PACKAGE_FILES is None and cp._RUNTIME_FILES is None


@pytest.mark.parametrize('requested', [True, 0, -1, 1.5, '4'])
def test_invalid_budgets_fail_without_native_observation(native, requested):
    with pytest.raises(ValueError):
        phase._noncheckpoint_preflight_final_native_threads(requested, NeverFit())
    assert native.full == native.reads == native.scopes == []


def test_nested_jobs_remain_rejected_at_preflight_and_actual_entry(native):
    with pytest.raises(ValueError, match='n_jobs=1'):
        phase._noncheckpoint_preflight_final_native_threads(4, NeverFit(jobs=2))
    assert native.reads == []
    pre = preflight(native)
    reads = len(native.reads)
    with pytest.raises(ValueError, match='n_jobs=1'):
        with fitting(native, pre, {}, NeverFit(jobs=2)):
            pytest.fail('Invalid nested jobs reached fitting')
    assert len(native.reads) == reads


@pytest.mark.parametrize('problem', ['cap', 'mixed', 'empty', 'openmp-only',
                                      'path', 'metadata', 'added', 'removed', 'artifact'])
def test_native_admission_refusals_restore_before_any_fit(native, problem):
    if problem == 'cap':
        native.cap = 2
    elif problem == 'mixed':
        native.mixed_cap = True
    elif problem in ('empty', 'openmp-only'):
        native.pool_kind = problem
    else:
        native.mutation = problem
    with pytest.raises(cp.CheckpointError):
        preflight(native)
    assert native.active == 3 and native.full == []


def test_duplicate_pool_multiplicity_is_not_normalized(native, monkeypatch):
    original = native.rows
    def duplicate():
        rows = original()
        if native.active > 1:
            rows.append(dict(rows[0]))
        return rows
    monkeypatch.setattr(native, 'rows', duplicate)
    with pytest.raises(cp.CheckpointError, match='changed beyond'):
        preflight(native)
    assert native.active == 3


@pytest.mark.parametrize('change', ['version', 'architecture', 'path', 'add', 'remove', 'duplicate'])
def test_cross_cv_native_drift_refuses_before_actual_fit(native, monkeypatch, change):
    pre = preflight(native)
    original = native.rows
    def changed():
        rows = original()
        if change in ('version', 'architecture'):
            rows[0][change] = 'changed after CV'
        elif change == 'path':
            rows[0]['filepath'] = native.paths[2]
        elif change == 'add':
            rows.append(dict(rows[0], filepath=native.paths[2]))
        elif change == 'remove':
            rows.pop()
        else:
            rows.append(dict(rows[0]))
        return rows
    monkeypatch.setattr(native, 'rows', changed)
    with pytest.raises(cp.CheckpointError, match='changed since preflight'):
        with fitting(native, pre, {}):
            pytest.fail('Changed native authority reached actual fit')
    assert native.active == 3


def test_fit_time_native_drift_refuses_and_restores(native):
    pre = preflight(native)
    receipt = {}
    with pytest.raises(cp.CheckpointError, match='changed beyond'):
        with fitting(native, pre, receipt):
            native.mutation = 'metadata'
    assert receipt['state'] == 'failed' and receipt['restored'] is True
    assert native.active == 3


@pytest.mark.parametrize('error', [ValueError('fit failed'), KeyboardInterrupt('interrupt'), SystemExit(17)])
def test_primary_fit_exception_wins_over_observation_and_restoration_errors(native, monkeypatch, error):
    pre = preflight(native)
    original = native.rows
    changed = False
    def observe():
        if changed and native.active == 1:
            raise LookupError('secondary restoration error')
        return original()
    monkeypatch.setattr(native, 'rows', observe)
    receipt = {}
    with pytest.raises(type(error)) as caught:
        with fitting(native, pre, receipt):
            native.mutation = 'metadata'
            changed = True
            raise error
    assert caught.value is error and native.active == 3
    assert receipt['state'] == 'failed' and receipt['restored'] is False
    assert receipt['fit_exception'] == type(error).__name__
    assert 'observation_error' in receipt and 'secondary restoration error' in receipt['restoration_error']


def test_restoration_error_raises_without_a_primary_fit_error(native, monkeypatch):
    pre = preflight(native)
    original = native.rows
    changed = False
    failure = LookupError('restore failed')
    def observe():
        if changed and native.active == 1:
            raise failure
        return original()
    monkeypatch.setattr(native, 'rows', observe)
    receipt = {}
    with pytest.raises(LookupError) as caught:
        with fitting(native, pre, receipt):
            changed = True
    assert caught.value is failure and receipt['restored'] is False
    assert receipt['state'] == 'failed' and native.active == 3


@pytest.mark.parametrize('stage', ['entry', 'exit'])
def test_scope_errors_preserve_restoration(native, stage):
    native.scope_error = stage
    with pytest.raises(LookupError, match='scope '+stage):
        preflight(native)
    assert native.active == 3 and native.full == []


@pytest.mark.parametrize('wrong', [None, {}, {'state': 'feasible', 'requested_native_threads': 3},
    {'state': 'feasible', 'requested_native_threads': 4, 'runtime_probe': {'before_scope': {'native_pools': []}}}])
def test_native_scope_requires_its_matching_preflight(native, wrong):
    with pytest.raises(ValueError, match='preflight|baseline'):
        with phase._noncheckpoint_final_fit_threads(4, NeverFit(), {}, preflight=wrong):
            pytest.fail('Missing native authority reached fit')
    assert native.reads == []


@pytest.mark.parametrize('frozen', [False, True])
def test_strict_public_wrappers_retain_source_authority_even_when_not_frozen(native, frozen):
    expected = None
    if frozen:
        with native.limits(limits=1):
            expected = cp._stable(native.fingerprint())
    phase.preflight_final_native_threads(4, NeverFit(), frozen_runtime=expected)
    assert native.full
    native.source = 'changed installed source'
    with native.limits(limits=1):
        with pytest.raises(cp.CheckpointError, match='source changed'):
            with phase.final_fit_threads(4, NeverFit(), {}, frozen_runtime=expected):
                pytest.fail('Strict source change reached fit')


@pytest.mark.parametrize('artifact', ['source', 'metadata'])
def test_real_source_changes_are_outside_C_but_still_refused_by_strict(tiny_runtime, artifact):
    cp.runtime_fingerprint()  # Establish the original complete source authority.
    before = phase._native_resource_runtime(1)
    target = getattr(tiny_runtime, artifact)
    target.write_bytes(target.read_bytes()+b'\nowned source policy difference\n')
    assert phase._native_resource_runtime(1) == before
    with pytest.raises(cp.CheckpointError, match='Runtime artifact changed'):
        cp.runtime_fingerprint(check_inventory=False)


def test_real_native_binary_change_remains_a_C_refusal(tiny_runtime):
    phase._native_resource_runtime(1)
    tiny_runtime.binary.write_bytes(tiny_runtime.binary.read_bytes()+b'owned native mutation')
    with pytest.raises(cp.CheckpointError, match='Runtime artifact changed'):
        phase._native_resource_runtime(1)


def test_real_loaded_native_budget_and_restoration_without_full_inventory(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Native scope must not request full source authority')
    monkeypatch.setattr(cp, 'runtime_fingerprint', forbidden)
    requested = int(os.environ.get('DWD_TEST_NATIVE_THREADS', '2'))
    assert requested > 1
    with threadpool_limits(limits=1):
        before = observed_counts()
        pre = phase._noncheckpoint_preflight_final_native_threads(requested, NeverFit())
        receipt = {}
        with phase._noncheckpoint_final_fit_threads(requested, NeverFit(), receipt, preflight=pre):
            assert all(count == requested for _, count in observed_counts())
        assert observed_counts() == before
    assert receipt['restored'] is True
    assert cp._RUNTIME_PACKAGE_FILES is None and cp._RUNTIME_FILES is None


def test_noncheckpoint_workflow_uses_only_native_policy_and_records_it(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('Legacy workflow incorrectly used full installation authority')
    monkeypatch.setattr(cp, 'runtime_fingerprint', forbidden)
    # Explicit controlled fixture: use observed real native counts, no full-observer
    # callback. This synthetic source authority is never presented as real evidence.
    monkeypatch.setattr(cp, '_native_pool_fingerprint',
                        lambda files, *, native_threads: observed_runtime(native_threads=native_threads)['native_pools'])
    result = workflow.run(legacy=True, budget=1)
    assert (workflow.cv_count, workflow.final_count, workflow.held_out_count) == (6, 1, 1)
    assert result['completed']['held_out_accuracy'] == 1.
    for filename in ('run.json', 'refit.json', 'completed.json'):
        artifact = json.loads((workflow.root/'run'/filename).read_text())
        policy = artifact['native_runtime'] if filename == 'refit.json' else artifact
        key = 'runtime_policy' if filename == 'refit.json' else 'final_thread_runtime_policy'
        assert policy[key] == 'noncheckpoint-native-scope-v1'
        assert policy['source_installation_attestation'] is False


def test_adaptive_initial_preflight_stays_strict_without_frozen_runtime(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, monkeypatch)
    error = cp.CheckpointError('strict adaptive authority requested')
    def refuse(*args, **kwargs):
        raise error
    monkeypatch.setattr(cp, 'runtime_fingerprint', refuse)
    with pytest.raises(cp.CheckpointError) as caught:
        workflow.run(legacy=False, budget=1)
    assert caught.value is error
    assert (workflow.cv_count, workflow.final_count, workflow.held_out_count) == (0, 0, 0)


@pytest.mark.parametrize('mutation', ['none', 'list', 'running', 'not-restored',
                                      'different-after', 'wrong-policy', 'source-claim'])
def test_incomplete_or_inconsistent_private_preflight_refuses_before_observation(native, mutation):
    pre = preflight(native)
    changed = deepcopy(pre)
    probe = changed['runtime_probe']
    if mutation == 'none':
        changed['runtime_probe'] = None
    elif mutation == 'list':
        changed['runtime_probe'] = []
    elif mutation == 'running':
        probe['state'] = 'running'
    elif mutation == 'not-restored':
        probe['restored'] = False
    elif mutation == 'different-after':
        probe['after_scope']['native_pools'][0]['version'] = 'different'
    else:
        for key in ('before_scope', 'after_scope'):
            if mutation == 'wrong-policy':
                probe[key]['runtime_policy'] = 'execution-role-v1'
            else:
                probe[key]['source_installation_attestation'] = True
    reads = len(native.reads)
    with pytest.raises(ValueError, match='preflight|baseline'):
        with phase._noncheckpoint_final_fit_threads(4, NeverFit(), {}, preflight=changed):
            pytest.fail('Malformed internal preflight reached actual fit')
    assert len(native.reads) == reads and native.active == 3
