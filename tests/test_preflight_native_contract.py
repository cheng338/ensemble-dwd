"""Native-only preflight observations retain complete runtime boundaries.

The controller/artifact seam leaves the actual native observer intact. Full
package inventory is represented separately, so native-only observations cannot
satisfy source-boundary assertions by reusing a synthetic full fingerprint.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import cv_checkpoint as cp
import phase_threads as phase
import tune_mnist

for _module in (adaptive_search, cp, phase, tune_mnist):
    assert Path(_module.__file__).resolve().parent == EXAMPLES.resolve()


class NeverFit:
    def __init__(self, jobs=1):
        self.jobs = jobs
        self.validations = []

    def get_params(self, deep=True):
        self.validations.append(deep)
        return {'outer__pair__n_jobs': self.jobs}

    def fit(self, *args, **kwargs):
        raise AssertionError('preflight must never fit')


class RuntimeModel:
    """Independent closed pool model; file bytes/source authority are explicit."""
    def __init__(self, monkeypatch, tmp_path):
        self.active = 3
        self.scopes = []
        self.full = []
        self.reads = []
        self.wide_reads = 0
        self.cap = None
        self.mixed_cap = False
        self.pool_kind = 'normal'
        self.mutation = None
        self.source = 'initial-source'
        self.seed = None
        self.on_wide = None
        self.restore_error = None
        self.scope_error = None
        self.paths = [str((tmp_path / name).absolute())
                      for name in ('blas.bin', 'openmp.bin', 'other.bin')]
        self.hashes = {path: str(i + 1) * 64 for i, path in enumerate(self.paths)}
        self.native = cp._native_pool_fingerprint
        monkeypatch.setattr(phase, 'threadpool_limits', self.limits)
        monkeypatch.setattr(cp, 'runtime_fingerprint', self.fingerprint)
        monkeypatch.setattr(cp, 'ThreadpoolController', self.controller)
        monkeypatch.setattr(cp, '_file_hash', self.file_hash)
        monkeypatch.setattr(cp, '_runtime_artifact_identity', self.artifact)
        # Restore observer-owned module state as well as callables on teardown.
        monkeypatch.setattr(cp, '_RUNTIME_CONTROLLER', cp._RUNTIME_CONTROLLER)
        monkeypatch.setattr(cp, '_RUNTIME_MODULE_COUNT', cp._RUNTIME_MODULE_COUNT)

    @contextmanager
    def limits(self, *, limits):
        old = self.active
        self.scopes.append(limits)
        self.active = min(limits, self.cap) if self.cap else limits
        try:
            if limits > 1 and self.scope_error == 'entry':
                raise LookupError('scope entry')
            yield
            if limits > 1 and self.scope_error == 'exit':
                raise LookupError('scope exit')
        finally:
            self.active = old

    def controller(self):
        model = self

        class Controller:
            def info(self):
                return model.rows()
        return Controller()

    def rows(self):
        wide = self.active > 1
        if wide:
            self.wide_reads += 1
            if self.on_wide is not None:
                self.on_wide(self)
        self.reads.append(self.active)
        rows = [dict(filepath=self.paths[0], user_api='blas', internal_api='test-blas',
                     num_threads=self.active, version='1'),
                dict(filepath=self.paths[1], user_api='openmp', internal_api='test-omp',
                     num_threads=min(self.active, 2) if self.mixed_cap else self.active,
                     version='1')]
        if self.pool_kind == 'empty':
            rows = []
        elif self.pool_kind == 'openmp-only':
            rows = rows[1:]
        if wide and self.mutation == 'path':
            rows[0]['filepath'] = self.paths[2]
        elif wide and self.mutation == 'metadata':
            rows[0]['version'] = 'changed'
        elif wide and self.mutation == 'added':
            rows.append(dict(rows[0], filepath=self.paths[2]))
        elif wide and self.mutation == 'removed':
            rows = rows[:1]
        return rows

    def file_hash(self, path):
        return self.hashes[str(Path(path).absolute())]

    def artifact(self, path):
        name = str(Path(path).absolute())
        digest = self.hashes[name]
        if self.active > 1 and self.mutation == 'artifact':
            digest = '9' * 64
        return name, (1, 1, 1, 1, 1), digest

    def observed_pools(self, requested):
        # Execute the production native loop; only its external inputs are fake.
        return self.native({}, native_threads=requested)

    def fingerprint(self, *, check_inventory=True, native_threads=1):
        self.full.append((check_inventory, native_threads))
        if self.seed is None:
            self.seed = self.source
        elif self.seed != self.source:
            raise cp.CheckpointError('controlled source changed after initial inventory')
        if native_threads == 1 and self.wide_reads and self.restore_error is not None:
            raise self.restore_error
        rows = self.observed_pools(native_threads)
        return {'source': self.source,
                'native_pools': sorted(rows, key=lambda row: row['filepath'])}


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    return RuntimeModel(monkeypatch, tmp_path)


@pytest.mark.parametrize('requested', [2, 4])
@pytest.mark.parametrize('frozen', [False, True], ids=['fresh', 'frozen'])
def test_success_keeps_full_boundaries_and_truthful_observation_receipts(runtime, requested, frozen):
    expected = None
    if frozen:
        with runtime.limits(limits=1):
            expected = cp._stable(runtime.fingerprint())
        runtime.full.clear()
        runtime.reads.clear()
        runtime.scopes.clear()
    estimator = NeverFit()
    result = phase.preflight_final_native_threads(requested, estimator, frozen_runtime=expected)
    probe = result['runtime_probe']
    assert result['fit_performed'] is False and result['state'] == 'feasible'
    assert estimator.validations == [True, True]
    assert runtime.scopes == [1, requested] and runtime.active == 3
    assert runtime.full[0] == runtime.full[-1] == (not frozen, 1)
    assert probe['before_scope'] == probe['after_scope']
    assert probe['state'] == 'completed' and probe['restored'] is True
    assert all(row['num_threads'] == requested for row in probe['before_fit']['native_pools'])
    assert runtime.full == [(not frozen, 1), (not frozen, 1)]
    assert runtime.wide_reads == 2
    assert probe['observation'] == 'native_budget_only'
    assert set(probe['before_fit']) == set(probe['after_fit']) == {'native_pools'}
    assert probe['before_fit'] == probe['after_fit']


@pytest.mark.parametrize('value', [1, np.int64(1)])
def test_default_never_enters_scope_or_observes_any_runtime(monkeypatch, value):
    def forbidden(*args, **kwargs):
        raise AssertionError('default must not observe or change limits')
    monkeypatch.setattr(phase, 'threadpool_limits', forbidden)
    monkeypatch.setattr(cp, 'runtime_fingerprint', forbidden)
    monkeypatch.setattr(cp, '_native_pool_fingerprint', forbidden)
    estimator = NeverFit(jobs=8)
    assert phase.preflight_final_native_threads(value, estimator, frozen_runtime=('old',)) == {
        'requested_native_threads': 1, 'fit_performed': False,
        'state': 'existing_cv_scope', 'runtime_probe': None}
    assert estimator.validations == []


@pytest.mark.parametrize('value', [True, False, 0, -1, 2.0, None, '2'])
def test_invalid_budget_fails_before_scope_or_inventory(runtime, value):
    with pytest.raises(ValueError, match='positive integer'):
        phase.preflight_final_native_threads(value, NeverFit())
    assert runtime.scopes == runtime.full == [] and runtime.active == 3


@pytest.mark.parametrize('jobs', [2, -1, True, None])
def test_nondefault_rejects_nonserial_nested_estimators_early(runtime, jobs):
    with pytest.raises(ValueError, match='n_jobs=1'):
        phase.preflight_final_native_threads(4, NeverFit(jobs))
    assert runtime.scopes == runtime.full == [] and runtime.active == 3


def test_second_validation_occurs_inside_one_thread_scope(runtime):
    class Changing(NeverFit):
        def get_params(self, deep=True):
            self.validations.append(runtime.active)
            return {'pair__n_jobs': 1 if len(self.validations) == 1 else 2}
    estimator = Changing()
    with pytest.raises(ValueError, match='n_jobs=1'):
        phase.preflight_final_native_threads(4, estimator)
    assert estimator.validations == [3, 1]
    assert runtime.scopes == [1] and runtime.full == [] and runtime.active == 3


def test_wrong_frozen_runtime_fails_before_requested_scope(runtime):
    with pytest.raises(cp.CheckpointError, match='frozen CV identity'):
        phase.preflight_final_native_threads(4, NeverFit(), frozen_runtime=('wrong',))
    assert runtime.scopes == [1] and runtime.wide_reads == 0 and runtime.active == 3


@pytest.mark.parametrize('condition', ['all-capped', 'mixed-capped', 'empty', 'openmp-only'])
def test_all_loaded_pools_and_blas_presence_are_required(runtime, condition):
    if condition == 'all-capped':
        runtime.cap = 2
    elif condition == 'mixed-capped':
        runtime.mixed_cap = True
    else:
        runtime.pool_kind = condition
    with pytest.raises(cp.CheckpointError):
        phase.preflight_final_native_threads(4, NeverFit())
    assert runtime.active == 3


@pytest.mark.parametrize('mutation', ['path', 'metadata', 'added', 'removed', 'artifact'])
def test_native_identity_cannot_change_with_only_budget_normalized(runtime, mutation):
    runtime.mutation = mutation
    with pytest.raises(cp.CheckpointError):
        phase.preflight_final_native_threads(4, NeverFit())
    assert runtime.active == 3


@pytest.mark.parametrize('kind', [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('boundary', ['requested', 'restored'])
def test_observation_exceptions_restore_exact_caller_and_preserve_identity(runtime, kind, boundary):
    error = kind('original observation')
    if boundary == 'requested':
        def fail(model):
            raise error
        runtime.on_wide = fail
    else:
        runtime.restore_error = error
    with pytest.raises(kind) as caught:
        phase.preflight_final_native_threads(4, NeverFit())
    assert caught.value is error
    assert runtime.active == 3


def test_primary_observation_error_survives_restoration_observer_error(runtime):
    original = KeyboardInterrupt('original')
    runtime.restore_error = ValueError('secondary restoration error')
    def fail(model):
        raise original
    runtime.on_wide = fail
    with pytest.raises(KeyboardInterrupt) as caught:
        phase.preflight_final_native_threads(4, NeverFit())
    assert caught.value is original and runtime.active == 3
    assert runtime.full[-1] == (True, 1)


@pytest.mark.parametrize('kind', [ValueError, KeyboardInterrupt, SystemExit])
def test_second_native_observation_error_is_preserved(runtime, kind):
    error = kind('second requested observation')
    def fail_second(model):
        if model.wide_reads == 2:
            raise error
    runtime.on_wide = fail_second
    with pytest.raises(kind) as caught:
        phase.preflight_final_native_threads(4, NeverFit())
    assert caught.value is error and runtime.wide_reads == 2
    assert runtime.active == 3


@pytest.mark.parametrize('boundary', ['entry', 'exit'])
def test_scope_failure_restores_caller_without_fitting(runtime, boundary):
    runtime.scope_error = boundary
    with pytest.raises(LookupError, match='scope ' + boundary):
        phase.preflight_final_native_threads(4, NeverFit())
    assert runtime.active == 3


def test_persistent_source_change_during_probe_is_rejected_at_full_boundary(runtime):
    def mutate(model):
        model.source = 'changed-during-probe'
    runtime.on_wide = mutate
    with pytest.raises(cp.CheckpointError):
        phase.preflight_final_native_threads(4, NeverFit())
    assert runtime.active == 3 and runtime.full[-1] == (True, 1)


def test_preflight_seeds_source_authority_used_before_actual_final_fit(runtime):
    assert runtime.seed is None
    phase.preflight_final_native_threads(4, NeverFit())
    assert runtime.seed == 'initial-source'
    runtime.source = 'changed-during-cv'
    called = False
    with runtime.limits(limits=1):
        with pytest.raises(cp.CheckpointError, match='source changed'):
            with phase.final_fit_threads(4, NeverFit(), {}):
                called = True
    assert not called and runtime.active == 3


@pytest.mark.parametrize('legacy', [False, True], ids=['adaptive', 'legacy'])
def test_cap_failure_precedes_workflow_fingerprints_cv_and_persisted_artifacts(
        runtime, tmp_path, monkeypatch, legacy):
    from test_final_thread_preflight import Workflow
    from test_phase_threads import PhaseSurface, PhaseTransform
    monkeypatch.setattr(PhaseSurface, 'fits', [])
    monkeypatch.setattr(PhaseSurface, 'predictions', [])
    monkeypatch.setattr(PhaseTransform, 'fits', [])
    workflow = Workflow(tmp_path, monkeypatch)
    # Workflow installs explicit synthetic full/native fixtures. Restore the
    # production native helper for this independently controlled seam.
    monkeypatch.setattr(phase, 'threadpool_limits', runtime.limits)
    monkeypatch.setattr(cp, 'runtime_fingerprint', runtime.fingerprint)
    monkeypatch.setattr(cp, '_native_pool_fingerprint', runtime.native)
    runtime.cap = 2
    def forbidden(*args, **kwargs):
        raise AssertionError('workflow data work/publication occurred before failed probe')
    owner = tune_mnist if legacy else adaptive_search
    monkeypatch.setattr(owner, 'data_fingerprint', forbidden)
    monkeypatch.setattr(owner, 'write_json_atomic', forbidden)
    array = np.array
    def no_heldout_copy(value, *args, **kwargs):
        if value is workflow.test_X or value is workflow.test_y:
            forbidden()
        return array(value, *args, **kwargs)
    monkeypatch.setattr(np, 'array', no_heldout_copy)
    with pytest.raises(cp.CheckpointError):
        workflow.run(legacy=legacy, budget=4)
    assert workflow.cv_count == workflow.final_count == workflow.held_out_count == 0
    assert workflow.hashes() == {} and runtime.active == 3


@pytest.mark.parametrize('budget_kind', ['one', 'two', 'host'])
def test_real_scope_only_budget_and_restore(budget_kind, record_property):
    from test_phase_threads import observed_counts
    from ensemble_dwd._cache import _initialize_numerical_runtime
    from threadpoolctl import threadpool_limits
    _initialize_numerical_runtime()
    budget = {'one': 1, 'two': 2, 'host': int(os.environ.get('DWD_TEST_NATIVE_THREADS', '2'))}[budget_kind]
    assert budget >= 1
    with threadpool_limits(limits=1):
        before = observed_counts()
        result = phase.preflight_final_native_threads(budget, NeverFit())
        assert observed_counts() == before
    assert result['fit_performed'] is False
    if budget > 1:
        assert result['runtime_probe']['restored'] is True
        assert all(p['num_threads'] == budget
                   for p in result['runtime_probe']['before_fit']['native_pools'])
    record_property('preflight_scope_receipt', json.dumps(result, sort_keys=True))


@pytest.mark.parametrize('order', ['fake-before-reference', 'fake-after-reference'])
def test_synthetic_native_fixture_resolves_current_full_observer(monkeypatch, order):
    import test_phase_threads as phase_tests
    calls = []
    rows = [{'filepath': 'synthetic', 'user_api': 'blas', 'num_threads': 4}]

    def fake_full(*, check_inventory=True, native_threads=1):
        calls.append((check_inventory, native_threads))
        return {'source': 'explicitly synthetic', 'native_pools': rows}

    if order == 'fake-before-reference':
        monkeypatch.setattr(cp, 'runtime_fingerprint', fake_full)
    observer = phase_tests.synthetic_native_pools
    if order == 'fake-after-reference':
        monkeypatch.setattr(cp, 'runtime_fingerprint', fake_full)
    files = {}
    assert observer(files, native_threads=4) is rows
    assert calls == [(False, 4)] and files == {}


def test_synthetic_full_fixture_preserves_fields_duplicates_and_input_state(monkeypatch):
    from copy import deepcopy
    import test_phase_threads as phase_tests
    rows = [{'filepath': 'z', 'user_api': 'blas', 'num_threads': 4, 'version': 'z'},
            {'filepath': 'a', 'user_api': 'openmp', 'num_threads': 4, 'version': 'first-a'},
            {'filepath': 'a', 'user_api': 'openmp', 'num_threads': 4, 'version': 'second-a'}]
    before = deepcopy(rows)
    expected = phase_tests.synthetic_runtime(source='phase-test-runtime', pools=rows)
    monkeypatch.setattr(phase_tests, 'threadpool_info', lambda: rows)
    actual = phase_tests.observed_runtime(native_threads=4)
    assert actual == {**expected, 'native_pools': [expected['native_pools'][1],
                                                  expected['native_pools'][2],
                                                  expected['native_pools'][0]]}
    assert len(actual['native_pools']) == 3 and rows == before
    assert actual['native_pools'] is not rows
    assert all(row is not original for row in actual['native_pools'] for original in rows)


def test_synthetic_full_fixture_keeps_wrong_budget_refusal(monkeypatch):
    import test_phase_threads as phase_tests
    rows = [{'filepath': 'z', 'user_api': 'blas', 'num_threads': 4},
            {'filepath': 'a', 'user_api': 'openmp', 'num_threads': 4}]
    monkeypatch.setattr(phase_tests, 'threadpool_info', lambda: rows)
    with pytest.raises(cp.CheckpointError, match='wrong active budget'):
        phase_tests.observed_runtime(native_threads=2)
