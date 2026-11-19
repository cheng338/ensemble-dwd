"""Final native-thread feasibility, phase ordering, restoration and resume.

Synthetic caps model a controlled library response. Real scope checks default
to two threads; DWD_TEST_NATIVE_THREADS may request another positive integer.
The requested budget must be feasible and restored exactly, without fitting.
"""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from path_support import extended_path
from time import perf_counter

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits as real_limits

# This maintained sibling supplies only small counting estimators and observers.
# Import no unittest TestCase classes, so their tests are not collected here.
from test_phase_threads import (
    PhaseSurface, PhaseTransform, observed_counts, observed_runtime, synthetic_native_pools,
    adaptive_search, cv_checkpoint, phase_threads, tune_mnist,
    new_run_directory,
)


@pytest.fixture(autouse=True)
def reset_counting_estimators():
    old = (PhaseSurface.fits, PhaseSurface.predictions, PhaseSurface.fail_final,
           PhaseTransform.fits)
    PhaseSurface.fits, PhaseSurface.predictions = [], []
    PhaseSurface.fail_final, PhaseTransform.fits = None, []
    try:
        yield
    finally:
        (PhaseSurface.fits, PhaseSurface.predictions, PhaseSurface.fail_final,
         PhaseTransform.fits) = old


class SyntheticRuntime:
    """Closed two-pool response model; does not replace model arithmetic."""
    def __init__(self):
        self.active = 3
        self.cap = None
        self.mixed_cap = False
        self.empty = False
        self.calls = []
        self.scopes = []
        self.wide_calls = 0
        self.fail_at = None
        self.failure = None
        self.drift = False

    @contextmanager
    def limits(self, *, limits):
        old = self.active
        self.scopes.append(limits)
        self.active = min(limits, self.cap) if self.cap else limits
        try:
            yield
        finally:
            self.active = old

    def fingerprint(self, *, native_threads=1, **kwargs):
        self.calls.append((self.active, native_threads))
        if native_threads > 1:
            self.wide_calls += 1
            if self.wide_calls == self.fail_at:
                raise self.failure
        pools = [] if self.empty else [
            dict(filepath='synthetic-blas', user_api='blas', num_threads=self.active),
            dict(filepath='synthetic-openmp', user_api='openmp',
                 num_threads=min(self.active, 2) if self.mixed_cap else self.active),
        ]
        # Model the existing observer's exact per-pool contract, including BLAS.
        if not pools or any(p['num_threads'] != native_threads for p in pools):
            raise cv_checkpoint.CheckpointError('Controlled effective native budget mismatch')
        return dict(source='changed' if self.drift and self.wide_calls else 'stable',
                    native_pools=pools)


@pytest.fixture
def synthetic(monkeypatch):
    runtime = SyntheticRuntime()
    monkeypatch.setattr(phase_threads, 'threadpool_limits', runtime.limits)
    monkeypatch.setattr(cv_checkpoint, 'runtime_fingerprint', runtime.fingerprint)
    monkeypatch.setattr(cv_checkpoint, '_native_pool_fingerprint', synthetic_native_pools)
    return runtime


def assert_no_fit():
    assert PhaseSurface.fits == PhaseTransform.fits == PhaseSurface.predictions == []


def test_preflight_api_is_available():
    assert callable(phase_threads.preflight_final_native_threads)


@pytest.mark.parametrize('requested', [1, 2, 4])
def test_component_feasible_without_fitting(synthetic, requested):
    result = phase_threads.preflight_final_native_threads(requested, PhaseSurface())
    assert result['requested_native_threads'] == requested
    assert result['fit_performed'] is False
    assert synthetic.active == 3
    if requested == 1:
        assert result['state'] == 'existing_cv_scope'
        assert result['runtime_probe'] is None
        assert synthetic.calls == synthetic.scopes == []
    else:
        assert result['state'] == 'feasible'
        raw = result['runtime_probe']
        assert raw['state'] == 'completed' and raw['restored'] is True
        assert raw['before_scope'] == raw['after_scope']
        assert all(p['num_threads'] == requested for p in raw['before_fit']['native_pools'])
    assert_no_fit()


@pytest.mark.parametrize('mixed', [False, True], ids=['all-capped', 'one-pool-capped'])
def test_component_cap_refuses_without_silent_clamp(synthetic, mixed):
    synthetic.cap = None if mixed else 2
    synthetic.mixed_cap = mixed
    with pytest.raises(cv_checkpoint.CheckpointError, match='effective native budget'):
        phase_threads.preflight_final_native_threads(4, PhaseSurface())
    assert synthetic.active == 3
    assert_no_fit()


def test_empty_runtime_inventory_refuses(synthetic):
    synthetic.empty = True
    with pytest.raises(cv_checkpoint.CheckpointError):
        phase_threads.preflight_final_native_threads(4, PhaseSurface())
    assert synthetic.active == 3
    assert_no_fit()


@pytest.mark.parametrize('error_type', [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('boundary', [1, 2], ids=['entry-observation', 'exit-observation'])
def test_preflight_observation_errors_restore_exact_caller(
        synthetic, error_type, boundary):
    original = error_type('controlled observation failure')
    synthetic.fail_at, synthetic.failure = boundary, original
    with pytest.raises(error_type) as caught:
        phase_threads.preflight_final_native_threads(4, PhaseSurface())
    assert caught.value is original
    assert synthetic.active == 3
    assert_no_fit()


def test_nonthread_runtime_drift_is_rejected_at_full_restored_boundary(synthetic):
    synthetic.drift = True
    # Native-only wide observations do not claim source authority. Persistent
    # source drift must still fail the complete restored boundary.
    with pytest.raises(cv_checkpoint.CheckpointError, match='did not restore the CV runtime'):
        phase_threads.preflight_final_native_threads(4, PhaseSurface())
    assert synthetic.active == 3
    assert_no_fit()


def test_default_does_not_observe_or_change_scope(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('default preflight must add no inventory or scope')
    monkeypatch.setattr(phase_threads, 'threadpool_limits', forbidden)
    monkeypatch.setattr(cv_checkpoint, 'runtime_fingerprint', forbidden)
    result = phase_threads.preflight_final_native_threads(1, PhaseSurface(), frozen_runtime=('existing',))
    assert result == dict(state='existing_cv_scope', requested_native_threads=1,
                          fit_performed=False, runtime_probe=None)
    assert_no_fit()


class Workflow:
    def __init__(self, tmp_path, monkeypatch):
        self.root = Path(extended_path(tmp_path))
        self.cap = 4
        self.cap_after_cv = None
        self.scopes = []
        rows = np.arange(20)
        self.X = np.column_stack((rows, rows % 2))
        self.y = rows % 2
        test_rows = np.arange(100, 108)
        self.test_X = np.column_stack((test_rows, test_rows % 2))
        self.test_y = test_rows % 2
        self.folds = [(np.setdiff1d(rows, fold), fold)
                      for fold in np.array_split(rows, 3)]
        self.pipeline = Pipeline([('transform', PhaseTransform()),
                                  ('ensemble', PhaseSurface())])
        self.grid = {'ensemble__lambd': [1., 4.], 'ensemble__gamma': [4.]}
        self.config = adaptive_search.AdaptiveSearchConfig(refine=False, max_candidates=2)
        original_stage = adaptive_search.next_search_stage

        def accept_small_fixture(results, grid, config, state):
            # Same finite policy-control seam as test_selection_resume_process.
            # This test concerns phase ordering, not adaptive search heuristics.
            if len(results['params']) == 2:
                return dict(state=state, review=dict(state='accepted', reasons=[],
                            stopping_reason='controlled finite test acceptance'))
            return original_stage(results, grid, config, state)

        monkeypatch.setattr(adaptive_search, 'next_search_stage', accept_small_fixture)
        monkeypatch.setattr(cv_checkpoint, 'runtime_fingerprint', observed_runtime)
        monkeypatch.setattr(cv_checkpoint, '_native_pool_fingerprint', synthetic_native_pools)
        monkeypatch.setattr(phase_threads, 'threadpool_limits', self.limits)

    @contextmanager
    def limits(self, *, limits):
        cap = self.cap_after_cv if self.cap_after_cv is not None and self.cv_count >= 6 else self.cap
        self.scopes.append(dict(requested=limits, effective=min(limits, cap)))
        with real_limits(limits=min(limits, cap)):
            yield

    @property
    def cv_count(self):
        return sum(n != 20 for n, pools in PhaseSurface.fits)

    @property
    def final_count(self):
        return sum(n == 20 for n, pools in PhaseSurface.fits)

    @property
    def held_out_count(self):
        return sum(n == 8 for n, pools in PhaseSurface.predictions)

    def run(self, *, legacy=False, resume=False, budget=4, name='run'):
        folder = new_run_directory(self.root / name, resume=resume)
        with redirect_stdout(io.StringIO()):
            args = (self.pipeline, self.grid, self.X, self.y, self.test_X,
                    self.test_y, self.folds, folder)
            if legacy:
                return tune_mnist.run_search_workflow(
                    *args, refine=False, verbose=0, final_native_threads=budget)
            return adaptive_search.run_adaptive_search(
                *args, config=self.config, resume=resume,
                reuse_equivalent_folds=False, final_native_threads=budget)

    def hashes(self, *, cv_only=False):
        root = self.root / 'run'
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.pkl')
                and (not cv_only or 'folds' in p.relative_to(root).parts
                     or p.name == 'identity.json')}


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    return Workflow(tmp_path, monkeypatch)


@pytest.mark.parametrize('legacy', [False, True], ids=['adaptive', 'legacy'])
def test_cap_discovery_occurs_at_intended_phase(workflow, legacy, record_property):
    workflow.cap = 2
    with real_limits(limits=3):
        before = observed_counts()
        with pytest.raises(cv_checkpoint.CheckpointError, match='wrong active budget'):
            workflow.run(legacy=legacy)
        assert observed_counts() == before
    assert workflow.final_count == workflow.held_out_count == 0
    assert workflow.cv_count == 0
    files = workflow.hashes()
    assert not files, 'Fresh infeasibility must precede run/status/checkpoint persistence.'
    assert PhaseSurface.predictions == PhaseTransform.fits == []
    record_property('ir16_cap_evidence', json.dumps(dict(
        synthetic_requested=4, synthetic_effective=2,
        legacy=legacy, cv_fits=workflow.cv_count, final_fits=workflow.final_count,
        held_out_predictions=workflow.held_out_count, files=files,
        caller_restored=True), sort_keys=True))


@pytest.mark.parametrize('legacy', [False, True], ids=['adaptive', 'legacy'])
def test_feasible_workflow_keeps_phase_budgets_and_original_data(workflow, legacy):
    snapshots = [a.copy() for a in (workflow.X, workflow.y, workflow.test_X, workflow.test_y)]
    with real_limits(limits=3):
        before = observed_counts()
        workflow.run(legacy=legacy, budget=2)
        assert observed_counts() == before
    assert workflow.cv_count == 6 and workflow.final_count == workflow.held_out_count == 1
    for events in (PhaseSurface.fits, PhaseTransform.fits):
        for n, pools in events:
            assert all(count == (2 if n == 20 else 1) for path, count in pools)
    assert all(count == 1 for n, pools in PhaseSurface.predictions for path, count in pools)
    for original, current in zip(snapshots, (workflow.X, workflow.y, workflow.test_X, workflow.test_y)):
        np.testing.assert_array_equal(current, original)


def test_late_cap_drift_refuses_then_same_budget_resumes_without_cv_replay(workflow):
    workflow.cap_after_cv = 2
    with pytest.raises(cv_checkpoint.CheckpointError, match='wrong active budget'):
        workflow.run()
    assert workflow.cv_count == 6 and workflow.final_count == workflow.held_out_count == 0
    retained = workflow.hashes(cv_only=True)
    assert retained and any('folds/' in key for key in retained)
    workflow.cap_after_cv = None
    result = workflow.run(resume=True)
    assert workflow.cv_count == 6 and workflow.final_count == workflow.held_out_count == 1
    assert workflow.hashes(cv_only=True) == retained
    assert result['completed']['held_out_accuracy'] == 1.


@pytest.mark.parametrize('completed', [False, True], ids=['unfinished', 'completed'])
def test_corrected_budget_still_refuses_identity_without_mutation(workflow, completed):
    if not completed:
        PhaseSurface.fail_final = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            workflow.run()
        PhaseSurface.fail_final = None
    else:
        workflow.run()
    before = workflow.hashes()
    calls = deepcopy((PhaseSurface.fits, PhaseSurface.predictions, workflow.scopes))
    with pytest.raises(cv_checkpoint.CheckpointError):
        workflow.run(resume=True, budget=2)
    assert workflow.hashes() == before
    assert (PhaseSurface.fits, PhaseSurface.predictions, workflow.scopes) == calls


def test_completed_replay_does_not_probe_now_unavailable_budget(workflow):
    first = workflow.run()
    calls = deepcopy((PhaseSurface.fits, PhaseSurface.predictions, workflow.scopes))
    retained = workflow.hashes(cv_only=True)
    workflow.cap = 2
    resumed = workflow.run(resume=True)
    assert resumed['resumed_completed'] is True
    assert resumed['refit'] == first['refit']
    assert (PhaseSurface.fits, PhaseSurface.predictions, workflow.scopes) == calls
    assert workflow.hashes(cv_only=True) == retained


def test_unfinished_resume_preflight_preserves_saved_folds(workflow):
    PhaseSurface.fail_final = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        workflow.run()
    assert workflow.cv_count == 6 and workflow.final_count == 1
    PhaseSurface.fail_final = None
    retained = workflow.hashes(cv_only=True)
    predictions = deepcopy(PhaseSurface.predictions)
    workflow.cap = 2
    with pytest.raises(cv_checkpoint.CheckpointError, match='wrong active budget'):
        workflow.run(resume=True)
    assert workflow.cv_count == 6 and workflow.final_count == 1
    assert PhaseSurface.predictions == predictions
    assert workflow.hashes(cv_only=True) == retained


@pytest.mark.parametrize('budget_kind', ['two', 'host', 'default'])
def test_real_loaded_pool_feasibility_and_restoration(budget_kind, monkeypatch, record_property):
    if budget_kind == 'default':
        monkeypatch.delenv('DWD_TEST_NATIVE_THREADS', raising=False)
    configured = os.environ.get('DWD_TEST_NATIVE_THREADS', '2')
    try:
        configured_budget = int(configured)
    except ValueError:
        pytest.fail('DWD_TEST_NATIVE_THREADS must be a positive integer.')
    assert configured_budget > 0, 'DWD_TEST_NATIVE_THREADS must be a positive integer.'
    budget = 2 if budget_kind == 'two' else configured_budget
    if budget_kind == 'default':
        assert configured == '2' and budget == 2
    # A scope-only real feasibility observation; no model or BLAS work benchmark.
    with real_limits(limits=1):
        before = observed_counts()
        inventory_started = perf_counter()
        frozen = cv_checkpoint._stable(cv_checkpoint.runtime_fingerprint())
        inventory_seconds = perf_counter() - inventory_started
        model = PhaseSurface()
        started = perf_counter()
        result = phase_threads.preflight_final_native_threads(budget, model, frozen_runtime=frozen)
        elapsed_seconds = perf_counter() - started
        # The requested count must be supported; observation or restoration
        # failures remain visible test failures.
        assert result['fit_performed'] is False
        if budget == 1:
            assert result == dict(state='existing_cv_scope', requested_native_threads=1,
                                  fit_performed=False, runtime_probe=None)
        else:
            raw = result['runtime_probe']
            assert result['state'] == 'feasible'
            assert all(p['num_threads'] == budget for p in raw['before_fit']['native_pools'])
            assert raw['restored'] is True
        evidence = dict(status=result['state'], requested=budget, receipt=result,
                        pre_probe_inventory_seconds=inventory_seconds,
                        probe_elapsed_seconds=elapsed_seconds,
                        timing_scope='one descriptive scope-only observation; no benchmark')
        assert observed_counts() == before
    assert_no_fit()
    evidence.update(budget_kind=budget_kind, configured_budget=configured_budget,
                    workload='scope-only; no model fit', restored=True)
    record_property('ir16_real_pool_evidence', json.dumps(evidence, sort_keys=True))
