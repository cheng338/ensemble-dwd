"""Regression tests for caller failures partway through parallel base submission."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from threading import Barrier, Event, Timer, current_thread

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from ensemble_dwd import EnsembleKernelDWD
from ensemble_dwd import _parallel_base as parallel
from ensemble_dwd import estimator
from ensemble_dwd import profiling
from ensemble_dwd._failure_context import failure_context
from profiling_test_support import synthetic_fit_profile


@contextmanager
def rescue_after_failure(release):
    """Bound an old implementation's blocked join, and report any intervention."""
    rescued = Event()

    def rescue():
        rescued.set()
        release()

    timer = Timer(3, rescue)
    try:
        yield timer.start, rescued
    finally:
        timer.cancel()
        if timer.ident is not None:
            timer.join()


@pytest.fixture
def failed_refit(monkeypatch):
    X = np.random.RandomState(17).normal(size=(96, 4))
    y = np.tile([0, 1], 48)
    model = EnsembleKernelDWD(
        M=6, theta=.125, sv_selection_ratio=1., lambd=.01, gamma=.05,
        stopping="fixed", max_iter=8, random_state=51, base_jobs=1,
    )
    with threadpool_limits(3):
        model.fit(X, y)
        assert model.final_estimator_ is not None
        model.set_params(base_jobs=2)
        runtime = threadpool_info()

        def run(injected, submit, *, collect=True):
            accepted = []
            workers = []

            class FaultPool(ThreadPoolExecutor):
                def __init__(self, *args, initializer, **kwargs):
                    def observed_initializer(*initargs):
                        workers.append(current_thread())
                        initializer(*initargs)

                    super().__init__(*args, initializer=observed_initializer, **kwargs)

                def submit(self, fn, *args, **kwargs):
                    def enqueue():
                        future = super(FaultPool, self).submit(fn, *args, **kwargs)
                        accepted.append(future)
                        return future

                    return submit(len(accepted) + 1, enqueue, tuple(accepted))

            monkeypatch.setattr(parallel, "ThreadPoolExecutor", FaultPool)
            clock = profiling.perf_counter
            with pytest.raises(type(injected)) as caught:
                with synthetic_fit_profile() if collect else nullcontext() as profile:
                    model.fit(X, y)
            assert profiling.perf_counter is clock
            assert caught.value is injected
            assert not [name for name in vars(model) if name.endswith("_")]
            assert accepted and all(future.done() for future in accepted)
            assert workers and all(not thread.is_alive() for thread in workers)
            assert threadpool_info() == runtime
            return failure_context(injected), profile.as_dict() if collect else None

        yield run


def assert_work(result, scored_by_position):
    """Every listed worker returned from fit. A zero means scoring raised."""
    context, profile = result
    fits = len(scored_by_position)
    scores = sum(scored_by_position)
    if profile is not None:
        assert profile["state"] == "aborted"
        assert profile["stages"]["parallel_base_work"]["aborted_calls"] == 1
        assert profile["counts"]["final_fit_completed"] == 0
        assert profile["counts"]["base_fit_completed"] == fits
        assert profile["counts"]["base_scoring_completed"] == scores
        workers = profile["parallel_worker_profiles"]
        assert [w["counts"]["base_fit_completed"] for w in workers] == [1] * fits
        assert [w["counts"]["base_scoring_completed"] for w in workers] == scored_by_position
    assert context["stage"] == "base_work"
    assert context["base_position"] is None
    assert context["base_work_source"] == "current"
    assert context["cached_base_learners"] == context["final_fit_calls_completed"] == 0
    assert context["base_fit_calls_completed"] == fits
    assert context["base_score_calls_completed"] == scores


def test_submit_fails_with_first_worker_waiting_for_missing_initializer(monkeypatch, failed_refit):
    arrived = Event()
    barriers = []
    injected = RuntimeError("second submit failed before starting a worker")

    class ObservedBarrier(Barrier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            barriers.append(self)

        def wait(self, *args, **kwargs):
            arrived.set()
            return super().wait(*args, **kwargs)

    monkeypatch.setattr(parallel, "Barrier", ObservedBarrier)
    with rescue_after_failure(lambda: barriers[0].abort()) as (arm, rescued):
        def submit(number, enqueue, previous):
            if number == 1:
                return enqueue()
            assert number == 2 and len(previous) == 1 and arrived.wait(15)
            arm()
            raise injected

        result = failed_refit(injected, submit)
    assert not rescued.is_set(), "initializer still blocked when test watchdog released it"
    assert_work(result, [])


@pytest.mark.parametrize("collect", [True, False])
def test_submit_fails_after_two_completed_futures(failed_refit, collect):
    injected = RuntimeError("third submit failed before enqueue")

    def submit(number, enqueue, previous):
        if number <= 2:
            return enqueue()
        assert number == 3
        for future in previous:
            future.result(timeout=15)
        raise injected

    assert_work(failed_refit(injected, submit, collect=collect), [1, 1])


@pytest.mark.parametrize("score_failure", [False, True])
def test_submit_enqueues_work_but_interrupts_before_returning_future(
    monkeypatch, failed_refit, score_failure,
):
    injected = KeyboardInterrupt("third submit interrupted after enqueue")
    third = Event()
    real_scores = estimator._binary_scores

    def score(*args, **kwargs):
        if third.is_set() and score_failure:
            raise ValueError("scoring failed inside the unreturned future")
        return real_scores(*args, **kwargs)

    monkeypatch.setattr(estimator, "_binary_scores", score)

    def submit(number, enqueue, previous):
        if number <= 2:
            return enqueue()
        assert number == 3
        for future in previous:
            future.result(timeout=15)
        third.set()
        enqueue().result(timeout=15)
        raise injected

    assert_work(failed_refit(injected, submit), [1, 1, int(not score_failure)])


@pytest.mark.parametrize("return_queued_future", [True, False])
def test_submit_interrupt_cancels_queued_future_and_joins_running_work(
    monkeypatch, failed_refit, return_queued_future,
):
    injected = KeyboardInterrupt("submit interrupted while older work is running")
    ready = Barrier(3)
    release = Event()
    queued = []
    real_scores = estimator._binary_scores

    def score(*args, **kwargs):
        if not release.is_set():
            ready.wait(timeout=15)
            assert release.wait(15)
        return real_scores(*args, **kwargs)

    monkeypatch.setattr(estimator, "_binary_scores", score)
    with rescue_after_failure(release.set) as (arm, rescued):
        def submit(number, enqueue, previous):
            if number <= 2 or (number == 3 and return_queued_future):
                return enqueue()
            assert number == (4 if return_queued_future else 3)
            ready.wait(timeout=15)
            pending = previous[2] if return_queued_future else enqueue()
            assert not pending.running() and not pending.done()
            queued.append(pending)
            pending.add_done_callback(lambda f: release.set() if f.cancelled() else None)
            arm()
            raise injected

        result = failed_refit(injected, submit)
    assert not rescued.is_set(), "submitted queued work remained uncancelled until test rescue"
    assert queued[0].cancelled()
    assert_work(result, [1, 1])
