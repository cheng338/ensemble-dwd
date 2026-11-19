"""Caller NumPy error-policy regressions for parallel ensemble base fits."""
import pickle
import threading
import warnings
from collections import Counter

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits
from ensemble_dwd._failure_context import failure_context
from ensemble_dwd.profiling import fit_profile

from parallel_error_policy_support import (DEFAULT, STRICT, Sink, PoolObserver, data, model, trigger,
                          caught, install_fit_probe, install_score_probe, assert_same_fit,
                          restore_runtime)



@pytest.mark.parametrize("policy", (DEFAULT, STRICT))
@pytest.mark.parametrize("offset", (0., float(2**40)))
def test_real_fits_preserve_policy_and_match_sequential(monkeypatch, policy, offset):
    X, y = data(offset=offset)
    rows = install_fit_probe(monkeypatch, len(y))
    with np.errstate(**policy):
        sequential = model(base_jobs=1).fit(X, y)
        parallel = model().fit(X, y)
        assert np.geterr() == policy and np.geterrcall() is None
        assert_same_fit(sequential, parallel, X)
    assert len(rows) == 6 and all(row["policy"] == policy for row in rows)
    assert all(row["handler"] is None for row in rows)
    assert parallel.base_execution_["effective_jobs"] == 2


@pytest.mark.parametrize("category", ("divide", "over", "under", "invalid"))
@pytest.mark.parametrize("mode", ("warn", "raise", "call", "log", "ignore", "print"))
def test_real_numpy_events_obey_each_caller_mode(monkeypatch, capfd, category, mode):
    X, y = data()
    rows = install_fit_probe(monkeypatch, len(y), event=category)
    observer, sink = PoolObserver(monkeypatch), Sink()
    np.seterrcall(sink)
    policy = dict(DEFAULT, **{category: mode})
    with np.errstate(**policy), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if mode == "raise":
            error = caught(lambda: model().fit(X, y))
            assert isinstance(error, FloatingPointError)
            context = failure_context(error)
            assert context["stage"] == "base_fit"
            assert context["base_fit_calls_completed"] == context["final_fit_calls_completed"] == 0
        else:
            model().fit(X, y)
        assert np.geterr() == policy and np.geterrcall() is sink
    assert rows and all(row["policy"] == policy and row["handler"] is sink for row in rows)
    word = {"over": "overflow", "under": "underflow"}.get(category, category)
    if mode == "warn":
        assert any(word in str(item.message).lower() for item in seen)
    if mode in ("call", "log"):
        assert any(word in str(event).lower() for _, event in sink.events)
    if mode == "ignore":
        assert not any(word in str(item.message).lower() for item in seen)
        assert not sink.events
    if mode == "print":
        captured = capfd.readouterr()
        assert word in (captured.out + captured.err).lower()
    observer.assert_restored_and_joined()


@pytest.mark.parametrize("category", ("over", "invalid"))
@pytest.mark.parametrize("mode", ("raise", "call"))
def test_scoring_preserves_existing_over_invalid_mask(monkeypatch, category, mode):
    X, y = data()
    rows = install_score_probe(monkeypatch, event=category)
    sink = Sink(fail=True)
    np.seterrcall(sink)
    policy = dict(DEFAULT, **{category: mode})
    with np.errstate(**policy):
        sequential = model(base_jobs=1).fit(X, y)
        parallel = model().fit(X, y)
        assert np.geterr() == policy and np.geterrcall() is sink
    assert len(rows) == 4
    assert all(row["policy"] == dict(policy, over="ignore", invalid="ignore") for row in rows)
    assert all(row["handler"] is sink for row in rows) and not sink.events
    np.testing.assert_allclose(sequential.consensus_scores_, parallel.consensus_scores_, rtol=2e-8, atol=2e-10)


def test_inactive_nonpickleable_handler_is_shared_but_not_stored_in_fitted_model(monkeypatch):
    X, y = data()
    rows, sink = install_fit_probe(monkeypatch, len(y)), Sink()
    np.seterrcall(sink)
    fitted = model().fit(X, y)
    assert all(row["handler"] is sink for row in rows) and not sink.events
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(fitted.predict(X), restored.predict(X))
    assert np.geterrcall() is sink


@pytest.mark.parametrize("mode", ("call", "log"))
def test_missing_handler_matches_numpy_exception_and_restores_workers(monkeypatch, mode):
    X, y = data()
    install_fit_probe(monkeypatch, len(y), event="divide")
    # A dirty pre-existing worker handler makes explicit call=None observable.
    observer = PoolObserver(monkeypatch, seed_handler=Sink())
    with np.errstate(divide=mode, call=None):
        expected = caught(lambda: trigger("divide"))
        actual = caught(lambda: model().fit(X, y))
        assert type(actual) is type(expected)
        assert failure_context(actual)["stage"] == "base_fit"
        assert np.geterrcall() is None
    observer.assert_restored_and_joined()


@pytest.mark.parametrize("mode", ("call", "log"))
def test_handler_failure_propagates_with_counts_and_no_cache_publication(monkeypatch, tmp_path, mode):
    X, y = data()
    install_fit_probe(monkeypatch, len(y), event="divide")
    observer, sink = PoolObserver(monkeypatch), Sink(fail=True)
    np.seterrcall(sink)
    fitted = model(cache_dir=tmp_path)
    with np.errstate(divide=mode), fit_profile() as profile:
        error = caught(lambda: fitted.fit(X, y))
        assert isinstance(error, RuntimeError) and "IR4 callback sentinel" in str(error)
        assert np.geterrcall() is sink and np.geterr()["divide"] == mode
    context = failure_context(error)
    assert context["stage"] == "base_fit" and context["base_position"] >= 1
    assert context["base_fit_calls_completed"] == context["final_fit_calls_completed"] == 0
    assert profile.as_dict()["counts"].get("base_fit_completed", 0) == 0
    assert not [name for name in vars(fitted) if name.endswith("_")]
    assert not list(tmp_path.rglob("*.joblib"))
    observer.assert_restored_and_joined()


@pytest.mark.parametrize("failure_type", (None, RuntimeError, KeyboardInterrupt))
def test_reused_workers_restore_policy_after_mutation_and_failure(monkeypatch, failure_type):
    X, y = data(n=40)
    fits = install_fit_probe(monkeypatch, len(y), mutate_after=True)
    install_score_probe(monkeypatch, mutate=True, fail_type=failure_type)
    observer, sink = PoolObserver(monkeypatch), Sink()
    np.seterrcall(sink)
    with np.errstate(**STRICT), fit_profile() as profile:
        if failure_type is None:
            fitted = model(M=5, theta=.2).fit(X, y)
            assert len(fitted.estimators_) == 5
        else:
            error = caught(lambda: model(M=5, theta=.2).fit(X, y))
            assert isinstance(error, failure_type)
            context = failure_context(error)
            assert context["stage"] == "base_scoring" and context["final_fit_calls_completed"] == 0
            assert context["base_fit_calls_completed"] == profile.as_dict()["counts"]["base_fit_completed"]
        assert np.geterr() == STRICT and np.geterrcall() is sink
    bases = [row for row in fits if row["size"] < len(y)]
    assert bases and all(row["policy"] == STRICT and row["handler"] is sink for row in bases)
    if failure_type is None:
        assert max(Counter(row["thread"] for row in bases).values()) >= 3
        assert profile.as_dict()["counts"]["base_fit_completed"] == 5
        assert profile.as_dict()["counts"]["base_scoring_completed"] == 5
    observer.assert_restored_and_joined()


@pytest.mark.parametrize("mode", ("call", "log"))
def test_active_handler_bypasses_cache_and_runs_real_base_work_each_time(monkeypatch, tmp_path, mode):
    X, y = data()
    rows = install_fit_probe(monkeypatch, len(y), event="divide")
    sink = Sink()
    np.seterrcall(sink)
    with np.errstate(divide=mode):
        first = model(cache_dir=tmp_path).fit(X, y)
        second = model(cache_dir=tmp_path).fit(X, y)
    assert len([row for row in rows if row["size"] < len(y)]) == 4
    assert sink.events
    assert first.cache_key_ is None and second.cache_key_ is None
    assert first.cache_status_.startswith("bypassed:") and second.cache_status_.startswith("bypassed:")
    assert not list(tmp_path.rglob("*.joblib"))


def test_real_rbf_underflow_fails_at_base_fit_without_scoring_or_final_fit(monkeypatch):
    X, y = data(far=True)
    observer = PoolObserver(monkeypatch)
    with np.errstate(under="raise"):
        for jobs in (1, 2):
            with fit_profile() as profile:
                error = caught(lambda: model(base_jobs=jobs, gamma=.5).fit(X, y))
            assert isinstance(error, FloatingPointError) and "underflow" in str(error)
            context = failure_context(error)
            assert context["stage"] == "base_fit"
            assert context["base_fit_calls_completed"] == context["base_score_calls_completed"] == 0
            assert context["final_fit_calls_completed"] == 0
            assert profile.as_dict()["counts"].get("base_scoring_completed", 0) == 0
    observer.assert_restored_and_joined()


def test_outer_python_thread_keeps_existing_sequential_fallback(monkeypatch):
    X, y = data()
    rows = install_fit_probe(monkeypatch, len(y))
    sink, answers, errors = Sink(), [], []

    def run():
        try:
            with np.errstate(call=sink, **STRICT):
                answers.append(model().fit(X, y))
                assert np.geterr() == STRICT and np.geterrcall() is sink
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive() and not errors and len(answers) == 1
    assert answers[0].base_execution_["effective_jobs"] == 1
    assert all(row["policy"] == STRICT and row["handler"] is sink for row in rows)
    assert np.geterr() == DEFAULT and np.geterrcall() is None


def test_real_sixteen_worker_fit_uses_native_one_and_restores_native_sixteen(monkeypatch):
    X, y = data(n=128)
    rows = install_fit_probe(monkeypatch, len(y))
    observer = PoolObserver(monkeypatch)
    with threadpool_limits(16):
        before = threadpool_info()
        assert before and all(pool["num_threads"] == 16 for pool in before)
        with np.errstate(**STRICT), fit_profile() as profile:
            fitted = model(M=16, theta=1/16, base_jobs=16, max_iter=2).fit(X, y)
            assert np.geterr() == STRICT
        assert threadpool_info() == before
    assert fitted.base_execution_["effective_jobs"] == 16
    assert fitted.base_execution_["native_threads"] == 1
    assert len(observer.initialized) == 16
    assert len([row for row in rows if row["size"] < len(y)]) == 16
    assert all(row["policy"] == STRICT for row in rows)
    for runtime in fitted.base_execution_["worker_runtimes"]:
        assert runtime and all(pool["num_threads"] == 1 for pool in runtime)
    assert profile.as_dict()["counts"]["base_fit_completed"] == 16
    assert profile.as_dict()["counts"]["final_fit_completed"] == 1
    observer.assert_restored_and_joined()
