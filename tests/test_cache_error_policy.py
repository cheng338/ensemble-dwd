"""IR-3 cache correctness under caller NumPy policies and active handlers.

Nondefault policies may either receive a distinct cache identity or bypass
persistent caching. Tests do not require the documentary conservative candidate
to retain nondefault hits. Real-fit tests leave production provenance intact;
small contract tests stub only heavyweight package/native identity discovery.
"""
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dwd.gen_kern_dwd import KernGDWD
from sklearn.svm import SVC
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM
from ensemble_dwd import _cache as cache


DEFAULT = dict(divide="warn", over="warn", under="ignore", invalid="warn")
STRICT = dict(DEFAULT, divide="raise")
CATEGORIES = ("divide", "over", "under", "invalid")


@pytest.fixture(autouse=True)
def restore_numpy_policy():
    policy, handler = np.geterr(), np.geterrcall()
    np.seterr(**DEFAULT)
    np.seterrcall(None)
    try:
        yield
    finally:
        np.seterr(**policy)
        np.seterrcall(handler)


@pytest.fixture
def stable_runtime(monkeypatch):
    # Keep policy collection, provenance assembly, verifier and cache I/O real.
    monkeypatch.setattr(cache, "_initialize_numerical_runtime", lambda: None)
    monkeypatch.setattr(cache, "_package_identity", lambda: {"fixture": "package"})
    monkeypatch.setattr(cache, "_external_native_identity", lambda pools: {"fixture": "native"})
    monkeypatch.setattr(cache, "threadpool_info", lambda: [{"num_threads": 1, "internal_api": "fixture"}])


def observe(factory=KernGDWD):
    return cache._runtime_provenance(factory)


def model_class(family):
    return EnsembleKernelDWD if family == "dwd" else EnsembleKernelSVM


def probe_model(family, directory):
    return model_class(family)(M=2, theta=.25, sv_selection_ratio=.5,
                               random_state=7, gamma=.4, cache_dir=directory)


def base_result(model):
    X = np.arange(16, dtype=float).reshape(8, 2) / 8.
    y = np.tile([0, 1], 4)
    arguments = {"gamma": .4} if isinstance(model, EnsembleKernelDWD) else .4
    return model._base_result(X, y, np.array([0, 1]),
                              [np.arange(4), np.arange(4, 8)], arguments)


def payload(number):
    # SVM validates fit_status on hits; retain that actual consumer requirement.
    return {"number": number, "fit_status": np.zeros(2, dtype=int)}


def cache_payloads(directory):
    return sorted(p for p in Path(directory).rglob("*")
                  if p.is_file() and p.suffix in (".joblib", ".json"))


class Handler:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail

    def _record(self, event):
        self.events.append(event)
        if self.fail:
            raise RuntimeError("IR-3 handler sentinel")

    def __call__(self, description, flag):
        self._record((description, flag))

    def write(self, description):
        self._record(description)


def trigger(category):
    if category == "divide":
        return np.divide(1., 0.)
    if category == "over":
        return np.multiply(np.finfo(float).max, 2.)
    if category == "under":
        return np.multiply(np.finfo(float).tiny, np.finfo(float).tiny)
    return np.sqrt(-1.)


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("mode", ("ignore", "warn", "raise", "print"))
def test_each_policy_changes_identity_or_explicitly_bypasses(stable_runtime, category, mode):
    before = observe()
    target = dict(DEFAULT, **{category: mode})
    with np.errstate(**target):
        try:
            changed = observe()
        except cache._CacheIdentityError:
            # A default-only cache is conservative, not incorrect.
            assert target != DEFAULT
        else:
            if target == DEFAULT:
                assert changed == before
            else:
                assert changed != before
                assert joblib.hash(changed) != joblib.hash(before)
    assert observe() == before
    assert np.geterr() == DEFAULT


def test_recorded_policy_is_a_snapshot_not_live_or_cached_state(stable_runtime):
    original = observe()
    frozen = deepcopy(original)
    with np.errstate(**STRICT):
        try:
            changed = observe()
        except cache._CacheIdentityError:
            changed = None
        assert original == frozen
        assert changed is None or changed != original
    assert original == frozen == observe()


@pytest.mark.parametrize("family", ("dwd", "svm"))
@pytest.mark.parametrize("mode", ("call", "log"))
@pytest.mark.parametrize("category", CATEGORIES)
def test_active_handlers_execute_each_time_and_never_publish(
        tmp_path, monkeypatch, stable_runtime, family, mode, category):
    sink, calls = Handler(), []
    np.seterrcall(sink)

    def compute(self, *unused):
        calls.append(1)
        trigger(category)
        return payload(len(calls))

    monkeypatch.setattr(model_class(family), "_base_stage", compute)
    with np.errstate(**dict(DEFAULT, **{category: mode})):
        first = base_result(probe_model(family, tmp_path))
        second = base_result(probe_model(family, tmp_path))
        assert np.geterr()[category] == mode
        assert np.geterrcall() is sink
    assert len(calls) == len(sink.events) == 2
    assert first[0]["number"] == 1 and second[0]["number"] == 2
    for result in (first, second):
        assert result[1] is False and result[2] is None
        assert result[3].startswith("bypassed:")
    assert cache_payloads(tmp_path) == []


@pytest.mark.parametrize("family", ("dwd", "svm"))
@pytest.mark.parametrize("mode", ("call", "log"))
def test_handler_exception_propagates_without_a_cache_entry(
        tmp_path, monkeypatch, stable_runtime, family, mode):
    sink, calls = Handler(fail=True), []
    np.seterrcall(sink)

    def compute(self, *unused):
        calls.append(1)
        trigger("divide")
        raise AssertionError("Handler exception was swallowed")

    monkeypatch.setattr(model_class(family), "_base_stage", compute)
    with np.errstate(**dict(DEFAULT, divide=mode)):
        with pytest.raises(RuntimeError, match="IR-3 handler sentinel"):
            base_result(probe_model(family, tmp_path))
    assert len(calls) == len(sink.events) == 1
    assert cache_payloads(tmp_path) == []


@pytest.mark.parametrize("family", ("dwd", "svm"))
@pytest.mark.parametrize("mode", ("call", "log"))
def test_active_mode_is_uncacheable_even_without_registered_handler(
        tmp_path, monkeypatch, stable_runtime, family, mode):
    calls = []

    def compute(self, *unused):
        calls.append(1)
        return payload(len(calls))

    monkeypatch.setattr(model_class(family), "_base_stage", compute)
    assert np.geterrcall() is None
    with np.errstate(**dict(DEFAULT, divide=mode)):
        first = base_result(probe_model(family, tmp_path))
        second = base_result(probe_model(family, tmp_path))
    assert len(calls) == 2
    assert all(not r[1] and r[2] is None and r[3].startswith("bypassed:") for r in (first, second))
    assert cache_payloads(tmp_path) == []


@pytest.mark.parametrize("family", ("dwd", "svm"))
def test_inactive_handler_changes_preserve_default_cache_reuse(
        tmp_path, monkeypatch, stable_runtime, family):
    calls = []

    def compute(self, *unused):
        calls.append(1)
        return payload(len(calls))

    monkeypatch.setattr(model_class(family), "_base_stage", compute)
    first_handler, second_handler = Handler(), Handler()
    np.seterrcall(first_handler)
    first = base_result(probe_model(family, tmp_path))
    np.seterrcall(second_handler)
    second = base_result(probe_model(family, tmp_path))
    assert len(calls) == 1
    assert first[1] is False and second[1] is True
    assert first[2] == second[2] and first[2] is not None
    assert first_handler.events == second_handler.events == []


@pytest.mark.parametrize("offset", (0., 1.5e5), ids=("ordinary", "large-offset"))
@pytest.mark.parametrize("reverse", (False, True), ids=("default-to-strict", "strict-to-default"))
def test_real_rbf_cache_matches_fresh_target_policy(tmp_path, offset, reverse):
    # Full production provenance, native discovery, cache I/O, fitting and
    # prediction are intentionally unpatched in this group.
    X = np.random.default_rng(3).normal(size=(30, 3)) + offset
    y = np.array([0] * 15 + [1] * 15)
    query = X[[1, 6, 11, 18, 24]] + np.array([.03, -.02, .01])
    settings = dict(M=2, theta=.2, sv_selection_ratio=.5, kernel="rbf",
                    gamma=.4, lambd=.1, max_iter=4, stopping="fixed",
                    acceleration=None, random_state=7, store_base_estimators=True,
                    base_jobs=1)
    first_policy, target_policy = (STRICT, DEFAULT) if reverse else (DEFAULT, STRICT)
    with np.errstate(**first_policy):
        first = EnsembleKernelDWD(**settings, cache_dir=tmp_path).fit(X, y)
    with np.errstate(**target_policy):
        target = EnsembleKernelDWD(**settings, cache_dir=tmp_path).fit(X, y)
        fresh = EnsembleKernelDWD(**settings, cache_dir=None).fit(X, y)
        again = EnsembleKernelDWD(**settings, cache_dir=tmp_path).fit(X, y)
        assert target.cache_hit_ is False
        assert target.cache_key_ is None or target.cache_key_ != first.cache_key_
        expected = "guarded_norm_sum_rel_v2"
        assert [m.kernel_computation_ for m in target.estimators_] == [expected] * 2
        assert [m.kernel_computation_ for m in fresh.estimators_] == [expected] * 2
        assert target.final_estimator_.kernel_computation_ == expected
        for other in (target, again):
            assert_allclose(other.consensus_scores_, fresh.consensus_scores_, rtol=1e-12, atol=1e-12)
            assert_array_equal(other.selected_indices_, fresh.selected_indices_)
            assert_allclose(other.decision_function(query), fresh.decision_function(query), rtol=1e-12, atol=1e-12)
            for actual_rows, expected_rows in zip(other.subset_indices_, fresh.subset_indices_):
                assert_array_equal(actual_rows, expected_rows)
        if target.cache_key_ is None:
            assert again.cache_key_ is None and not again.cache_hit_
        else:
            assert again.cache_hit_ and again.cache_key_ == target.cache_key_


@pytest.mark.parametrize("reverse", (False, True), ids=("default-to-strict", "strict-to-default"))
def test_real_svm_shared_cache_obeys_target_policy(tmp_path, reverse):
    X = np.random.default_rng(17).normal(size=(40, 3))
    y = np.tile([0, 1], 20)
    settings = dict(M=2, theta=.2, sv_selection_ratio=1., kernel="rbf",
                    gamma=.3, C=.5, tol=1e-6, random_state=7)
    first_policy, target_policy = (STRICT, DEFAULT) if reverse else (DEFAULT, STRICT)
    with np.errstate(**first_policy):
        first = EnsembleKernelSVM(**settings, cache_dir=tmp_path).fit(X, y)
    with np.errstate(**target_policy):
        target = EnsembleKernelSVM(**settings, cache_dir=tmp_path).fit(X, y)
        fresh = EnsembleKernelSVM(**settings, cache_dir=None).fit(X, y)
        again = EnsembleKernelSVM(**settings, cache_dir=tmp_path).fit(X, y)
        assert not target.cache_hit_
        assert target.cache_key_ is None or target.cache_key_ != first.cache_key_
        for other in (target, again):
            assert_allclose(other.consensus_scores_, fresh.consensus_scores_, rtol=1e-12, atol=1e-12)
            assert_array_equal(other.selected_indices_, fresh.selected_indices_)
            assert_allclose(other.decision_function(X[:7]), fresh.decision_function(X[:7]), rtol=1e-12, atol=1e-12)
        if target.cache_key_ is None:
            assert again.cache_key_ is None and not again.cache_hit_
        else:
            assert again.cache_hit_ and again.cache_key_ == target.cache_key_


@pytest.mark.parametrize("factory", (KernGDWD, SVC), ids=("dwd", "svm"))
def test_missing_historical_policy_field_cannot_hit_new_identity(tmp_path, stable_runtime, factory):
    provenance = observe(factory)
    historical = dict(provenance)
    historical.pop("numpy_error_state", None)
    old, hit, old_key = cache._cached_base_stage(tmp_path, {"runtime": historical}, lambda: {"old": True})
    assert old == {"old": True} and not hit
    old_files = {p: p.read_bytes() for p in cache_payloads(tmp_path)}
    current, hit, key = cache._cached_base_stage(tmp_path, {"runtime": provenance}, lambda: {"current": True})
    assert not hit and key != old_key
    assert current == {"current": True}
    assert all(p.read_bytes() == data for p, data in old_files.items())


@pytest.mark.parametrize("boundary", ("before-lookup", "after-read", "after-compute", "after-serialize"))
def test_policy_changes_at_cache_boundaries_do_not_replay_or_publish(
        tmp_path, monkeypatch, stable_runtime, boundary):
    provenance = observe()
    key_data = {"runtime": provenance}
    prior = {}
    if boundary == "after-read":
        cache._cached_base_stage(tmp_path, key_data, lambda: {"cached": True})
        prior = {p: p.read_bytes() for p in cache_payloads(tmp_path)}
        original_load = cache.joblib.load

        def load(*args, **kwargs):
            value = original_load(*args, **kwargs)
            np.seterr(**STRICT)
            return value

        monkeypatch.setattr(cache.joblib, "load", load)
    elif boundary == "after-serialize":
        original_dump = cache.joblib.dump

        def dump(*args, **kwargs):
            value = original_dump(*args, **kwargs)
            np.seterr(**STRICT)
            return value

        monkeypatch.setattr(cache.joblib, "dump", dump)
    calls = []

    def compute():
        calls.append(1)
        if boundary == "after-compute":
            np.seterr(**STRICT)
        return {"computed": len(calls)}

    verifier = cache._RuntimeVerifier(observe, provenance)
    if boundary == "before-lookup":
        np.seterr(**STRICT)
    result, hit, key = cache._cached_base_stage(tmp_path, key_data, compute, verify=verifier)
    assert result == {"computed": 1} and calls == [1]
    assert not hit and key is None and verifier.failure
    assert {p: p.read_bytes() for p in cache_payloads(tmp_path)} == prior


@pytest.mark.parametrize("mode", ("call", "log"))
def test_active_handler_bypass_precedes_runtime_initialization(monkeypatch, mode):
    calls = []

    def initialize():
        calls.append(1)
        raise AssertionError("Runtime initialization ran before handler bypass")

    monkeypatch.setattr(cache, "_initialize_numerical_runtime", initialize)
    with np.errstate(**dict(DEFAULT, divide=mode)):
        with pytest.raises(cache._CacheIdentityError):
            observe()
    assert calls == []
