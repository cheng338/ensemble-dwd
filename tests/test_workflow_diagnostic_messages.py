"""Failure-only reporting contracts exercised through the search workflows."""

from contextlib import nullcontext
import json
from unittest.mock import Mock
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from ensemble_dwd import IneligibleSelectionError
from test_final_thread_preflight import Workflow
from test_phase_threads import PhaseSurface, PhaseTransform
from test_final_recovery import RecoveryFixture, protect_cv
import adaptive_search
import cv_checkpoint
import evaluation_reporting
import final_recovery
import phase_threads
import search_support
import work_reporting


FALLBACK = "Error message could not be formatted"


class BrokenMessage(ValueError):
    def __str__(self):
        raise RuntimeError("formatting failed")


class InvalidMessage(ValueError):
    def __str__(self):
        return 42


class InterruptingMessage(ValueError):
    def __str__(self):
        raise KeyboardInterrupt("interrupt from primary message formatter")


class BrokenInterrupt(KeyboardInterrupt):
    def __str__(self):
        raise RuntimeError("formatting the interruption failed")


class BrokenSelection(IneligibleSelectionError):
    def __str__(self):
        raise RuntimeError("formatting the ineligible selection failed")


class SimpleClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] > 0).astype(int)


def fail(error):
    def raise_error(*args, **kwargs):
        raise error
    return raise_error


def raised(call):
    # Capture even a misplaced formatter interruption so a failing control
    # reports a normal assertion failure and the rest of the suite still runs.
    with pytest.raises(BaseException) as caught:
        call()
    return caught.value


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(PhaseSurface, "fits", [])
    monkeypatch.setattr(PhaseSurface, "predictions", [])
    monkeypatch.setattr(PhaseSurface, "fail_final", None)
    monkeypatch.setattr(PhaseTransform, "fits", [])
    return Workflow(tmp_path, monkeypatch)


@pytest.mark.parametrize(("error", "message"), [
    (FloatingPointError("ordinary numerical text"), "ordinary numerical text"),
    (InvalidMessage(), FALLBACK), (InterruptingMessage(), FALLBACK),
])
def test_exception_details_preserves_message_category_and_notes(error, message):
    error.add_note("original note")
    try:
        details = search_support.exception_details(error)
    except BaseException as escaped:
        pytest.fail("message conversion escaped as " + type(escaped).__name__)
    assert details == {
        "category": "numerical_error" if isinstance(error, FloatingPointError) else "unexpected_error",
        "exception_type": type(error).__module__ + "." + type(error).__qualname__,
        "message": message, "exception_notes": ["original note"],
    }


@pytest.mark.parametrize("phase", ["cv_fit", "cv_predict", "cv_scoring"])
def test_evaluator_preserves_primary_and_writes_failure(tmp_path, monkeypatch, phase):
    X, y = np.array([[-1], [1]]), np.array([0, 1])
    evaluator = search_support.CandidateEvaluator(SimpleClassifier(), failure_dir=tmp_path).fit(X, y)
    original = BrokenMessage()
    cause = ArithmeticError("original cause")
    original.__cause__ = cause
    if phase == "cv_fit":
        monkeypatch.setattr(SimpleClassifier, "fit", fail(original))
        call = lambda: evaluator.fit(X, y)
    elif phase == "cv_predict":
        monkeypatch.setattr(SimpleClassifier, "predict", fail(original))
        call = lambda: evaluator.predict(X)
    else:
        monkeypatch.setattr(search_support, "accuracy_score", fail(original))
        call = lambda: search_support.eligible_accuracy(evaluator, X, y)
    assert raised(call) is original
    assert original.__cause__ is cause
    paths = list(tmp_path.glob("failure-*.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_bytes())
    assert record["phase"] == phase and record["message"] == FALLBACK
    assert record["exception_type"].endswith(".BrokenMessage")
    assert record["category"] == "unexpected_error"
    assert record["training_data"] == search_support.data_fingerprint(X, y)
    if phase == "cv_fit":
        assert not evaluator.eligible_ and not hasattr(evaluator, "estimator_")
        assert evaluator.failure_["message"] == FALLBACK
    elif phase == "cv_scoring":
        assert evaluator.accuracy_counts_ is None


def test_broken_selection_retains_explicit_nan_policy(monkeypatch):
    X, y = np.array([[-1], [1]]), np.array([0, 1])
    primary = BrokenSelection()
    monkeypatch.setattr(SimpleClassifier, "fit", fail(primary))
    model = search_support.CandidateEvaluator(SimpleClassifier()).fit(X, y)
    assert not model.eligible_ and model.failure_["category"] == "ineligible_selection"
    assert model.failure_["message"] == FALLBACK
    assert np.isnan(search_support.eligible_accuracy(model, X, y))
    assert raised(lambda: model.set_params(error_policy="raise").fit(X, y)) is primary


@pytest.mark.parametrize(("origin", "kind"), [
    ("factory", KeyboardInterrupt), ("factory", SystemExit),
    ("formatter", KeyboardInterrupt), ("sink", SystemExit),
])
def test_optional_interrupt_propagates_from_its_origin(monkeypatch, origin, kind):
    signal = kind("optional reporting interruption")

    class InterruptedFormatting(ValueError):
        def __str__(self):
            raise signal

    error = {"factory": signal, "formatter": InterruptedFormatting(),
             "sink": ValueError("ordinary message")}[origin]
    if origin == "sink":
        monkeypatch.setattr(warnings, "warn", fail(signal))
    assert raised(lambda: final_recovery._optional_report(fail(error))) is signal


@pytest.mark.parametrize(("error", "expected"), [(InvalidMessage(), FALLBACK),
                                                 (ValueError("usual text"), "usual text")])
def test_optional_warning_failure_preserves_fallback(monkeypatch, error, expected):
    sink = Mock(side_effect=RuntimeError("warning sink failed"))
    monkeypatch.setattr(warnings, "warn", sink)
    assert final_recovery._optional_report(fail(error)) == {
        "coverage": "unavailable", "reporting_error": expected,
    }
    if expected != FALLBACK:
        assert sink.call_args.args == ("Work/report metadata unavailable: usual text", RuntimeWarning)


@pytest.mark.parametrize("boundary", ["write_failure_json", "record_failure"])
@pytest.mark.parametrize(("origin", "kind"), [
    ("formatter", KeyboardInterrupt), ("formatter", SystemExit), ("sink", KeyboardInterrupt),
])
def test_final_estimator_error_during_failure_record_write(workflow, monkeypatch, boundary, origin, kind):
    primary = FloatingPointError("failed final estimator fit")
    signal = kind("interruption in failure logging")
    original_fit = PhaseSurface.fit
    original_write = search_support.write_json_atomic
    original_show = warnings.showwarning
    attempted = []
    warned = []

    class UnprintableWriteError(OSError):
        def __str__(self):
            raise signal

    def fail_final(model, X, y):
        value = original_fit(model, X, y)
        if len(y) == len(workflow.y):
            raise primary
        return value

    def write(path, payload, **kw):
        target = (path.name == "refit.json" if boundary == "write_failure_json" else
                  path.name.startswith("failure-") and payload.get("phase") == "final_refit")
        if target:
            attempted.append(path.name)
            if origin == "formatter":
                raise UnprintableWriteError()
            raise OSError("failure record disk error")
        return original_write(path, payload, **kw)

    def show(message, category, *args, **kw):
        if category is RuntimeWarning and str(message).startswith("Could not write the failure record:"):
            warned.append(str(message))
            if origin == "sink":
                raise signal
            return
        return original_show(message, category, *args, **kw)

    monkeypatch.setattr(PhaseSurface, "fit", fail_final)
    monkeypatch.setattr(search_support, "write_json_atomic", write)
    with warnings.catch_warnings():
        warnings.simplefilter("always", RuntimeWarning)
        monkeypatch.setattr(warnings, "showwarning", show)
        caught = raised(lambda: workflow.run(budget=1, legacy=True))
    assert len(attempted) == 1
    assert caught is (signal if origin == "sink" else primary)
    expected = "failure record disk error" if origin == "sink" else FALLBACK
    assert warned == ["Could not write the failure record: " + expected]
    assert workflow.final_count == 1 and workflow.held_out_count == 0


@pytest.mark.parametrize("interrupt", [False, True])
def test_fold_failure_preserves_completed_receipt_and_retries_only_missing(
    workflow, monkeypatch, interrupt,
):
    original = BrokenInterrupt() if interrupt else BrokenMessage()
    real_fit = PhaseSurface.fit
    injected = []

    def fit(model, X, y):
        value = real_fit(model, X, y)
        if workflow.cv_count == 2 and not injected:
            injected.append(True)
            raise original
        return value

    monkeypatch.setattr(PhaseSurface, "fit", fit)
    assert raised(lambda: workflow.run(budget=1)) is original
    run = workflow.root / "run"
    status = json.loads((run / "status.json").read_bytes())
    expected = "interrupted" if interrupt else "failed"
    assert status["state"] == expected and status["failure"]["message"] == FALLBACK
    attempts = [json.loads(p.read_bytes())["payload"] for p in
                (run / "cv-checkpoints/attempts").glob("*.json")]
    assert len([r for r in attempts if r["state"] == expected and
                r["failure"]["message"] == FALLBACK]) == 1
    kept = {p: p.read_bytes() for p in (run / "cv-checkpoints/folds").glob("*.json")}
    assert len(kept) == 1
    resumed = workflow.run(resume=True, budget=1)
    assert resumed["completed"]["held_out_accuracy"] == 1.0
    assert (workflow.cv_count, workflow.final_count, workflow.held_out_count) == (7, 1, 1)
    assert all(p.read_bytes() == raw for p, raw in kept.items())


@pytest.mark.parametrize("legacy", [False, True], ids=["adaptive", "noncheckpoint"])
def test_failed_final_refit_preserves_original_without_scoring(workflow, monkeypatch, legacy):
    original = BrokenMessage()
    real_fit = PhaseSurface.fit

    def fit(model, X, y):
        value = real_fit(model, X, y)
        if len(y) == len(workflow.y):
            raise original
        return value

    monkeypatch.setattr(PhaseSurface, "fit", fit)
    assert raised(lambda: workflow.run(budget=1, legacy=legacy)) is original
    folder = workflow.root / "run"
    for name in ("status", "refit"):
        record = json.loads((folder / (name + ".json")).read_bytes())
        assert record["state"] == "failed" and record["failure"]["message"] == FALLBACK
    assert workflow.final_count == 1 and workflow.held_out_count == 0
    assert not (folder / "completed.json").exists()


@pytest.mark.parametrize("snapshot", [False, True], ids=["optional-summaries", "work-snapshot"])
def test_adaptive_optional_failures_commit_and_replay_once(workflow, monkeypatch, snapshot):
    names = (["aggregate_work_summaries"] if snapshot else
             ["summarize_cache_diagnostics", "numerical_check_summary",
              "selection_composition", "classification_report"])
    for name in names:
        monkeypatch.setattr(adaptive_search, name, fail(BrokenMessage()))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        outcome = workflow.run(budget=1)
        replay = workflow.run(budget=1, resume=True)
    fallback = {"coverage": "unavailable", "reporting_error": FALLBACK}
    work = outcome["completion_work_summary"]
    if snapshot:
        assert work == {"schema": 1, "scope": "current_invocation", "state": "completed", **fallback}
    else:
        assert work["final_refit"]["base_cache"] == fallback
        assert outcome["refit"]["numerical_checks"] == outcome["refit"]["selection_composition"] == fallback
        assert outcome["completed"]["held_out_classification"] == fallback
    assert replay["resumed_completed"] and replay["completed"] == outcome["completed"]
    assert (workflow.cv_count, workflow.final_count, workflow.held_out_count) == (6, 1, 1)


@pytest.mark.parametrize(("boundary", "primary_type", "state"), [
    ("snapshot", ValueError, "failed"), ("mirror", KeyboardInterrupt, "interrupted"),
])
def test_failed_adaptive_fit_retains_primary_during_work_publication(
    workflow, monkeypatch, boundary, primary_type, state,
):
    primary = primary_type("original final fit failure")
    secondary = SystemExit("formatter interrupted while reporting the earlier failure")
    fit_failed = []
    faults = []
    real_fit = PhaseSurface.fit
    real_aggregate = adaptive_search.aggregate_work_summaries
    real_write = adaptive_search.write_json_atomic

    class UnprintableWorkError(OSError):
        def __str__(self):
            raise secondary

    def fit(model, X, y):
        value = real_fit(model, X, y)
        if len(y) == len(workflow.y):
            fit_failed.append(True)
            raise primary
        return value

    def aggregate(*args, **kw):
        if fit_failed and boundary == "snapshot":
            faults.append(boundary)
            raise UnprintableWorkError()
        return real_aggregate(*args, **kw)

    def write(path, payload, **kw):
        if (fit_failed and boundary == "mirror" and path.name == "work-summary.json"
                and payload.get("state") == state):
            faults.append(boundary)
            raise UnprintableWorkError()
        return real_write(path, payload, **kw)

    monkeypatch.setattr(PhaseSurface, "fit", fit)
    monkeypatch.setattr(adaptive_search, "aggregate_work_summaries", aggregate)
    monkeypatch.setattr(adaptive_search, "write_json_atomic", write)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        error = raised(lambda: workflow.run(budget=1))
    assert faults == [boundary] and error is primary
    status = json.loads((workflow.root / "run/status.json").read_bytes())
    assert status["state"] == state and status["phase"] == "final_refit"
    assert status["failure"]["message"] == "original final fit failure"
    assert (workflow.cv_count, workflow.final_count, workflow.held_out_count) == (6, 1, 0)


def test_recovery_optional_failures_preserve_final_claim_and_replay(workflow, monkeypatch):
    workflow.cap_after_cv = 2
    with pytest.raises(cv_checkpoint.CheckpointError, match="wrong active budget"):
        workflow.run(budget=4)
    workflow.cap_after_cv = None
    recovery = RecoveryFixture(workflow)
    protect_cv(monkeypatch)
    for module, name in ((work_reporting, "summarize_cache_diagnostics"),
                         (evaluation_reporting, "numerical_check_summary"),
                         (evaluation_reporting, "selection_composition"),
                         (evaluation_reporting, "classification_report")):
        monkeypatch.setattr(module, name, fail(InvalidMessage()))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        completed = recovery.recover()["completed"]
        replay = recovery.recover(resume=True)
    fallback = {"coverage": "unavailable", "reporting_error": FALLBACK}
    assert all(completed["refit"][key] == fallback for key in
               ("base_cache", "numerical_checks", "selection_composition"))
    assert completed["held_out_classification"] == fallback
    assert completed["held_out_accuracy"] == 1.0 and recovery.barrier.is_file()
    assert replay["resumed_completed"] and replay["completed"] == completed
    assert (workflow.cv_count, workflow.final_count, workflow.held_out_count) == (6, 1, 1)
    recovery.unchanged()


@pytest.mark.parametrize("preflight", [False, True], ids=["fit-scope", "preflight"])
@pytest.mark.parametrize("with_primary", [False, True], ids=["restoration-only", "active-primary"])
def test_unprintable_native_observation_and_restoration(monkeypatch, preflight, with_primary):
    primary = KeyboardInterrupt("original interruption")
    restoration = InterruptingMessage()
    baseline = {"native_pools": [{"filepath": "synthetic", "num_threads": 1}]}
    wide = {"native_pools": [{"filepath": "synthetic", "num_threads": 2}]}
    receipt = {}
    monkeypatch.setattr(phase_threads, "threadpool_limits", lambda **kw: nullcontext())
    if preflight:
        native = Mock(side_effect=[primary] if with_primary else [wide["native_pools"]] * 2)
        monkeypatch.setattr(cv_checkpoint, "_native_pool_fingerprint", native)
        observe = Mock(side_effect=[baseline, restoration])
        def call():
            phase_threads.preflight_final_native_threads(2, SimpleClassifier())
    else:
        observe = Mock(side_effect=[baseline, wide, BrokenMessage() if with_primary else wide, restoration])
        def call():
            with phase_threads.final_fit_threads(2, SimpleClassifier(), receipt):
                if with_primary:
                    raise primary
    monkeypatch.setattr(cv_checkpoint, "runtime_fingerprint", observe)
    assert raised(call) is (primary if with_primary else restoration)
    assert observe.call_count == (2 if preflight else 4)
    if not preflight:
        assert receipt["state"] == "failed" and receipt["restored"] is False
        assert receipt["restoration_error"] == FALLBACK
        if with_primary:
            assert receipt["observation_error"] == FALLBACK
