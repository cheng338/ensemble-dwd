"""Optional, execution-local timing of binary ensemble DWD fit stages.

Use ``with fit_profile() as profile: model.fit(X, y)`` and then
``profile.as_dict()``. This also collects the binary fits of a sequential
one-vs-one wrapper. It does not change estimator options, numerical operations,
sampling, or cached results, and never stores the profile on the estimator.

Each stage reports inclusive, child, and self wall time. Inclusive times overlap;
sum self times, not inclusive times, to allocate the measured fit work. Cache
hits have cache-stage time and reused learner counts, without invented base-fit
time. The context's wall time can include unrelated work (including a pipeline's
scaler); the ``fit`` stage measures only the binary ensemble estimator.

Profiles apply only to the thread and asyncio task that entered the context.
New threads, processes, and tasks need their own contexts. Nested contexts are
independent and the innermost context collects its work. Profiles contain only
timings and counts, not training data or numerical model state.
"""

import asyncio
from contextlib import nullcontext
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from threading import get_ident
from time import perf_counter


_ACTIVE = ContextVar("ensemble_dwd_fit_profile", default=None)
_FRAMES = ContextVar("ensemble_dwd_fit_profile_frames", default=())
_DISABLED = nullcontext()


def _execution_identity():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return get_ident(), task


def _active_profile():
    profile = _ACTIVE.get()
    if (profile is not None and profile._state == "active"
            and profile._owner == _execution_identity()):
        return profile
    return None


class _Stage:
    def __init__(self, profile, name):
        self.profile, self.name = profile, name
        self.child_seconds = 0.0

    def __enter__(self):
        frames = _FRAMES.get()
        self.parent = frames[-1] if frames and frames[-1].profile is self.profile else None
        self.token = _FRAMES.set(frames + (self,))
        self.started = perf_counter()
        return self

    def __exit__(self, error_type, error, traceback):
        elapsed = perf_counter() - self.started
        _FRAMES.reset(self.token)
        if self.parent is not None:
            self.parent.child_seconds += elapsed
        record = self.profile._stages.setdefault(self.name, {
            "calls": 0, "completed_calls": 0, "aborted_calls": 0,
            "inclusive_seconds": 0.0, "child_seconds": 0.0,
            "self_seconds": 0.0, "aborted_seconds": 0.0,
            "exception_types": {}})
        record["calls"] += 1
        record["inclusive_seconds"] += elapsed
        record["child_seconds"] += self.child_seconds
        record["self_seconds"] += elapsed - self.child_seconds
        if error_type is None:
            record["completed_calls"] += 1
        else:
            record["aborted_calls"] += 1
            record["aborted_seconds"] += elapsed
            name = type.__getattribute__(error_type, "__name__")
            record["exception_types"][name] = record["exception_types"].get(name, 0) + 1
        return False


def _stage(name):
    profile = _active_profile()
    return _DISABLED if profile is None else _Stage(profile, name)


def _profiled(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            with _stage(name):
                return function(*args, **kwargs)
        return call
    return decorate


def _record_cache_result(hit, status, reused_learners):
    profile = _active_profile()
    if profile is not None:
        profile._cache_outcomes[status] = profile._cache_outcomes.get(status, 0) + 1
        if hit:
            profile._reused_learners += reused_learners


class _FitProfile:
    def __init__(self):
        self._state = "not_entered"
        self._stages = {}
        self._cache_outcomes = {}
        self._reused_learners = 0
        self._parallel_profiles = []
        self._wall_seconds = 0.0

    def __enter__(self):
        if self._state != "not_entered":
            raise RuntimeError("A fit profile context can only be entered once.")
        self._owner = _execution_identity()
        self._started = perf_counter()
        self._token = _ACTIVE.set(self)
        self._state = "active"
        return self

    def __exit__(self, error_type, error, traceback):
        self._wall_seconds = perf_counter() - self._started
        _ACTIVE.reset(self._token)
        self._token = None
        self._owner = None
        self._state = "completed" if error_type is None else "aborted"
        return False

    def as_dict(self):
        """Return detached, JSON-compatible counters and completed stage timings.

        While the context is active, open stages are not included. An aborted
        stage is measured through its exception; it is not a completed fit.
        Self time includes uninstrumented work and profiling overhead.
        """
        elapsed = self._wall_seconds
        if self._state == "active" and self._owner == _execution_identity():
            elapsed = perf_counter() - self._started
        stages = deepcopy(self._stages)
        counts = {name + "_completed": stages.get(name, {}).get("completed_calls", 0)
                  for name in ("fit", "base_fit", "base_scoring", "final_fit")}
        for worker in self._parallel_profiles:
            for key in ("base_fit_completed", "base_scoring_completed"):
                counts[key] += worker["counts"].get(key, 0)
        counts["reused_base_learners"] = self._reused_learners
        return {"schema_version": 1, "state": self._state,
                "scope": "binary_ensemble_dwd_fit_in_current_execution_context",
                "context_wall_seconds": elapsed, "stages": stages,
                "counts": counts, "cache_outcomes": dict(self._cache_outcomes),
                "parallel_worker_profiles": deepcopy(self._parallel_profiles),
                "parallel_timing_semantics": "Worker times overlap; never add their self times to parent wall-clock stages.",
                "timing_semantics": {
                    "inclusive_seconds": "includes nested stages; do not sum across stages",
                    "child_seconds": "direct child stage wall time",
                    "self_seconds": "inclusive minus child; includes uninstrumented work and overhead",
                    "aborted_seconds": "inclusive time of aborted calls; overlaps inclusive_seconds",
                    "context_wall_seconds": "entire context; may include work outside estimator.fit"}}


def fit_profile():
    """Create a single-use context for optional ensemble DWD fit-stage profiling.

    Profiling is off unless this context is entered. Disabled instrumentation
    makes no clock calls. Managed base-fit workers attach their individual
    profiles to the parent collector. For independent parallel fits, use a
    separate context in each worker and collect those profiles explicitly.
    """
    return _FitProfile()


def _record_parallel_profiles(profiles):
    profile = _active_profile()
    if profile is not None:
        profile._parallel_profiles.extend(deepcopy(profiles))
