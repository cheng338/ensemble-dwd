"""Small failure records without retaining observations or fitted models.

Records describe returned calls, not solver iterations or the amount of work
inside a failed call. Collection is best effort and runs only on failure paths.
"""

import math
from inspect import getattr_static
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


_MAX_INTEGER = (1 << 63) - 1
_MAX_LABEL_LENGTH = 160
_STAGES = frozenset((
    "options", "input_validation", "setup", "sampling", "base_work",
    "base_setup", "base_fit", "base_validation", "base_scoring",
    "base_diagnostics", "base_consensus", "base_summary", "base_cache",
    "selection", "final_fit", "final_validation", "final_diagnostics",
    "finalization",
))
_COUNTS = frozenset((
    "base_fit_calls_completed", "base_score_calls_completed",
    "cached_base_learners", "final_fit_calls_completed",
))
_CLASS_COUNTS = frozenset((
    "training_class_counts", "subset_class_counts", "selected_class_counts",
))
_FIELDS = frozenset(("schema", "scope", "family", "stage", "base_position",
                     "base_work_source", "classes")) | _COUNTS | _CLASS_COUNTS
_NUMPY_INTS = frozenset(np.dtype(name).type for name in (
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"))
_NUMPY_FLOATS = frozenset(np.dtype(name).type for name in (
    "float16", "float32", "float64"))
_SCALAR_TYPES = frozenset((bool, int, float, str, bytes, np.bool_, np.str_, np.bytes_)) | _NUMPY_INTS | _NUMPY_FLOATS
_EXCEPTION_DICT = BaseException.__dict__["__dict__"]
_ACTIVE_SCOPE = ContextVar("ensemble_failure_context_scope", default=None)
_OWNER_KEY = "_ensemble_failure_context_owner"


def uses_implementation(instance, name, implementation):
    """Inspect a method without invoking a custom callable's attribute hooks."""
    try:
        return getattr_static(instance, name) is implementation
    except Exception:
        return False


def _is_type(kind, allowed):
    # Identity avoids equality/hash hooks on a user-defined metaclass.
    return any(kind is candidate for candidate in allowed)


def _integer(value, *, minimum=0, maximum=_MAX_INTEGER):
    return type(value) is int and minimum <= value <= maximum


def _label(value):
    """A bounded primitive label; None means its description is unavailable."""
    kind = type(value)
    if kind is bool or kind is np.bool_:
        return bool(value)
    if kind is int or _is_type(kind, _NUMPY_INTS):
        return int(value) if -_MAX_INTEGER <= value <= _MAX_INTEGER else None
    if kind is float or _is_type(kind, _NUMPY_FLOATS):
        result = float(value)
        return result if math.isfinite(result) else None
    if kind is str or kind is np.str_:
        return str(value) if len(value) <= _MAX_LABEL_LENGTH else None
    # Bytes and unfamiliar labels are deliberately not stringified.
    return None


def _validated(context):
    if type(context) is not dict or len(context) > len(_FIELDS):
        return None
    for key in context:
        if type(key) is not str or key not in _FIELDS:
            return None
    if (type(context.get("schema")) is not int or context["schema"] != 1
            or type(context.get("scope")) is not str
            or context["scope"] != "binary_ensemble_fit"
            or type(context.get("family")) is not str
            or context["family"] not in ("ensemble_dwd", "ensemble_svm")
            or type(context.get("stage")) is not str
            or context["stage"] not in _STAGES):
        return None
    result = {}
    for key, value in context.items():
        if key in _COUNTS:
            maximum = 1 if key == "final_fit_calls_completed" else _MAX_INTEGER
            if value is not None and not _integer(value, maximum=maximum):
                return None
        elif key == "base_position":
            if value is not None and not _integer(value, minimum=1):
                return None
        elif key == "base_work_source":
            if type(value) is not str or value not in ("current", "cache", "unknown"):
                return None
        elif key in _CLASS_COUNTS:
            if value is not None:
                if type(value) is not list or not 1 <= len(value) <= 2:
                    return None
                if any(not _integer(count) for count in value):
                    return None
                value = value.copy()
        elif key == "classes":
            if value is not None:
                if type(value) is not list or not 1 <= len(value) <= 2:
                    return None
                for label in value:
                    if label is not None and (not _is_type(type(label), (bool, int, float, str))
                                              or _label(label) is None):
                        return None
                value = value.copy()
        result[key] = value
    classes = result.get("classes")
    for key in _CLASS_COUNTS:
        counts = result.get(key)
        if counts is not None and (classes is None or len(counts) != len(classes)):
            return None
    return result


def _exception_dict(error):
    if not issubclass(type(error), BaseException):
        return None
    # Read the built-in storage directly: custom attribute hooks and properties
    # on an exception subclass must not run to read supplementary metadata.
    values = _EXCEPTION_DICT.__get__(error)
    return values if type(values) is dict else None


@contextmanager
def failure_context_scope():
    """Keep inner facts authoritative only within the current fit invocation.

    A reused exception starts fresh in every fit. On propagation from a nested
    fit, promote its marker to the parent scope so the outer handler can add
    missing facts without replacing the more precise inner failure stage.
    """
    parent = _ACTIVE_SCOPE.get()
    owner = object()
    token = _ACTIVE_SCOPE.set(owner)
    try:
        yield
    except BaseException as error:
        if parent is not None:
            try:
                values = _exception_dict(error)
                if values is not None and values.get(_OWNER_KEY) is owner:
                    values[_OWNER_KEY] = parent
            except BaseException:
                pass
        raise
    finally:
        _ACTIVE_SCOPE.reset(token)


def failure_context(error):
    """Return a fresh validated record, or None, without following causes."""
    try:
        values = _exception_dict(error)
        return _validated(values.get("ensemble_failure_context")) if values is not None else None
    except BaseException:
        return None


def _raw_classes(classes):
    if type(classes) is np.ndarray:
        if classes.ndim != 1 or not 1 <= classes.size <= 2:
            return None
        values = list(classes)
    elif _is_type(type(classes), (list, tuple)) and 1 <= len(classes) <= 2:
        values = list(classes)
    else:
        return None
    return values if all(_is_type(type(value), _SCALAR_TYPES) for value in values) else None


def _class_counts(y, classes, indices=None):
    if type(y) is not np.ndarray or y.ndim != 1:
        return None
    if y.dtype.kind not in "biufUSO":
        return None
    if y.dtype.kind == "O" and any(not _is_type(type(value), _SCALAR_TYPES) for value in y):
        return None
    if indices is not None:
        if (type(indices) is not np.ndarray or indices.ndim != 1
                or indices.dtype.kind not in "iu"):
            return None
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= y.size):
            return None
        y = y[indices]
    counts = [int(np.count_nonzero(y == label)) for label in classes]
    return counts if sum(counts) == y.size else None


def attach_failure_context(error, context, *, y=None, classes=None,
                           subset=None, selected=None):
    """Attach bounded facts; inner facts from the current invocation win.

    ``subset`` and ``selected`` are integer row indices into ``y``. Class counts
    align with ``classes``; a None label description does not change that order.
    Unknown facts are omitted or None, never inferred as zero.
    """
    try:
        values = _exception_dict(error)
        if values is None or type(context) is not dict:
            return
        if len(context) > len(_FIELDS):
            return
        # Reject exotic keys before copying or looking them up.
        if any(type(key) is not str or key not in _FIELDS for key in context):
            return
        proposed = context.copy()
        proposed.setdefault("schema", 1)
        proposed.setdefault("scope", "binary_ensemble_fit")
        raw_classes = _raw_classes(classes)
        if raw_classes is not None:
            proposed.setdefault("classes", [_label(value) for value in raw_classes])
            proposed.setdefault("training_class_counts", _class_counts(y, raw_classes))
            if subset is not None:
                proposed.setdefault("subset_class_counts", _class_counts(y, raw_classes, subset))
            if selected is not None:
                proposed.setdefault("selected_class_counts", _class_counts(y, raw_classes, selected))
        proposed = _validated(proposed)
        if proposed is None:
            return
        owner = _ACTIVE_SCOPE.get()
        previous = (failure_context(error)
                    if values.get(_OWNER_KEY) is owner else None)
        if previous is not None:
            proposed.update(previous)
        combined = _validated(proposed)
        if combined is not None:
            values["ensemble_failure_context"] = combined
            values[_OWNER_KEY] = owner
    except BaseException:
        # Failure reporting must not replace even an interruption being handled.
        return


def copy_failure_context(source, target):
    """Preserve a bounded record when another exception wraps the original."""
    try:
        record = failure_context(source)
        if record is not None:
            attach_failure_context(target, record)
    except BaseException:
        return
