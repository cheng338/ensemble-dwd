"""Fit-free estimates and supplementary records for proposed CV stages.

Counts describe one cold pass through every requested candidate/seed/fold.
They do not measure executed work, retries, CPU time, elapsed time or savings.
This module never fits, transforms, samples, loads caches or reads checkpoints.
"""
import json
import math
from numbers import Integral
import warnings

from joblib import hash as parameter_hash

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.multiclass import check_classification_targets

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier
from search_support import _io_path, write_json_atomic
from near_coordinates import console_notice


COUNT_FIELDS = (
    "binary_models", "base_learner_fits", "final_learner_fits",
    "full_learner_fits", "total_learner_fits",
)
_BINARY_TYPES = (KernGDWD, EnsembleKernelDWD, EnsembleKernelSVM, SVC)


def _counts(value=0):
    return dict.fromkeys(COUNT_FIELDS, value)


def _class_name(value):
    cls = type(value)
    return cls.__module__ + "." + cls.__qualname__


def _display(value, depth=0):
    """JSON-safe parameter display, deliberately not a checkpoint identity."""
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else {"nonfinite": str(float(value))}
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if depth < 8 and type(value) in (list, tuple):
        return [_display(item, depth + 1) for item in value]
    if depth < 8 and type(value) is dict and all(type(key) is str for key in value):
        return {key: _display(item, depth + 1) for key, item in value.items()}
    if depth < 8 and type(value) is np.ndarray:
        return {"array_dtype": str(value.dtype), "array_shape": list(value.shape),
                "values": _display(value.tolist(), depth + 1)}
    return {"python_type": _class_name(value), "value_not_serialized": True}


def _seeds(value):
    if value is None:
        return None
    if type(value) is np.ndarray:
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise ValueError("seeds must be a one-dimensional integer array")
    elif type(value) not in (list, tuple):
        raise ValueError("seeds must be an ordered list, tuple or integer array")
    if any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
           or not 0 <= seed < 2 ** 32 for seed in value):
        raise ValueError("seeds must contain integers between 0 and 2**32 - 1")
    result = [int(seed) for seed in value]
    if not result or len(set(result)) != len(result):
        raise ValueError("seeds must be nonempty and contain no duplicates")
    return result


def _fold_classes(y, folds):
    labels = np.asarray(y)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("y must be a nonempty one-dimensional label vector")
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise ValueError("y must not contain nonfinite labels")
    check_classification_targets(labels)
    result = []
    for index, fold in enumerate(folds):
        try:
            training, validation = fold
        except (TypeError, ValueError) as error:
            raise ValueError("each fold must contain training and validation indices") from error
        normalized = []
        for indices in (training, validation):
            array = np.asarray(indices)
            if array.ndim != 1 or array.dtype.kind not in "iu" or not len(array):
                raise ValueError("fold indices must be nonempty integer vectors")
            if (np.any(array < 0) or np.any(array >= len(labels))
                    or len(np.unique(array)) != len(array)):
                raise ValueError("fold indices must be unique and within y")
            normalized.append(array)
        if np.intersect1d(*normalized).size:
            raise ValueError("training and validation indices must be disjoint")
        try:
            class_count = len(np.unique(labels[normalized[0]]))
        except (TypeError, ValueError) as error:
            raise ValueError("training labels must have comparable class values") from error
        result.append({"fold_index": index, "training_rows": len(normalized[0]),
                       "validation_rows": len(normalized[1]),
                       "training_class_count": class_count})
    if not result:
        raise ValueError("at least one fold is required")
    return result


def _model_shape(estimator, *, validate_M=True):
    """Recognize exact known types only; no custom estimator methods are used."""
    terminal = estimator
    if type(terminal) is Pipeline:
        steps = terminal.steps
        if (not isinstance(steps, (list, tuple)) or not steps
                or any(not isinstance(step, (list, tuple)) or len(step) != 2
                       for step in steps)):
            return None, "unsupported pipeline structure"
        if any(type(step) is not StandardScaler for _, step in steps[:-1]):
            return None, "unsupported preprocessing; only exact StandardScaler is counted"
        terminal = steps[-1][1]
    ovo = type(terminal) is OvOClassifier
    binary = terminal.estimator if ovo else terminal
    if type(binary) not in _BINARY_TYPES:
        return None, "unsupported estimator type: " + _class_name(binary)
    kind = ("ensemble_dwd" if type(binary) is EnsembleKernelDWD else
            "ensemble_svm" if type(binary) is EnsembleKernelSVM else
            "full_dwd" if type(binary) is KernGDWD else "full_svm")
    ensemble = type(binary) in (EnsembleKernelDWD, EnsembleKernelSVM)
    M = binary.M if ensemble else None
    valid_M = (not ensemble or (not isinstance(M, (bool, np.bool_))
                               and isinstance(M, Integral) and M >= 1))
    if validate_M and not valid_M:
        return None, "invalid M; a positive integer is required"
    return {"kind": kind, "ovo": ovo, "ensemble": ensemble,
            "M": int(M) if ensemble and valid_M else None}, None


def _unsupported_replacement(parameters):
    # Unknown estimator values may customize cloning; do not invoke them merely
    # to prepare an optional report. Normal numeric/kernel parameters are read.
    pending = list(parameters.values())
    visited = set()
    while pending:
        value = pending.pop()
        if type(value) in (list, tuple, dict):
            if id(value) in visited:
                continue
            visited.add(id(value))
            pending.extend(value.values() if type(value) is dict else value)
            continue
        if isinstance(value, BaseEstimator) and type(value) is not StandardScaler:
            _, reason = _model_shape(value, validate_M=False)
            if reason is not None:
                return reason
    return None


def workload_plan(pipeline, candidates, y, folds, *, seeds=None, reuse=None):
    """Describe nominal cold learner counts without running any estimator.

    Malformed indices/seeds raise ValueError instead of silently changing the
    request. Unsupported estimators, invalid M and unsupported training class
    counts remain visible with unknown counts. All nominal pipeline evaluation
    slots are retained even when a candidate cannot be structurally counted.
    No reuse deduction is made: successful fits, valid recovery payloads and
    cache hits cannot be established from this input alone.
    """
    seeds = _seeds(seeds)
    seed_count = 1 if seeds is None else len(seeds)
    fold_info = _fold_classes(y, folds)
    candidates = list(candidates)
    _, original_reason = _model_shape(pipeline, validate_M=False)
    plans = []
    known_total = _counts()
    for index, parameters in enumerate(candidates):
        if type(parameters) is not dict or any(type(key) is not str for key in parameters):
            raise ValueError("each candidate must be a dictionary with string parameter names")
        reason = original_reason or _unsupported_replacement(parameters)
        model = None
        if reason is None:
            try:
                effective = clone(pipeline).set_params(**parameters)
                model, reason = _model_shape(effective)
            except Exception as error:
                reason = "candidate configuration could not be counted: " + type(error).__name__
        per_fold = []
        candidate_counts = _counts()
        reasons = [] if reason is None else [reason]
        for info in fold_info:
            fold_reason = reason
            class_count = info["training_class_count"]
            if fold_reason is None and (class_count < 2 or (not model["ovo"] and class_count != 2)):
                fold_reason = ("at least two training classes are required" if class_count < 2 else
                               "direct binary learner requires exactly two training classes")
            counts = _counts(None)
            if fold_reason is None:
                pairs = class_count * (class_count - 1) // 2 if model["ovo"] else 1
                binary_count = seed_count * pairs
                counts = {"binary_models": binary_count,
                          "base_learner_fits": binary_count * model["M"] if model["ensemble"] else 0,
                          "final_learner_fits": binary_count if model["ensemble"] else 0,
                          "full_learner_fits": 0 if model["ensemble"] else binary_count,
                          "total_learner_fits": binary_count * (model["M"] + 1) if model["ensemble"] else binary_count}
                for name, value in counts.items():
                    candidate_counts[name] += value
            elif fold_reason not in reasons:
                reasons.append(fold_reason)
            per_fold.append({**info, "status": "known" if fold_reason is None else "unknown",
                             "reason": fold_reason, "counts": counts})
        known = not reasons
        if known:
            for name, value in candidate_counts.items():
                known_total[name] += value
        plans.append({"candidate_index": index, "parameters": _display(parameters),
                      "parameter_display_is_checkpoint_identity": False,
                      "status": "known" if known else "unknown", "reasons": reasons,
                      "model": model, "folds": per_fold,
                      "counts": candidate_counts if known else _counts(None)})
    known_count = sum(plan["status"] == "known" for plan in plans)
    return {"schema": 1, "scope": "proposed_cv_stage",
            "basis": "nominal_cold_single_pass_before_reuse_or_failure",
            "interpretation": ("Requested learner fits assuming successful fitting, before checkpoint, "
                               "base-cache, equivalent-fold or completed-pair reuse. Counts are not "
                               "executed work, retries, CPU time, elapsed time or an ETA."),
            "candidate_count": len(plans), "fold_count": len(fold_info),
            "seed_count": seed_count, "seeds": seeds,
            "requested_pipeline_evaluations": len(plans) * len(fold_info) * seed_count,
            "reuse_options": _display(reuse), "reuse_deductions_applied": False,
            "known_candidate_count": known_count,
            "unknown_candidate_count": len(plans) - known_count,
            "totals": known_total.copy() if known_count == len(plans) else _counts(None),
            "known_candidate_subtotal": known_total, "candidates": plans}


def encode(value):
    """Describe exact grid literals; display precision never merges candidates."""
    if isinstance(value, np.generic):
        return {"numpy_dtype": str(value.dtype), "bytes_hex": value.tobytes().hex()}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return {"opaque_type": _class_name(value), "identity": parameter_hash(value),
            "reconstructable": False}


def grid_proposal(grids, seen, limit, candidates, *, scan_limit=2048):
    """Describe an already-decided grid proposal without changing its decision.

    ``candidates`` is the policy's authoritative result: None means its complete
    stage exceeded the candidate allowance. Extra enumeration is bounded by raw
    entries, including duplicates. The encoded generating grids and excluded
    identities remain proportional to the supplied input, not the scan limit.
    An incomplete scan gives a lower bound, never an exact refused-stage size.
    """
    if type(scan_limit) is not int or scan_limit < 1:
        raise ValueError("scan_limit must be a positive Python integer")
    definitions = [grids] if isinstance(grids, dict) else list(grids)
    generator = ParameterGrid(definitions)
    raw_count = sum(math.prod(len(values) for values in grid.values())
                    for grid in definitions)
    new, keys, scanned = [], set(), 0
    iterator = iter(generator)
    for _ in range(min(raw_count, scan_limit)):
        parameters = next(iterator)
        scanned += 1
        identity = parameter_hash(parameters)
        if identity in seen or identity in keys:
            continue
        keys.add(identity)
        new.append(parameters)
    complete = scanned == raw_count
    report = {
        "proposal_kind": "cartesian_grid_union_minus_seen",
        "generating_grids": encode(definitions),
        "generating_grids_identity": parameter_hash(definitions),
        "excluded_candidate_identities": sorted(seen),
        "remaining_candidate_allowance": limit,
        "raw_cartesian_entries": raw_count,
        "scanned_entries": scanned, "scan_limit": scan_limit,
        "count_status": "exact" if complete else "lower_bound",
        "new_candidate_count": len(new) if complete else None,
        "new_candidate_count_lower_bound": len(new),
        "candidate_list_complete": complete,
        "candidate_identities": [parameter_hash(parameters) for parameters in new],
        "candidate_parameters": encode(new),
        "candidate_cap_refused": candidates is None,
        "interpretation": ("Distinct exact identities only; no nearby-value merging. "
                           "An incomplete scan is not an exact refused-stage size."),
    }
    if candidates is None:
        # The authoritative policy already observed allowance + 1 new entries.
        report["new_candidate_count_lower_bound"] = max(len(new), limit + 1)
    else:
        report.update(count_status="exact", new_candidate_count=len(candidates),
                      new_candidate_count_lower_bound=len(candidates),
                      candidate_list_complete=True,
                      candidate_identities=[parameter_hash(item) for item in candidates],
                      candidate_parameters=encode(candidates))
    return report


def explicit_proposal(candidates, seen, limit, kind):
    """Count a materialized proposal, excluding only exact seen/duplicate keys."""
    new, keys = [], set()
    for candidate in candidates:
        identity = parameter_hash(candidate)
        if identity in seen or identity in keys:
            continue
        keys.add(identity)
        new.append(candidate)
    return {
        "proposal_kind": kind, "count_status": "exact",
        "new_candidate_count": len(new), "new_candidate_count_lower_bound": len(new),
        "candidate_list_complete": True,
        "candidate_parameters": encode(new),
        "candidate_identities": [parameter_hash(item) for item in new],
        "excluded_candidate_identities": sorted(seen),
        "remaining_candidate_allowance": limit,
        "candidate_cap_refused": len(new) > limit,
    }


def candidate_budget(maximum, completed, proposed, admitted):
    """Describe counts; an unknown refused proposal spends no candidate budget."""
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or value < 0 for value in (maximum, completed)):
        raise ValueError("candidate counts must be nonnegative integers")
    if type(admitted) is not bool:
        raise ValueError("admitted must be a boolean")
    if not (proposed is None and not admitted):
        if (isinstance(proposed, (bool, np.bool_)) or not isinstance(proposed, Integral)
                or proposed < 0):
            raise ValueError("proposed must be a nonnegative integer, or None if refused")
        proposed = int(proposed)
    maximum, completed = int(maximum), int(completed)
    allowance = maximum - completed
    return {
        "maximum_unique_candidates": maximum, "completed_unique_candidates": completed,
        "remaining_before": allowance, "proposed_count": proposed, "admitted": admitted,
        "remaining_after_admission": allowance - proposed if admitted else allowance,
        "incomplete_proposals_not_charged": True, "optimality_claim": False,
    }


def _failure_record(error):
    try:
        message = str(error)
    except Exception:
        message = "Error message could not be formatted"
    return {"reporting_status": "unavailable", "reporting_only": True,
            "error_type": type(error).__name__, "message": message}


def warn_reporting_failure(error):
    """A warning filter or failed warning sink cannot interrupt a search."""
    try:
        warnings.warn("Supplementary search reporting failed: "
                      + _failure_record(error)["message"], RuntimeWarning, stacklevel=2)
    except Exception:
        pass


def safe_report(factory):
    """Return supplementary metadata or an error record; preserve interruptions."""
    try:
        return factory()
    except Exception as error:
        return _failure_record(error)


def publish_report(path, factory, *, announce=False):
    """Build and replace supplementary JSON without making it authoritative.

    The destination directory must already exist. Ordinary build, I/O, warning
    and announcement failures do not change a search decision. KeyboardInterrupt
    and other BaseException interruptions still propagate.
    """
    try:
        report = factory()
    except Exception as error:
        report = _failure_record(error)
        warn_reporting_failure(error)
    # Existing supplementary output can suppress a repeated notice on replay.
    # It has no authority over fitting, checkpoint identity or search decisions.
    notice = None
    if announce and isinstance(report, dict) and isinstance(report.get('near_coordinates'), dict):
        try:
            coordinates = report['near_coordinates']
            notice = console_notice(coordinates)
            if notice is not None:
                prior_path = _io_path(path)
                if prior_path.is_file() and prior_path.stat().st_size <= 8 * 1024 * 1024:
                    previous = json.loads(prior_path.read_text(encoding='utf-8'))
                    prior_coordinates = previous.get('near_coordinates', {})
                    if (coordinates.get('input_identity') is not None
                            and prior_coordinates.get('input_identity') == coordinates['input_identity']):
                        notice = None
        except Exception:
            # A missing, stale or malformed report only permits another notice.
            pass
    try:
        write_json_atomic(path, report, replace=True)
    except Exception as error:
        warn_reporting_failure(error)
        return _failure_record(error)
    if announce and isinstance(report, dict):
        try:
            parts = []
            plan = report.get("workload_plan")
            if isinstance(plan, dict) and "candidate_count" in plan:
                counts = plan.get("totals", {})
                total = counts.get("total_learner_fits")
                parts.append(f"{plan['candidate_count']} candidates, "
                             f"{plan['fold_count']} folds, {plan['seed_count']} seeds; "
                             f"nominal cold learner fits: {total if total is not None else 'unknown'}")
                names = ("base_learner_fits", "final_learner_fits", "full_learner_fits")
                if all(counts.get(name) is not None for name in names):
                    parts.append(f"base/final/full learner fits: {counts[names[0]]}/"
                                 f"{counts[names[1]]}/{counts[names[2]]}")
            budget = report.get("budget")
            if isinstance(budget, dict) and "remaining_before" in budget:
                proposed = budget.get("proposed_count")
                parts.append(f"proposed candidates: {proposed if proposed is not None else 'unknown'}; "
                             f"candidate allowance before stage: {budget['remaining_before']}; "
                             f"after admission: {budget['remaining_after_admission']}")
            if report.get("stopping_reason"):
                parts.append(f"stopping reason: {report['stopping_reason']}")
            if parts:
                print("Proposed CV stage: " + ". ".join(parts) + f". Report: {path}", flush=True)
            if notice is not None:
                print(notice + f" Report: {path}", flush=True)
        except Exception as error:
            warn_reporting_failure(error)
    return report
