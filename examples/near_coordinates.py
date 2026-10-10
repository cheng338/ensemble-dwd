"""Fit-free notices for close, distinct search coordinates.

The threshold describes represented binary64 values, not equivalent models.
No candidate is rounded, merged, replaced, skipped or given a new identity.
Historical pairs are excluded so a later stage does not repeat their notices.
"""
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Integral, Real

from joblib import hash as parameter_hash
import numpy as np


THRESHOLD = Fraction(1, 10**14)


def _limit(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(name + " must be a nonnegative integer.")
    return int(value)


def _positive_float(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("A reported coordinate must be a finite positive binary64 real number.")
    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError("A reported coordinate must be a finite positive binary64 real number.") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError("A reported coordinate must be a finite positive binary64 real number.")
    if isinstance(value, Integral):
        exact = Fraction(int(value))
    elif hasattr(value, "as_integer_ratio"):
        exact = Fraction(*value.as_integer_ratio())
    else:
        exact = Fraction(value)
    if exact != Fraction(number):
        raise ValueError("A reported coordinate must be exactly representable as binary64.")
    return number


def close_distinct(first, second):
    """Exact observation test: 0 < (upper-lower)/upper <= 1e-14."""
    lower, upper = sorted((_positive_float(first), _positive_float(second)))
    return lower != upper and Fraction(upper) - Fraction(lower) <= THRESHOLD * Fraction(upper)


def axis_report(values, max_details=6, *, previous=()):
    """Count new close unique-coordinate pairs with a sorted sliding window.

    The union of current and historical coordinates is inspected. A pair whose
    two coordinates already occur in previous is excluded. Dense groups are
    counted without enumerating every pair; only bounded details are sampled.
    """
    max_details = _limit(max_details, "max_details")
    old = {_positive_float(value) for value in previous}
    current = {_positive_float(value) for value in values}
    ordered = sorted(current | old)
    exact = [Fraction(value) for value in ordered]
    new_positions = [index for index, value in enumerate(ordered) if value not in old]
    old_prefix = [0]
    for value in ordered:
        old_prefix.append(old_prefix[-1] + int(value in old))
    start, count, details = 0, 0, []
    for end, upper in enumerate(exact):
        while start < end and upper - exact[start] > THRESHOLD * upper:
            start += 1
        old_upper = ordered[end] in old
        count += end - start - (old_prefix[end] - old_prefix[start] if old_upper else 0)
        remaining = max_details - len(details)
        if not remaining:
            continue
        if old_upper:
            first_new = bisect_left(new_positions, start)
            last_new = bisect_left(new_positions, end)
            lower_indices = new_positions[first_new:min(last_new, first_new + remaining)]
        else:
            lower_indices = range(start, min(end, start + remaining))
        for index in lower_indices:
            lower, higher = ordered[index], ordered[end]
            details.append({"lower": lower, "upper": higher,
                            "lower_hex": lower.hex(), "upper_hex": higher.hex(),
                            "relative_separation": float((upper - exact[index]) / upper)})
    return {"unique_coordinate_count": len(ordered),
            "previous_unique_coordinate_count": len(old),
            "new_unique_coordinate_count": len(current - old),
            "close_distinct_pair_count": count, "details": details,
            "details_omitted": count - len(details)}


def _whole_candidate_report(current, previous, axes, max_details, max_pair_checks):
    groups = defaultdict(dict)
    old_keys = set()
    for old, rows in ((True, previous), (False, current)):
        for row in rows:
            key = parameter_hash(row)
            context = parameter_hash({name: value for name, value in row.items() if name not in axes})
            groups[context][key] = row
            if old:
                old_keys.add(key)
    separated = []
    needed = 0
    for context in sorted(groups):
        items = sorted(groups[context].items())
        old = [(key, row) for key, row in items if key in old_keys]
        new = [(key, row) for key, row in items if key not in old_keys]
        needed += len(old) * len(new) + len(new) * (len(new) - 1) // 2
        separated.append((old, new))
    result = {"potential_pair_checks": needed, "pair_check_limit": max_pair_checks,
              "pair_scope": "distinct candidate identities with matching non-axis parameter hashes; excludes previous-previous pairs"}
    if needed > max_pair_checks:
        return {**result, "status": "not_checked_pair_limit", "close_distinct_pair_count": None,
                "details": [], "details_omitted": None}
    count, details = 0, []

    def inspect(first_key, first, second_key, second):
        nonlocal count
        differing = [axis for axis in axes if float(first[axis]) != float(second[axis])]
        if not differing or not all(close_distinct(first[axis], second[axis]) for axis in differing):
            return
        count += 1
        if len(details) < max_details:
            details.append({"first_candidate_hash": first_key, "second_candidate_hash": second_key,
                            "differing_axes": differing,
                            "first_coordinates": {axis: float(first[axis]) for axis in axes},
                            "second_coordinates": {axis: float(second[axis]) for axis in axes},
                            "first_coordinates_hex": {axis: float(first[axis]).hex() for axis in axes},
                            "second_coordinates_hex": {axis: float(second[axis]).hex() for axis in axes}})

    for old, new in separated:
        for index, (first_key, first) in enumerate(new):
            for second_key, second in old:
                inspect(first_key, first, second_key, second)
            for second_key, second in new[index + 1:]:
                inspect(first_key, first, second_key, second)
    return {**result, "status": "complete", "close_distinct_pair_count": count,
            "details": details, "details_omitted": count - len(details)}


def disclose(candidates, axes, *, previous=(), max_details=6, max_pair_checks=250000):
    """Observe only close pairs introduced by candidates relative to previous.

    Axis counts describe unique represented coordinate pairs. Conditional
    counts restrict these pairs to identical hashes of every other parameter.
    Whole-candidate counts instead use distinct original candidate identities,
    and allow several close axes to differ. Typed candidate identities remain
    separate even when their represented coordinate values are the same.
    All counts are observations, never evidence authorizing reuse of a fit.
    """
    max_details = _limit(max_details, "max_details")
    max_pair_checks = _limit(max_pair_checks, "max_pair_checks")
    if isinstance(axes, str):
        raise ValueError("axes must be an ordered collection of unique parameter names.")
    axes = tuple(axes)
    if any(not isinstance(axis, str) or not axis for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError("axes must contain unique nonempty parameter names.")
    current, history = list(candidates), list(previous)
    for row in current + history:
        if not isinstance(row, Mapping):
            raise ValueError("Candidates must be parameter mappings.")
        for axis in axes:
            if axis not in row:
                raise ValueError("Each reported candidate must contain axis " + axis + ".")
            _positive_float(row[axis])
    current_keys = {parameter_hash(row) for row in current}
    previous_keys = {parameter_hash(row) for row in history}
    result = {"threshold": "1e-14", "relative_separation_definition": "(upper-lower)/upper",
              "input_identity": parameter_hash((axes, history, current, max_details, max_pair_checks)),
              "threshold_role": "observation only; not equivalence or permission for reuse",
              "pair_scope": "newly introduced pairs; previous-previous pairs excluded",
              "candidate_count": len(current), "candidate_unique_identity_count": len(current_keys),
              "previous_candidate_count": len(history),
              "previous_candidate_unique_identity_count": len(previous_keys),
              "new_candidate_unique_identity_count": len(current_keys - previous_keys),
              "axes": {}, "conditional_pair_count": 0,
              "conditional_pair_scope": "unique represented coordinate pairs within identical hashes of all other parameters; excludes previously present pairs",
              "source_ancestry": "unknown unless supplied separately; close user inputs may be intentional"}
    for axis in axes:
        old_groups, current_groups = defaultdict(list), defaultdict(list)
        for rows, groups in ((history, old_groups), (current, current_groups)):
            for row in rows:
                context = parameter_hash({name: item for name, item in row.items() if name != axis})
                groups[context].append(row[axis])
        facts = axis_report((row[axis] for row in current), max_details,
                            previous=(row[axis] for row in history))
        conditional_count, conditional_details = 0, []
        for context in sorted(current_groups):
            conditional = axis_report(current_groups[context], max_details - len(conditional_details),
                                      previous=old_groups.get(context, ()))
            conditional_count += conditional["close_distinct_pair_count"]
            conditional_details.extend({"other_parameters_hash": context, **item}
                                       for item in conditional["details"])
        facts.update({"conditional_pair_count": conditional_count,
                      "conditional_details": conditional_details,
                      "conditional_details_omitted": conditional_count - len(conditional_details)})
        result["axes"][axis] = facts
        result["conditional_pair_count"] += conditional_count
    result["whole_candidates"] = _whole_candidate_report(current, history, axes, max_details, max_pair_checks)
    return result


def console_notice(report):
    """A brief notice only when close pairs were found; publication is external."""
    axis_count = sum(item["close_distinct_pair_count"] for item in report["axes"].values())
    conditional = report["conditional_pair_count"]
    whole = report["whole_candidates"]["close_distinct_pair_count"]
    if not axis_count and not conditional and not whole:
        return None
    whole_text = "unknown (comparison limit)" if whole is None else str(whole)
    return ("Close coordinates: " + str(axis_count) + " new pairs across axes; "
            + str(conditional) + " within fixed contexts; " + whole_text + " whole-candidate pairs. "
            "Candidates remain distinct; closeness does not establish equivalent fits.")
