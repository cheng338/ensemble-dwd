"""Read-only evaluation summaries; none of these values selects a candidate.

Use the predictions already obtained for ordinary accuracy. Numerical counters
describe observed learner records, not a replacement for measured pipeline time.
"""
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import chain, repeat
import json
import math
from numbers import Real
from operator import is_
from sys import modules

import numpy as np

from dwd._validation import (_PLAIN_REAL_TYPES, _SelectedTemporalObservations, _custom_comparison_types,
                             _is_boxed_nat, _is_temporal_nat, _label_source_and_target, _numeric_abc_instance,
                             _original_label_vector, _pandas_target_array, _python_label, _sparse_text_fill,
                             _temporal_label_key, _unbox_numeric_array, exact_label_matches, exact_sparse_uint64_target)
from ensemble_dwd._validation import _class_mask, _label_value


def _report_input(value):
    if type(value) is np.ndarray:
        if value.dtype.kind != 'O':
            return value, value, None
        if value.ndim == 1 and value.size and isinstance(value[0], (str, bytes)):
            families = _label_families(value)
            if families <= {'bytes', 'text'}:
                return value, value, families
    source, target = _label_source_and_target(value, 'score')
    if isinstance(target, _SelectedTemporalObservations) and source is not target:
        # Scoring can replace boxed times with native ticks, even when no
        # shared unit exists and the result still has object dtype.
        pd = modules.get('pandas')
        storage = _pandas_target_array(source)
        categorical = pd is not None and isinstance(storage, pd.Categorical)
        levels = storage.categories._values if categorical else storage
        if storage is None or isinstance(levels, pd.arrays.SparseArray) and levels.dtype.subtype.kind == 'O':
            # A custom projection has already supplied its selected source.
            # Ordinary sparse categories supply only their occupied levels.
            rows = _original_label_vector(source)
            if categorical:
                rows[storage._codes == -1] = pd.NaT
            target = rows.reshape(target.shape)
    return source, np.asarray(target), None


def _label_families(values):
    kind = values.dtype.kind
    if kind != 'O':
        if kind in 'UT':
            return {'text'}
        if kind == 'S':
            return {'bytes'}
        if kind in 'Mm':
            return {'native time'}
        return {'number'} if kind in 'biuf' else {'other'}
    # Boxed Python strings are common. A type-identity scan avoids running
    # scalar hooks or building the same family set for every row.
    if values.size and type(values[0]) is str and all(map(is_, map(type, values), repeat(str))):
        return {'text'}
    families = set()
    temporal_types = {id(date), id(datetime), id(timedelta)}
    pd = modules.get('pandas')
    if pd is not None:
        temporal_types.update((id(pd.Timestamp), id(pd.Timedelta), id(type(pd.NaT))))
    for value in values:
        if value is None:
            families.add('none')
        elif isinstance(value, str):
            families.add('text')
        elif isinstance(value, bytes):
            families.add('bytes')
        elif type(value) is np.datetime64 or type(value) is np.timedelta64:
            families.add('native time')
        elif id(type(value)) in temporal_types:
            families.add('Python time')
        elif isinstance(value, (date, timedelta, np.datetime64, np.timedelta64)):
            # Keep a temporal subclass's public equality and ordering.
            families.add('custom time')
        elif isinstance(value, (np.bool_, Decimal)) or _numeric_abc_instance(value, Real):
            families.add('number')
        else:
            families.add('other')
    return families


def _mixed_report_classes(actual, predicted, *, compare=exact_label_matches, identify=None):
    """Group unlike label families by scalar equality, in first-observed order."""
    values = list(map(_unbox_numeric_array, [*actual, *predicted]))
    codes = np.full(len(values), -1, dtype=np.intp)
    labels, witnesses = [], []
    custom = _custom_comparison_types(values, ()) if identify is None else ()
    text_cache = {}
    conflict = 'Mixed labels compare inconsistently and do not define disjoint report classes.'
    for i, value in enumerate(values):
        group = codes[i]
        kind = id(type(value))
        # Active comparators may distinguish values of the same scalar type.
        if identify is None:
            witness = id(value) if kind in custom and not _sparse_text_fill(value, 'score', text_cache) else kind
        else:
            witness = identify(value)
        if group >= 0 and witness in witnesses[group]:
            continue
        # In particular a bytes subclass can equal a NumPy integer but differ
        # from the equal-valued Python int. Check each represented scalar type
        # within a class against the full observed membership of that class.
        matches = compare(values, [value])
        if not matches[i] or not np.array_equal(matches, compare([value] * len(values), values)):
            raise ValueError(conflict)
        if group < 0:
            if np.any(codes[matches] >= 0):
                raise ValueError(conflict)
            group = len(labels)
            labels.append(value)
            witnesses.append(set())
            codes[matches] = group
        elif not np.array_equal(matches, codes == group):
            raise ValueError(conflict)
        witnesses[group].add(witness)
    truth, guesses = codes[:len(actual)], codes[len(actual):]
    if not np.array_equal(compare(actual, predicted), truth == guesses):
        raise ValueError(conflict)
    return labels, truth, guesses


def _temporal_report_classes(actual, predicted, families):
    """Encode exact instants or durations before comparing or counting rows."""
    pd = modules.get('pandas')
    display_families = families
    if actual.dtype.kind in 'Mm' and predicted.dtype.kind in 'Mm' and actual.dtype.kind != predicted.dtype.kind:
        raise ValueError('Temporal observations and predictions have incompatible units.')
    if actual.dtype == predicted.dtype and actual.dtype.kind in 'Mm':
        labels, codes = np.unique(np.concatenate((actual, predicted)), return_inverse=True)
    elif families == {'native time'}:
        # Different native units can round or overflow on promotion. Use the
        # same arbitrary-precision instant/duration keys as estimator scoring.
        representatives = {}
        keys = []
        for value in chain(actual, predicted):
            key = None if _is_temporal_nat(value, pd) else _temporal_label_key(value, pd)
            keys.append(key)
            representatives.setdefault(key, value)
        ordered = sorted(key for key in representatives if key is not None)
        if ordered and ordered[0][0] != ordered[-1][0]:
            raise ValueError('Temporal observations and predictions have incompatible units.')
        if None in representatives:
            ordered.append(None)
        labels = [representatives[key] for key in ordered]
        positions = {key: i for i, key in enumerate(ordered)}
        codes = np.fromiter(map(positions.__getitem__, keys), dtype=np.intp, count=len(keys))
    else:
        labels, codes, missing_class, display_families = _object_temporal_classes(actual, predicted, families, pd)
    if families == {'native time'}:
        missing_class = len(labels) - 1 if _is_temporal_nat(labels[-1], pd) else None
    displayed = _temporal_report_labels(labels, display_families, missing_class)
    return displayed, codes[:len(actual)], codes[len(actual):], missing_class


def _report_time_key(value, pd):
    # A Python date denotes a civil day when compared to a native instant.
    return _temporal_label_key(np.datetime64(value, 'D') if type(value) is date else value, pd)


def _object_temporal_classes(actual, predicted, families, pd):
    left, right = list(actual), list(predicted)
    rows = left + right
    first = rows[0]
    if families == {'Python time'} and not _is_temporal_nat(first, pd) and all(
            type(v) is type(first) and getattr(v, 'tzinfo', None) is getattr(first, 'tzinfo', None)
            and not _is_temporal_nat(v, pd) for v in rows):
        # One concrete scalar type and timezone has a common equality/hash
        # contract. Python and pandas can disagree within a repeated hour.
        labels = sorted(dict.fromkeys(rows))
        positions = {v: i for i, v in enumerate(labels)}
        codes = np.fromiter(map(positions.__getitem__, rows), dtype=np.intp, count=len(rows))
        return labels, codes, None, families
    # Dates denote civil days when meeting a native instant. Python's own
    # date/datetime equality still distinguishes the two Python types.
    plain = {id(date), id(datetime), id(timedelta), id(np.datetime64), id(np.timedelta64)}
    if pd is not None:
        plain.update((id(pd.Timestamp), id(pd.Timedelta), id(type(pd.NaT))))
    domains = set()
    keys = {}
    for value in rows:
        if id(value) in keys:
            continue
        if _is_temporal_nat(value, pd):
            # A custom scalar can define a reflexive class of its own even
            # with NaT storage. Nonreflexive NaTs all share the missing class.
            if _is_boxed_nat(value, pd):
                keys[id(value)] = None
        elif id(type(value)) in plain:
            key = _report_time_key(value, pd)
            keys[id(value)] = key
            domains.add(key[0])
    if len(domains) > 1:
        raise ValueError('Temporal observations and predictions have incompatible units.')

    def native(value):
        return type(value) is np.datetime64 or type(value) is np.timedelta64

    def missing(value):
        return id(value) in keys and keys[id(value)] is None

    def equal(a, b):
        if missing(a) or missing(b):
            return missing(a) and missing(b)
        if id(a) in keys and id(b) in keys and (native(a) or native(b)):
            return keys[id(a)] == keys[id(b)]
        return a == b

    def compare(a, b):
        others = repeat(b[0]) if len(b) == 1 else b
        return np.fromiter((equal(x, y) for x, y in zip(a, others)), dtype=bool, count=len(a))

    def identity(value):
        if missing(value):
            return None
        if id(value) not in keys:
            return id(value)
        if native(value):
            return (value.dtype, int(value.view('i8')))
        # Equal civil times in a repeated hour can have different behavior
        # against another timezone. Keep these comparison witnesses distinct.
        return (id(type(value)), value, getattr(value, 'fold', 0), id(getattr(value, 'tzinfo', None)))

    labels, truth, guesses = _mixed_report_classes(left, right, compare=compare, identify=identity)
    order = [i for i, v in enumerate(labels) if not missing(v)]
    missing_class = len(order) if len(order) < len(labels) else None
    display_families = families
    if missing_class is not None:
        # Ignore the missing marker's type. Keep types of all finite values,
        # including aliases whose class representative has a different type.
        samples = {id(type(v)): v for v in rows if not missing(v)}
        display_families = _label_families(np.fromiter(samples.values(), dtype=object, count=len(samples)))
    if all(id(labels[i]) in keys for i in order):
        order.sort(key=lambda i: keys[id(labels[i])])
    elif 'custom time' in display_families and display_families <= {'Python time', 'custom time'}:
        order.sort(key=labels.__getitem__)
    order.extend(i for i, v in enumerate(labels) if missing(v))
    codes = np.argsort(order)[np.concatenate((truth, guesses))]
    return [labels[i] for i in order], codes, missing_class, display_families


def _temporal_report_label(value):
    if isinstance(value, (np.datetime64, np.timedelta64)):
        # .item() discards units for sub-microsecond and out-of-range labels.
        return {'dtype': str(value.dtype), 'ticks': int(value.view('i8'))}
    if isinstance(value, date):
        return value.isoformat()
    return str(value) if isinstance(value, timedelta) else _label_value(value)


def _temporal_report_labels(values, families, missing_class):
    def temporal_value(i):
        return None if i == missing_class else _temporal_report_label(values[i])

    if 'custom time' in families and 'native time' not in families:
        # Preserve the representative of a user-defined class where the
        # workflow can publish it distinctly.
        labels = [None if i == missing_class else _label_value(v) for i, v in enumerate(values)]
    else:
        labels = [temporal_value(i) for i in range(len(values))]
    if families == {'native time'} or len(labels) < 2:
        return labels
    from search_support import json_safe

    def encoded(value):
        return json.dumps(json_safe(value), sort_keys=True, allow_nan=False)

    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(encoded(label), []).append(i)
    collisions = [indices for indices in groups.values() if len(indices) > 1]
    for indices in collisions:
        for i in indices:
            value = values[i]
            labels[i] = {'type': type(value).__module__ + '.' + type(value).__qualname__,
                         'value': temporal_value(i)}
    if collisions and len(set(map(encoded, labels))) != len(labels):
        raise ValueError('Distinct temporal classes cannot be represented by distinct published report labels.')
    return labels


def classification_report(actual, predicted):
    """Compute secondary metrics from one existing set of class predictions.

    Balanced accuracy is the mean recall over classes present in ``actual``.
    A predicted-only class has no observed recall and is recorded with null.
    Mixed numeric, byte and text classes have no shared sorting order and
    are listed in their first-observed order, starting with the actual labels.
    Native temporal units are compared exactly. Python temporal objects keep
    their scalar equality, including daylight-saving folds. Mixed custom and
    native time labels must define disjoint classes. When both occur among the
    nonmissing observations or predictions, the report sorts by exact time if
    every chosen representative has a time key. Otherwise it uses first-observed
    order. Python-only time labels can retain custom scalar ordering. Missing
    times come last and receive zero credit. Matching Python None labels receive credit.
    NumPy temporal labels retain their dtype and tick count. If two classes
    have identical JSON labels even after including their type and temporal
    value, the report raises ValueError.
    """
    same_input = actual is predicted
    actual_input = _report_input(actual)
    actual_source, actual, actual_families = actual_input
    predicted_source, predicted, predicted_families = actual_input if same_input else _report_input(predicted)
    if actual.ndim != 1 or actual.shape != predicted.shape or not actual.size:
        raise ValueError('Classification reporting requires equal nonempty 1D labels.')
    families = (actual_families or _label_families(actual)) | (predicted_families or _label_families(predicted))
    missing_class = None
    temporal = {'native time', 'Python time', 'custom time'}
    custom_time = bool(families & {'custom time', 'other', 'none'})
    if (families & temporal and families <= temporal | {'none'}
            or 'native time' in families and custom_time):
        labels, actual, predicted, missing_class = _temporal_report_classes(actual, predicted, families)
        class_operands = np.arange(len(labels))
    elif len(families) > 1 and families <= {'text', 'bytes', 'number'}:
        truth = _original_label_vector(actual_source)
        guesses = truth if same_input else _original_label_vector(predicted_source)
        labels, actual, predicted = _mixed_report_classes(truth, guesses)
        class_operands = np.arange(len(labels))
    else:
        if families & temporal and families - temporal and not custom_time:
            raise ValueError('Temporal reporting requires temporal labels on both sides.')
        class_operands, actual, predicted = _ordered_report_classes(actual, predicted)
        labels = class_operands
    rows = []
    for i, label in enumerate(labels):
        representative = class_operands[i:i + 1]
        actual_mask = actual == representative
        predicted_mask = predicted == representative
        support = int(np.count_nonzero(actual_mask))
        true_positive = 0 if i == missing_class else int(np.count_nonzero(actual_mask & predicted_mask))
        rows.append({'label': _label_value(label), 'support': support,
                     'predicted_count': int(np.count_nonzero(predicted_mask)),
                     'correct': true_positive,
                     'recall': true_positive / support if support else None})
    matches = actual == predicted
    if missing_class is not None:
        matches &= actual != missing_class
    return {'schema': 1, 'n_samples': len(actual),
            'accuracy': float(np.mean(matches)),
            'balanced_accuracy': float(np.mean([row['recall'] for row in rows if row['support']])),
            'per_class': rows, 'role': 'descriptive_only; ordinary accuracy selects candidates'}


def _ordered_report_classes(actual, predicted):
    dtype = np.result_type(actual.dtype, predicted.dtype)
    if dtype.kind == 'f' and any(v.dtype.kind in 'iu' and
            np.iinfo(v.dtype).bits > np.finfo(dtype).nmant + 1 for v in (actual, predicted)):
        # The common float cannot represent every integer in the source dtype.
        dtype = object
    joined = np.concatenate((actual, predicted), dtype=dtype)
    if (joined.dtype.kind == 'O' and actual.dtype.kind in 'biufO' and predicted.dtype.kind in 'biufO'
            and all(id(type(v)) in _PLAIN_REAL_TYPES for v in joined)):
        # Even inside an object array, NumPy scalars can compare after float
        # promotion. Use exact numeric operands for both the union and counts.
        # Leave custom comparison operands intact.
        joined = np.fromiter(map(_python_label, joined), dtype=object, count=len(joined))
        actual, predicted = joined[:len(actual)], joined[len(actual):]
    return np.unique(joined), actual, predicted


def selection_composition(diagnostics, y):
    """Compare training and selected class mixtures using recorded row counts.

    OvO diagnostics use the estimator's binary remapping: the first pair label
    maps to +1 and the second to -1. Both original and encoded labels are kept.
    Missing counts remain unavailable; absent classes count as zero only when
    the complete recorded selected counts agree with the recorded set size.
    """
    facts = diagnostics if isinstance(diagnostics, Mapping) else {}
    labels = np.asarray(exact_sparse_uint64_target(y))
    if labels.ndim != 1 or not labels.size:
        return {'available': False, 'reason': 'training labels unavailable'}
    if 'pairwise_estimators' in facts:
        pairs = facts['pairwise_estimators']
        if not isinstance(pairs, list):
            return {'available': False, 'reason': 'pair diagnostics unavailable'}
        result = []
        for row in pairs:
            pair = row.get('pair') if isinstance(row, Mapping) else None
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                result.append({'available': False, 'reason': 'pair labels unavailable'})
                continue
            first = _class_mask(labels, pair[0])
            included = first | _class_mask(labels, pair[1])
            pair_y = np.where(first[included], 1, -1)
            summary = selection_composition(row.get('diagnostics'), pair_y)
            summary.update(pair=list(pair), label_mapping=[
                {'encoded': 1, 'original': pair[0]}, {'encoded': -1, 'original': pair[1]}])
            result.append(summary)
        return {'pairwise': result}
    selected = facts.get('selected_rows')
    if not isinstance(selected, Mapping) or not isinstance(selected.get('class_counts'), list):
        return {'available': False, 'reason': 'selected class counts unavailable'}
    size = selected.get('size')
    if type(size) is not int or size < 0 or size > len(labels):
        return {'available': False, 'reason': 'selected size unavailable or invalid'}
    classes, counts = np.unique(labels, return_counts=True)
    recorded = selected['class_counts']
    if any(not isinstance(row, Mapping) or type(row.get('count')) is not int
           or row['count'] < 0 or 'label' not in row for row in recorded):
        return {'available': False, 'reason': 'selected class counts invalid'}
    selected_counts = []
    used = 0
    for label, count in zip(classes, counts):
        matches = [row['count'] for row in recorded if row['label'] == _label_value(label)]
        if len(matches) > 1:
            return {'available': False, 'reason': 'duplicate selected class labels'}
        selected_count = matches[0] if matches else 0
        if selected_count > count:
            return {'available': False, 'reason': 'selected count exceeds training count'}
        used += len(matches)
        selected_counts.append(selected_count)
    if used != len(recorded) or sum(selected_counts) != size:
        return {'available': False, 'reason': 'selected class counts do not cover the selected set'}
    return {'available': True, 'training_size': len(labels), 'selected_size': size,
            'selection_ratio_realized': size / len(labels),
            'classes': [{'label': _label_value(label), 'training_count': int(count),
                         'training_fraction': int(count) / len(labels),
                         'selected_count': retained,
                         'selected_fraction': retained / size if size else None,
                         'class_retention_fraction': retained / int(count)}
                        for label, count, retained in zip(classes, counts, selected_counts)],
            'interpretation': 'Observed selection composition; no class balancing or ratio adjustment is performed.'}


CHECK_FIELDS = ('native_refinement_trials', 'native_refinement_discarded',
                'native_refinement_acceptances', 'native_refinement_seconds',
                'compensated_residual_checks', 'compensated_residual_seconds',
                'native_residual_checks', 'native_residual_seconds',
                'adaptive_residual_probes', 'adaptive_residual_probe_errors',
                'adaptive_residual_probe_invalidations', 'adaptive_residual_probe_acceptances',
                'adaptive_native_trials_skipped', 'adaptive_residual_probe_fallbacks',
                'adaptive_residual_probe_reuses', 'adaptive_residual_history_resets')
CHECK_SUBTOTAL = 'refinement_and_compensated_seconds'


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    try:
        return _label_value(value) if math.isfinite(value) and value >= 0 else None
    except (OverflowError, TypeError, ValueError):
        return None


def _learner(node, path, origin):
    node = node if isinstance(node, Mapping) else {}
    solver = node.get('solver_diagnostics', node.get('diagnostics', {}))
    linear = solver.get('linear_system_diagnostics', {}) if isinstance(solver, Mapping) else {}
    linear = linear if isinstance(linear, Mapping) else {}
    values = {field: _number(linear.get(field)) for field in CHECK_FIELDS}
    first, second = values['native_refinement_seconds'], values['compensated_residual_seconds']
    values[CHECK_SUBTOTAL] = first + second if first is not None and second is not None else None
    return {'learner': path, 'work_origin': origin, 'values': values}


def _learners(facts, path='model', origin='current_fit'):
    if not isinstance(facts, Mapping):
        return [_learner(None, path, origin)], True
    if 'pairwise_estimators' in facts:
        pairs = facts['pairwise_estimators']
        if not isinstance(pairs, list) or not pairs:
            return [], True
        rows, incomplete = [], False
        for index, pair in enumerate(pairs):
            pair = pair if isinstance(pair, Mapping) else {}
            pair_origin = 'historical_reference' if pair.get('work_origin') == 'recovered' else origin
            children, missing = _learners(pair.get('diagnostics'), f'{path}/pair-{index}', pair_origin)
            rows.extend(children)
            incomplete |= missing
        return rows, incomplete
    ensemble = any(name in facts for name in ('base_diagnostics_', 'subset_sizes', 'selected_rows', 'final_estimator'))
    if not ensemble:
        return [_learner(facts, path, origin)], False
    bases = facts.get('base_diagnostics_')
    bases = bases if isinstance(bases, list) else []
    sizes = facts.get('subset_sizes')
    expected = len(sizes) if isinstance(sizes, list) else len(bases)
    incomplete = not expected or len(bases) != expected or 'final_estimator' not in facts
    expected = max(expected, len(bases))
    base_origin = origin
    if origin == 'current_fit':
        flag, status = facts.get('cache_hit_'), facts.get('cache_status_')
        if flag is True and (status is None or status == 'hit'):
            base_origin = 'historical_reference'
        elif flag is not False or status == 'hit':
            base_origin = 'unknown_origin'
    rows = [_learner(bases[index] if index < len(bases) else None,
                     f'{path}/base-{index}', base_origin) for index in range(expected)]
    rows.append(_learner(facts.get('final_estimator'), f'{path}/final', origin))
    return rows, incomplete


def _totals(rows):
    result = {}
    for field in (*CHECK_FIELDS, CHECK_SUBTOTAL):
        known = [row['values'][field] for row in rows if row['values'][field] is not None]
        result[field] = {'sum': sum(known) if known else None,
                         'reported_learners': len(known), 'unknown_learners': len(rows) - len(known)}
    return result


def numerical_check_summary(diagnostics, *, historical=False):
    """Visit every base and final learner exactly once, retaining work origin.

    ``native_residual_seconds`` can overlap compensated residual time and is
    never added to the refinement-plus-compensated subtotal. The adaptive probe
    timer also overlaps compensated checks and is excluded from this summary.
    Historical values
    are references, possibly repeated across cache hits, not new or unique work.
    Missing counters/timers stay null and carry explicit coverage counts.
    """
    rows, incomplete = _learners(diagnostics, origin='historical_reference' if historical else 'current_fit')
    groups = {}
    for origin in ('current_fit', 'historical_reference', 'unknown_origin'):
        children = [row for row in rows if row['work_origin'] == origin]
        groups[origin] = {'learner_records': len(children), 'fields': _totals(children)}
    return {'schema': 1, 'learners': rows, 'learner_inventory_incomplete': incomplete,
            'by_origin': groups,
            'interpretation': 'Diagnostic counters are observed records. Missing values are unknown. Historical references may repeat cached fits. Timers are components of fitting, not additional wall time; native residual time is not added to compensated residual time.'}
