"""Read-only evaluation summaries; none of these values selects a candidate.

Use the predictions already obtained for ordinary accuracy. Numerical counters
describe observed learner records, not a replacement for measured pipeline time.
"""
from collections.abc import Mapping
import math
from numbers import Real

import numpy as np


def _label(value):
    return value.item() if isinstance(value, np.generic) else value


def classification_report(actual, predicted):
    """Compute secondary metrics from one existing set of class predictions.

    Balanced accuracy is the mean recall over classes present in ``actual``.
    A predicted-only class has no observed recall and is recorded with null.
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or actual.shape != predicted.shape or not actual.size:
        raise ValueError('Classification reporting requires equal nonempty 1D labels.')
    labels = np.unique(np.concatenate((actual, predicted)))
    rows = []
    for label in labels:
        support = int(np.count_nonzero(actual == label))
        true_positive = int(np.count_nonzero((actual == label) & (predicted == label)))
        rows.append({'label': _label(label), 'support': support,
                     'predicted_count': int(np.count_nonzero(predicted == label)),
                     'correct': true_positive,
                     'recall': true_positive / support if support else None})
    return {'schema': 1, 'n_samples': len(actual),
            'accuracy': float(np.mean(actual == predicted)),
            'balanced_accuracy': float(np.mean([row['recall'] for row in rows if row['support']])),
            'per_class': rows, 'role': 'descriptive_only; ordinary accuracy selects candidates'}


def selection_composition(diagnostics, y):
    """Compare training and selected class mixtures using recorded row counts.

    OvO diagnostics use the estimator's binary remapping: the first pair label
    maps to +1 and the second to -1. Both original and encoded labels are kept.
    Missing counts remain unavailable; absent classes count as zero only when
    the complete recorded selected counts agree with the recorded set size.
    """
    facts = diagnostics if isinstance(diagnostics, Mapping) else {}
    labels = np.asarray(y)
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
            included = (labels == pair[0]) | (labels == pair[1])
            pair_y = np.where(labels[included] == pair[0], 1, -1)
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
        matches = [row['count'] for row in recorded if row['label'] == _label(label)]
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
            'classes': [{'label': _label(label), 'training_count': int(count),
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
        return _label(value) if math.isfinite(value) and value >= 0 else None
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
