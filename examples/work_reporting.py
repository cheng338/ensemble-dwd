"""Describe completed checkpoint work without changing its numerical results.

Worker durations are sums of elapsed intervals reported by workers, not CPU
time or parallel wall time. Historical durations describe replayed receipts;
an equivalent alias's source fit/score references are never added to totals.
Cache flags describe only reported base-cache observations, not cache-loading
time, avoided fits, or measured time savings.
"""
from collections.abc import Mapping


def empty_work_summary():
    """Return a fresh, JSON-compatible summary of no completed search work."""
    return {
        'schema': 1,
        'evaluations': {
            'total': 0, 'executed': 0, 'equivalent_aliases_created': 0,
            'checkpoint_replayed': 0, 'executed_ineligible': 0,
            'replayed_ineligible': 0, 'replayed_equivalent_aliases': 0,
        },
        'current_worker_seconds': {'fit': 0., 'score': 0., 'equivalent_alias': 0.},
        'historical_receipt_seconds': {
            'replayed_fit': 0., 'replayed_score': 0., 'replayed_alias': 0.,
        },
        'base_cache': {
            'reported_hits': 0, 'reported_misses': 0,
            'reported_disabled': 0, 'reported_bypassed': 0,
            'reported_non_hits_unknown_status': 0, 'unknown_diagnostic_nodes': 0,
            'evaluations_with_complete_cache_flags': 0,
            'evaluations_with_partial_cache_flags': 0,
            'evaluations_with_no_cache_flags': 0,
        },
        'search_call_wall_seconds': 0.,
        'search_call_count': 0,
    }


_CACHE_OBSERVATIONS = ('reported_hits', 'reported_misses', 'reported_disabled',
                       'reported_bypassed', 'reported_non_hits_unknown_status',
                       'unknown_diagnostic_nodes')


def _cache_nodes(diagnostics):
    """Count observed leaves, preserving evidence of unavailable diagnostics."""
    result = {name: 0 for name in _CACHE_OBSERVATIONS}
    if not isinstance(diagnostics, dict):
        return result, True
    if 'pairwise_estimators' in diagnostics:
        pairs = diagnostics['pairwise_estimators']
        if not isinstance(pairs, list) or not pairs:
            return result, True
        unavailable = False
        for pair in pairs:
            child = pair.get('diagnostics') if isinstance(pair, dict) else None
            child_counts, child_unavailable = _cache_nodes(child)
            for name, value in child_counts.items():
                result[name] += value
            unavailable = unavailable or child_unavailable
        return result, unavailable
    # Binary ensembles expose one base-cache flag. Their base learners and
    # final estimator are not additional base-cache observations.
    flag, status = diagnostics.get('cache_hit_'), diagnostics.get('cache_status_')
    if flag is True and (status is None or status == 'hit'):
        category = 'reported_hits'
    elif flag is False:
        if status == 'miss':
            category = 'reported_misses'
        elif status == 'disabled':
            category = 'reported_disabled'
        elif isinstance(status, str) and status.startswith('bypassed:'):
            category = 'reported_bypassed'
        else:
            category = 'reported_non_hits_unknown_status'
    else:
        category = 'unknown_diagnostic_nodes'
    result[category] = 1
    return result, False


def summarize_cache_diagnostics(diagnostics):
    """Summarize cache flags for one observed evaluation or fitted model.

    Explicit bool flags report a hit or non-hit; only a matching ``miss``
    status establishes a miss. Disabled/bypassed caches and non-hits without
    a known reason are counted separately. A present diagnostic leaf without
    a usable flag is unknown. Missing/empty pair collections provide no
    number of absent pairs, so no missing pair-node count is invented. The
    coverage counters still mark the evaluation as partial or unreported.
    """
    result = empty_work_summary()['base_cache']
    counts, unavailable = _cache_nodes(diagnostics)
    result.update(counts)
    known = sum(counts[name] for name in _CACHE_OBSERVATIONS
                if name != 'unknown_diagnostic_nodes')
    if not known:
        coverage = 'evaluations_with_no_cache_flags'
    elif counts['unknown_diagnostic_nodes'] or unavailable:
        coverage = 'evaluations_with_partial_cache_flags'
    else:
        coverage = 'evaluations_with_complete_cache_flags'
    result[coverage] = 1
    return result


def summarize_records(records):
    """Summarize validated ``(receipt, replayed)`` pairs exactly once.

    Accept a mapping of evaluation keys to pairs or an iterable of pairs.
    Execution, newly created equivalent aliases, and existing receipt replay
    are mutually exclusive. Cache observations concern newly executed records
    only. Ineligible evaluations remain executed work. No fit is performed,
    and neither receipt contents nor legacy timing fields are modified.
    """
    result = empty_work_summary()
    values = records.values() if isinstance(records, Mapping) else records
    counts = result['evaluations']
    current = result['current_worker_seconds']
    historical = result['historical_receipt_seconds']
    for record, replayed in values:
        counts['total'] += 1
        alias = 'equivalent_reuse' in record
        ineligible = record['state'] == 'ineligible'
        if replayed:
            counts['checkpoint_replayed'] += 1
            counts['replayed_ineligible'] += int(ineligible)
            counts['replayed_equivalent_aliases'] += int(alias)
            historical['replayed_fit'] += record['fit_time']
            historical['replayed_score'] += record['score_time']
            if alias:
                historical['replayed_alias'] += record['equivalent_reuse']['reuse_seconds']
        elif alias:
            counts['equivalent_aliases_created'] += 1
            current['equivalent_alias'] += record['equivalent_reuse']['reuse_seconds']
        else:
            counts['executed'] += 1
            counts['executed_ineligible'] += int(ineligible)
            current['fit'] += record['fit_time']
            current['score'] += record['score_time']
            cache = summarize_cache_diagnostics(record.get('diagnostics'))
            for name, value in cache.items():
                result['base_cache'][name] += value
    return result


def aggregate_work_summaries(summaries):
    """Sum completed-call summaries without revisiting their receipt records.

    Pass each completed search call once. In particular, a multi-seed search
    already reports every candidate/seed/fold and must not be added once per
    seed. Summed call wall durations describe those calls, not the enclosing
    workflow's elapsed time.
    """
    result = empty_work_summary()
    for summary in summaries:
        if summary['schema'] != result['schema']:
            raise ValueError('Unsupported work summary schema')
        for group in ('evaluations', 'current_worker_seconds',
                      'historical_receipt_seconds', 'base_cache'):
            for name in result[group]:
                result[group][name] += summary[group][name]
        result['search_call_wall_seconds'] += summary['search_call_wall_seconds']
        result['search_call_count'] += summary['search_call_count']
    return result
