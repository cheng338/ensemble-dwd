"""Bounded, training-only adaptive grids with durable fixed-fold evaluation.

Completed candidates are reused within one run. The original search retains its
estimator and numerical controls. Optional numerical confirmation uses a separate
iteration-cap context without replacing original scores or final-fit settings.
A plateau is a search stopping decision, not a certificate of optimal parameters.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from fractions import Fraction
import hashlib
import math
import struct
import uuid
import warnings
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import hash as parameter_hash, parallel_config
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from cv_checkpoint import (CheckpointedCV, validate_cv_seeds, validate_seed,
                           validate_seed_parameters)

from search_support import (_io_path, _resolved_public_path, bare_parameters, data_fingerprint,
                            exception_details, parameter_snapshot,
                            record_failure, write_failure_json,
                            write_json_atomic)
from tune_mnist import fully_valid_candidate_mask, ranked_valid_indices
from search_support import score_value, score_token, token_value, comparable_tokens, equal_score_tokens
from search_support import score_nonimproving
from ratio_refinement import (RatioRefinementConfig, next_ratio_stage,
                              validate_ratio_grid)
from numerical_confirmation import (NumericalConfirmationConfig, validate_confirmation,
                                    run_numerical_confirmation)
from work_reporting import (aggregate_work_summaries, summarize_cache_diagnostics)
from stage_reporting import (candidate_budget, grid_proposal, publish_report,
                             safe_report, workload_plan)
from near_coordinates import disclose
from phase_threads import (final_fit_threads, final_thread_policy_sha256,
                           validate_final_native_threads)


@dataclass(frozen=True)
class AdaptiveSearchConfig:
    regularization: str = 'lambd'
    prefix: str = 'ensemble__'
    max_expansions: int = 3
    max_candidates: int = 500
    top_k: int = 3
    plateau_rounds: int = 2
    improvement_tol: float = 0.0
    tie_tol: float = 1e-12
    refine: bool = True
    bracket_factor: float = 2.0
    stable_refinement_rounds: int = 2
    allow_boundary_plateau: bool = True

    def __post_init__(self):
        if self.regularization not in ('lambd', 'C') or not isinstance(self.prefix, str):
            raise ValueError('Use regularization lambd or C and a string prefix.')
        for name in ('max_expansions', 'max_candidates', 'top_k', 'plateau_rounds',
                     'stable_refinement_rounds'):
            value = getattr(self, name)
            minimum = 0 if name == 'max_expansions' else 1
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        for name in ('improvement_tol', 'tie_tol'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        if not isinstance(self.refine, (bool, np.bool_)):
            raise ValueError('refine must be boolean.')
        if not isinstance(self.allow_boundary_plateau, (bool, np.bool_)):
            raise ValueError('allow_boundary_plateau must be boolean.')
        if (isinstance(self.bracket_factor, (bool, np.bool_))
                or not isinstance(self.bracket_factor, Real)
                or not math.isfinite(self.bracket_factor) or self.bracket_factor <= 1):
            raise ValueError('bracket_factor must be finite and greater than one.')

    @property
    def axes(self):
        return (self.prefix + self.regularization, self.prefix + 'gamma')


def _initial_grid(grid, config):
    if not isinstance(grid, dict):
        raise ValueError('The initial adaptive grid must be one parameter dictionary.')
    result = {name: list(values) for name, values in grid.items()}
    # Let sklearn validate empty lists and unsupported grid shapes before work.
    ParameterGrid(result)
    for name in config.axes:
        if name not in result:
            raise ValueError(f'The initial grid must contain {name}.')
        values = result[name]
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f'{name} values must be finite positive numbers.')
        result[name] = sorted(set(float(value) for value in values))
    return result


def _results(rows, n_folds):
    keys = ['mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time',
            'mean_score_time', 'std_score_time']
    keys += [f'split{fold}_test_score' for fold in range(n_folds)]
    return {'params': [row['params'] for row in rows],
            'accuracy_evidence': [row.get('accuracy_evidence') for row in rows],
            **{key: np.asarray([row[key] for row in rows], dtype=float) for key in keys}}


def _boundary_directions(results, axes, config):
    ranked = ranked_valid_indices(results)
    if not len(ranked):
        return []
    best = score_value(results, ranked[0])
    tied = [int(index) for index in ranked
            if best - score_value(results, index) <= config.tie_tol]
    directions = []
    valid = fully_valid_candidate_mask(results)
    for name in config.axes:
        for position, side in enumerate(('low', 'high')):
            unresolved = False
            for index in tied:
                parameters = results['params'][index]
                conditional = _conditional_values(results, parameters, name)
                neighbor = _adjacent(parameters[name], conditional)[position]
                unresolved |= neighbor is None or not valid[conditional[neighbor]]
            if unresolved:
                directions.append((name, side))
    return directions


def _rounded_geometric_extension(ordered, side):
    """Round the capped geometric extension once, without logarithm loss."""
    edge = ordered[0] if side == 'low' else ordered[-1]
    exact_edge = Fraction(edge)
    if len(ordered) == 1:
        factor = Fraction(10)
    else:
        neighbor = Fraction(ordered[1] if side == 'low' else ordered[-2])
        factor = neighbor / exact_edge if side == 'low' else exact_edge / neighbor
        factor = min(factor, Fraction(10))
    exact_value = exact_edge / factor if side == 'low' else exact_edge * factor
    try:
        value = float(exact_value)
    except OverflowError:
        value = math.inf
    if not math.isfinite(value) or value <= 0 or not (value < edge if side == 'low' else value > edge):
        raise ValueError('Further boundary values are outside representable positive arithmetic.')
    return value


def _outside(values, side):
    ordered = sorted(values)
    edge = ordered[0] if side == 'low' else ordered[-1]
    if len(ordered) == 1:
        step = math.log(10.)
    elif side == 'low':
        step = math.log(ordered[1]) - math.log(edge)
    else:
        step = math.log(edge) - math.log(ordered[-2])
    # Sparse initial grids can have multi-decade gaps. Extend by at most one
    # decade so a coarse grid does not force an unnecessary extreme jump.
    step = min(step, math.log(10.))
    try:
        value = math.exp(math.log(edge) + (-step if side == 'low' else step))
    except OverflowError:
        value = math.inf
    if not math.isfinite(value) or value <= 0 or not (value < edge if side == 'low' else value > edge):
        # Nearby floats can have identical logarithms. Preserve every existing
        # successful coordinate; recover only a false arithmetic refusal using
        # the same geometric rule, rather than selecting an arbitrary neighbor.
        return _rounded_geometric_extension(ordered, side)
    return value


def _expansion_grid(axes, fixed, directions, config):
    added = {name: [] for name in config.axes}
    for name, side in directions:
        added[name].append(_outside(axes[name], side))
    first, second = config.axes
    # Full new strips include corners when both axes expand; no unchanged
    # interior Cartesian product is scheduled again.
    grids = []
    if added[first]:
        grids.append({**fixed, first: added[first],
                      second: sorted(set(axes[second]) | set(added[second]))})
    if added[second]:
        grids.append({**fixed, first: sorted(axes[first]), second: added[second]})
    return grids


def _neighbors(value, values):
    ordered = sorted(values)
    position = ordered.index(value)
    result = [value]
    for adjacent in (position - 1, position + 1):
        if 0 <= adjacent < len(ordered):
            midpoint = _midpoint(value, ordered[adjacent])
            if midpoint is not None:
                result.append(midpoint)
    return sorted(set(result))


def _refinement_grid(results, axes, fixed, config):
    parents, grids = set(), []
    first, second = config.axes
    for index in ranked_valid_indices(results):
        parameters = results['params'][int(index)]
        parent = (parameters[first], parameters[second])
        if parent in parents:
            continue
        parents.add(parent)
        # Coordinates from another conditional slice do not establish support.
        grids.append({**fixed,
                      first: _neighbors(parent[0], _conditional_values(results, parameters, first)),
                      second: _neighbors(parent[1], _conditional_values(results, parameters, second))})
        if len(parents) == config.top_k:
            break
    return grids


def _new_candidates(grids, seen, limit):
    """Return a complete new stage or None when its whole stage exceeds budget."""
    pending, keys = [], set()
    for parameters in ParameterGrid(grids):
        key = parameter_hash(parameters)
        if key in seen or key in keys:
            continue
        keys.add(key)
        pending.append(parameters)
        if len(pending) > limit:
            return None
    return pending


def _context(parameters, excluded):
    return {name: value for name, value in parameters.items() if name not in excluded}


def _conditional_values(results, anchor, axis):
    signature = parameter_hash(_context(anchor, (axis,)))
    index = results.get('_conditional_indexes', {}).get(axis)
    if index is not None:
        return index.get(signature, {})
    return {float(parameters[axis]): index
            for index, parameters in enumerate(results['params'])
            if parameter_hash(_context(parameters, (axis,))) == signature}


def _adjacent(value, values):
    lower = [point for point in values if point < value]
    upper = [point for point in values if point > value]
    return max(lower) if lower else None, min(upper) if upper else None


def _rounded_geometric_midpoint(lower, upper):
    """Round sqrt(lower * upper) to binary64 using exact comparisons.

    Positive finite float bit patterns are ordered like their values. At most
    63 bisections find the neighboring floats around the exact root without
    overflowing or underflowing the product. This is only a fallback when
    log/exp rounding loses an available interior coordinate.
    """
    target = Fraction(lower) * Fraction(upper)
    low = struct.unpack('>Q', struct.pack('>d', lower))[0]
    high = struct.unpack('>Q', struct.pack('>d', upper))[0]

    def value(bits):
        return struct.unpack('>d', struct.pack('>Q', bits))[0]

    while high - low > 1:
        middle = (low + high) // 2
        if Fraction(value(middle)) ** 2 <= target:
            low = middle
        else:
            high = middle
    below, above = value(low), value(high)
    threshold = ((Fraction(below) + Fraction(above)) / 2) ** 2
    if target < threshold or (target == threshold and low % 2 == 0):
        return below
    return above


def _midpoint(first, second):
    lower, upper = sorted((first, second))
    value = math.exp((math.log(lower) + math.log(upper)) / 2.)
    if lower < value < upper:
        return value
    if not lower < math.nextafter(lower, upper) < upper:
        return None
    value = _rounded_geometric_midpoint(lower, upper)
    return value if lower < value < upper else None


def _policy_records(results):
    keys = sorted(name for name in results if name.startswith('split') and name.endswith('_test_score'))
    return {parameter_hash(parameters):
            (parameters, float(results['mean_test_score'][index]),
             tuple(float(results[name][index]) for name in keys),
             results.get('accuracy_evidence', [None] * len(results['params']))[index])
            for index, parameters in enumerate(results['params'])}


def _evidence(records, keys=None):
    keys = sorted(records) if keys is None else sorted(keys)
    return parameter_hash([(key, records.get(key)) for key in keys])


def _bracket(results, anchor, config):
    valid = fully_valid_candidate_mask(results)
    details = {}
    for axis in config.axes:
        conditional = _conditional_values(results, anchor, axis)
        lower, upper = _adjacent(anchor[axis], conditional)
        lower_valid = lower is not None and bool(valid[conditional[lower]])
        upper_valid = upper is not None and bool(valid[conditional[upper]])
        narrow = (lower_valid and upper_valid
                  and anchor[axis] / lower <= config.bracket_factor * (1 + 1e-14)
                  and upper / anchor[axis] <= config.bracket_factor * (1 + 1e-14))
        details[axis] = {'lower': lower, 'upper': upper, 'lower_valid': lower_valid,
                         'upper_valid': upper_valid, 'within_factor': bool(narrow)}
    return all(item['within_factor'] for item in details.values()), details


def _in_region(parameters, region, config):
    return (region is not None
            and parameter_hash(_context(parameters, config.axes)) == region['context']
            and all(lower <= parameters[axis] <= upper
                    for axis, (lower, upper) in region['bounds'].items()))


def _slice_key(anchor, axis, side):
    return parameter_hash((axis, side, _context(anchor, (axis,))))


def _full_strip_certificates(results, anchors, records, stages, leader, best, config):
    """Check actual current slices against two completed outward grid levels.

    A new corner slice may have been evaluated together in the later full strip.
    Its scores are checked here, never inferred from another gamma or fraction.
    """
    valid = fully_valid_candidate_mask(results)
    leader_index = next(index for index, parameters in enumerate(results['params'])
                        if parameter_hash(parameters) == parameter_hash(leader))
    best_value = score_value(results, leader_index)
    best_token = score_token(results, leader_index)
    certificates = []
    for anchor in anchors:
        for axis in config.axes:
            conditional = _conditional_values(results, anchor, axis)
            for side in ('low', 'high'):
                levels = [(stage['label'], item) for stage in stages
                          if equal_score_tokens(best_token, stage.get('score_token'), best, stage['mean_score'])
                          and stage['leader_key'] == parameter_hash(leader)
                          for item in stage['levels'] if item['axis'] == axis and item['side'] == side]
                levels = levels[-config.plateau_rounds:]
                if (len(levels) < config.plateau_rounds or levels[-1][1]['to'] != anchor[axis]
                        or any(left[1]['to'] != right[1]['from'] for left, right in zip(levels, levels[1:]))):
                    continue
                start, end = levels[0][1]['from'], levels[-1][1]['to']
                required = {start, *(item['to'] for _, item in levels)}
                if not required.issubset(conditional):
                    continue
                points = sorted((value for value in conditional if min(start, end) <= value <= max(start, end)),
                                reverse=side == 'low')
                indices = [conditional[value] for value in points]
                means = [float(results['mean_test_score'][index]) for index in indices]
                values = [score_value(results, index) for index in indices]
                if (not all(valid[index] for index in indices)
                        or any(not score_nonimproving(after, before, config.improvement_tol)
                               for before, after in zip(values, values[1:]))
                        or any(not score_nonimproving(score, best_value, config.improvement_tol) for score in values)):
                    continue
                keys = [parameter_hash(results['params'][index]) for index in indices]
                certificates.append({'slice_key': _slice_key(anchor, axis, side), 'axis': axis, 'side': side,
                    'edge_key': parameter_hash(anchor), 'leader_key': parameter_hash(leader),
                    'conditional_parameters': _context(anchor, (axis,)), 'mean_score': best,
                    'completed_outward_rounds': len(levels), 'stage_labels': [label for label, _ in levels],
                    'coordinates': points, 'conditional_mean_scores': means,
                    'evidence_keys': keys, 'evidence': _evidence(records, keys),
                    'basis': 'full_strip_plateau: completed outward stages and valid nonimproving same-slice evidence'})
    return certificates


def _conditional_repairs(results, anchors, axes, fixed, config, certificates=()):
    """Resolve each winning slice; invalid attempts never become valid support."""
    valid = fully_valid_candidate_mask(results)
    grids, issues, waivers = [], [], []
    for anchor in anchors:
        for axis in config.axes:
            conditional = _conditional_values(results, anchor, axis)
            neighbors = _adjacent(anchor[axis], conditional)
            global_neighbors = _adjacent(anchor[axis], axes[axis])
            for side, neighbor, global_neighbor in zip(('low', 'high'), neighbors, global_neighbors):
                if neighbor is not None and valid[conditional[neighbor]]:
                    continue
                if neighbor is not None:
                    point = _midpoint(anchor[axis], neighbor)
                    issue = 'invalid_conditional_neighbor'
                elif global_neighbor is not None:
                    point = global_neighbor
                    issue = 'missing_conditional_neighbor'
                else:
                    certificate = next((item for item in certificates
                                        if item['slice_key'] == _slice_key(anchor, axis, side)
                                        and item['edge_key'] == parameter_hash(anchor)), None)
                    if certificate is not None:
                        waivers.append(certificate)
                        continue
                    try:
                        point = _outside(conditional, side)
                    except ValueError:
                        point = None
                    issue = 'winning_boundary'
                candidate = None if point is None else {**anchor, axis: point}
                issues.append({'parameters': dict(anchor), 'axis': axis, 'side': side,
                               'issue': issue, 'representable': point is not None,
                               'candidate': candidate})
                if point is not None:
                    pair = {name: [point if name == axis else anchor[name]] for name in config.axes}
                    grids.append({**fixed, **pair})
    return grids, issues, waivers


def next_search_stage(results, initial_grid, config, state=None):
    """Pure, bounded training-only policy over completed candidate evidence.

    A pending stage earns credit only after all its candidates have records.
    Repairs at existing coordinates are finite and do not earn refinement credit.
    A documented empirical plateau may waive a remote outer edge only after
    repeated valid outward trials in exactly that conditional slice. It never
    waives the canonical winner's bilateral bracket or invalid/missing evidence.
    """
    grid = _initial_grid(initial_grid, config)
    # Build each conditional index once; a flat surface can have many ties.
    indexes = {axis: {} for axis in config.axes}
    for row, parameters in enumerate(results['params']):
        for axis in config.axes:
            signature = parameter_hash(_context(parameters, (axis,)))
            indexes[axis].setdefault(signature, {})[float(parameters[axis])] = row
    results = {**results, '_conditional_indexes': indexes}
    state = deepcopy({} if state is None else state)
    if state.get('schema_version', 1) != 1:
        raise ValueError('Unsupported search policy state schema.')
    state['schema_version'] = 1
    identity = parameter_hash((asdict(config), grid))
    if state.get('identity', identity) != identity:
        raise ValueError('Search policy state belongs to different settings or initial grid.')
    state.setdefault('identity', identity)
    state.setdefault('expansion_rounds', 0)
    state.setdefault('refinement_rounds', 0)
    state.setdefault('repair_rounds', 0)
    state.setdefault('stable_rounds', 0)
    state.setdefault('stalled_expansions', 0)
    state.setdefault('full_strip_stages', [])
    records = _policy_records(results)
    seen = set(records)
    ranked = ranked_valid_indices(results) if records else []
    best = None if not len(ranked) else float(results['mean_test_score'][ranked[0]])
    best_token = None if not len(ranked) else score_token(results, ranked[0])
    best_value = token_value(best_token, best)
    leader = None if not len(ranked) else results['params'][int(ranked[0])]
    valid_keys = {parameter_hash(results['params'][int(index)]) for index in ranked}

    def finish(reason, accepted=False):
        review = {'state': 'accepted' if accepted else 'needs_review',
                  'reasons': [] if accepted else [reason],
                  'stopping_reason': reason,
                  'plateau_is_optimality_certificate': False,
                  'plateau_observed': state['stalled_expansions'] >= config.plateau_rounds,
                  'final_refit_performed': False, 'held_out_scored': False}
        state['last_reason'] = reason
        return {'state': state, 'phase': None, 'candidates': [], 'review': review}

    pending = state.get('pending')
    if pending:
        missing = [p for p in pending['candidates'] if parameter_hash(p) not in seen]
        if missing:
            return {'state': state, 'phase': pending['label'], 'candidates': missing, 'review': None}
        unchanged = _evidence(records, pending['baseline_keys']) == pending['baseline_evidence']
        current, previous = comparable_tokens(best_token, pending.get('baseline_score_token'),
                                               best, pending['baseline_best'])
        same_score = current == previous
        local_evidence = [parameter_hash(p) for p in pending['candidates']
                          if _in_region(p, pending.get('region'), config)]
        steady = (unchanged and same_score and leader is not None
                  and _in_region(leader, pending.get('region'), config))
        if pending['kind'] == 'refinement':
            if steady and local_evidence and all(key in valid_keys for key in local_evidence):
                state['stable_rounds'] += 1
            else:
                state['stable_rounds'] = 0
                state.pop('stability_region', None)
        elif not (pending['kind'] == 'conditional_repair' and steady):
            state['stable_rounds'] = 0
            state.pop('stability_region', None)
        if pending['kind'] == 'expansion':
            all_valid = all(parameter_hash(p) in valid_keys for p in pending['candidates'])
            if unchanged and all_valid and score_nonimproving(current, previous, config.improvement_tol):
                state['stalled_expansions'] += 1
            else:
                state['stalled_expansions'] = 0
        elif not unchanged or not same_score:
            state['stalled_expansions'] = 0
        if not unchanged or not same_score:
            state['full_strip_stages'] = []
        elif pending['kind'] == 'expansion':
            state['full_strip_stages'].append({'label': pending['label'],
                'mean_score': best, 'score_token': best_token, 'leader_key': parameter_hash(leader),
                'levels': pending.get('outward_levels', [])})
        state['pending'] = None
    elif state.get('observed_evidence') not in (None, _evidence(records)):
        state['stable_rounds'] = 0
        state['stalled_expansions'] = 0
        state['full_strip_stages'] = []
        state.pop('stability_region', None)
    state['observed_evidence'] = _evidence(records)

    def schedule(grids, kind):
        candidates = _new_candidates(grids, seen, config.max_candidates - len(records))

        def reported(answer):
            # Supplementary metadata never enters policy state or its identity.
            # Scheduled/pending answers retain their original replay contract;
            # their workload is reported separately just before dispatch.
            if answer['review'] is None:
                return answer
            def describe():
                proposal = grid_proposal(grids, seen, config.max_candidates - len(records), candidates)
                admitted = answer['review'] is None
                return {'kind': kind, 'decision': 'scheduled' if admitted else 'refused',
                        'stopping_reason': None if admitted else answer['review']['stopping_reason'],
                        'proposal': proposal,
                        'budget': candidate_budget(config.max_candidates, len(records),
                                                   proposal['new_candidate_count'], admitted)}
            answer['proposal_report'] = safe_report(describe)
            return answer

        if candidates is None:
            return reported(finish('candidate_budget: the complete ' + kind + ' stage exceeds max_candidates'))
        if not candidates:
            return reported(finish('parameter_limit: no new representable point resolves ' + kind))
        if kind == 'expansion':
            if state['expansion_rounds'] >= config.max_expansions:
                return reported(finish('expansion_budget: a best or tied-best conditional boundary remains unresolved'))
            state['expansion_rounds'] += 1
            label = 'expansion-' + str(state['expansion_rounds'])
        elif kind == 'refinement':
            state['refinement_rounds'] += 1
            label = 'local' if state['refinement_rounds'] == 1 else 'local-' + str(state['refinement_rounds'])
        elif kind == 'coarse':
            label = 'coarse'
        else:
            state['repair_rounds'] += 1
            label = 'conditional-repair-' + str(state['repair_rounds'])
        state['pending'] = {'kind': kind, 'label': label, 'candidates': candidates,
                            'baseline_keys': sorted(records), 'baseline_evidence': _evidence(records),
                            'baseline_best': best, 'baseline_score_token': best_token,
                            'region': deepcopy(state.get('stability_region')),
                            'outward_probes': [deepcopy(item) for item in state.get('boundary_issues', [])
                                               if kind == 'expansion' and item['issue'] == 'winning_boundary'
                                               and item['candidate'] is not None],
                            'outward_levels': []}
        if kind == 'expansion':
            for axis in config.axes:
                for side, select in (('low', min), ('high', max)):
                    probes = [item for item in state['pending']['outward_probes']
                              if item['axis'] == axis and item['side'] == side]
                    if probes:
                        state['pending']['outward_levels'].append({'axis': axis, 'side': side,
                            'from': select(p[axis] for p in results['params']),
                            'to': select(item['candidate'][axis] for item in probes)})
        state['last_reason'] = kind
        return reported({'state': state, 'phase': label, 'candidates': candidates, 'review': None})

    if not records:
        return schedule(grid, 'coarse')
    if not len(ranked):
        return finish('ineligible: no candidate has finite scores on every validation fold')
    axes = {axis: sorted({float(p[axis]) for p in results['params']}) for axis in config.axes}
    fixed = {name: values for name, values in grid.items() if name not in config.axes}
    tied = [results['params'][int(index)] for index in ranked
            if best_value - score_value(results, index) <= config.tie_tol]
    certificates = []
    if config.allow_boundary_plateau:
        certificates = _full_strip_certificates(results, tied, records, state['full_strip_stages'],
                                                leader, best, config)
    certificates = [item for item in certificates if item['edge_key'] != parameter_hash(leader)]
    repairs, issues, waivers = _conditional_repairs(results, tied, axes, fixed, config, certificates)
    bracketed, bracket = _bracket(results, leader, config)
    state['winner_bracket'] = bracket
    state['winner_bracket_resolved'] = bracketed
    state['boundary_issues'] = issues
    state['boundary_plateau_evidence'] = waivers
    state['best_parameters'] = dict(leader)
    state['best_mean_score'] = best
    repair_only = bool(issues) and all(item['issue'] == 'missing_conditional_neighbor' for item in issues)
    if state.get('stability_region') is not None and not _in_region(leader, state['stability_region'], config):
        state['stable_rounds'] = 0
        state.pop('stability_region', None)
    if bracketed and (not issues or repair_only) and state.get('stability_region') is None:
        state['stability_region'] = {'context': parameter_hash(_context(leader, config.axes)),
                                     'bounds': {axis: [item['lower'], item['upper']]
                                                for axis, item in bracket.items()}}
    if issues:
        if not repair_only:
            state['stable_rounds'] = 0
            state.pop('stability_region', None)
        # Fill existing-coordinate gaps before extending either axis. Otherwise
        # new coordinates can continually create more missing cross-combinations.
        gaps = [item for item in issues if item['issue'] != 'winning_boundary']
        if gaps:
            repair_grids = [{**fixed, **{axis: [item['candidate'][axis]] for axis in config.axes}}
                            for item in gaps if item['candidate'] is not None]
            return schedule(repair_grids, 'conditional_repair')
        added = {axis: set() for axis in config.axes}
        for item in issues:
            if item['candidate'] is not None:
                added[item['axis']].add(item['candidate'][item['axis']])
        first, second = config.axes
        expansion_grids = []
        if added[first]:
            expansion_grids.append({**fixed, first: sorted(added[first]),
                                    second: sorted(set(axes[second]) | added[second])})
        if added[second]:
            expansion_grids.append({**fixed, first: axes[first], second: sorted(added[second])})
        return schedule(expansion_grids, 'expansion')
    if not config.refine:
        reason = 'valid canonical conditional neighbors; refinement disabled'
        if waivers:
            reason += '; empirical boundary plateau (not an optimality certificate)'
        return finish(reason, accepted=True)
    if bracketed and state['stable_rounds'] >= config.stable_refinement_rounds:
        reason = 'valid conditional bracket and stable completed local rounds'
        if waivers:
            reason += '; empirical boundary plateau (not an optimality certificate)'
        return finish(reason, accepted=True)
    return schedule(_refinement_grid(results, axes, fixed, config), 'refinement')


def _check_folds(folds, n_rows):
    materialized = []
    for train, validation in folds:
        pair = []
        for indices in (train, validation):
            indices = np.asarray(indices)
            if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices):
                raise ValueError('CV folds must contain nonempty integer index vectors.')
            if np.any(indices < 0) or np.any(indices >= n_rows) or len(np.unique(indices)) != len(indices):
                raise ValueError('CV indices must be unique and inside the training set.')
            pair.append(indices.astype(np.int64, copy=True))
        if np.intersect1d(*pair).size:
            raise ValueError('CV training and validation rows must be disjoint.')
        materialized.append(tuple(pair))
    if len(materialized) < 2:
        raise ValueError('At least two fixed CV folds are required.')
    return materialized


def run_adaptive_search(pipeline, grid, X_train, y_train, X_test, y_test, folds,
                        run_dir, *, config=None, jobs=1, error_policy='nan',
                        settings=None, verbose=0, resume=False,
                        reuse_equivalent_folds=False, cv_seeds=None, refit_seed=None,
                        ratio_refinement=None, numerical_confirmation=None, final_native_threads=1,
                        pair_recovery=False, pair_recovery_max_bytes=None):
    """Evaluate only new candidates on fixed training folds, then refit once.

    The candidate/expansion budgets hold final fitting and held-out scoring if
    review remains unfinished. A documented same-slice empirical plateau may
    end outward checks at a tied edge; local bracketing and stability still
    apply. Explicit resume reuses only completed, verified candidate/fold fits
    under the same data, folds, parameters, policy and runtime identity.
    Optional equivalent-fold reuse also shares completed results between
    supported selection ratios that retain exactly the same training rows.
    All candidate labels remain visible, and final fitting uses the selected
    original ratio on the complete training set.
    Optional cv_seeds repeat each candidate on the same folds with the specified
    sampling seeds. Scores are averaged across seeds within each fold, then
    across folds; every requested fit must be valid. Seeds are not ranked.
    The final refit uses refit_seed, or the pipeline's original integer seed.
    These options are recorded in the run identity and cannot change on resume.
    Optional pair_recovery restores completed binary pair models inside an
    unfinished multiclass CV fold. It requires the supported deterministic DWD
    pipeline without a base-stage cache; equivalent-fold reuse and multi-seed
    CV are not supported together with it. The optional per-fold byte quota
    limits committed pair payloads and metadata, not transient memory. Final
    refitting is unaffected. Pair recovery is disabled by default.
    Optional ratio_refinement adds bounded joint local rounds after the original
    coefficient policy accepts. It shares the total candidate budget and never
    widens the initial grid. Unresolved follow-up work holds the final fit.
    Optional numerical_confirmation repeats a frozen ranked shortlist with a
    higher iteration cap in a separate CV context. A changed ranking or an
    invalid confirmation holds final fitting. Original scores, selection and
    final-fit settings are retained; stability does not certify convergence.
    An interrupted final refit restarts; it does not use pair-model persistence.
    final_native_threads changes only that full-training pipeline fit; CV
    and held-out prediction stay at one native thread. It defaults to one
    and is frozen on resume. Wider budgets require explicit n_jobs=1 for
    every exposed nested estimator; runtime observations are recorded.
    """
    final_native_threads = validate_final_native_threads(final_native_threads, pipeline)
    invocation_started = perf_counter()
    invocation_id = uuid.uuid4().hex
    config = AdaptiveSearchConfig() if config is None else config
    if not isinstance(config, AdaptiveSearchConfig):
        raise TypeError('config must be AdaptiveSearchConfig.')
    if isinstance(jobs, (bool, np.bool_)) or not isinstance(jobs, Integral) or jobs < 1:
        raise ValueError('jobs must be a positive integer.')
    if error_policy not in ('nan', 'raise'):
        raise ValueError('error_policy must be nan or raise.')
    if not isinstance(reuse_equivalent_folds, (bool, np.bool_)):
        raise ValueError('reuse_equivalent_folds must be boolean.')
    reuse_equivalent_folds = bool(reuse_equivalent_folds)
    if type(pair_recovery) is not bool:
        raise ValueError('pair_recovery must be boolean.')
    if pair_recovery_max_bytes is not None:
        if not pair_recovery:
            raise ValueError('pair_recovery_max_bytes requires pair_recovery.')
        if (isinstance(pair_recovery_max_bytes, (bool, np.bool_))
                or not isinstance(pair_recovery_max_bytes, Integral) or pair_recovery_max_bytes < 0):
            raise ValueError('pair_recovery_max_bytes must be a nonnegative integer or None.')
        pair_recovery_max_bytes = int(pair_recovery_max_bytes)
    if pair_recovery and (reuse_equivalent_folds or cv_seeds is not None):
        raise ValueError('Pair recovery does not support equivalent-fold reuse or multi-seed CV.')
    pair_options = ({'pair_recovery': True, 'pair_recovery_max_bytes': pair_recovery_max_bytes}
                    if pair_recovery else {})
    grid = _initial_grid(grid, config)
    ratio_bounds = None
    if ratio_refinement is not None:
        if not isinstance(ratio_refinement, RatioRefinementConfig):
            raise TypeError('ratio_refinement must be RatioRefinementConfig or None.')
        if not config.refine:
            raise ValueError('Ratio refinement requires coefficient refinement.')
        ratio_bounds = validate_ratio_grid(grid, config, ratio_refinement)
    confirmation_profile = None
    if numerical_confirmation is not None:
        if not isinstance(numerical_confirmation, NumericalConfirmationConfig):
            raise TypeError('numerical_confirmation must be NumericalConfirmationConfig or None.')
        confirmation_profile = validate_confirmation(
            pipeline, grid, config, numerical_confirmation)
    cv_seeds = validate_cv_seeds(cv_seeds)
    seed_parameter = None
    if cv_seeds is None:
        if refit_seed is not None:
            raise ValueError('refit_seed requires cv_seeds.')
    else:
        seed_parameter = config.prefix + 'random_state'
        parameters = pipeline.get_params(deep=True)
        if seed_parameter not in parameters:
            raise ValueError(f'The pipeline must expose {seed_parameter} for multi-seed CV.')
        original_seed = validate_seed(parameters[seed_parameter], name=seed_parameter)
        refit_seed = original_seed if refit_seed is None else validate_seed(refit_seed, name='refit_seed')
        validate_seed_parameters(grid, seed_parameter)
    run_dir = _resolved_public_path(run_dir)
    failure_dir = _io_path(run_dir / 'failures')
    if not failure_dir.is_dir():
        raise ValueError('Create a fresh run directory with new_run_directory first.')
    folds = _check_folds(folds, len(y_train))
    # Own the held-out values too: the saved identity must describe what is
    # eventually scored even if a caller modifies its arrays during CV.
    X_test, y_test = np.array(X_test, copy=True), np.array(y_test, copy=True)
    X_test.setflags(write=False)
    y_test.setflags(write=False)
    identity = {'training': data_fingerprint(X_train, y_train),
                'held_out': data_fingerprint(X_test, y_test)}
    metadata = {'schema_version': 1, 'policy': asdict(config), 'initial_grid': grid,
                'workflow_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'final_thread_policy_sha256': final_thread_policy_sha256(),
                'data': identity, 'folds': [{'train': t, 'validation': v} for t, v in folds],
                'settings': {'jobs': jobs, 'native_threads': 1, 'scoring': 'accuracy',
                             'final_native_threads': final_native_threads, 'held_out_native_threads': 1,
                             'error_policy': error_policy, 'parameters': parameter_snapshot(pipeline),
                             'user': {} if settings is None else settings}}
    if reuse_equivalent_folds:
        metadata['settings']['reuse_equivalent_folds'] = True
    if pair_recovery:
        metadata['settings']['pair_recovery'] = {'enabled': True, 'max_bytes': pair_recovery_max_bytes,
                                                'scope': 'unfinished_cv_folds'}
    if cv_seeds is not None:
        metadata['settings']['sampling'] = {
            'cv_seeds': list(cv_seeds), 'seed_parameter': seed_parameter,
            'refit_seed': refit_seed, 'aggregation': 'mean_seeds_within_fold_then_mean_folds'}
    if ratio_refinement is not None:
        metadata['ratio_refinement'] = {
            'config': asdict(ratio_refinement), 'bounds': list(ratio_bounds),
            'policy_sha256': hashlib.sha256(Path(__file__).with_name('ratio_refinement.py').read_bytes()).hexdigest()}
    if numerical_confirmation is not None:
        metadata['numerical_confirmation'] = {
            'config': asdict(numerical_confirmation), 'profile': confirmation_profile,
            'policy_sha256': hashlib.sha256(Path(__file__).with_name('numerical_confirmation.py').read_bytes()).hexdigest()}
    rows, stages, seen = [], [], set()
    searches = {}
    axes = {name: set(grid[name]) for name in config.axes}
    policy_state = {}
    ratio_policy_state = {}
    final_runtime = None
    phase = 'coarse_cv'
    last_search = None
    review = None

    work_stages = []
    confirmation_work = {'state': 'not_requested' if numerical_confirmation is None else 'not_started',
                         'wall_seconds': 0.0, 'work_summary': None}
    refit_work = {'state': 'not_started', 'attempted': False, 'wall_seconds': 0.0}
    scoring_work = {'state': 'not_started', 'attempted': False, 'wall_seconds': 0.0}
    active_work = None
    historical_work = None

    def work_snapshot(state, *, snapshot_point='before_summary_publication'):
        now = perf_counter()
        incomplete = None
        confirmation = deepcopy(confirmation_work)
        refit, scoring = deepcopy(refit_work), deepcopy(scoring_work)
        if active_work is not None:
            kind = active_work['kind']
            if kind in ('original_search', 'numerical_confirmation'):
                incomplete = {key: value for key, value in active_work.items() if key != 'started'}
                incomplete['elapsed_seconds'] = now - active_work['started']
                incomplete['unmeasured_evaluations'] = True
                if kind == 'numerical_confirmation':
                    confirmation.update(state=state, wall_seconds=incomplete['elapsed_seconds'])
            elif kind in ('final_refit', 'held_out_scoring'):
                target = refit if kind == 'final_refit' else scoring
                target.update(state=state, wall_seconds=now - active_work['started'])
        known = [stage['work_summary'] for stage in work_stages if stage['work_summary'] is not None]
        missing = [stage['name'] for stage in work_stages if stage['work_summary'] is None]
        original = aggregate_work_summaries(known)
        if confirmation['work_summary'] is not None:
            known.append(confirmation['work_summary'])
        elif confirmation['state'] in ('stable', 'needs_review') and confirmation.get('candidate_count', 0) >= 2:
            missing.append('numerical_confirmation')
        return {
            'schema': 1, 'scope': 'current_invocation', 'state': state,
            'invocation_id': invocation_id, 'elapsed_seconds': now - invocation_started,
            'snapshot_point': snapshot_point,
            'elapsed_boundary': 'run_adaptive_search entry through this snapshot; excludes subsequent summary publication and context teardown, and excludes caller data loading',
            'coverage': 'completed_stages_only' if incomplete is not None or missing else 'complete',
            'incomplete_phase': incomplete, 'stages_without_work_details': missing,
            'original_search': {'stages': deepcopy(work_stages), 'work_summary': original},
            'numerical_confirmation': confirmation, 'cv_totals': aggregate_work_summaries(known),
            'final_refit': refit, 'held_out_scoring': scoring,
            'historical_completion_summary_available': historical_work is not None,
            'interpretation': 'Worker durations are summed elapsed work, not CPU or wall time. Historical receipt times are not current fitting. Incomplete-stage work and cumulative elapsed time across interruptions are not inferred.'}

    def warn_report(error):
        try:
            warnings.warn('Could not publish the work summary: ' + str(error), RuntimeWarning)
        except Exception:
            pass

    def safe_work_snapshot(state, **kwargs):
        # Reporting cannot turn a successful fit into a failure or hide the
        # original exception, including before the authoritative commit.
        try:
            return work_snapshot(state, **kwargs)
        except Exception as error:
            warn_report(error)
            return {'schema': 1, 'scope': 'current_invocation', 'state': state,
                    'reporting_error': str(error), 'coverage': 'unavailable'}

    def publish_work(state):
        summary = safe_work_snapshot(state)
        try:
            write_json_atomic(run_dir / 'work-summary.json', summary, replace=True)
        except Exception as error:
            warn_report(error)
        return summary

    def evaluate(label, candidates, proposal_report=None):
        nonlocal phase, last_search, active_work
        phase = label + '_cv'
        write_json_atomic(run_dir / 'status.json', {
            'state': 'running', 'phase': phase, 'new_candidates': len(candidates),
            'completed_candidates': len(rows)}, replace=True)
        candidate_grid = [{name: [value] for name, value in parameters.items()}
                          for parameters in candidates]
        seed_note = '' if cv_seeds is None else f' and {len(cv_seeds)} sampling seeds'
        print(f'{label}: checking {len(candidates)} candidates on {len(folds)} fixed folds{seed_note}.', flush=True)
        publish_report(run_dir / (label + '-plan.json'), lambda: {
            'schema': 1, 'scope': 'original_search_stage', 'stage': label,
            'budget': candidate_budget(config.max_candidates, len(seen), len(candidates), True),
            'proposal_report': proposal_report,
            'near_coordinates': safe_report(lambda: disclose(
                candidates, config.axes + ((config.prefix + 'sv_selection_ratio',)
                    if ratio_refinement is not None else ()),
                previous=[row['params'] for row in rows])),
            'workload_plan': workload_plan(checkpoint.estimator, candidates, checkpoint.y, folds,
                seeds=cv_seeds, reuse={'completed_fold_receipts': resume,
                                      'equivalent_folds': reuse_equivalent_folds,
                                      'completed_pairs': pair_recovery}),
        }, announce=True)
        started = perf_counter()
        active_work = {'kind': 'original_search', 'name': label, 'started': started}
        search = checkpoint.search(candidate_grid, stage=label)
        elapsed = perf_counter() - started
        work_stages.append({'name': label, 'wall_seconds': elapsed,
                            'work_summary': deepcopy(search.checkpoint_summary_.get('work_summary'))})
        active_work = None
        current = search.cv_results_
        keys = [key for key in _results([], len(folds)) if key not in ('params', 'accuracy_evidence')]
        for index, parameters in enumerate(current['params']):
            bare = bare_parameters(parameters)
            rows.append({'params': bare, 'stage': label, 'stage_index': index,
                         'accuracy_evidence': deepcopy(current.get('accuracy_evidence', [None] * len(current['params']))[index]),
                         **{key: float(current[key][index]) for key in keys}})
            seen.add(parameter_hash(bare))
            for name in axes:
                axes[name].add(float(bare[name]))
        stage = {'name': label, 'new_candidates': len(candidates),
                 'total_candidates': len(rows), 'wall_seconds': elapsed,
                 'parameters': candidates, 'checkpoint': search.checkpoint_summary_}
        stages.append(stage)
        write_json_atomic(run_dir / (label + '-cv.json'), {
            **stage, 'cv_results': {**current, 'params': [bare_parameters(p) for p in current['params']]},
            'fully_valid': fully_valid_candidate_mask(current)}, replace=resume)
        write_json_atomic(run_dir / 'accumulated-cv.json', {
            'cv_results': _results(rows, len(folds)), 'stages': stages}, replace=True)
        last_search = search
        searches[label] = search
        ranked = ranked_valid_indices(_results(rows, len(folds)))
        if not len(ranked):
            raise ValueError('No candidate has finite scores on every validation fold.')
        print(f'{label}: best accumulated CV accuracy {rows[int(ranked[0])]["mean_test_score"]:.6f}; '
              f'{elapsed:.3f} seconds.', flush=True)

    def execute():
        nonlocal phase, policy_state, ratio_policy_state, review, active_work, historical_work, final_runtime
        with parallel_config(backend='loky', inner_max_num_threads=1), threadpool_limits(limits=1):
            while review is None:
                results = _results(rows, len(folds))
                proposal = next_search_stage(results, grid, config, policy_state)
                policy_state = proposal['state']
                review = proposal['review']
                if review is None:
                    evaluate(proposal['phase'], proposal['candidates'], proposal.get('proposal_report'))
                elif review['state'] != 'accepted':
                    publish_report(run_dir / 'coefficient-refused-stage.json', lambda: {
                        'schema': 1, 'scope': 'refused_coefficient_stage',
                        'decision': review['state'], 'stopping_reason': review['stopping_reason'],
                        'budget': (proposal.get('proposal_report') or {}).get('budget')
                                  or candidate_budget(config.max_candidates, len(seen), None, False),
                        'proposal_report': proposal.get('proposal_report'), 'refused_stage_fits_dispatched': 0,
                    }, announce=True)

            coefficient_review = deepcopy(review)
            ratio_started = ratio_refinement is not None and review['state'] == 'accepted'
            if ratio_started:
                while True:
                    proposal = next_ratio_stage(_results(rows, len(folds)), grid, config,
                                                ratio_refinement, ratio_policy_state)
                    ratio_policy_state = proposal['state']
                    review = proposal['review']
                    if review is not None:
                        if review['state'] != 'accepted':
                            publish_report(run_dir / 'ratio-refused-stage.json', lambda: {
                                'schema': 1, 'scope': 'refused_ratio_stage',
                                'decision': review['state'], 'stopping_reason': review['stopping_reason'],
                                'budget': (proposal.get('proposal_report') or {}).get('budget')
                                          or candidate_budget(config.max_candidates, len(seen), None, False),
                                'proposal_report': proposal.get('proposal_report'), 'refused_stage_fits_dispatched': 0,
                            }, announce=True)
                        break
                    evaluate(proposal['phase'], proposal['candidates'], proposal.get('proposal_report'))

            results = _results(rows, len(folds))
            ranked = ranked_valid_indices(results) if rows else []
            index = int(ranked[0]) if len(ranked) else None
            selection = None if index is None else {
                'candidate_index': index, 'parameters': rows[index]['params'],
                'source_stage': rows[index]['stage'],
                'mean_cv_accuracy': rows[index]['mean_test_score']}
            if selection is not None and cv_seeds is not None:
                selection.update(cv_seeds=list(cv_seeds), seed_parameter=seed_parameter,
                                 refit_seed=refit_seed)
            boundary_evidence = (policy_state.get('boundary_issues', [])
                                 + policy_state.get('boundary_plateau_evidence', []))
            review.update({'evaluated_candidates': len(rows),
                           'expansion_rounds': policy_state.get('expansion_rounds', 0),
                           'refinement_completed': policy_state.get('refinement_rounds', 0) > 0,
                           'refinement_rounds': policy_state.get('refinement_rounds', 0),
                           'stable_refinement_rounds': policy_state.get('stable_rounds', 0),
                           'winner_bracket': policy_state.get('winner_bracket'),
                           'boundary_plateau_evidence': policy_state.get('boundary_plateau_evidence', []),
                           'boundary_directions': sorted({(item['axis'], item['side'])
                                                          for item in boundary_evidence})})
            ratio_payload = {}
            if ratio_refinement is not None:
                review['coefficient_search'] = {'review': coefficient_review,
                                               'winner_bracket': policy_state.get('winner_bracket'),
                                               'boundary_plateau_evidence': policy_state.get('boundary_plateau_evidence', [])}
                review['ratio_refinement'] = {
                    'config': asdict(ratio_refinement), 'bounds': list(ratio_bounds),
                    'started': ratio_started, 'completed_rounds': ratio_policy_state.get('rounds', 0),
                    'stable_rounds': ratio_policy_state.get('stable_rounds', 0)}
                if ratio_started:
                    # The earlier coefficient bracket describes its old winner.
                    # Report the follow-up winner's actual conditional evidence.
                    review['winner_bracket'] = ratio_policy_state.get('winner_bracket')
                    review['boundary_directions'] = sorted({(item['axis'], item['side'])
                        for item in ratio_policy_state.get('boundary_directions', [])})
                    review['boundary_plateau_evidence'] = []
                ratio_payload = {'ratio_policy_state': ratio_policy_state}
                write_json_atomic(run_dir / 'ratio-refinement.json', {
                    **review['ratio_refinement'], 'state': ratio_policy_state,
                    'review_state': review['state'], 'stopping_reason': review['stopping_reason']}, replace=resume)
            confirmation_payload = {}
            if numerical_confirmation is not None:
                search_review = deepcopy(review)
                if review['state'] == 'accepted':
                    phase = 'numerical_confirmation_cv'
                    write_json_atomic(run_dir / 'status.json', {
                        'state': 'running', 'phase': phase,
                        'max_confirmation_candidates': numerical_confirmation.top_k,
                        'confirmation_max_iter': numerical_confirmation.max_iter}, replace=True)
                    active_work = {'kind': 'numerical_confirmation', 'name': 'numerical_confirmation',
                                   'started': perf_counter()}
                    confirmation_report = run_numerical_confirmation(
                        pipeline=checkpoint.estimator, results=results, rows=rows, searches=searches,
                        folds=folds, X=checkpoint.X, y=checkpoint.y, run_dir=run_dir,
                        config=numerical_confirmation, search_config=config, jobs=jobs,
                        error_policy=error_policy, resume=resume, cv_seeds=cv_seeds,
                        seed_parameter=seed_parameter, reuse_equivalent_folds=reuse_equivalent_folds,
                        parent_identity=checkpoint.identity_sha256, **pair_options)
                    confirmation_work.update(
                        state=confirmation_report['state'], candidate_count=confirmation_report['candidate_count'],
                        wall_seconds=perf_counter() - active_work['started'],
                        work_summary=deepcopy(confirmation_report.get('checkpoint_summary', {}).get('work_summary')))
                    active_work = None
                    if confirmation_report['state'] not in ('stable', 'needs_review'):
                        raise ValueError('Unexpected numerical confirmation state.')
                else:
                    confirmation_report = {
                        'state': 'not_started', 'reasons': ['Original parameter search requires review.'],
                        'stopping_reason': 'original_search_requires_review',
                        'candidate_count': 0, 'profile': confirmation_profile,
                        'config': asdict(numerical_confirmation)}
                if confirmation_report['state'] == 'needs_review':
                    review.update(state='needs_review', reasons=list(confirmation_report['reasons']),
                                  stopping_reason='numerical_confirmation: ' + confirmation_report['stopping_reason'])
                review['search_review'] = search_review
                review['winner_bracket_context'] = 'original_search_budget'
                review['numerical_confirmation'] = {
                    key: confirmation_report[key] for key in
                    ('state', 'reasons', 'stopping_reason', 'candidate_count', 'profile')}
                review['numerical_confirmation']['stability_is_convergence_certificate'] = False
                confirmation_payload = {'numerical_confirmation': confirmation_report}
                write_json_atomic(run_dir / 'numerical-confirmation.json', confirmation_report, replace=resume)
            write_json_atomic(run_dir / 'selection.json', selection, replace=resume)
            write_json_atomic(run_dir / 'review.json', review, replace=resume)
            result = {'model': None, 'completed': None, 'selection': selection,
                      'review': review, 'run_dir': run_dir, 'cv_results': results,
                      'policy_state': policy_state,
                      'stages': stages,
                      'coarse_search': searches.get('coarse'),
                      'selected_search': None if selection is None else searches[selection['source_stage']],
                      'last_search': last_search, **ratio_payload, **confirmation_payload}
            if review['state'] == 'needs_review':
                write_json_atomic(run_dir / 'status.json', {**review, 'phase': 'parameter_review'}, replace=True)
                print('Search held: ' + '; '.join(review['reasons']), flush=True)
                result['work_summary'] = publish_work('needs_review')
                return result
            phase = 'final_refit'
            write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
            refit_work.update(state='running', attempted=True)
            started = perf_counter()
            active_work = {'kind': 'final_refit', 'started': started}
            refit_parameters = dict(selection['parameters'])
            if cv_seeds is not None:
                refit_parameters[seed_parameter] = refit_seed
            try:
                model = clone(checkpoint.estimator).set_params(**refit_parameters)
                final_runtime = {}
                with final_fit_threads(final_native_threads, model, final_runtime,
                                       frozen_runtime=checkpoint.runtime):
                    model.fit(checkpoint.X, checkpoint.y)
            except Exception as error:
                write_failure_json(run_dir / 'refit.json', {'state': 'failed', **selection,
                    'wall_seconds': perf_counter() - started, 'failure': exception_details(error),
                    'native_runtime': final_runtime}, replace=True)
                raise
            refit_seconds = perf_counter() - started
            refit = {'state': 'completed', **selection, 'wall_seconds': refit_seconds,
                     'diagnostics': checkpoint.diagnostics(model, y_train), 'native_runtime': final_runtime}
            refit_work.update(state='completed', wall_seconds=refit_seconds)
            try:
                refit_work['base_cache'] = summarize_cache_diagnostics(refit['diagnostics'])
            except Exception as error:
                warn_report(error)
                refit_work['base_cache'] = {'reporting_error': str(error), 'coverage': 'unavailable'}
            active_work = None
            if cv_seeds is not None:
                refit['refit_parameters'] = refit_parameters
            write_json_atomic(run_dir / 'refit.json', refit, replace=resume)
            review.update(final_refit_performed=True)
            write_json_atomic(run_dir / 'review.json', review, replace=True)
            phase = 'held_out_scoring'
            write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
            checkpoint.check_runtime()
            scoring_work.update(state='running', attempted=True)
            active_work = {'kind': 'held_out_scoring', 'started': perf_counter()}
            accuracy = float(accuracy_score(y_test, model.predict(X_test)))
            scoring_work.update(state='completed', wall_seconds=perf_counter() - active_work['started'])
            active_work = None
            completed = {**selection, 'refit_seconds': refit_seconds, 'held_out_accuracy': accuracy,
                         'final_native_threads': final_native_threads}
            review.update(final_refit_performed=True, held_out_scored=True)
            # Commit the authoritative completion receipt before public mirrors.
            # A verified completed resume never fits or scores the test set again.
            completion_work = safe_work_snapshot('completed', snapshot_point='before_completion_receipt')
            checkpoint.write_artifact('completed', {
                'completion_work_summary': completion_work,
                'completed': completed, 'review': review, 'selection': selection,
                'refit': refit, 'cv_results': results, 'policy_state': policy_state,
                'stages': stages, **ratio_payload, **confirmation_payload})
            write_json_atomic(run_dir / 'review.json', review, replace=True)
            write_json_atomic(run_dir / 'completed.json', completed, replace=resume)
            write_json_atomic(run_dir / 'status.json', {'state': 'completed', 'phase': phase,
                                                       'stopping_reason': review['stopping_reason']}, replace=True)
            historical_work = completion_work
            result.update(model=model, completed=completed, refit=refit,
                          completion_work_summary=completion_work, work_summary=publish_work('completed'))
            return result
    # Acquire exclusive ownership and verify identity before changing run/status
    # files. A mismatch or another live coordinator leaves their state untouched.
    with CheckpointedCV(run_dir, pipeline, X_train, y_train, folds, jobs=jobs,
                        error_policy=error_policy, resume=resume,
                        provenance=metadata,
                        reuse_equivalent_folds=reuse_equivalent_folds,
                        cv_seeds=cv_seeds, seed_parameter=seed_parameter, **pair_options) as checkpoint:
        saved = checkpoint.read_artifact('completed')
        if saved is not None:
            historical_work = saved.get('completion_work_summary')
            refit_work['state'] = scoring_work['state'] = 'not_performed_replay'
            confirmation_work['state'] = 'not_performed_replay'
            for name in ('selection', 'review', 'refit', 'completed'):
                write_json_atomic(run_dir / (name + '.json'), saved[name], replace=True)
            if 'numerical_confirmation' in saved:
                write_json_atomic(run_dir / 'numerical-confirmation.json',
                                  saved['numerical_confirmation'], replace=True)
            if 'ratio_policy_state' in saved:
                write_json_atomic(run_dir / 'ratio-refinement.json', {
                    **saved['review']['ratio_refinement'], 'state': saved['ratio_policy_state'],
                    'review_state': saved['review']['state'],
                    'stopping_reason': saved['review']['stopping_reason']}, replace=True)
            write_json_atomic(run_dir / 'status.json', {
                'state': 'completed', 'phase': 'held_out_scoring',
                'stopping_reason': saved['review']['stopping_reason']}, replace=True)
            return {**saved, 'model': None, 'run_dir': run_dir, 'resumed_completed': True,
                    'work_summary': publish_work('completed_replay'),
                    'selected_search': None, 'coarse_search': None, 'last_search': None}
        write_json_atomic(run_dir / 'run.json', {
            **metadata, 'checkpoint_identity_sha256': checkpoint.identity_sha256}, replace=resume)
        try:
            return execute()
        except (KeyboardInterrupt, SystemExit) as error:
            write_failure_json(run_dir / 'status.json', {
                'state': 'interrupted', 'phase': phase,
                'failure': exception_details(error), 'final_runtime': final_runtime,
                'recovery': 'Resume with the same settings and output directory; completed folds are retained.'},
                replace=True)
            publish_work('interrupted')
            raise
        except Exception as error:
            record_failure(failure_dir, error, phase=phase, estimator=pipeline,
                           training_data=identity['training'])
            write_failure_json(run_dir / 'status.json', {'state': 'failed', 'phase': phase,
                                                        'failure': exception_details(error)}, replace=True)
            publish_work('failed')
            raise
