"""Bounded, training-only adaptive grids with durable fixed-fold evaluation.

Completed candidates are reused within one run. The estimator and its numerical
controls are never modified by the search policy. A plateau is a search stopping
decision, not a certificate of optimal hyperparameters.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import hash as parameter_hash, parallel_config
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from cv_checkpoint import CheckpointedCV

from search_support import (bare_parameters, data_fingerprint,
                            exception_details, parameter_snapshot,
                            record_failure, write_failure_json,
                            write_json_atomic)
from tune_mnist import fully_valid_candidate_mask, ranked_valid_indices


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
            **{key: np.asarray([row[key] for row in rows], dtype=float) for key in keys}}


def _boundary_directions(results, axes, config):
    ranked = ranked_valid_indices(results)
    if not len(ranked):
        return []
    best = float(results['mean_test_score'][ranked[0]])
    tied = [int(index) for index in ranked
            if best - float(results['mean_test_score'][index]) <= config.tie_tol]
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
        raise ValueError('Further boundary values are outside representable positive arithmetic.')
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
            midpoint = math.exp((math.log(value) + math.log(ordered[adjacent])) / 2.)
            if min(value, ordered[adjacent]) < midpoint < max(value, ordered[adjacent]):
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


def _midpoint(first, second):
    lower, upper = sorted((first, second))
    value = math.exp((math.log(lower) + math.log(upper)) / 2.)
    return value if lower < value < upper else None


def _policy_records(results):
    keys = sorted(name for name in results if name.startswith('split') and name.endswith('_test_score'))
    return {parameter_hash(parameters):
            (parameters, float(results['mean_test_score'][index]),
             tuple(float(results[name][index]) for name in keys))
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
    certificates = []
    for anchor in anchors:
        for axis in config.axes:
            conditional = _conditional_values(results, anchor, axis)
            for side in ('low', 'high'):
                levels = [(stage['label'], item) for stage in stages
                          if stage['mean_score'] == best and stage['leader_key'] == parameter_hash(leader)
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
                if (not all(valid[index] for index in indices)
                        or any(after > before + config.improvement_tol for before, after in zip(means, means[1:]))
                        or any(score > best + config.improvement_tol for score in means)):
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
        same_score = best == pending['baseline_best']
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
            if unchanged and all_valid and best <= pending['baseline_best'] + config.improvement_tol:
                state['stalled_expansions'] += 1
            else:
                state['stalled_expansions'] = 0
        elif not unchanged or not same_score:
            state['stalled_expansions'] = 0
        if not unchanged or not same_score:
            state['full_strip_stages'] = []
        elif pending['kind'] == 'expansion':
            state['full_strip_stages'].append({'label': pending['label'],
                'mean_score': best, 'leader_key': parameter_hash(leader),
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
        if candidates is None:
            return finish('candidate_budget: the complete ' + kind + ' stage exceeds max_candidates')
        if not candidates:
            return finish('parameter_limit: no new representable point resolves ' + kind)
        if kind == 'expansion':
            if state['expansion_rounds'] >= config.max_expansions:
                return finish('expansion_budget: a best or tied-best conditional boundary remains unresolved')
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
                            'baseline_best': best, 'region': deepcopy(state.get('stability_region')),
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
        return {'state': state, 'phase': label, 'candidates': candidates, 'review': None}

    if not records:
        return schedule(grid, 'coarse')
    if not len(ranked):
        return finish('ineligible: no candidate has finite scores on every validation fold')
    axes = {axis: sorted({float(p[axis]) for p in results['params']}) for axis in config.axes}
    fixed = {name: values for name, values in grid.items() if name not in config.axes}
    tied = [results['params'][int(index)] for index in ranked
            if best - float(results['mean_test_score'][index]) <= config.tie_tol]
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
                        settings=None, verbose=0, resume=False):
    """Evaluate only new candidates on fixed training folds, then refit once.

    The candidate/expansion budgets hold final fitting and held-out scoring if
    review remains unfinished. A documented same-slice empirical plateau may
    end outward checks at a tied edge; local bracketing and stability still
    apply. Explicit resume reuses only completed, verified candidate/fold fits
    under the same data, folds, parameters, policy and runtime identity.
    An interrupted final refit restarts; fitted models are not serialized.
    """
    config = AdaptiveSearchConfig() if config is None else config
    if not isinstance(config, AdaptiveSearchConfig):
        raise TypeError('config must be AdaptiveSearchConfig.')
    if isinstance(jobs, (bool, np.bool_)) or not isinstance(jobs, Integral) or jobs < 1:
        raise ValueError('jobs must be a positive integer.')
    if error_policy not in ('nan', 'raise'):
        raise ValueError('error_policy must be nan or raise.')
    grid = _initial_grid(grid, config)
    run_dir = Path(run_dir).resolve()
    failure_dir = run_dir / 'failures'
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
                'data': identity, 'folds': [{'train': t, 'validation': v} for t, v in folds],
                'settings': {'jobs': jobs, 'native_threads': 1, 'scoring': 'accuracy',
                             'error_policy': error_policy, 'parameters': parameter_snapshot(pipeline),
                             'user': {} if settings is None else settings}}
    rows, stages, seen = [], [], set()
    searches = {}
    axes = {name: set(grid[name]) for name in config.axes}
    policy_state = {}
    phase = 'coarse_cv'
    last_search = None
    review = None

    def evaluate(label, candidates):
        nonlocal phase, last_search
        phase = label + '_cv'
        write_json_atomic(run_dir / 'status.json', {
            'state': 'running', 'phase': phase, 'new_candidates': len(candidates),
            'completed_candidates': len(rows)}, replace=True)
        candidate_grid = [{name: [value] for name, value in parameters.items()}
                          for parameters in candidates]
        print(f'{label}: checking {len(candidates)} candidates on {len(folds)} fixed folds.', flush=True)
        started = perf_counter()
        search = checkpoint.search(candidate_grid, stage=label)
        elapsed = perf_counter() - started
        current = search.cv_results_
        keys = [key for key in _results([], len(folds)) if key != 'params']
        for index, parameters in enumerate(current['params']):
            bare = bare_parameters(parameters)
            rows.append({'params': bare, 'stage': label, 'stage_index': index,
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
        nonlocal phase, policy_state, review
        with parallel_config(backend='loky', inner_max_num_threads=1), threadpool_limits(limits=1):
            while review is None:
                results = _results(rows, len(folds))
                proposal = next_search_stage(results, grid, config, policy_state)
                policy_state = proposal['state']
                review = proposal['review']
                if review is None:
                    evaluate(proposal['phase'], proposal['candidates'])

            results = _results(rows, len(folds))
            ranked = ranked_valid_indices(results) if rows else []
            index = int(ranked[0]) if len(ranked) else None
            selection = None if index is None else {
                'candidate_index': index, 'parameters': rows[index]['params'],
                'source_stage': rows[index]['stage'],
                'mean_cv_accuracy': rows[index]['mean_test_score']}
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
            write_json_atomic(run_dir / 'selection.json', selection, replace=resume)
            write_json_atomic(run_dir / 'review.json', review, replace=resume)
            result = {'model': None, 'completed': None, 'selection': selection,
                      'review': review, 'run_dir': run_dir, 'cv_results': results,
                      'policy_state': policy_state,
                      'stages': stages,
                      'coarse_search': searches.get('coarse'),
                      'selected_search': None if selection is None else searches[selection['source_stage']],
                      'last_search': last_search}
            if review['state'] == 'needs_review':
                write_json_atomic(run_dir / 'status.json', {**review, 'phase': 'parameter_review'}, replace=True)
                print('Search held: ' + '; '.join(review['reasons']), flush=True)
                return result
            phase = 'final_refit'
            write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
            started = perf_counter()
            try:
                model = clone(checkpoint.estimator).set_params(**selection['parameters'])
                model.fit(checkpoint.X, checkpoint.y)
            except Exception as error:
                write_failure_json(run_dir / 'refit.json', {'state': 'failed', **selection,
                    'wall_seconds': perf_counter() - started, 'failure': exception_details(error)}, replace=True)
                raise
            refit_seconds = perf_counter() - started
            refit = {'state': 'completed', **selection, 'wall_seconds': refit_seconds,
                     'diagnostics': checkpoint.diagnostics(model)}
            write_json_atomic(run_dir / 'refit.json', refit, replace=resume)
            review.update(final_refit_performed=True)
            write_json_atomic(run_dir / 'review.json', review, replace=True)
            phase = 'held_out_scoring'
            write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
            checkpoint.check_runtime()
            accuracy = float(accuracy_score(y_test, model.predict(X_test)))
            completed = {**selection, 'refit_seconds': refit_seconds, 'held_out_accuracy': accuracy}
            review.update(final_refit_performed=True, held_out_scored=True)
            # Commit the authoritative completion receipt before public mirrors.
            # A verified completed resume never fits or scores the test set again.
            checkpoint.write_artifact('completed', {
                'completed': completed, 'review': review, 'selection': selection,
                'refit': refit, 'cv_results': results, 'policy_state': policy_state,
                'stages': stages})
            write_json_atomic(run_dir / 'review.json', review, replace=True)
            write_json_atomic(run_dir / 'completed.json', completed, replace=resume)
            write_json_atomic(run_dir / 'status.json', {'state': 'completed', 'phase': phase,
                                                       'stopping_reason': review['stopping_reason']}, replace=True)
            result.update(model=model, completed=completed, refit=refit)
            return result
    # Acquire exclusive ownership and verify identity before changing run/status
    # files. A mismatch or another live coordinator leaves their state untouched.
    with CheckpointedCV(run_dir, pipeline, X_train, y_train, folds, jobs=jobs,
                        error_policy=error_policy, resume=resume,
                        provenance=metadata) as checkpoint:
        saved = checkpoint.read_artifact('completed')
        if saved is not None:
            for name in ('selection', 'review', 'refit', 'completed'):
                write_json_atomic(run_dir / (name + '.json'), saved[name], replace=True)
            write_json_atomic(run_dir / 'status.json', {
                'state': 'completed', 'phase': 'held_out_scoring',
                'stopping_reason': saved['review']['stopping_reason']}, replace=True)
            return {**saved, 'model': None, 'run_dir': run_dir, 'resumed_completed': True,
                    'selected_search': None, 'coarse_search': None, 'last_search': None}
        write_json_atomic(run_dir / 'run.json', {
            **metadata, 'checkpoint_identity_sha256': checkpoint.identity_sha256}, replace=resume)
        try:
            return execute()
        except (KeyboardInterrupt, SystemExit) as error:
            write_failure_json(run_dir / 'status.json', {
                'state': 'interrupted', 'phase': phase,
                'failure': exception_details(error),
                'recovery': 'Resume with the same settings and output directory; completed folds are retained.'},
                replace=True)
            raise
        except Exception as error:
            record_failure(failure_dir, error, phase=phase, estimator=pipeline,
                           training_data=identity['training'])
            write_failure_json(run_dir / 'status.json', {'state': 'failed', 'phase': phase,
                                                        'failure': exception_details(error)}, replace=True)
            raise
