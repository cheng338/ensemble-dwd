"""Optional, bounded local refinement of selection ratio and coefficients.

This policy runs after the ordinary coefficient search has accepted its result.
It changes candidate selection only; fitting, fold scoring and numerical
controls remain the responsibility of the existing checkpointed CV workflow.
An accepted local search is empirical evidence, not an optimality certificate.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import math
from numbers import Integral, Real

import numpy as np
from joblib import hash as parameter_hash

from search_support import json_safe
from search_support import score_token, comparable_tokens


@dataclass(frozen=True)
class RatioRefinementConfig:
    bounds: tuple | None = None
    max_rounds: int = 3
    ratio_tol: float = .01
    stable_rounds: int = 2

    def __post_init__(self):
        for name in ('max_rounds', 'stable_rounds'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        try:
            finite_tolerance = math.isfinite(self.ratio_tol)
        except (TypeError, OverflowError):
            finite_tolerance = False
        if (isinstance(self.ratio_tol, (bool, np.bool_))
                or not isinstance(self.ratio_tol, Real)
                or not finite_tolerance or self.ratio_tol <= 0):
            raise ValueError('ratio_tol must be finite and positive.')
        if self.bounds is not None:
            if not isinstance(self.bounds, (tuple, list)) or len(self.bounds) != 2:
                raise ValueError('Ratio bounds must contain a lower and upper endpoint.')
            if any(type(value) not in (int, float) for value in self.bounds):
                raise ValueError('Ratio bounds must be finite Python numbers.')
            lower, upper = self.bounds
            if not 0 < lower < upper <= 1:
                raise ValueError('Ratio bounds must satisfy 0 < lower < upper <= 1.')
            object.__setattr__(self, 'bounds', (float(lower), float(upper)))


def validate_ratio_grid(initial_grid, coefficient_config, ratio_config):
    """Validate the opt-in policy and return its closed, declared ratio range."""
    # Imported here because adaptive_search exposes this policy to its callers.
    from adaptive_search import _initial_grid
    if not isinstance(ratio_config, RatioRefinementConfig):
        raise ValueError('ratio_config must be a RatioRefinementConfig.')
    if not coefficient_config.refine:
        raise ValueError('Ratio refinement requires coefficient refinement.')
    grid = _initial_grid(initial_grid, coefficient_config)
    name = coefficient_config.prefix + 'sv_selection_ratio'
    if name not in grid:
        raise ValueError(f'The initial grid must contain {name}.')
    values = grid[name]
    if any(type(value) not in (int, float) or not 0 < value <= 1
           for value in values):
        raise ValueError('Initial selection ratios must be finite Python numbers in (0, 1].')
    if len(set(values)) < 2:
        raise ValueError('Ratio refinement requires at least two distinct initial ratios.')
    bounds = ((float(min(values)), float(max(values))) if ratio_config.bounds is None
              else ratio_config.bounds)
    if any(not bounds[0] <= value <= bounds[1] for value in values):
        raise ValueError('Ratio bounds must contain every initial selection ratio.')
    return bounds


def _state_hash(state):
    content = {key: value for key, value in state.items() if key != 'state_hash'}
    encoded = json.dumps(json_safe(content), sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _arithmetic_midpoint(first, second):
    lower, upper = sorted((float(first), float(second)))
    value = lower + (upper - lower) / 2.
    return value if lower < value < upper else None


def _brackets(results, anchor, axes, bounds, coefficient_config, ratio_config):
    from adaptive_search import _adjacent, _conditional_values
    from tune_mnist import fully_valid_candidate_mask
    valid = fully_valid_candidate_mask(results)
    details, boundaries = {}, []
    for axis in axes:
        ratio = axis == axes[-1]
        value = float(anchor[axis])
        conditional = _conditional_values(results, anchor, axis)
        neighbors = _adjacent(value, conditional)
        side_details, satisfied = {}, []
        for position, side in enumerate(('low', 'high')):
            neighbor = neighbors[position]
            at_bound = ratio and value == bounds[position]
            finite = neighbor is not None and bool(valid[conditional[neighbor]])
            if ratio:
                narrow = finite and abs(neighbor - value) <= ratio_config.ratio_tol
            else:
                factor = (value / neighbor if position == 0 else neighbor / value) if finite else math.inf
                narrow = finite and factor <= coefficient_config.bracket_factor * (1 + 1e-14)
            resolved = at_bound or narrow
            satisfied.append(resolved)
            side_details.update({('lower' if position == 0 else 'upper'): neighbor,
                                 ('lower_valid' if position == 0 else 'upper_valid'): finite})
            if at_bound or not resolved:
                boundaries.append({'axis': axis, 'side': side,
                    'issue': 'declared_range_boundary' if at_bound else
                             'missing_neighbor' if neighbor is None else
                             'invalid_neighbor' if not finite else 'wide_bracket',
                    'resolved': bool(at_bound), 'coordinate': value,
                    'neighbor': neighbor})
        side_details['resolved'] = bool(all(satisfied))
        side_details['within_tolerance' if ratio else 'within_factor'] = bool(all(satisfied))
        if ratio:
            side_details['bounds'] = list(bounds)
            side_details['ratio_tol'] = float(ratio_config.ratio_tol)
        details[axis] = side_details
    return all(item['resolved'] for item in details.values()), details, boundaries


def _patches(results, parents, axes, bounds):
    """Return at most 27 tuple-preserving local points per selected parent."""
    from adaptive_search import (_adjacent, _conditional_values, _context,
                                 _midpoint, _outside)
    all_parameters = results['params']
    patches, exhausted = [], []
    for parent in parents:
        parent_key = parameter_hash(parent)
        context = parameter_hash(_context(parent, axes))
        support = [parameters for parameters in all_parameters
                   if parameter_hash(_context(parameters, axes)) == context]
        choices = []
        for axis in axes:
            value = float(parent[axis])
            ratio = axis == axes[-1]
            conditional = _conditional_values(results, parent, axis)
            neighbors = _adjacent(value, conditional)
            shared = sorted({float(parameters[axis]) for parameters in support})
            global_neighbors = _adjacent(value, shared)
            points = [parent[axis]]
            for position, side in enumerate(('low', 'high')):
                neighbor = neighbors[position]
                generated = None
                if ratio:
                    if value == bounds[position]:
                        continue
                    if neighbor is None:
                        generated = bounds[position]
                    else:
                        generated = _arithmetic_midpoint(value, neighbor)
                elif neighbor is not None:
                    # Invalid neighbors are narrowed inward too. They never
                    # establish support until a new finite score is observed.
                    generated = _midpoint(value, neighbor)
                elif global_neighbors[position] is not None:
                    # Another ratio or coefficient slice supplies only a
                    # coordinate to evaluate, never evidence for this slice.
                    generated = global_neighbors[position]
                else:
                    try:
                        generated = _outside(shared, side)
                    except ValueError:
                        pass
                if generated is None:
                    exhausted.append({'parent_key': parent_key, 'axis': axis,
                                      'side': side, 'issue': 'unrepresentable_midpoint_or_extension'})
                elif generated not in points:
                    points.append(float(generated))
            choices.append(sorted(points))
        for coordinates in product(*choices):
            overrides = {axis: float(value) for axis, value in zip(axes, coordinates)
                         if value != parent[axis]}
            candidate = {**deepcopy(parent), **overrides}
            patches.append((candidate, {'parent_key': parent_key, 'axes': overrides,
                                       'key': parameter_hash(candidate)}))
    return patches, exhausted


def next_ratio_stage(results, initial_grid, coefficient_config, ratio_config, state=None):
    """Propose one complete bounded local stage, or return a terminal review.

Only finite complete fold records can rank candidates or establish a local
bracket. Every scheduled stage must finish before it earns stability credit.
The saved policy state is JSON round-trip safe; candidate reconstruction uses
unchanged parent records to retain the original fixed parameter values/types.
"""
    from adaptive_search import (_evidence, _initial_grid,
                                 _policy_records)
    from tune_mnist import fully_valid_candidate_mask, ranked_valid_indices
    bounds = validate_ratio_grid(initial_grid, coefficient_config, ratio_config)
    grid = _initial_grid(initial_grid, coefficient_config)
    axes = (*coefficient_config.axes, coefficient_config.prefix + 'sv_selection_ratio')
    state = deepcopy({} if state is None else state)
    if state and state.get('state_hash') != _state_hash(state):
        raise ValueError('Ratio policy state checksum does not match its contents.')
    if state.get('schema_version', 1) != 1:
        raise ValueError('Unsupported ratio policy state schema.')
    identity = parameter_hash((asdict(coefficient_config), asdict(ratio_config), grid))
    if state.get('identity', identity) != identity:
        raise ValueError('Ratio policy state belongs to different settings or initial grid.')
    state.setdefault('schema_version', 1)
    state.setdefault('identity', identity)
    state.setdefault('rounds', 0)
    state.setdefault('stable_rounds', 0)
    state.setdefault('pending', None)
    state['bounds'] = list(bounds)

    # Validate input shapes before building records, so duplicate keys cannot
    # silently replace earlier observations or bypass the candidate budget.
    valid = fully_valid_candidate_mask(results)
    records = _policy_records(results)
    if len(records) != len(results['params']):
        raise ValueError('Ratio policy results contain duplicate candidate identities.')
    for parameters in results['params']:
        for axis in axes:
            if axis not in parameters or isinstance(parameters[axis], (bool, np.bool_)):
                raise ValueError('Ratio policy results are missing a valid numerical axis.')
            value = parameters[axis]
            if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError('Ratio policy axes must be finite positive numbers.')
        if not bounds[0] <= parameters[axes[-1]] <= bounds[1]:
            raise ValueError('A completed selection ratio is outside the declared bounds.')
    seen = set(records)
    ranked = ranked_valid_indices(results)
    leader = None if not len(ranked) else results['params'][int(ranked[0])]
    leader_key = None if leader is None else parameter_hash(leader)
    best = None if leader is None else float(results['mean_test_score'][ranked[0]])
    best_token = None if leader is None else score_token(results, ranked[0])
    valid_keys = {parameter_hash(results['params'][index]) for index in np.flatnonzero(valid)}
    parent_lookup = {parameter_hash(parameters): parameters for parameters in results['params']}
    proposal_description = None

    def answer(phase=None, candidates=None, review=None):
        state['state_hash'] = _state_hash(state)
        result = {'state': state, 'phase': phase,
                  'candidates': [] if candidates is None else candidates, 'review': review}
        if proposal_description is not None and review is not None:
            from stage_reporting import candidate_budget, safe_report
            result['proposal_report'] = safe_report(lambda: {
                'kind': 'ratio_refinement', 'decision': 'scheduled' if review is None else 'refused',
                'stopping_reason': None if review is None else review['stopping_reason'],
                'proposal': proposal_description,
                'budget': candidate_budget(coefficient_config.max_candidates, len(records),
                                           proposal_description.get('new_candidate_count'), review is None),
            })
        return result

    def finish(reason, accepted=False):
        state['last_reason'] = reason
        review = {'state': 'accepted' if accepted else 'needs_review',
                  'reasons': [] if accepted else [reason], 'stopping_reason': reason,
                  'plateau_is_optimality_certificate': False,
                  'final_refit_performed': False, 'held_out_scored': False,
                  'range_boundary': [item for item in state.get('boundary_directions', [])
                                     if item['issue'] == 'declared_range_boundary']}
        return answer(review=review)

    if len(records) > coefficient_config.max_candidates:
        return finish('candidate_budget: completed records already exceed max_candidates')
    pending = state.get('pending')
    if pending is not None:
        if _evidence(records, pending['baseline_keys']) != pending['baseline_evidence']:
            state['stable_rounds'] = 0
            return finish('changed_evidence: earlier candidate scores or parameters changed')
        pending_keys = {item['key'] for item in pending['candidates']}
        if seen - set(pending['baseline_keys']) - pending_keys:
            state['stable_rounds'] = 0
            return finish('changed_evidence: records appeared outside the pending local stage')
        missing = []
        for item in pending['candidates']:
            if item['key'] in seen:
                continue
            candidate = {**deepcopy(parent_lookup[item['parent_key']]), **item['axes']}
            if parameter_hash(candidate) != item['key']:
                raise ValueError('Ratio policy pending candidate identity changed.')
            missing.append(candidate)
        if missing:
            return answer(phase=pending['label'], candidates=missing)
        current, previous = comparable_tokens(best_token, pending.get('baseline_score_token'),
                                               best, pending['baseline_best'])
        steady = (leader_key == pending['baseline_leader'] and current == previous
                  and pending_keys.issubset(valid_keys))
        state['stable_rounds'] = state['stable_rounds'] + 1 if steady else 0
        state['rounds'] += 1
        state['pending'] = None
    elif state.get('observed_evidence') not in (None, _evidence(records)):
        state['stable_rounds'] = 0
        return finish('changed_evidence: observations changed outside a pending local stage')
    state['observed_evidence'] = _evidence(records)

    if leader is None:
        return finish('ineligible: no candidate has finite scores on every validation fold')
    bracketed, bracket, boundaries = _brackets(
        results, leader, axes, bounds, coefficient_config, ratio_config)
    state.update(winner_bracket=bracket, winner_bracket_resolved=bracketed,
                 boundary_directions=boundaries, best_parameters=json_safe(leader),
                 best_mean_score=best)
    if bracketed and state['stable_rounds'] >= ratio_config.stable_rounds:
        return finish('valid local coefficient and ratio brackets with stable completed rounds; '
                      'restricted to the declared ratio range, not an optimality certificate', accepted=True)
    if state['rounds'] >= ratio_config.max_rounds:
        return finish('ratio_round_budget: local bracket or stability remains unresolved')

    parents = [results['params'][int(index)] for index in ranked[:coefficient_config.top_k]]
    patches, exhausted = _patches(results, parents, axes, bounds)
    state['parameter_limits'] = exhausted
    if exhausted:
        from stage_reporting import encode, safe_report
        proposal_description = safe_report(lambda: {
            'proposal_kind': 'ratio_refinement', 'count_status': 'unavailable_parameter_limit',
            'new_candidate_count': None, 'candidate_list_complete': False,
            'generated_partial_parameters': encode([candidate for candidate, _ in patches]),
            'parameter_limits': encode(exhausted),
            'remaining_candidate_allowance': coefficient_config.max_candidates - len(records),
        })
        return finish('parameter_limit: a complete local neighborhood exceeds representable arithmetic')
    candidates, descriptors, new_keys = [], [], set()
    for candidate, descriptor in patches:
        key = descriptor['key']
        if key in seen or key in new_keys:
            continue
        new_keys.add(key)
        candidates.append(candidate)
        descriptors.append(descriptor)
    if not candidates or len(records) + len(candidates) > coefficient_config.max_candidates:
        from stage_reporting import explicit_proposal, safe_report
        proposal_description = safe_report(lambda: explicit_proposal(
            candidates, seen, coefficient_config.max_candidates - len(records), 'ratio_refinement'))
    if not candidates:
        return finish('parameter_limit: no new representable local candidate is available')
    if len(records) + len(candidates) > coefficient_config.max_candidates:
        return finish('candidate_budget: the complete ratio refinement stage exceeds max_candidates')
    label = 'ratio-local-' + str(state['rounds'] + 1)
    state['pending'] = {'label': label, 'candidates': descriptors,
                        'baseline_keys': sorted(records), 'baseline_evidence': _evidence(records),
                        'baseline_leader': leader_key, 'baseline_best': best,
                        'baseline_score_token': best_token}
    state['last_reason'] = 'local_ratio_refinement'
    return answer(phase=label, candidates=candidates)
