"""Bounded, training-only adaptive grids around external GridSearchCV.

Completed candidates are reused within one run. The estimator and its numerical
controls are never modified by the search policy. A plateau is a search stopping
decision, not a certificate of optimal hyperparameters.
"""
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import hash as parameter_hash, parallel_config
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from threadpoolctl import threadpool_limits

from search_support import (CandidateEvaluator, bare_parameters, data_fingerprint,
                            eligible_accuracy, exception_details, parameter_snapshot,
                            prefix_grid, record_failure, write_failure_json,
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

    def __post_init__(self):
        if self.regularization not in ('lambd', 'C') or not isinstance(self.prefix, str):
            raise ValueError('Use regularization lambd or C and a string prefix.')
        for name in ('max_expansions', 'max_candidates', 'top_k', 'plateau_rounds'):
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
    for name in config.axes:
        for side, edge in (('low', min(axes[name])), ('high', max(axes[name]))):
            if any(results['params'][index][name] == edge for index in tied):
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
        grids.append({**fixed, first: _neighbors(parent[0], axes[first]),
                      second: _neighbors(parent[1], axes[second])})
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
                        settings=None, verbose=0):
    """Evaluate only new candidates on fixed training folds, then refit once.

    The candidate/expansion budgets hold final fitting and held-out scoring if
    review remains unfinished. A documented plateau may end expansion even
    with a tied boundary. Saved results are reused within this run; interrupted
    stages are not automatically resumed across processes or new invocations.
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
    identity = {'training': data_fingerprint(X_train, y_train),
                'held_out': data_fingerprint(X_test, y_test)}
    metadata = {'schema_version': 1, 'policy': asdict(config), 'initial_grid': grid,
                'data': identity, 'folds': [{'train': t, 'validation': v} for t, v in folds],
                'settings': {'jobs': jobs, 'native_threads': 1, 'scoring': 'accuracy',
                             'error_policy': error_policy, 'parameters': parameter_snapshot(pipeline),
                             'user': {} if settings is None else settings}}
    write_json_atomic(run_dir / 'run.json', metadata)
    rows, stages, seen = [], [], set()
    searches = {}
    axes = {name: set(grid[name]) for name in config.axes}
    fixed = {name: values for name, values in grid.items() if name not in axes}
    expansion_count, refined = 0, False
    stalled = {}
    phase = 'coarse_cv'
    last_search = None
    review = None

    def evaluate(label, candidates):
        nonlocal phase, last_search
        phase = label + '_cv'
        write_json_atomic(run_dir / 'status.json', {
            'state': 'running', 'phase': phase, 'new_candidates': len(candidates),
            'completed_candidates': len(rows)}, replace=True)
        evaluator = CandidateEvaluator(pipeline, error_policy=error_policy,
                                       failure_dir=str(failure_dir))
        candidate_grid = [{name: [value] for name, value in parameters.items()}
                          for parameters in candidates]
        search = GridSearchCV(evaluator, prefix_grid(candidate_grid), cv=folds,
                              scoring=eligible_accuracy, n_jobs=jobs, pre_dispatch=jobs,
                              error_score='raise', refit=False, verbose=verbose)
        print(f'{label}: evaluating {len(candidates)} new candidates on {len(folds)} fixed folds.', flush=True)
        started = perf_counter()
        search.fit(X_train, y_train)
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
                 'parameters': candidates}
        stages.append(stage)
        write_json_atomic(run_dir / (label + '-cv.json'), {
            **stage, 'cv_results': {**current, 'params': [bare_parameters(p) for p in current['params']]},
            'fully_valid': fully_valid_candidate_mask(current)})
        write_json_atomic(run_dir / 'accumulated-cv.json', {
            'cv_results': _results(rows, len(folds)), 'stages': stages}, replace=True)
        last_search = search
        searches[label] = search
        ranked = ranked_valid_indices(_results(rows, len(folds)))
        if not len(ranked):
            raise ValueError('No candidate has finite scores on every validation fold.')
        print(f'{label}: best accumulated CV accuracy {rows[int(ranked[0])]["mean_test_score"]:.6f}; '
              f'{elapsed:.3f} seconds.', flush=True)

    def hold(reason):
        return {'state': 'needs_review', 'reasons': [reason],
                'final_refit_performed': False, 'held_out_scored': False}

    try:
        with parallel_config(backend='loky', inner_max_num_threads=1), threadpool_limits(limits=1):
            pending = _new_candidates(grid, seen, config.max_candidates)
            if pending is None:
                review = hold('candidate_budget: the complete initial grid exceeds max_candidates')
            else:
                evaluate('coarse', pending)
            while review is None:
                results = _results(rows, len(folds))
                ranked = ranked_valid_indices(results)
                best = float(results['mean_test_score'][ranked[0]])
                directions = _boundary_directions(results, axes, config)
                plateau = bool(directions) and all(stalled.get(direction, 0) >= config.plateau_rounds
                                                   for direction in directions)
                if directions and not plateau:
                    if expansion_count >= config.max_expansions:
                        review = hold('expansion_budget: best or tied-best parameters still reach the boundary')
                        break
                    try:
                        proposal = _expansion_grid(axes, fixed, directions, config)
                    except ValueError as error:
                        review = hold('parameter_limit: ' + str(error))
                        break
                    pending = _new_candidates(proposal, seen, config.max_candidates - len(rows))
                    if pending is None:
                        review = hold('candidate_budget: a complete boundary expansion would exceed max_candidates')
                        break
                    expansion_count += 1
                    evaluate(f'expansion-{expansion_count}', pending)
                    updated = max(row['mean_test_score'] for row in rows
                                  if math.isfinite(row['mean_test_score'])
                                  and all(math.isfinite(row[f'split{fold}_test_score']) for fold in range(len(folds))))
                    if updated > best + config.improvement_tol:
                        stalled.clear()
                    else:
                        stalled = {direction: stalled.get(direction, 0) + 1 for direction in directions}
                    continue
                if config.refine and not refined:
                    proposal = _refinement_grid(results, axes, fixed, config)
                    pending = _new_candidates(proposal, seen, config.max_candidates - len(rows))
                    if pending is None:
                        review = hold('candidate_budget: the complete local refinement would exceed max_candidates')
                        break
                    refined = True
                    if pending:
                        evaluate('local', pending)
                        updated_results = _results(rows, len(folds))
                        updated_best = updated_results['mean_test_score'][ranked_valid_indices(updated_results)[0]]
                        if updated_best > best + config.improvement_tol:
                            stalled.clear()
                        continue
                review = {'state': 'accepted', 'reasons': [],
                          'stopping_reason': 'plateau' if plateau else 'interior',
                          'plateau_is_optimality_certificate': False,
                          'final_refit_performed': False, 'held_out_scored': False}

            results = _results(rows, len(folds))
            ranked = ranked_valid_indices(results) if rows else []
            index = int(ranked[0]) if len(ranked) else None
            selection = None if index is None else {
                'candidate_index': index, 'parameters': rows[index]['params'],
                'source_stage': rows[index]['stage'],
                'mean_cv_accuracy': rows[index]['mean_test_score']}
            review.update({'evaluated_candidates': len(rows), 'expansion_rounds': expansion_count,
                           'refinement_completed': refined,
                           'boundary_directions': _boundary_directions(results, axes, config) if rows else []})
            write_json_atomic(run_dir / 'selection.json', selection)
            write_json_atomic(run_dir / 'review.json', review)
            result = {'model': None, 'completed': None, 'selection': selection,
                      'review': review, 'run_dir': run_dir, 'cv_results': results,
                      'stages': stages,
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
                model = clone(pipeline).set_params(**selection['parameters'])
                model.fit(X_train, y_train)
            except Exception as error:
                write_failure_json(run_dir / 'refit.json', {'state': 'failed', **selection,
                    'wall_seconds': perf_counter() - started, 'failure': exception_details(error)})
                raise
            refit_seconds = perf_counter() - started
            write_json_atomic(run_dir / 'refit.json', {'state': 'completed', **selection,
                                                      'wall_seconds': refit_seconds})
            review.update(final_refit_performed=True)
            write_json_atomic(run_dir / 'review.json', review, replace=True)
            phase = 'held_out_scoring'
            write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
            accuracy = float(accuracy_score(y_test, model.predict(X_test)))
            completed = {**selection, 'refit_seconds': refit_seconds, 'held_out_accuracy': accuracy}
            review.update(final_refit_performed=True, held_out_scored=True)
            write_json_atomic(run_dir / 'review.json', review, replace=True)
            write_json_atomic(run_dir / 'completed.json', completed)
            write_json_atomic(run_dir / 'status.json', {'state': 'completed', 'phase': phase,
                                                       'stopping_reason': review['stopping_reason']}, replace=True)
            result.update(model=model, completed=completed)
            return result
    except Exception as error:
        record_failure(failure_dir, error, phase=phase, estimator=locals().get('model', pipeline),
                       training_data=identity['training'])
        write_failure_json(run_dir / 'status.json', {'state': 'failed', 'phase': phase,
                                                    'failure': exception_details(error)}, replace=True)
        raise
