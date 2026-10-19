"""Optional training-only sensitivity checks for a larger DWD iteration cap.

The original search remains authoritative. Confirmation never promotes a new
winner, alters stopping tolerances, certifies convergence, or scores held-out
data. Its independent checkpoint context cannot replay original-budget folds.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral
from pathlib import Path
from time import perf_counter

from joblib import hash as parameter_hash
import numpy as np
from sklearn.base import clone

from cv_checkpoint import CheckpointError, CheckpointedCV
from ensemble_dwd import EnsembleKernelDWD
from search_support import _io_path, bare_parameters, json_safe
from search_support import score_value, score_ranks, exact_accuracy_available


@dataclass(frozen=True)
class NumericalConfirmationConfig:
    """Declare the higher iteration cap and shortlist size before searching."""

    max_iter: int
    top_k: int = 3

    def __post_init__(self):
        for name, minimum in (('max_iter', 1), ('top_k', 2)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Integral) or value < minimum):
                raise ValueError(f'{name} must be an integer >= {minimum}.')


def _target_path(search_config):
    prefix = search_config.prefix
    if not isinstance(prefix, str) or (prefix and not prefix.endswith('__')):
        raise ValueError('The DWD parameter prefix must be empty or end in __.')
    path = prefix[:-2] if prefix else ''
    if path and any(not part for part in path.split('__')):
        raise ValueError('The DWD parameter prefix contains an empty component.')
    return path


def _forbidden_names(search_config):
    path = _target_path(search_config)
    parts = path.split('__') if path else []
    ancestors = {'__'.join(parts[:length]) for length in range(1, len(parts) + 1)}
    replaced_steps = {'__'.join(parts[:length] + ['steps'])
                      for length in range(len(parts) + 1)}
    return ancestors | replaced_steps | {search_config.prefix + 'max_iter'}


def validate_confirmation(pipeline, initial_grid, search_config, confirmation):
    """Return the fixed baseline profile, rejecting ambiguous budget changes."""
    if not isinstance(confirmation, NumericalConfirmationConfig):
        raise TypeError('confirmation must be NumericalConfirmationConfig.')
    if search_config.regularization != 'lambd':
        raise ValueError('Numerical-budget confirmation supports ensemble DWD only.')
    if not isinstance(initial_grid, dict):
        raise ValueError('The initial confirmation grid must be one dictionary.')
    conflicts = _forbidden_names(search_config).intersection(initial_grid)
    if conflicts:
        raise ValueError('Confirmation cannot tune max_iter or replace its estimator path: '
                         + ', '.join(sorted(conflicts)))
    parameters = pipeline.get_params(deep=True)
    path = _target_path(search_config)
    target = parameters.get(path) if path else pipeline
    if type(target) is not EnsembleKernelDWD:
        raise ValueError('Confirmation requires an exact EnsembleKernelDWD at the configured prefix.')
    baseline = target.max_iter
    if (isinstance(baseline, (bool, np.bool_)) or not isinstance(baseline, Integral)
            or baseline < 1):
        raise ValueError('The original DWD max_iter must be a positive integer.')
    if confirmation.max_iter <= baseline:
        raise ValueError('The confirmation max_iter must be strictly larger than the original cap.')
    return json_safe({'parameter': search_config.prefix + 'max_iter',
                      'max_iter': baseline, 'confirmation_max_iter': confirmation.max_iter,
                      'stopping': target.stopping, 'obj_tol': target.obj_tol,
                      'tol': target.tol, 'acceleration': target.acceleration,
                      'backend': target.backend,
                      'residual_check_order': target.residual_check_order})


def _slice_results(results, indices, parameters):
    """Keep fold and timing evidence in original candidate order and labels."""
    names = ('mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time',
             'mean_score_time', 'std_score_time')
    selected = {'params': deepcopy(parameters)}
    if 'accuracy_evidence' in results:
        selected['accuracy_evidence'] = [deepcopy(results['accuracy_evidence'][index]) for index in indices]
    for key in results:
        if key in names or (key.startswith('split') and key.endswith('_test_score')):
            selected[key] = np.asarray(results[key])[indices].copy()
    return selected


def _candidate_evidence(search, index, n_folds):
    current = search.cv_results_
    summary = search.checkpoint_summary_
    result = {'mean_cv_accuracy': float(current['mean_test_score'][index]),
              'accuracy_evidence': deepcopy(current.get('accuracy_evidence', [None] * len(current['params']))[index]),
              'fold_scores': [float(current[f'split{fold}_test_score'][index])
                              for fold in range(n_folds)],
              'diagnostics': deepcopy(summary.get('diagnostics', [])[index])}
    for name in ('mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'):
        if name in current:
            result[name] = float(current[name][index])
    seed = getattr(search, 'seed_results_', None)
    if seed is not None:
        result['seed_results'] = {
            'axes': ['seed', 'fold'], 'seeds': list(seed['seeds']),
            'seed_parameter': seed['seed_parameter'],
            'aggregation_policy': seed['aggregation_policy']}
        for key in ('scores', 'accuracy_counts', 'fit_times', 'score_times', 'states', 'diagnostics',
                    'receipt_task_sha256', 'receipt_parameters', 'reused', 'equivalent_reuse'):
            if key in seed:
                result['seed_results'][key] = deepcopy(seed[key][index])
    return result


def _diagnostic_summary(evidence, max_iter):
    """Count reported learner facts once, including every OvO pair and base."""
    reasons = Counter()
    facts = {'learner_records': 0, 'converged_true': 0, 'converged_false': 0,
             'converged_unknown': 0, 'objective_tolerance_true': 0,
             'objective_tolerance_false': 0, 'objective_tolerance_unknown': 0,
             'optimality_true': 0, 'optimality_false': 0, 'optimality_unknown': 0,
             'criterion_reached_true': 0, 'criterion_reached_false': 0,
             'criterion_reached_unknown': 0, 'iteration_cap_reached': 0}

    def visit(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        # Wrapper arrays summarize children. Count the original learner records
        # only, rather than counting the same final model twice.
        if 'pairwise_estimators' in value:
            for item in value['pairwise_estimators']:
                visit(item.get('diagnostics', {}))
            return
        if 'base_diagnostics_' in value or 'final_estimator' in value:
            visit(value.get('base_diagnostics_', []))
            visit(value.get('final_estimator', {}))
            return
        relevant = ('n_iter', 'n_iter_', 'termination_reason', 'termination_reason_',
                    'stopping_reason', 'stopping_reason_', 'converged', 'converged_')
        if not any(name in value for name in relevant):
            return
        facts['learner_records'] += 1
        reason = next((value[name] for name in ('termination_reason_', 'termination_reason',
                      'stopping_reason_', 'stopping_reason') if name in value), None)
        if isinstance(reason, str):
            reasons[reason] += 1
        converged = value.get('converged_', value.get('converged'))
        if isinstance(converged, (bool, np.bool_)):
            facts['converged_true' if converged else 'converged_false'] += 1
        else:
            facts['converged_unknown'] += 1
        for field, prefix in (('objective_tolerance_met', 'objective_tolerance'),
                              ('optimality_met', 'optimality'), ('criterion_reached', 'criterion_reached')):
            flag = value.get(field + '_', value.get(field))
            suffix = ('true' if flag else 'false') if isinstance(flag, (bool, np.bool_)) else 'unknown'
            facts[prefix + '_' + suffix] += 1
        iterations = value.get('n_iter_', value.get('n_iter'))
        if isinstance(iterations, Integral) and not isinstance(iterations, (bool, np.bool_)):
            facts['iteration_cap_reached'] += int(iterations >= max_iter)

    for item in evidence:
        visit(item['diagnostics'])
    return {**facts, 'termination_reasons': dict(sorted(reasons.items())),
            'interpretation': 'Reported facts only; cap sensitivity and stable accuracy do not establish convergence.'}


def _ranks(means):
    means = np.asarray(means, dtype=float)
    return [1 + int(np.sum(means > value)) if np.isfinite(value) else None for value in means]


def compare_rankings(baseline, confirmation):
    """Compare the complete ordering, including exact ties, of one shortlist."""
    from tune_mnist import fully_valid_candidate_mask, ranked_valid_indices

    if not fully_valid_candidate_mask(baseline).all():
        raise ValueError('Confirmation requires a fully valid original shortlist.')
    eligible = fully_valid_candidate_mask(confirmation)
    exact = exact_accuracy_available(baseline, confirmation)
    before = [score_value(baseline, index, exact=exact) for index in range(len(baseline['params']))]
    after = [score_value(confirmation, index, exact=exact) for index in range(len(confirmation['params']))]
    if len(before) != len(after):
        raise ValueError('Confirmation changed the shortlist length.')
    def ordering(values):
        return [[int(first > second) - int(first < second) for second in values] for first in values]
    changed = bool(not eligible.all() or ordering(before) != ordering(after))
    reasons = []
    if not eligible.all():
        reasons.append('At least one shortlist candidate is ineligible at the confirmation cap.')
    if changed and eligible.all():
        reasons.append('The shortlist ordering or exact ties changed at the confirmation cap.')
    return {'state': 'needs_review' if changed else 'stable', 'reasons': reasons,
            'stopping_reason': 'confirmation_requires_review' if changed else 'confirmation_ranking_stable',
            'ranking_changed': changed, 'eligible': eligible,
            'comparison': 'exact_accuracy_counts' if exact else 'reported_scalar_scores',
            'baseline_order': sorted(range(len(before)), key=lambda i: before[i], reverse=True),
            'confirmation_order': sorted(np.flatnonzero(eligible).tolist(), key=lambda i: after[i], reverse=True),
            'baseline_ranks': score_ranks(baseline, exact=exact).tolist(),
            'confirmation_ranks': [int(rank) if ok else None for rank, ok in
                                   zip(score_ranks(confirmation, exact=exact), eligible)]}


def run_numerical_confirmation(*, pipeline, results, rows, searches, folds, X, y,
                               run_dir, config, search_config, jobs, error_policy,
                               resume, cv_seeds, seed_parameter,
                               reuse_equivalent_folds, parent_identity,
                               pair_recovery=False, pair_recovery_max_bytes=None):
    """Evaluate a frozen shortlist in an independent, resumable CV context."""
    from tune_mnist import ranked_valid_indices

    if type(pair_recovery) is not bool:
        raise ValueError('pair_recovery must be boolean.')
    if pair_recovery_max_bytes is not None:
        if not pair_recovery:
            raise ValueError('pair_recovery_max_bytes requires pair_recovery.')
        if type(pair_recovery_max_bytes) is not int or pair_recovery_max_bytes < 0:
            raise ValueError('pair_recovery_max_bytes must be a nonnegative integer or None.')
    if pair_recovery and cv_seeds is not None:
        raise ValueError('Pair recovery does not support multi-seed CV.')
    started = perf_counter()
    profile = validate_confirmation(pipeline, {}, search_config, config)
    indices = [int(index) for index in ranked_valid_indices(results)[:config.top_k]]
    parameters = [deepcopy(results['params'][index]) for index in indices]
    for candidate in parameters:
        validate_confirmation(pipeline, candidate, search_config, config)
    hashes = [parameter_hash(candidate) for candidate in parameters]
    if len(set(hashes)) != len(hashes):
        raise ValueError('The confirmation shortlist contains duplicate candidate identities.')
    report = {'config': asdict(config), 'profile': profile,
              'profile_scope': 'Original pipeline defaults; each shortlist profile includes its candidate overrides.',
              'candidate_count': len(indices), 'shortlist_indices': indices,
              'shortlist': [], 'parent_identity_sha256': parent_identity,
              'final_model_policy': 'Retain the original selected parameters and iteration cap; hold final fitting if review is required.',
              'interpretation': 'This training-only shortlist diagnostic measures iteration-cap sensitivity. It does not certify convergence, a global optimum, or new-budget parameter brackets.',
              'held_out_scored': False, 'final_refit_performed': False}
    if len(indices) < 2:
        return json_safe({**report, 'state': 'needs_review',
                          'reasons': ['Confirmation requires at least two fully valid original candidates.'],
                          'stopping_reason': 'insufficient_confirmation_candidates'})

    baseline_evidence, baseline_stage_summaries = [], {}
    for index, candidate, digest in zip(indices, parameters, hashes, strict=True):
        row = rows[index]
        stage, local_index = row['stage'], row['stage_index']
        search = searches[stage]
        original = bare_parameters(search.cv_results_['params'][local_index])
        if (parameter_hash(row['params']) != digest or parameter_hash(original) != digest
                or any(float(results[f'split{fold}_test_score'][index]) !=
                       float(search.cv_results_[f'split{fold}_test_score'][local_index])
                       for fold in range(len(folds)))
                or float(results['mean_test_score'][index]) !=
                   float(search.cv_results_['mean_test_score'][local_index])
                or results.get('accuracy_evidence', [None] * len(results['params']))[index] !=
                   search.cv_results_.get('accuracy_evidence', [None] * len(search.cv_results_['params']))[local_index]):
            raise ValueError('The confirmation baseline does not match its original stage evidence.')
        evidence = _candidate_evidence(search, local_index, len(folds))
        baseline_evidence.append(evidence)
        report['shortlist'].append({'candidate_index': index, 'parameters': candidate,
                                   'parameter_hash': digest, 'source_stage': stage,
                                   'source_stage_index': int(local_index), 'baseline': evidence,
                                   'profile': validate_confirmation(
                                       clone(pipeline).set_params(**candidate), {}, search_config, config)})
        baseline_stage_summaries[stage] = {
            'scope': 'Entire original stage, including candidates outside the shortlist.',
            # Detailed diagnostics and seed receipts are already retained for
            # the shortlist. Do not copy unrelated candidates into this report.
            'checkpoint_summary': {name: deepcopy(search.checkpoint_summary_[name])
                for name in ('stage', 'candidate_count', 'fold_count', 'seed_count',
                             'evaluation_count', 'new_folds', 'reused_folds',
                             'executed_folds', 'equivalent_reused_folds',
                             'wall_seconds', 'seed_policy', 'seed_counts',
                             'equivalent_fold_policy', 'timing_policy', 'axes')
                if name in search.checkpoint_summary_}}

    higher = clone(pipeline).set_params(**{profile['parameter']: config.max_iter})
    child_dir = Path(run_dir) / 'numerical-confirmation'
    checkpoint_dir = _io_path(child_dir / 'cv-checkpoints')
    child_exists = (checkpoint_dir / 'identity.json').exists()
    if not child_exists and any(
            path.is_file() for name in ('folds', 'attempts', 'artifacts')
            for path in (checkpoint_dir / name).rglob('*')):
        raise CheckpointError('Confirmation receipts exist without their checkpoint identity; use a new run directory.')
    # The parent can resume before ever reaching this optional stage. Only a
    # pre-existing child identity requests child resume; corruption is not reset.
    child_resume = bool(resume and child_exists)
    source_hash = hashlib.sha256(_io_path(Path(__file__)).read_bytes()).hexdigest()
    baseline_results = _slice_results(results, indices, parameters)
    provenance = {'purpose': 'dwd-iteration-cap-confirmation-v1',
                  'parent_identity_sha256': parent_identity, 'config': asdict(config),
                  'profile': profile, 'shortlist_indices': indices,
                  'shortlist_parameter_hashes': hashes,
                  'baseline_scores_hash': parameter_hash({key: value for key, value in baseline_results.items()
                    if key in ('mean_test_score', 'accuracy_evidence') or (key.startswith('split') and key.endswith('_test_score'))}),
                  'source_sha256': source_hash}
    pair_options = {}
    if pair_recovery:
        pair_options = {'pair_recovery': True, 'pair_recovery_max_bytes': pair_recovery_max_bytes}
        provenance['pair_recovery'] = {'enabled': True, 'max_bytes': pair_recovery_max_bytes,
                                      'scope': 'unfinished_cv_folds'}
    with CheckpointedCV(child_dir, higher, X, y, folds, jobs=jobs,
                        error_policy=error_policy, resume=child_resume,
                        provenance=provenance, reuse_equivalent_folds=reuse_equivalent_folds,
                        cv_seeds=cv_seeds, seed_parameter=seed_parameter, **pair_options) as checkpoint:
        _io_path(child_dir / 'failures').mkdir(parents=True, exist_ok=True)
        confirmation = checkpoint.search(
            [{name: [value] for name, value in candidate.items()} for candidate in parameters],
            stage='numerical_confirmation')
        report['checkpoint_identity_sha256'] = checkpoint.identity_sha256
        report['checkpoint_summary'] = deepcopy(confirmation.checkpoint_summary_)
        report['checkpoint_summary'].setdefault(
            'executed_folds', report['checkpoint_summary']['new_folds'])
        report['checkpoint_summary'].setdefault('equivalent_reused_folds', 0)

    confirmed = _slice_results(confirmation.cv_results_, list(range(len(indices))), parameters)
    comparison = compare_rankings(baseline_results, confirmed)
    confirmation_evidence = [_candidate_evidence(confirmation, index, len(folds))
                             for index in range(len(indices))]
    for position, item in enumerate(report['shortlist']):
        before, after = baseline_evidence[position], confirmation_evidence[position]
        item.update(confirmation=after, eligible=bool(comparison['eligible'][position]),
                    baseline_rank=comparison['baseline_ranks'][position],
                    confirmation_rank=comparison['confirmation_ranks'][position],
                    mean_score_delta=after['mean_cv_accuracy'] - before['mean_cv_accuracy'],
                    fold_score_deltas=(np.asarray(after['fold_scores']) - np.asarray(before['fold_scores'])))
        if 'seed_results' in before:
            item['seed_score_deltas'] = (np.asarray(after['seed_results']['scores'])
                                        - np.asarray(before['seed_results']['scores']))
    report.update({name: comparison[name] for name in
                   ('state', 'reasons', 'stopping_reason', 'ranking_changed', 'comparison')})
    report.update(cv_results={'baseline': baseline_results, 'confirmation': confirmed},
                  wall_seconds=perf_counter() - started,
                  baseline_ranking=[indices[index] for index in comparison['baseline_order']],
                  confirmation_ranking=[indices[index] for index in comparison['confirmation_order']],
                  baseline_stage_summaries=baseline_stage_summaries,
                  diagnostics_summary={
                      'baseline': _diagnostic_summary(baseline_evidence, profile['max_iter']),
                      'confirmation': _diagnostic_summary(confirmation_evidence, config.max_iter)})
    return json_safe(report)
