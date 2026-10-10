"""Adaptive external search on a local MNIST archive.

Archive keys may be x_train/y_train/x_test/y_test (standard Keras format),
X_train/y_train/X_test/y_test, or X/y. Images may be flattened or have shape
(n, 28, 28). With X/y, a stratified 25% test split is made before CV. A supplied
test split remains held out. Data are never downloaded.
The CLI expands lambda/gamma boundaries and repeatedly refines joint local
settings on the same training folds. Unresolved boundaries or exhausted budgets hold final
fitting/test scoring. A plateau does not prove global optimality. The legacy
run_search_workflow helper remains available as a nonresumable coarse/local workflow.
Explicit --resume --run reuses completed folds from --output-dir under identical
settings, budgets and runtime. Interrupted final fitting restarts; a committed
completed result resumes without fitting or scoring and returns model=None.

Preview: python examples/tune_mnist.py --refine
Run: python examples/tune_mnist.py --data mnist.npz --refine --run --jobs 1
"""
import argparse
from copy import deepcopy
from pathlib import Path
import pickle
import re
import sys
from time import perf_counter

import numpy as np
from joblib import hash as joblib_hash, parallel_config
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelDWD, mnist_param_grid

# Also support loading this example by its file path, as the example tests do.
EXAMPLES_DIRECTORY = str(Path(__file__).resolve().parent)
if EXAMPLES_DIRECTORY not in sys.path:
    sys.path.insert(0, EXAMPLES_DIRECTORY)
from search_support import (_io_path, _resolved_public_path, CandidateEvaluator,
                             bare_parameters, data_fingerprint,
                            eligible_accuracy, exception_details, new_run_directory,
                            parameter_snapshot, prefix_grid, record_failure,
                            write_failure_json, write_json_atomic)
from search_support import score_value

from phase_threads import (add_final_thread_argument, report_final_threads,
                           final_fit_threads, final_thread_policy_sha256,
                           validate_final_native_threads)


def select_pair(X, y, first, second):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim < 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError('Expected an image/feature array and one label per row.')
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError('MNIST features must be real numbers.')
    if not np.isfinite(X).all() or any(size == 0 for size in X.shape[1:]):
        raise ValueError('MNIST features must be nonempty and finite.')
    if np.iscomplexobj(y):
        raise ValueError('MNIST labels must be real integer digit values.')
    labels = np.asarray(y, dtype=np.float64)
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('MNIST labels must be finite integer digit values.')
    keep = (labels == first) | (labels == second)
    if np.unique(labels[keep]).size != 2:
        raise ValueError('Both requested digits must occur in every supplied split.')
    with np.errstate(over='ignore', invalid='ignore'):
        features = np.asarray(X[keep].reshape(np.count_nonzero(keep), -1), dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError('MNIST features must remain finite when converted to float64.')
    return features, labels[keep].astype(np.int64)


def load_local_data(path, first, second):
    with np.load(path, allow_pickle=False) as archive:
        for train_key, test_key in (('x_train', 'x_test'), ('X_train', 'X_test')):
            if {train_key, 'y_train', test_key, 'y_test'} <= set(archive.files):
                X_train, y_train = select_pair(archive[train_key], archive['y_train'], first, second)
                X_test, y_test = select_pair(archive[test_key], archive['y_test'], first, second)
                if X_train.shape[1] != X_test.shape[1]:
                    raise ValueError('Training and test feature counts differ.')
                return X_train, X_test, y_train, y_test
        if {'X', 'y'} <= set(archive.files):
            X, y = select_pair(archive['X'], archive['y'], first, second)
            return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    raise ValueError('Archive must contain X/y, x_train/y_train/x_test/y_test, '
                     'or X_train/y_train/X_test/y_test.')


def positive_int(value):
    if isinstance(value, (bool, float, np.floating)):
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Expected a positive integer.') from exc
    if number < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return number


def seed_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Expected an integer seed between 0 and 4294967295.') from exc
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError('Expected an integer seed between 0 and 4294967295.')
    return number


def add_seed_arguments(parser):
    parser.add_argument('--cv-seeds', type=seed_int, nargs='+',
                        help='Optional distinct sampling seeds shared by every candidate; average seeds within each fixed fold. Default: one seed, 42.')
    parser.add_argument('--refit-seed', type=seed_int,
                        help='Seed for the separate full-training refit; requires --cv-seeds. Default: 42.')


def seed_search_options(parser, args):
    from cv_checkpoint import validate_cv_seeds
    try:
        seeds = validate_cv_seeds(args.cv_seeds)
    except ValueError as error:
        parser.error(str(error))
    if seeds is None:
        if args.refit_seed is not None:
            parser.error('--refit-seed requires --cv-seeds.')
        return {}
    return {'cv_seeds': seeds, 'refit_seed': 42 if args.refit_seed is None else args.refit_seed}


def report_seed_options(options):
    if options:
        print(f"Sampling seeds: {list(options['cv_seeds'])}; average seeds within each fold, then folds. "
              f"Separate final-refit seed: {options['refit_seed']}.")
        print('Candidate budgets count parameter sets; CV fits also multiply by the number of seeds. '
              'Every requested fold and seed must be valid; seeds are never selected by validation accuracy.')


def add_ratio_arguments(parser):
    parser.add_argument('--refine-ratio', action='store_true',
                        help='After coefficient review, refine selection ratios in bounded joint local rounds; disabled by default.')
    parser.add_argument('--ratio-bounds', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                        help='Declared ratio search bounds in (0, 1], containing the initial grid; requires --refine-ratio. Default: initial ratio range.')
    parser.add_argument('--max-ratio-rounds', type=positive_int,
                        help='Maximum completed ratio refinement rounds; requires --refine-ratio. Default: 3.')
    parser.add_argument('--ratio-tol', type=float,
                        help='Target absolute gap to valid ratio neighbors; requires --refine-ratio. Default: 0.01 (one percentage point).')


def ratio_search_options(parser, args, grid, *, prefix='ensemble__', regularization='lambd', smoke=False):
    modified = any(value is not None for value in (args.ratio_bounds, args.max_ratio_rounds, args.ratio_tol))
    if not args.refine_ratio:
        if modified:
            parser.error('Ratio controls require --refine-ratio.')
        return {}
    if not args.refine or smoke:
        parser.error('--refine-ratio requires coefficient refinement and is incompatible with --no-refine or --smoke-grid.')
    from adaptive_search import AdaptiveSearchConfig
    from ratio_refinement import RatioRefinementConfig, validate_ratio_grid
    try:
        policy = RatioRefinementConfig(
            bounds=None if args.ratio_bounds is None else tuple(args.ratio_bounds),
            max_rounds=3 if args.max_ratio_rounds is None else args.max_ratio_rounds,
            ratio_tol=.01 if args.ratio_tol is None else args.ratio_tol)
        coefficient_config = AdaptiveSearchConfig(prefix=prefix, regularization=regularization)
        bounds = validate_ratio_grid(grid, coefficient_config, policy)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(f'Optional ratio refinement: bounds {bounds[0]:g} to {bounds[1]:g}; '
          f'at most {policy.max_rounds} completed local rounds; absolute ratio gap {policy.ratio_tol:g}.')
    print('Targeted joint neighborhoods share the total candidate budget and fixed folds/seeds. '
          'Unresolved follow-up work holds final fitting; local stopping does not establish a global optimum.')
    return {'ratio_refinement': policy}


def add_confirmation_arguments(parser):
    parser.add_argument('--confirm-max-iter', type=positive_int,
                        help='Optional DWD iteration cap for a separate shortlist confirmation; must exceed the original cap. Disabled by default.')
    parser.add_argument('--confirm-top-k', type=positive_int,
                        help='Maximum ranked candidates in the confirmation shortlist, including the original winner; minimum 2, default 3. Requires --confirm-max-iter.')


def confirmation_search_options(parser, args):
    if args.confirm_max_iter is None:
        if args.confirm_top_k is not None:
            parser.error('--confirm-top-k requires --confirm-max-iter.')
        return {}
    from numerical_confirmation import NumericalConfirmationConfig
    try:
        confirmation = NumericalConfirmationConfig(
            max_iter=args.confirm_max_iter,
            top_k=3 if args.confirm_top_k is None else args.confirm_top_k)
        baseline = EnsembleKernelDWD().max_iter
        if confirmation.max_iter <= baseline:
            raise ValueError(f'--confirm-max-iter must exceed the original DWD cap ({baseline}).')
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return {'numerical_confirmation': confirmation}


def report_confirmation_options(options, *, folds, seed_count, pairs=1):
    if not options:
        return
    confirmation = options['numerical_confirmation']
    fits = confirmation.top_k * folds * seed_count
    kind = 'ensemble' if pairs == 1 else 'OVO'
    print(f'Optional numerical confirmation: at most {confirmation.top_k} ranked candidates '
          f'at max_iter={confirmation.max_iter}; up to {fits} additional {kind} CV fits '
          f'and {fits * pairs * 6} DWD fits before reuse.')
    print('Confirmation has a separate evaluation budget and checkpoints, using the same folds and seeds. '
          'Rank or tie changes hold final fitting; stable ranking retains the original selected parameters and iteration cap.')
    print('Tolerances and stopping rules are unchanged. A larger cap may leave objective-tolerance exits unchanged; '
          'stable scores do not establish convergence.')


def fully_valid_candidate_mask(cv_results):
    """Require every fold score and the aggregate score to be finite."""
    split_keys = sorted(key for key in cv_results if re.fullmatch(r'split\d+_test_score', key))
    if not split_keys:
        raise ValueError('Cross-validation results contain no individual fold scores.')
    split_numbers = sorted(int(key[5:key.index('_')]) for key in split_keys)
    if split_numbers != list(range(len(split_keys))):
        raise ValueError('Cross-validation fold-score indices must be consecutive from zero.')
    means = np.ma.asarray(cv_results['mean_test_score'], dtype=float).filled(np.nan)
    if means.ndim != 1 or len(means) != len(cv_results['params']):
        raise ValueError('Cross-validation candidate and mean-score counts differ.')
    valid = np.isfinite(means)
    for key in split_keys:
        scores = np.ma.asarray(cv_results[key], dtype=float).filled(np.nan)
        if scores.shape != means.shape:
            raise ValueError('Cross-validation fold-score shapes differ.')
        valid &= np.isfinite(scores)
    return valid


def ranked_valid_indices(cv_results):
    """Rank only complete candidates, keeping original order for score ties."""
    indices = np.flatnonzero(fully_valid_candidate_mask(cv_results))
    return np.asarray(sorted(indices, key=lambda index: score_value(cv_results, index),
                             reverse=True), dtype=int)


def best_valid_candidate(cv_results):
    """GridSearchCV refit selector; no model is selected if all candidates fail."""
    ranked = ranked_valid_indices(cv_results)
    if not len(ranked):
        raise ValueError('No candidate has finite scores on every validation fold '
                         f"({len(cv_results['params'])} invalid candidates).")
    return int(ranked[0])


def local_refinement_grid(cv_results, prefix='ensemble__', top_k=3):
    """Union local neighborhoods around up to three fully valid coarse results.

    Every parent parameter is retained; only lambda and gamma are varied.
    No clipping to the coarse grid is applied. Formatting lambda/gamma
    deduplication keys to 15 significant digits merges floating-point variants
    of the same values. Other parameter values distinguish neighborhoods.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError('top_k must be a positive integer.')
    ranked = ranked_valid_indices(cv_results)
    if not len(ranked):
        raise ValueError('Local refinement requires a fully valid coarse candidate.')
    names = [prefix+name for name in ('lambd', 'gamma', 'theta', 'sv_selection_ratio')]
    grid, seen = [], set()
    for index in ranked[:top_k]:
        parent = cv_results['params'][int(index)]
        lambd, gamma, theta, ratio = [float(parent[name]) for name in names]
        if not np.isfinite([lambd, gamma, theta, ratio]).all() or min(lambd, gamma) <= 0:
            raise ValueError('Refinement requires finite parameters and positive lambda/gamma.')
        other_parameters = {name: value for name, value in parent.items()
                            if name not in names[:2]}
        try:
            other_key = joblib_hash(other_parameters)
        except pickle.PicklingError:
            # Local callable kernels are valid with serial CV. Keep opaque
            # parent neighborhoods separate rather than dropping parameters
            # or adding a requirement that they support standard pickle.
            other_key = ('opaque-parent', int(index))
        for lambda_factor in (0.25, 1., 4.):
            for gamma_factor in (10.**-0.5, 1., 10.**0.5):
                values = (lambd*lambda_factor, gamma*gamma_factor, theta, ratio)
                if not np.isfinite(values).all() or min(values[:2]) <= 0:
                    raise ValueError('Local lambda/gamma values are not representable.')
                key = (format(values[0], '.15g'), format(values[1], '.15g'), other_key)
                if key not in seen:
                    seen.add(key)
                    candidate = {name: [value] for name, value in deepcopy(parent).items()}
                    candidate.update({name: [value] for name, value in zip(names[:2], values[:2])})
                    grid.append(candidate)
    return grid


def report_search(label, search):
    valid = fully_valid_candidate_mask(search.cv_results_)
    print(f'{label}: {len(valid)} candidates; '
          f'{np.count_nonzero(~valid)} invalid candidates; '
          f'{np.count_nonzero(valid)} fully valid candidates.')
    best = best_valid_candidate(search.cv_results_)
    print(f'{label} best fully valid CV accuracy: '
          f"{search.cv_results_['mean_test_score'][best]:.4f}")
    print(f'{label} selected candidate mean fold-fit seconds: '
          f"{search.cv_results_['mean_fit_time'][best]:.4f}")
    return best


def boundary_parameters(parameters, grid, *, regularization='lambd', prefix='ensemble__'):
    """Identify selected regularization/gamma values at or beyond coarse endpoints.

    Values outside the coarse range can arise in the optional DWD local search.
    Sampling and selection fractions never trigger this review.
    """
    names = (prefix + regularization, prefix + 'gamma')
    return [name for name in names
            if parameters[name] <= min(grid[name]) or parameters[name] >= max(grid[name])]


def run_search_workflow(pipeline, grid, X_train, y_train, X_test, y_test, folds,
                        run_dir, *, jobs=1, error_policy='nan', refine=False,
                        settings=None, verbose=1, selection_review=None, final_native_threads=1):
    """Checkpoint completed CV stages before fitting the selected bare pipeline.

    ``run_dir`` must be newly created by ``new_run_directory``. Fold indices
    refer to the training arrays in their original order. No automatic fallback
    or resume is performed, and the held-out set is scored only after final fit.
    An optional selection_review(parameters) callback runs after the selection
    checkpoint and before final fitting. Return a list of review reasons to
    hold both final fitting and held-out scoring, or None/an empty list to
    continue. The default leaves the DWD workflow unchanged. KeyboardInterrupt
    and SystemExit preserve completed records, mark the run interrupted and
    propagate; this legacy helper does not support resuming the run.
    final_native_threads changes only the selected full-training pipeline
    fit. CV and held-out prediction remain at one native thread. The default
    is one; wider budgets require every exposed nested n_jobs to be one.
    """
    final_native_threads = validate_final_native_threads(final_native_threads, pipeline)
    run_dir = _resolved_public_path(run_dir)
    failure_dir = _io_path(run_dir / 'failures')
    if not failure_dir.is_dir():
        raise ValueError('Create a fresh run directory with new_run_directory first.')
    jobs = positive_int(jobs)
    if error_policy not in ('nan', 'raise'):
        raise ValueError("error_policy must be 'nan' or 'raise'.")
    if selection_review is not None and not callable(selection_review):
        raise TypeError('selection_review must be callable or None.')
    folds = [(np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
             for train, test in folds]
    identity = {'training': data_fingerprint(X_train, y_train),
                'held_out': data_fingerprint(X_test, y_test)}
    metadata = {
        'schema_version': 1, 'data': identity,
        'final_thread_policy_sha256': final_thread_policy_sha256(),
        'folds': [{'train': train, 'validation': test} for train, test in folds],
        'settings': {'jobs': jobs, 'error_policy': error_policy, 'refine': refine,
                     'selection_review_enabled': selection_review is not None,
                     'scoring': 'accuracy', 'native_threads': 1,
                      'final_native_threads': final_native_threads, 'held_out_native_threads': 1,
                     'parameters': parameter_snapshot(pipeline),
                     'user': {} if settings is None else settings},
    }
    write_json_atomic(run_dir / 'run.json', {**metadata, 'coarse_grid': grid})
    final_runtime = None
    phase = 'coarse_cv'
    write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase})

    def stage_search(label, candidate_grid, *, allow_all_ineligible=False):
        evaluator = CandidateEvaluator(pipeline, error_policy=error_policy,
                                       failure_dir=str(failure_dir))
        search = GridSearchCV(
            evaluator, prefix_grid(candidate_grid), cv=folds, scoring=eligible_accuracy,
            n_jobs=jobs, pre_dispatch=jobs, error_score='raise', refit=False,
            verbose=verbose,
        )
        start = perf_counter()
        search.fit(X_train, y_train)
        elapsed = perf_counter() - start
        valid = fully_valid_candidate_mask(search.cv_results_)
        ranked = ranked_valid_indices(search.cv_results_)
        selected = int(ranked[0]) if len(ranked) else None
        # Persist before reporting or raising for an all-ineligible search.
        write_json_atomic(run_dir / (label + '-cv.json'), {
            **metadata, 'stage': label, 'wall_seconds': elapsed,
            'grid': candidate_grid, 'cv_results': search.cv_results_,
            'fully_valid': valid, 'selected_index': selected,
            'selected_parameters': None if selected is None else
                bare_parameters(search.cv_results_['params'][selected]),
        })
        print(f'{label.title()} search wall seconds: {elapsed:.4f}')
        if selected is not None or not allow_all_ineligible:
            report_search(label.title(), search)
        else:
            print(f'{label.title()}: no fully valid candidates; retaining the coarse result.')
        return search

    try:
        # Limit native threads in both the parent refit and CV worker processes.
        with parallel_config(backend='loky', inner_max_num_threads=1), threadpool_limits(limits=1):
            coarse = stage_search('coarse', grid)
            search, stage = coarse, 'coarse'
            if refine:
                phase = 'local_cv'
                write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
                bare_results = {**coarse.cv_results_, 'params': [
                    bare_parameters(params) for params in coarse.cv_results_['params']]}
                local_grid = local_refinement_grid(bare_results)
                print(f'Local grid: {len(local_grid)} candidates after deduplication.')
                local = stage_search('local', local_grid, allow_all_ineligible=True)
                local_ranked = ranked_valid_indices(local.cv_results_)
                coarse_best = best_valid_candidate(coarse.cv_results_)
                # Preserve the existing local-stage preference on exact ties.
                if (len(local_ranked) and
                        local.cv_results_['mean_test_score'][local_ranked[0]] >=
                        coarse.cv_results_['mean_test_score'][coarse_best]):
                    search, stage = local, 'local'
            selected = best_valid_candidate(search.cv_results_)
            parameters = bare_parameters(search.cv_results_['params'][selected])
            selection = {'source_stage': stage, 'candidate_index': selected,
                         'parameters': parameters,
                         'mean_cv_accuracy': float(search.cv_results_['mean_test_score'][selected])}
            write_json_atomic(run_dir / 'selection.json', {**metadata, **selection})
            review = None
            if selection_review is not None:
                phase = 'parameter_review'
                write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
                reasons = selection_review(deepcopy(parameters))
                reasons = [] if reasons is None else reasons
                if not isinstance(reasons, (list, tuple)) or any(
                        not isinstance(reason, str) or not reason.strip() for reason in reasons):
                    raise ValueError('selection_review must return a list of nonempty strings or None.')
                review = {**selection, 'state': 'needs_review' if reasons else 'accepted',
                          'reasons': list(reasons), 'final_refit_performed': False,
                          'held_out_scored': False}
                write_json_atomic(run_dir / 'review.json', review)
                if reasons:
                    write_json_atomic(run_dir / 'status.json', {
                        'state': 'needs_review', 'phase': phase, 'reasons': list(reasons),
                        'final_refit_performed': False, 'held_out_scored': False}, replace=True)
                    print('CV selection requires review: ' + '; '.join(reasons))
                    print('Final fitting and held-out scoring were not run. Completed CV results are saved.')
                    return {'coarse_search': coarse, 'selected_search': search, 'model': None,
                            'completed': None, 'selection': selection, 'review': review, 'run_dir': run_dir}
            phase = 'final_refit'
            write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
            start = perf_counter()
            try:
                final_model = clone(pipeline).set_params(**parameters)
                final_runtime = {}
                with final_fit_threads(final_native_threads, final_model, final_runtime):
                    final_model.fit(X_train, y_train)
            except Exception as error:
                write_failure_json(run_dir / 'refit.json', {
                    'state': 'failed', **selection, 'wall_seconds': perf_counter() - start,
                    'failure': exception_details(error), 'native_runtime': final_runtime})
                raise
            refit_seconds = perf_counter() - start
            # This completion record exists even if later test scoring fails.
            write_json_atomic(run_dir / 'refit.json', {
                'state': 'completed', **selection, 'wall_seconds': refit_seconds,
                'native_runtime': final_runtime})
            if review is not None:
                review['final_refit_performed'] = True
                write_json_atomic(run_dir / 'review.json', review, replace=True)
            phase = 'held_out_scoring'
            write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
            test_accuracy = float(accuracy_score(y_test, final_model.predict(X_test)))
            completed = {**selection, 'refit_seconds': refit_seconds,
                         'held_out_accuracy': test_accuracy, 'final_native_threads': final_native_threads}
            write_json_atomic(run_dir / 'completed.json', completed)
            if review is not None:
                review['held_out_scored'] = True
                write_json_atomic(run_dir / 'review.json', review, replace=True)
            write_json_atomic(run_dir / 'status.json', {'state': 'completed', 'phase': phase}, replace=True)
    except (KeyboardInterrupt, SystemExit) as error:
        write_failure_json(run_dir / 'status.json', {
            'state': 'interrupted', 'phase': phase,
            'exception_type': type(error).__name__, 'resume_supported': False,
            'final_runtime': final_runtime}, replace=True)
        raise
    except Exception as error:
        record_failure(failure_dir, error, phase=phase,
                       estimator=locals().get('final_model', pipeline),
                       training_data=identity['training'])
        write_failure_json(run_dir / 'status.json', {
            'state': 'failed', 'phase': phase, 'failure': exception_details(error),
            'final_runtime': final_runtime}, replace=True)
        raise
    print(f"Best CV accuracy: {selection['mean_cv_accuracy']:.4f}")
    print(f'Best parameters: {parameters}')
    print(f'Final refit seconds: {refit_seconds:.4f}')
    print(f'Held-out test accuracy: {test_accuracy:.4f}')
    return {'coarse_search': coarse, 'selected_search': search, 'model': final_model,
            'completed': completed, 'selection': selection, 'review': review, 'run_dir': run_dir}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_final_thread_argument(parser)
    add_seed_arguments(parser)
    add_ratio_arguments(parser)
    add_confirmation_arguments(parser)
    parser.add_argument('--data', type=Path, help='Local NumPy .npz archive.')
    parser.add_argument('--digits', type=int, nargs=2, default=(7, 9), metavar=('FIRST', 'SECOND'))
    parser.add_argument('--jobs', type=positive_int, default=1,
                        help='Positive outer-worker count; choose a value that fits available RAM.')
    parser.add_argument('--error-score', choices=('raise', 'nan'), default='nan',
                        help='nan skips only ineligible subsets or selections; raise stops on every fit failure.')
    parser.add_argument('--output-dir', type=Path,
                        help='Fresh directory for a new run, or the original directory with --resume; defaults to results/mnist-*.')
    parser.add_argument('--resume', action='store_true',
                        help='Reuse completed folds from --output-dir with identical settings, budgets and runtime; requires --run.')
    parser.add_argument('--reuse-equivalent-folds', action='store_true',
                        help='Reuse completed CV results when selection ratios retain identical rows under the same settings; disabled by default.')
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--cache-dir', type=Path,
                       help='Shared base-stage cache; defaults to the new run directory/base-cache.')
    cache.add_argument('--no-cache', action='store_true', help='Disable base-stage caching.')
    refinement = parser.add_mutually_exclusive_group()
    refinement.add_argument('--refine', dest='refine', action='store_true',
                            help='Compatibility option: refinement is enabled by default.')
    refinement.add_argument('--no-refine', dest='refine', action='store_false',
                            help='Skip local rounds and bracket tightening; valid conditional neighbors are still required.')
    parser.set_defaults(refine=True)
    parser.add_argument('--max-expansions', type=int, default=3,
                        help='Maximum complete expansion rounds; default 3, minimum 0.')
    parser.add_argument('--max-candidates', type=positive_int, default=500,
                        help='Total unique candidate budget, including coarse CV; default 500.')
    parser.add_argument('--bracket-factor', type=float, default=2.0,
                        help='Target ratio to valid local neighbors when refining; default 2, greater than 1.')
    parser.add_argument('--stable-refinement-rounds', type=positive_int, default=2,
                        help='Completed stable local rounds required when refining; default 2.')
    parser.add_argument('--broad-grid', action='store_true',
                        help='Start from 405 coarse candidates instead of 125.')
    parser.add_argument('--run', action='store_true', help='Start the requested search stages.')
    args = parser.parse_args(argv)
    report_final_threads(args.final_native_threads)
    seed_options = seed_search_options(parser, args)
    confirmation_options = confirmation_search_options(parser, args)
    seed_count = len(seed_options.get('cv_seeds', (42,)))
    if args.resume and (not args.run or args.output_dir is None):
        parser.error('--resume requires --run and --output-dir.')
    if args.max_expansions < 0:
        parser.error('--max-expansions must be nonnegative.')
    if not np.isfinite(args.bracket_factor) or args.bracket_factor <= 1:
        parser.error('--bracket-factor must be finite and greater than 1.')
    if args.digits[0] == args.digits[1]:
        parser.error('Choose two distinct digits.')
    if any(digit < 0 or digit > 9 for digit in args.digits):
        parser.error('MNIST digit labels must be between 0 and 9.')

    grid = mnist_param_grid(prefix='ensemble__', broad=args.broad_grid)
    ratio_options = ratio_search_options(parser, args, grid)
    candidates = len(ParameterGrid(grid))
    print(f'Coarse: {candidates} candidates, {5*candidates*seed_count} ensemble CV fits, '
          f'up to {6*5*candidates*seed_count} DWD CV fits before cache reuse.')
    report_seed_options(seed_options)
    report_confirmation_options(confirmation_options, folds=5, seed_count=seed_count)
    print(f'Adaptive budget: {args.max_candidates} unique candidates total; '
          f'up to {args.max_expansions} complete expansion rounds; refinement={args.refine}.')
    print('M=5, theta=0.1; retention [0.1, 0.15, 0.2, 0.25, 0.3]. '
          'All stages share five folds and evaluate only new candidates.')
    print('Tied lambda/gamma boundaries expand together, including cross-combinations. '
          'Repeated joint refinement uses the top three distinct parameter pairs.')
    print(f'When refinement is enabled: valid conditional brackets within factor {args.bracket_factor:g}, '
          f'with {args.stable_refinement_rounds} completed stable local rounds.')
    print('Two completed nonimproving outward levels with valid same-slice evidence may waive distant tied edges; '
          'the selected candidate still needs valid neighbors. A plateau does not prove a global optimum. '
          'Unresolved boundaries or exhausted budgets hold final fitting/test scoring.')
    print('An accepted final selection adds six DWD fits; test scoring follows that fit.')
    if not args.run:
        print('Prepared only. Supply --data PATH --run to start; no data were loaded.')
        return
    if args.data is None:
        parser.error('--data is required with --run.')

    X_train, X_test, y_train, y_test = load_local_data(args.data, *args.digits)
    if np.min(np.unique(y_train, return_counts=True)[1]) < 5:
        parser.error('Each training class needs at least five examples for five-fold CV.')
    run_dir = new_run_directory(args.output_dir, resume=args.resume)
    cache_dir = None if args.no_cache else str(
        args.cache_dir.resolve() if args.cache_dir is not None else run_dir / 'base-cache')
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('ensemble', EnsembleKernelDWD(M=5, theta=0.1, cache_dir=cache_dir, store_base_estimators=False)),
    ])
    # Every adaptive stage uses these same training/validation rows.
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train))
    print(f'Training rows: {len(X_train)}; held-out test rows: {len(X_test)}.')
    print(f'Run records: {run_dir}')
    from adaptive_search import AdaptiveSearchConfig, run_adaptive_search
    config = AdaptiveSearchConfig(regularization='lambd', prefix='ensemble__',
        max_expansions=args.max_expansions, max_candidates=args.max_candidates,
        top_k=3, plateau_rounds=2, improvement_tol=0.0, tie_tol=1e-12, refine=args.refine,
        bracket_factor=args.bracket_factor, stable_refinement_rounds=args.stable_refinement_rounds)
    result = run_adaptive_search(
        pipeline, grid, X_train, y_train, X_test, y_test, folds, run_dir,
        jobs=args.jobs, error_policy=args.error_score, config=config, resume=args.resume,
        reuse_equivalent_folds=args.reuse_equivalent_folds,
        final_native_threads=args.final_native_threads,
        settings={'data_path': str(args.data.resolve()), 'digits': args.digits,
                  'cache_dir': cache_dir, 'store_base_estimators': False},
        **seed_options, **ratio_options, **confirmation_options,
    )
    if result.get('resumed_completed'):
        print('This run already completed; saved results were verified and no fits or test scoring were repeated.')
    elif result['model'] is None:
        print('Adaptive search requires training-only review: ' + ', '.join(result['review']['reasons']))
        print(f"Saved selection: {run_dir / 'selection.json'}; review: {run_dir / 'review.json'}")
        print('Review the saved search limits and unresolved boundaries; the test set was not scored.')


if __name__ == '__main__':
    main()
