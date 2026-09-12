"""Adaptive external search on a local MNIST archive.

Archive keys may be x_train/y_train/x_test/y_test (standard Keras format),
X_train/y_train/X_test/y_test, or X/y. Images may be flattened or have shape
(n, 28, 28). With X/y, a stratified 25% test split is made before CV. A supplied
test split remains held out. Data are never downloaded.
The CLI expands lambda/gamma boundaries and refines promising settings on the
same training folds. Unresolved boundaries or exhausted budgets hold final
fitting/test scoring. A plateau does not prove global optimality. The legacy
run_search_workflow helper remains available for reproducibility.

Preview: python examples/tune_mnist.py --refine
Run: python examples/tune_mnist.py --data mnist.npz --refine --run --jobs 1
"""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from joblib import parallel_config
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
from search_support import (CandidateEvaluator, bare_parameters, data_fingerprint,
                            eligible_accuracy, exception_details, new_run_directory,
                            parameter_snapshot, prefix_grid, record_failure,
                            write_failure_json, write_json_atomic)


def select_pair(X, y, first, second):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim < 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError('Expected an image/feature array and one label per row.')
    labels = np.asarray(y, dtype=np.float64)
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('MNIST labels must be finite integer digit values.')
    keep = (labels == first) | (labels == second)
    if np.unique(labels[keep]).size != 2:
        raise ValueError('Both requested digits must occur in every supplied split.')
    return np.asarray(X[keep].reshape(np.count_nonzero(keep), -1), dtype=np.float64), labels[keep].astype(np.int64)


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
    means = np.asarray(cv_results['mean_test_score'], dtype=float)
    return indices[np.argsort(-means[indices], kind='stable')]


def best_valid_candidate(cv_results):
    """GridSearchCV refit selector; no model is selected if all candidates fail."""
    ranked = ranked_valid_indices(cv_results)
    if not len(ranked):
        raise ValueError('No candidate has finite scores on every validation fold '
                         f"({len(cv_results['params'])} invalid candidates).")
    return int(ranked[0])


def local_refinement_grid(cv_results, prefix='ensemble__', top_k=3):
    """Union local neighborhoods around up to three fully valid coarse results.

    Each parent retains its own theta and retention fraction. No clipping to
    the coarse grid is applied. Formatting lambda/gamma deduplication keys to
    15 significant digits merges floating-point variants of the same values;
    theta/retention match exactly and the first candidate's values are retained.
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
        for lambda_factor in (0.25, 1., 4.):
            for gamma_factor in (10.**-0.5, 1., 10.**0.5):
                values = (lambd*lambda_factor, gamma*gamma_factor, theta, ratio)
                if not np.isfinite(values).all() or min(values[:2]) <= 0:
                    raise ValueError('Local lambda/gamma values are not representable.')
                key = (format(values[0], '.15g'), format(values[1], '.15g'), theta, ratio)
                if key not in seen:
                    seen.add(key)
                    grid.append({name: [value] for name, value in zip(names, values)})
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
                        settings=None, verbose=1, selection_review=None):
    """Checkpoint completed CV stages before fitting the selected bare pipeline.

    ``run_dir`` must be newly created by ``new_run_directory``. Fold indices
    refer to the training arrays in their original order. No automatic fallback
    or resume is performed, and the held-out set is scored only after final fit.
    An optional selection_review(parameters) callback runs after the selection
    checkpoint and before final fitting. Return a list of review reasons to
    hold both final fitting and held-out scoring, or None/an empty list to
    continue. The default leaves the DWD workflow unchanged.
    """
    run_dir = Path(run_dir).resolve()
    failure_dir = run_dir / 'failures'
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
        'folds': [{'train': train, 'validation': test} for train, test in folds],
        'settings': {'jobs': jobs, 'error_policy': error_policy, 'refine': refine,
                     'selection_review_enabled': selection_review is not None,
                     'scoring': 'accuracy', 'native_threads': 1,
                     'parameters': parameter_snapshot(pipeline),
                     'user': {} if settings is None else settings},
    }
    write_json_atomic(run_dir / 'run.json', {**metadata, 'coarse_grid': grid})
    phase = 'coarse_cv'
    write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase})

    def stage_search(label, candidate_grid):
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
        report_search(label.title(), search)
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
                search, stage = stage_search('local', local_grid), 'local'
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
                final_model.fit(X_train, y_train)
            except Exception as error:
                write_failure_json(run_dir / 'refit.json', {
                    'state': 'failed', **selection, 'wall_seconds': perf_counter() - start,
                    'failure': exception_details(error)})
                raise
            refit_seconds = perf_counter() - start
            # This completion record exists even if later test scoring fails.
            write_json_atomic(run_dir / 'refit.json', {
                'state': 'completed', **selection, 'wall_seconds': refit_seconds})
            phase = 'held_out_scoring'
            write_json_atomic(run_dir / 'status.json', {'state': 'running', 'phase': phase}, replace=True)
            test_accuracy = float(accuracy_score(y_test, final_model.predict(X_test)))
            completed = {**selection, 'refit_seconds': refit_seconds,
                         'held_out_accuracy': test_accuracy}
            write_json_atomic(run_dir / 'completed.json', completed)
            write_json_atomic(run_dir / 'status.json', {'state': 'completed', 'phase': phase}, replace=True)
    except Exception as error:
        record_failure(failure_dir, error, phase=phase,
                       estimator=locals().get('final_model', pipeline),
                       training_data=identity['training'])
        write_failure_json(run_dir / 'status.json', {
            'state': 'failed', 'phase': phase, 'failure': exception_details(error)}, replace=True)
        raise
    print(f"Best CV accuracy: {selection['mean_cv_accuracy']:.4f}")
    print(f'Best parameters: {parameters}')
    print(f'Final refit seconds: {refit_seconds:.4f}')
    print(f'Held-out test accuracy: {test_accuracy:.4f}')
    return {'coarse_search': coarse, 'selected_search': search, 'model': final_model,
            'completed': completed, 'selection': selection, 'review': review, 'run_dir': run_dir}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Local NumPy .npz archive.')
    parser.add_argument('--digits', type=int, nargs=2, default=(7, 9), metavar=('FIRST', 'SECOND'))
    parser.add_argument('--jobs', type=positive_int, default=1,
                        help='Positive outer-worker count; choose a value that fits available RAM.')
    parser.add_argument('--error-score', choices=('raise', 'nan'), default='nan',
                        help='nan skips only ineligible subsets or selections; raise stops on every fit failure.')
    parser.add_argument('--output-dir', type=Path,
                        help='New run directory; defaults to a unique results/mnist-* directory.')
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--cache-dir', type=Path,
                       help='Shared base-stage cache; defaults to the new run directory/base-cache.')
    cache.add_argument('--no-cache', action='store_true', help='Disable base-stage caching.')
    refinement = parser.add_mutually_exclusive_group()
    refinement.add_argument('--refine', dest='refine', action='store_true',
                            help='Compatibility option: refinement is enabled by default.')
    refinement.add_argument('--no-refine', dest='refine', action='store_false')
    parser.set_defaults(refine=True)
    parser.add_argument('--max-expansions', type=int, default=3,
                        help='Maximum complete expansion rounds; default 3, minimum 0.')
    parser.add_argument('--max-candidates', type=positive_int, default=500,
                        help='Total unique candidate budget, including coarse CV; default 500.')
    parser.add_argument('--broad-grid', action='store_true',
                        help='Start from 405 coarse candidates instead of 125.')
    parser.add_argument('--run', action='store_true', help='Start the requested search stages.')
    args = parser.parse_args(argv)
    if args.max_expansions < 0:
        parser.error('--max-expansions must be nonnegative.')
    if args.digits[0] == args.digits[1]:
        parser.error('Choose two distinct digits.')
    if any(digit < 0 or digit > 9 for digit in args.digits):
        parser.error('MNIST digit labels must be between 0 and 9.')

    grid = mnist_param_grid(prefix='ensemble__', broad=args.broad_grid)
    candidates = len(ParameterGrid(grid))
    print(f'Coarse: {candidates} candidates, {5*candidates} ensemble CV fits, '
          f'up to {6*5*candidates} DWD CV fits before cache reuse.')
    print(f'Adaptive budget: {args.max_candidates} unique candidates total; '
          f'up to {args.max_expansions} complete expansion rounds; refinement={args.refine}.')
    print('M=5, theta=0.1; retention [0.1, 0.15, 0.2, 0.25, 0.3]. '
          'All stages share five folds and evaluate only new candidates.')
    print('Tied lambda/gamma boundaries expand together, including cross-combinations. '
          'Refinement uses the top three distinct parameter pairs.')
    print('Two nonimproving full rounds may stop expansion; a plateau does not prove a global optimum. '
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
    run_dir = new_run_directory(args.output_dir)
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
        top_k=3, plateau_rounds=2, improvement_tol=0.0, tie_tol=1e-12, refine=args.refine)
    result = run_adaptive_search(
        pipeline, grid, X_train, y_train, X_test, y_test, folds, run_dir,
        jobs=args.jobs, error_policy=args.error_score, config=config,
        settings={'data_path': str(args.data.resolve()), 'digits': args.digits,
                  'cache_dir': cache_dir, 'store_base_estimators': False},
    )
    if result['model'] is None:
        print('Adaptive search requires training-only review: ' + ', '.join(result['review']['reasons']))
        print(f"Saved selection: {run_dir / 'selection.json'}; review: {run_dir / 'review.json'}")
        print('Review the saved search limits and unresolved boundaries; the test set was not scored.')


if __name__ == '__main__':
    main()
