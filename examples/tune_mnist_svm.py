"""Preview or run adaptive external fivefold CV for the ensemble SVM comparator.

Preview: python examples/tune_mnist_svm.py
Run: python examples/tune_mnist_svm.py --data mnist.npz --run --jobs 4

Uses a local MNIST archive and retains its supplied test split. The returned
grid is a coarse starting range. C/gamma boundaries expand on the same folds;
repeated joint local refinement is enabled by default. Unresolved boundaries or exhausted
budgets hold final fitting and test scoring for training-only review.
Explicit --resume --run reuses completed folds from --output-dir under identical
settings, budgets and runtime. Interrupted final fitting restarts; a committed
completed result resumes without fitting or scoring and returns model=None.
No data are downloaded.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_dwd import EnsembleKernelSVM, mnist_svm_param_grid

EXAMPLES_DIRECTORY = str(Path(__file__).resolve().parent)
if EXAMPLES_DIRECTORY not in sys.path:
    sys.path.insert(0, EXAMPLES_DIRECTORY)
from tune_mnist import (load_local_data, positive_int, add_seed_arguments,
                        seed_search_options, report_seed_options,
                        add_ratio_arguments, ratio_search_options,
                        boundary_parameters as _boundary_parameters)
from search_support import new_run_directory, write_json_atomic


def boundary_parameters(parameters, grid):
    """Identify coarse C/gamma endpoints that warrant training-only review."""
    return _boundary_parameters(parameters, grid, regularization='C')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_seed_arguments(parser)
    add_ratio_arguments(parser)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--digits", type=int, nargs=2, default=(7, 9))
    parser.add_argument("--jobs", type=positive_int, default=1,
                        help="Concurrent outer CV workers; pre_dispatch equals this count.")
    parser.add_argument("--error-score", choices=("raise", "nan"), default="nan",
                        help="nan skips only ineligible subsets or selections; unexpected failures stop.")
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh directory for a new run, or the original directory with --resume.")
    parser.add_argument('--resume', action='store_true',
                        help='Reuse completed folds from --output-dir with identical settings, budgets and runtime; requires --run.')
    parser.add_argument('--reuse-equivalent-folds', action='store_true',
                        help='Reuse completed CV results when selection ratios retain identical rows under the same settings; disabled by default.')
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument('--no-cache', action='store_true')
    refinement = parser.add_mutually_exclusive_group()
    refinement.add_argument('--refine', dest='refine', action='store_true',
                            help='Compatibility option: refinement is enabled by default.')
    refinement.add_argument('--no-refine', dest='refine', action='store_false',
                            help='Skip local rounds and bracket tightening; valid conditional neighbors are still required.')
    parser.set_defaults(refine=True)
    parser.add_argument('--max-expansions', type=int, default=3)
    parser.add_argument('--max-candidates', type=positive_int, default=500)
    parser.add_argument('--bracket-factor', type=float, default=2.0)
    parser.add_argument('--stable-refinement-rounds', type=positive_int, default=2)
    parser.add_argument('--broad-grid', action='store_true',
                        help='Start from 405 coarse candidates instead of 125.')
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    seed_options = seed_search_options(parser, args)
    seed_count = len(seed_options.get('cv_seeds', (42,)))
    if args.resume and (not args.run or args.output_dir is None):
        parser.error('--resume requires --run and --output-dir.')
    if args.max_expansions < 0:
        parser.error('--max-expansions must be nonnegative.')
    if not np.isfinite(args.bracket_factor) or args.bracket_factor <= 1:
        parser.error('--bracket-factor must be finite and greater than 1.')
    if args.no_cache and args.cache_dir is not None:
        parser.error('--cache-dir and --no-cache are mutually exclusive.')
    if len(set(args.digits)) != 2 or any(d < 0 or d > 9 for d in args.digits):
        parser.error("Choose two distinct MNIST digits between 0 and 9.")

    grid = mnist_svm_param_grid("ensemble__", broad=args.broad_grid)
    ratio_options = ratio_search_options(parser, args, grid, regularization='C')
    candidates = len(ParameterGrid(grid))
    print(f"Coarse: {candidates} candidates, {5*candidates*seed_count} ensemble CV fits, "
          f"up to {6*5*candidates*seed_count} SVC CV fits before cache reuse.")
    report_seed_options(seed_options)
    print("M=5, theta fixed at 0.1, retention over "
          "[0.1, 0.15, 0.2, 0.25, 0.3]. An accepted selection's new final refit adds six SVC fits.")
    print(f"Adaptive budget: {args.max_candidates} unique candidates total; "
          f"up to {args.max_expansions} complete expansion rounds; refinement={args.refine}.")
    print("All stages share five folds and evaluate only new candidates. Tied C/gamma boundaries "
          "expand together, including cross-combinations; repeated joint refinement uses the top three distinct pairs.")
    print(f'When refinement is enabled: valid conditional brackets within factor {args.bracket_factor:g}, '
          f'with {args.stable_refinement_rounds} completed stable local rounds.')
    print("Two completed nonimproving outward levels with valid same-slice evidence may waive distant tied edges; "
          "the selected candidate still needs valid neighbors. A plateau does not prove a global optimum. "
          "Unresolved boundaries or exhausted budgets hold final fitting/test scoring.")
    if not args.run:
        print("Prepared only. Supply --data PATH --run to fit; no data were loaded.")
        return
    if args.data is None:
        parser.error("--data is required with --run.")

    X_train, X_test, y_train, y_test = load_local_data(args.data, *args.digits)
    if min(np.unique(y_train, return_counts=True)[1]) < 5:
        parser.error("Each training class requires at least five rows.")
    run_dir = new_run_directory(args.output_dir, resume=args.resume)
    cache_dir = None if args.no_cache else str(args.cache_dir.resolve() if args.cache_dir else run_dir / "base-cache")
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("ensemble", EnsembleKernelSVM(M=5, theta=0.1, cache_dir=cache_dir, store_base_estimators=False)),
    ])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train))
    from adaptive_search import AdaptiveSearchConfig, run_adaptive_search
    config = AdaptiveSearchConfig(regularization='C', prefix='ensemble__',
        max_expansions=args.max_expansions, max_candidates=args.max_candidates,
        top_k=3, plateau_rounds=2, improvement_tol=0.0, tie_tol=1e-12, refine=args.refine,
        bracket_factor=args.bracket_factor, stable_refinement_rounds=args.stable_refinement_rounds)
    result = run_adaptive_search(
        pipeline, grid, X_train, y_train, X_test, y_test, folds, run_dir,
        jobs=args.jobs, error_policy=args.error_score, config=config, resume=args.resume,
        reuse_equivalent_folds=args.reuse_equivalent_folds,
        settings={"classifier": "EnsembleKernelSVM", "selection": "signed_margin",
                  "data_path": str(args.data.resolve()), "digits": args.digits,
                  "cache_dir": cache_dir},
        **seed_options, **ratio_options,
    )
    if result.get('resumed_completed'):
        print('This run already completed; saved results were verified and no fits or test scoring were repeated.')
    elif result["model"] is None:
        print("Adaptive search requires training-only review: " + ", ".join(result["review"]["reasons"]))
        print(f"Saved selection: {run_dir / 'selection.json'}; review: {run_dir / 'review.json'}")
        print("Review the saved search limits and unresolved boundaries; the test set was not scored.")
        return
    # The authoritative completion receipt also serves resumed runs with no model.
    diagnostics = result["refit"]["diagnostics"]
    review = result["review"]
    directions = review["boundary_directions"]
    boundaries = sorted({name for name, side in directions})
    write_json_atomic(run_dir / "svm-diagnostics.json", {
        "n_iter": diagnostics["n_iter_"], "fit_status": diagnostics["fit_status_"],
        "converged": diagnostics["converged_"], "support_counts": diagnostics["support_counts_"],
        "n_support": diagnostics["n_support_"], "gamma": diagnostics["gamma_"],
        "subset_sizes": diagnostics["subset_sizes"],
        "selected_size": diagnostics["selected_size"], "cache_hit": diagnostics["cache_hit_"],
        "boundary_parameters": boundaries,
        "boundary_directions": directions,
        "search_stopping_reason": review["stopping_reason"],
    }, replace=args.resume)
    print(f"SVC diagnostics: {run_dir / 'svm-diagnostics.json'}")
    if result.get('resumed_completed'):
        return result


if __name__ == "__main__":
    main()
