"""Preview or run external fivefold CV for the ensemble SVM comparator.

Preview: python examples/tune_mnist_svm.py
Run: python examples/tune_mnist_svm.py --data mnist.npz --run --jobs 4

Uses a local MNIST archive and retains its supplied test split. The returned
grid is a coarse starting range. A winning C/gamma endpoint holds final fitting
and test scoring for training-only review; the grid is not expanded automatically.
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
from tune_mnist import (load_local_data, positive_int, run_search_workflow,
                        boundary_parameters as _boundary_parameters)
from search_support import new_run_directory, write_json_atomic


def boundary_parameters(parameters, grid):
    """Identify coarse C/gamma endpoints that warrant training-only review."""
    return _boundary_parameters(parameters, grid, regularization='C')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--digits", type=int, nargs=2, default=(7, 9))
    parser.add_argument("--jobs", type=positive_int, default=1,
                        help="Concurrent outer CV workers; pre_dispatch equals this count.")
    parser.add_argument("--error-score", choices=("raise", "nan"), default="nan",
                        help="nan skips only ineligible selections; unexpected failures stop.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if len(set(args.digits)) != 2 or any(d < 0 or d > 9 for d in args.digits):
        parser.error("Choose two distinct MNIST digits between 0 and 9.")

    grid = mnist_svm_param_grid("ensemble__")
    candidates = len(ParameterGrid(grid))
    print(f"Coarse: {candidates} candidates, {5*candidates} ensemble CV fits, "
          f"up to {6*5*candidates} SVC CV fits before cache reuse.")
    print("M=5, theta searched over [0.1, 0.2], retention over "
          "[0.1, 0.15, 0.2, 0.25, 0.3]. An interior winner's final refit adds six SVC fits.")
    print("C/gamma boundary winners preserve CV results and hold final fitting/test scoring for review.")
    if not args.run:
        print("Prepared only. Supply --data PATH --run to fit; no data were loaded.")
        return
    if args.data is None:
        parser.error("--data is required with --run.")

    X_train, X_test, y_train, y_test = load_local_data(args.data, *args.digits)
    if min(np.unique(y_train, return_counts=True)[1]) < 5:
        parser.error("Each training class requires at least five rows.")
    run_dir = new_run_directory(args.output_dir)
    cache_dir = args.cache_dir.resolve() if args.cache_dir else run_dir / "base-cache"
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("ensemble", EnsembleKernelSVM(cache_dir=str(cache_dir), store_base_estimators=False)),
    ])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train))
    result = run_search_workflow(
        pipeline, grid, X_train, y_train, X_test, y_test, folds, run_dir,
        jobs=args.jobs, error_policy=args.error_score, refine=False,
        selection_review=lambda parameters: boundary_parameters(parameters, grid),
        settings={"classifier": "EnsembleKernelSVM", "selection": "signed_margin",
                  "data_path": str(args.data.resolve()), "digits": args.digits,
                  "cache_dir": str(cache_dir)},
    )
    if result["model"] is None:
        print("Review C/gamma endpoints using training-only CV: " + ", ".join(result["review"]["reasons"]))
        print(f"Saved selection: {run_dir / 'selection.json'}; review: {run_dir / 'review.json'}")
        print("Adjust the explicit grid if warranted before running a new search; the test set was not scored.")
        return
    model = result["model"].named_steps["ensemble"]
    boundaries = boundary_parameters(result["completed"]["parameters"], grid)
    write_json_atomic(run_dir / "svm-diagnostics.json", {
        "n_iter": model.n_iter_, "fit_status": model.fit_status_,
        "converged": model.converged_, "support_counts": model.support_counts_,
        "n_support": model.n_support_, "gamma": model.gamma_,
        "subset_sizes": [len(rows) for rows in model.subset_indices_],
        "selected_size": len(model.selected_indices_), "cache_hit": model.cache_hit_,
        "boundary_parameters": boundaries,
    })
    print(f"SVC diagnostics: {run_dir / 'svm-diagnostics.json'}")


if __name__ == "__main__":
    main()
