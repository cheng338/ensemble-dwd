"""Tune all ten MNIST digits using a supplied local train/test archive.

Use an archive containing the official MNIST split, with keys
x_train/y_train/x_test/y_test or X_train/y_train/X_test/y_test. Features may
be flattened or image arrays. Reduced archives are accepted when each split
still contains digits 0 through 9; array counts do not establish provenance.
The supplied splits remain separate. No data are downloaded or pooled.

By default, stratified subsampling selects 6,000 training rows and keeps every
supplied test row (10,000 for the complete official archive). Scaling and all
45 binary selection/refit procedures run inside each training fold. One
parameter setting is shared by every pair; there are no per-pair searches.

Preview: python examples/tune_mnist_multiclass.py
Run: python examples/tune_mnist_multiclass.py --data mnist.npz --run --jobs 1
Small validation grid: add --smoke-grid; this is not the broad search.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from ensemble_dwd import EnsembleKernelDWD, OvOClassifier, mnist_param_grid

EXAMPLES_DIRECTORY = str(Path(__file__).resolve().parent)
if EXAMPLES_DIRECTORY not in sys.path:
    sys.path.insert(0, EXAMPLES_DIRECTORY)
from tune_mnist import positive_int, run_search_workflow
from search_support import data_fingerprint, new_run_directory, write_json_atomic


DIGITS = np.arange(10, dtype=np.int64)
PARAMETER_PREFIX = 'ovo__estimator__'
RANDOM_STATE = 42


def class_counts(y):
    """Return JSON-friendly counts without remapping the digit labels."""
    labels, counts = np.unique(y, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def validate_split(X, y, name):
    """Validate one supplied split and flatten features without subsampling."""
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim < 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError(f'{name}: expected features and one label per row.')
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError(f'{name}: features must be finite real numbers.')
    if not np.isfinite(X).all() or np.prod(X.shape[1:]) == 0:
        raise ValueError(f'{name}: features must be nonempty and finite.')
    if np.iscomplexobj(y):
        raise ValueError(f'{name}: MNIST labels must be real integer digits 0 through 9.')
    try:
        labels = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name}: MNIST labels must be integer digits 0 through 9.') from error
    if (not np.isfinite(labels).all()
            or not np.equal(labels, np.floor(labels)).all()
            or not np.array_equal(np.unique(labels), DIGITS)):
        raise ValueError(f'{name}: all ten integer digit labels 0 through 9 must occur.')
    return X.reshape(len(y), -1), labels.astype(np.int64)


def stratified_indices(y, size=None, *, random_state=RANDOM_STATE):
    """Select rows within one split, returning indices in original row order.

    When fewer than ten rows are excluded, allocate proportional removal
    counts while retaining at least one row per digit. Randomize allocation
    ties and sample the excluded rows uniformly within each digit. This avoids
    requiring the small excluded complement to contain every class. Ordinary
    subsets continue to use sklearn's stratified train_test_split unchanged.
    """
    y = np.asarray(y)
    if y.ndim != 1 or not np.array_equal(np.unique(y), DIGITS):
        raise ValueError('Subsampling requires all ten digit labels 0 through 9.')
    if size is None:
        size = len(y)
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
        raise ValueError('Requested subset size must be an integer.')
    if size < 10 or size > len(y):
        raise ValueError(f'Requested subset size must be between 10 and {len(y)}; got {size}.')
    indices = np.arange(len(y), dtype=np.int64)
    excluded_size = len(y) - size
    if 0 < excluded_size < len(DIGITS):
        rng = check_random_state(random_state)
        counts = np.unique(y, return_counts=True)[1]
        quotas = counts * (excluded_size / len(y))
        excluded_counts = np.minimum(np.floor(quotas).astype(np.int64), counts - 1)
        while excluded_counts.sum() < excluded_size:
            available = excluded_counts < counts - 1
            remainder = quotas - excluded_counts
            largest = remainder[available].max()
            tied = np.flatnonzero(available & (remainder == largest))
            excluded_counts[rng.choice(tied)] += 1
        keep = np.ones(len(y), dtype=bool)
        for digit, count in zip(DIGITS, excluded_counts):
            if count:
                removed = rng.choice(indices[y == digit], size=int(count), replace=False)
                keep[removed] = False
        indices = indices[keep]
    elif size < len(y):
        indices, _ = train_test_split(indices, train_size=int(size), stratify=y,
                                      random_state=random_state)
        indices.sort()
    if not np.array_equal(np.unique(y[indices]), DIGITS):
        raise ValueError('The selected subset must contain all ten digits; increase its size.')
    return indices


def load_local_multiclass_data(path, train_size=6000, test_size=None, *, random_state=RANDOM_STATE):
    """Return selected arrays and provenance while preserving supplied splits.

    The first four results have the order X_train, X_test, y_train, y_test.
    The fifth records full-source fingerprints, class counts, and exact
    selected row indices relative to each supplied source split.
    """
    with np.load(path, allow_pickle=False) as archive:
        for train_key, test_key in (('x_train', 'x_test'), ('X_train', 'X_test')):
            if {train_key, 'y_train', test_key, 'y_test'} <= set(archive.files):
                X_train, y_train = validate_split(archive[train_key], archive['y_train'], 'Training split')
                X_test, y_test = validate_split(archive[test_key], archive['y_test'], 'Test split')
                break
        else:
            raise ValueError('Archive must contain x_train/y_train/x_test/y_test or '
                             'X_train/y_train/X_test/y_test. A pooled X/y archive is unsupported.')
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError('Training and test feature counts differ.')
    train_indices = stratified_indices(y_train, train_size, random_state=random_state)
    test_indices = stratified_indices(y_test, test_size, random_state=random_state)
    provenance = {
        'data_path': str(Path(path).resolve()),
        'split_policy': 'Preserve supplied training and test arrays; subsample each separately.',
        'archive_keys': {'training': [train_key, 'y_train'], 'test': [test_key, 'y_test']},
        'source_splits': {
            'training': {'rows': len(y_train), 'class_counts': class_counts(y_train),
                         'fingerprint': data_fingerprint(X_train, y_train)},
            'test': {'rows': len(y_test), 'class_counts': class_counts(y_test),
                     'fingerprint': data_fingerprint(X_test, y_test)},
        },
        'subsampling': {'random_state': random_state, 'requested_train_size': train_size,
                        'requested_test_size': test_size,
                        'train_indices': train_indices, 'test_indices': test_indices},
        'selected_splits': {
            'training': {'rows': len(train_indices), 'class_counts': class_counts(y_train[train_indices])},
            'test': {'rows': len(test_indices), 'class_counts': class_counts(y_test[test_indices])},
        },
    }
    return (np.asarray(X_train[train_indices], dtype=np.float64),
            np.asarray(X_test[test_indices], dtype=np.float64),
            y_train[train_indices], y_test[test_indices], provenance)


def search_grid(smoke=False):
    """Return the broad grid or an explicitly limited eight-candidate grid."""
    if not smoke:
        return mnist_param_grid(prefix=PARAMETER_PREFIX)
    return {
        PARAMETER_PREFIX + 'lambd': [2.0 ** -24, 2.0 ** -18],
        PARAMETER_PREFIX + 'gamma': [1e-4, 1e-3],
        PARAMETER_PREFIX + 'theta': [0.1, 0.2],
        PARAMETER_PREFIX + 'sv_selection_ratio': [0.3],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Local archive with separate training and test arrays.')
    parser.add_argument('--train-size', type=positive_int, default=6000,
                        help='Stratified training subset size; default 6000. Use 60000 for the complete official training set.')
    parser.add_argument('--test-size', type=positive_int,
                        help='Optional stratified test subset size; default keeps all supplied test rows.')
    parser.add_argument('--folds', type=positive_int, default=5,
                        help='Number of stratified CV folds; default 5, minimum 2.')
    parser.add_argument('--jobs', type=positive_int, default=1,
                        help='Positive outer CV worker count; pair fitting remains sequential.')
    parser.add_argument('--smoke-grid', action='store_true',
                        help='Use eight explicit validation candidates instead of the broad 810-candidate grid.')
    parser.add_argument('--error-score', choices=('raise', 'nan'), default='nan',
                        help='nan skips only ineligible selections; unexpected failures stop the run.')
    parser.add_argument('--output-dir', type=Path, help='New run directory; an existing directory is rejected.')
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--cache-dir', type=Path, help='Shared trusted base-stage cache directory.')
    cache.add_argument('--no-cache', action='store_true', help='Disable base-stage caching.')
    parser.add_argument('--run', action='store_true', help='Load local data and start the search.')
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error('--folds must be at least 2.')

    grid = search_grid(args.smoke_grid)
    candidates = len(ParameterGrid(grid))
    ovo_fits = args.folds * candidates
    print(f"{'Validation' if args.smoke_grid else 'Broad'} grid: {candidates} candidates, "
          f'{args.folds} folds, {ovo_fits:,} OVO CV fits.')
    print(f'Each OVO fit contains 45 binary ensembles with 5 bases and 1 final DWD: '
          f'up to {candidates} x {args.folds} x 45 x 6 = {ovo_fits * 45 * 6:,} DWD CV fits.')
    print('The selected full-training refit adds 45 x 6 = 270 DWD fits. '
          'Counts are upper bounds before cache reuse; there is no automatic CV refit.')
    print(f'Outer workers: {args.jobs}; pair workers: 1; native threads per worker: 1. '
          'Completed CV results and selection are saved before the final fit.')
    if not args.run:
        print('Prepared only. Supply --data PATH --run to start; no data were loaded.')
        return
    if args.data is None:
        parser.error('--data is required with --run.')

    try:
        X_train, X_test, y_train, y_test, provenance = load_local_multiclass_data(
            args.data, args.train_size, args.test_size)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if min(class_counts(y_train).values()) < args.folds:
        parser.error(f'Each selected training digit needs at least {args.folds} examples '
                     f'for {args.folds}-fold CV; increase --train-size or reduce --folds.')
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=RANDOM_STATE)
                 .split(X_train, y_train))
    run_dir = new_run_directory(args.output_dir)
    cache_dir = None if args.no_cache else str(
        args.cache_dir.resolve() if args.cache_dir is not None else run_dir / 'base-cache')
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('ovo', OvOClassifier(EnsembleKernelDWD(
            M=5, random_state=RANDOM_STATE, cache_dir=cache_dir,
            store_base_estimators=False), n_jobs=1, prediction_batch_size=1024)),
    ])
    settings = {
        **provenance, 'digits': DIGITS, 'n_pairs': 45, 'fold_count': args.folds,
        'cv_random_state': RANDOM_STATE, 'grid_kind': 'smoke' if args.smoke_grid else 'broad',
        'pair_jobs': 1, 'M': 5, 'cache_dir': cache_dir, 'store_base_estimators': False,
    }
    write_json_atomic(run_dir / 'data-selection.json', settings)
    print(f'Training rows: {len(y_train)}; held-out test rows: {len(y_test)}; all ten digits present.')
    print(f'Run records: {run_dir}')
    # Load BLAS's deferred OpenMP runtime before the workflow limits threads.
    np.dot(np.ones((2, 2)), np.ones((2, 2)))
    return run_search_workflow(
        pipeline, grid, X_train, y_train, X_test, y_test, folds, run_dir,
        jobs=args.jobs, error_policy=args.error_score, refine=False, settings=settings)


if __name__ == '__main__':
    main()
