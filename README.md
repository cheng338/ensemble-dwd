# ensemble-dwd

`EnsembleKernelDWD` is a binary classifier that uses several kernel DWD models
to select a smaller training set, then fits one final kernel DWD classifier.
It implements the four-stage method in Chapter 5 of Chang Cheng's dissertation,
*Overcoming the Curses of Dimensionality and Kernelization: Scalable DWD and
Supervised NMF* (Purdue University, August 2026).

This is private research software. Copyright 2026 Chang Cheng. All rights
reserved. A software license has not yet been selected or granted.

## Installation

Python 3.11 or later is required. From this source directory:

```shell
python -m pip install .
```

To build and install a local wheel:

```shell
python -m pip install build
python -m build
python -m pip install dist/ensemble_dwd-0.1.1-py3-none-any.whl
```

The dependency metadata pins the DWD implementation to commit
`9ed1892d74c4eda5ed2e058df239ffb7973c120a` of
[cheng338/dwd](https://github.com/cheng338/dwd/tree/9ed1892d74c4eda5ed2e058df239ffb7973c120a).
Installation downloads that source archive unless it is already available to
the installer. Install the package with its declared dependencies; installing
`dwd` by name separately can retrieve another implementation.

## Basic use

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ensemble_dwd import EnsembleKernelDWD

digits = load_digits()
keep = (digits.target == 7) | (digits.target == 9)
X_train, X_test, y_train, y_test = train_test_split(
    digits.data[keep], digits.target[keep],
    test_size=0.25, stratify=digits.target[keep], random_state=42,
)
model = make_pipeline(StandardScaler(), EnsembleKernelDWD())
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.score(X_test, y_test))
```

The bundled digits dataset requires no download. The classifier preserves the
two input class labels. Its `decision_function` and `predict` use the final
refitted model; the average base-model scores are used during selection.

## Defaults

| Parameter | Default | Meaning |
|---|---:|---|
| `M` | `5` | Number of base learners. |
| `theta` | `0.1` | Base subset fraction; size is `max(2, floor(theta * n))`. |
| `sv_selection_ratio` | `0.3` | Fraction of training rows retained for the final refit. |
| `lambd` | `1e-6` | Shared package DWD regularization parameter. |
| `q` | `1.0` | Ordinary DWD loss exponent. |
| `kernel` | `'rbf'` | Kernel used by every learner. |
| `gamma` | `'scale'` | Resolved once from the complete current training fold and shared. |
| `max_iter`, `obj_tol` | `100`, `1e-5` | Default objective-change stopping controls. |
| `stopping`, `tol` | `'objective'`, `1e-6` | Stopping policy and optional optimality tolerance. |
| `acceleration` | `None` | Ordinary MM updates. |
| `random_state` | `42` | Reproducible subset sampling. |
| `prediction_batch_size` | `1024` | Query-row batch size for decision evaluation. |
| `cache_dir` | `None` | Optional local directory for reusing identical base fits. |
| `store_base_estimators` | `True` | Retain fitted base models for inspection. |

Base learners and the final model share their DWD parameters. Every fit
explicitly uses `implementation='optimized'`; the defaults therefore perform
five optimized base fits and one optimized final fit when no cache entry is
reused. A shared stopping policy
does not require all six fits to take the same number of iterations.
All DWD fits use zero initialization; they do not draw random initial
coefficients. `random_state` controls the ensemble's subset sampling.

These are starting settings, not universal optima. `theta` and
`sv_selection_ratio` have separate roles. The method uses the arithmetic mean
of raw decision values, without normalizing individual learners' scores.

## Parameter search

Put preprocessing inside the cross-validation pipeline. Search the complete
ensemble and score its final refit's predictions:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ensemble_dwd import EnsembleKernelDWD, mnist_param_grid

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('ensemble', EnsembleKernelDWD(
        cache_dir='.ensemble-dwd-cache', store_base_estimators=False,
    )),
])
search = GridSearchCV(
    pipeline,
    mnist_param_grid(prefix='ensemble__'),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy', n_jobs=1, error_score='raise',
)
# Run explicitly when the data and computational budget are ready:
# search.fit(X_train, y_train)
```

The broad MNIST preset contains:

- `lambd`: `2**e` for `e` in `[-30, -26, -22, -18, -14, -10, -6, -2, 0]`.
- `gamma`: `10**e` for `e` in `[-8, -7, -6, -5, -4, -3, -2, -1, 0]`.
- `theta`: `[0.1, 0.2]`.
- `sv_selection_ratio`: `[0.1, 0.15, 0.2, 0.25, 0.3]`.

This is 810 candidates and 4,050 ensemble fits with five folds. With `M=5`,
the uncached budget is up to 24,300 DWD fits for cross-validation, plus six for
the selected candidate's refit. Reusing base fits across retention ratios can
reduce that work. This is a proposed search space, not a benchmark result.
The helper returns a parameter grid and does not launch fitting.

`examples/tune_mnist.py` accepts a local NumPy archive with standard Keras keys
`x_train/y_train/x_test/y_test`, capitalized `X_train/X_test` equivalents, or
`X/y`. It requires `--run` before starting a search and does not download
MNIST. Set `--jobs` to a positive worker count that fits the available RAM;
each worker trains its DWD models sequentially with one native BLAS thread.
`examples/basic_usage.py` demonstrates a single fit on bundled digits.

The broad search example defaults to `--error-score nan` for the expected
`IneligibleSelectionError`: a requested consensus set may contain too few rows
or only one class. Such candidates are reported and excluded. Unexpected
exceptions, including numerical or input errors, are recorded and stop the
search. Use `--error-score raise` to stop on ineligible selections as well.
Only candidates with finite scores in every fold and a finite mean may be
selected; an all-invalid search stops. Failures are never assigned zero
accuracy or replaced by fallback classifiers.

Add `--refine` for a second, explicit local search around the three best fully
valid coarse candidates. For each candidate, multiply lambda by
`[0.25, 1, 4]` and gamma by `[10**-0.5, 1, 10**0.5]`, keeping that candidate's
sampling and retention fractions. Duplicate combinations are removed, and
values may extend beyond the coarse grid. This adds at most 27 candidates,
135 ensemble CV fits, or up to 810 uncached DWD CV fits. Both stages reuse the
same training folds. Each completed stage writes its CV results, fold indices, data
fingerprint and selected parameters before the final refit. Both searches use
`refit=False`; the selected pipeline is then fitted explicitly on all training
rows. A final-fit failure leaves the completed CV checkpoint available, and
the held-out test set is scored only after a successful final fit. Running the
example without `--run`
prints these budgets without loading data or fitting models.

```shell
python examples/tune_mnist.py --refine
python examples/tune_mnist.py --data mnist.npz --jobs 2 --error-score nan --refine --run
```

The example creates a separate output directory for each run, uses a shared
base-fit cache inside it, and discards fitted base models after recording their
diagnostics. Use `--output-dir PATH` to choose a fresh run directory; otherwise
it creates a unique directory under `results/`. `--cache-dir PATH` selects a
trusted shared cache, and `--no-cache` disables reuse.

`run.json` records input hashes, folds and settings. `coarse-cv.json` and,
when requested, `local-cv.json` hold completed CV results. `selection.json` is
written before final fitting; `refit.json`, `status.json` and any
`failures/failure-*.json` records explain the outcome. `completed.json` contains
the held-out score after success. These files preserve completed work for an
explicit retry; automatic resume is not implemented. An interrupted or failed
CV stage may not have complete CV results, but its run and failure records
remain available. CV stays external through `GridSearchCV`.

## Reusing work and storing models

Set `cache_dir` to reuse the base stage when only the retention fraction
changes. The key includes the current training data and row order, labels,
subsets, kernel and solver settings, and numerical runtime. It excludes
`sv_selection_ratio`, because selection and the final fit are performed anew.
Entries are written atomically under a per-key operating-system lock so
parallel workers can share the directory. Caching is bypassed for
`random_state=None`, callable kernels or callable kernel keywords, and custom
estimator subclasses. Inspect `cache_hit_`, `cache_key_`, and `cache_status_`
after fitting to see whether the base stage was reused.

With `store_base_estimators=False`, `estimators_` is `None` after fitting.
Consensus scores, sampled and selected indices, iteration counts, convergence
flags, and the final predictor remain available. This reduces saved-model and
retained memory without changing selection or prediction. The default keeps
the base models for inspection.

Cache entries use pickle-based local serialization. Use a private directory
and load entries only from trusted sources. Lean entries do not directly store
the training feature matrix, but they still contain scores and index
diagnostics. Entries that retain base models also contain their training rows.

Training still requires dense kernel matrices. Limiting workers and prediction
batches controls some allocations; it is not an automatic RAM guarantee.
Choose `--jobs` for the available memory. Changing constructor parameters after
fitting takes effect on the next fit, including `prediction_batch_size`.

## Results and boundaries

After fitting, inspect `subset_indices_`, `selected_indices_`,
`consensus_scores_`, `estimators_`, `final_estimator_`, and `gamma_`.
`n_iter_` and `converged_` describe the `M` base learners followed by the final
learner. An objective stopping event does not itself certify optimality.
`gamma_` is `None` for kernels that do not consume top-level gamma, including
linear and callable kernels. Callable kernels receive only their explicitly
supplied `kernel_kws`; the top-level gamma is not injected into them.

The package accepts exactly two classes. Precomputed kernels are unsupported.
Base subsets are sampled uniformly without replacement conditional on
containing both classes. Final selection keeps exactly
`floor(sv_selection_ratio * n)` rows in stable order of absolute consensus
score. It raises an error if that set has fewer than two rows or only one
class; it does not substitute another classifier or expand the retained set.

See [the method and parameter conventions](docs/method.md) for details.

## Attribution

Chang Cheng developed the ensemble method. Kernel DWD fitting is provided by
the separately installed `cheng338/dwd` dependency, derived from
[slicersalt/dwd](https://github.com/slicersalt/dwd), originally authored by Iain
Carmichael with upstream maintenance by David Allemang / Kitware. The
dependency retains its own MIT license and attribution. Its license does not
grant rights to this separate package. Citation information is in
[CITATION.cff](CITATION.cff).
