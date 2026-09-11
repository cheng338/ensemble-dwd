# ensemble-dwd

`ensemble-dwd` implements Chang Cheng's ensemble kernel Distance Weighted
Discrimination for binary and multiclass classification. Use
`EnsembleKernelDWD` for two classes and
`OvOClassifier(EnsembleKernelDWD(...))` for multiclass classification.

`EnsembleKernelDWD` is a binary classifier that uses several kernel DWD models
to select a smaller training set, then fits one final kernel DWD classifier.
It implements the four-stage method in Chapter 5 of Chang Cheng's dissertation,
*Overcoming the Curses of Dimensionality and Kernelization: Scalable DWD and
Supervised NMF* (Purdue University, August 2026).

`OvOClassifier(EnsembleKernelDWD(...))` extends this procedure to multiclass
classification using the dissertation's one-versus-one maximum-wins vote.
For ten classes, it fits 45 binary ensembles.

This is private research software. Copyright 2026 Chang Cheng. All rights
reserved. A software license has not yet been selected or granted.

See the [release validation](docs/validation-0.3.3.md) for the justification,
tests and limitations of major implementation changes.

## Installation

Python 3.11 or later is required. From this source directory:

```shell
python -m pip install .
```

To build and install a local wheel:

```shell
python -m pip install build
python -m build
python -m pip install dist/ensemble_dwd-0.3.3-py3-none-any.whl
```

The dependency metadata pins SHA-256-verified wheels from
[cheng338/dwd 1.3.8](https://github.com/cheng338/dwd/releases/tag/v1.3.8).
Conventional GIL-enabled CPython on Windows AMD64 uses the wheel with the compiled prediction helper;
other platforms use the portable wheel. Both contain the optimized DWD
implementation. Install the package with its declared dependencies; installing
`dwd` by name separately can retrieve another implementation.

Free-threaded Windows CPython cannot load that `abi3` wheel, and Python's
dependency markers cannot distinguish it from the conventional build. On
that interpreter, install the portable DWD wheel and numerical dependencies
explicitly, then install this package without dependency resolution:

```shell
python -m pip install numpy scipy "scikit-learn>=1.6" "joblib>=1.3" "dwd @ https://github.com/cheng338/dwd/releases/download/v1.3.8/dwd-1.3.8-py3-none-any.whl#sha256=27eaa3f12e9b980e27f2113b0abf1b5376d53e717beb48303a2a669194c7b212"
python -m pip install --no-deps .
```

This installation workaround has not been runtime-tested on free-threaded
Python. The validated environment uses conventional CPython 3.12.

## Basic use

This example demonstrates binary DWD. See [Multiclass use](#multiclass-use)
for classification with more than two classes.

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

The following defaults apply to `EnsembleKernelDWD`, including each pairwise
DWD ensemble inside `OvOClassifier`.

| Parameter | Default | Meaning |
|---|---:|---|
| `M` | `5` | Number of base learners. |
| `theta` | `0.1` | Base subset fraction; size is `max(2, floor(theta * n))`. |
| `sv_selection_ratio` | `0.3` | Fraction of training rows retained for the final refit. |
| `lambd` | `1e-6` | Shared package DWD regularization parameter. |
| `q` | `1.0` | Ordinary DWD loss exponent. |
| `kernel` | `'rbf'` | Kernel used by every learner. |
| `gamma` | `'scale'` | Resolved once per binary fit and shared by its base and final learners. |
| `max_iter`, `obj_tol` | `100`, `1e-5` | Default objective-change stopping controls. |
| `stopping`, `tol` | `'objective'`, `1e-6` | Stopping policy and optional optimality tolerance. |
| `acceleration` | `None` | Ordinary MM updates. |
| `random_state` | `42` | Reproducible subset sampling. |
| `prediction_batch_size` | `1024` | Query-row batch size for decision evaluation. |
| `cache_dir` | `None` | Optional local directory for reusing identical base fits. |
| `store_base_estimators` | `True` | Retain fitted base models for inspection. |
| `backend` | `'auto'` | Shared optimized DWD backend for all base and final fits. |

Base learners and the final model share their DWD parameters. Every fit
explicitly uses `implementation='optimized'`; the defaults therefore perform
five optimized base fits and one optimized final fit when no cache entry is
reused. A shared stopping policy
does not require all six fits to take the same number of iterations.
All DWD fits use zero initialization; they do not draw random initial
coefficients. `random_state` controls the ensemble's subset sampling.

An explicit `backend='spectral'`, `'cholesky'`, or `'lbfgs'` selects an
alternative supported by the optimized DWD dependency. Runtime and
finite-budget decision scores can differ; validate the complete ensemble
when changing this choice. See [Numerical controls](docs/method.md#numerical-controls).

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

This minimal example stops on every fit error, including an expected
single-class selection. For a broad search, use the checkpointed example
below: it excludes `IneligibleSelectionError` and its `IneligibleSubsetError`
subclass while still stopping on unexpected failures.

The default MNIST grid starts with 125 candidates:

- `lambd`: `2**e` for `e` in `[-30, -22, -14, -6, 0]`.
- `gamma`: `10**e` for `e` in `[-8, -6, -4, -2, 0]`.
- `theta`: `[0.1]` (fixed at 10%).
- `sv_selection_ratio`: `[0.1, 0.15, 0.2, 0.25, 0.3]`.

Call `mnist_param_grid(prefix='ensemble__', broad=True)` to restore the
historical 405-candidate grid: lambda exponents
`[-30, -26, -22, -18, -14, -10, -6, -2, 0]` and every integer gamma exponent
from -8 through 0, with the same fractions. Both grids cover the same outer
range. Neither helper fits models or tunes parameters internally.

With five folds and `M=5`, the default coarse stage has 625 ensemble CV fits,
or up to 3,750 DWD fits before base-cache reuse. The broad coarse stage has
2,025 ensemble CV fits, or up to 12,150 DWD fits. These are initial-stage
budgets, not measured runtimes or the total cost of adaptive search.

`examples/tune_mnist.py` runs bounded adaptive search through external
`GridSearchCV`. Each stage uses the same saved stratified five-fold split,
fits preprocessing inside each training fold, and sets `refit=False`.
Only new complete parameter combinations are evaluated. Results from earlier
stages in that run remain eligible for selection; changing only retention
can also reuse the package's base-stage cache.

The policy expands lambda/gamma endpoints reached by the best or tied-best
fully valid candidates, including cross-combinations when both axes expand.
Each outward step is at most one decade (a factor of 10), or the smaller
adjacent logarithmic spacing when the explored grid is finer. Sampling stays at 10% and retention stays on the five specified fractions.
Local refinement is enabled by default: it takes the top three distinct
lambda/gamma pairs, proposes logarithmic midpoints within the explored bounds,
and evaluates those pairs with all fixed fraction choices. Duplicate complete
candidates are removed. `--no-refine` disables this stage; `--refine` remains
accepted for compatibility.

Defaults are three complete expansion rounds, 500 unique candidates total,
a top-three refinement, two nonimproving complete expansion rounds for the
plateau rule, improvement tolerance 0, and boundary tie tolerance `1e-12`.
Use `--max-expansions`, `--max-candidates`, and `--broad-grid` to choose an
explicit budget. A complete proposed stage that does not fit the remaining
budget is held rather than partially evaluated. Unresolved boundaries or
exhausted limits produce `review.json` with `needs_review`, without final
fitting or test scoring. A documented plateau can finish the search even
while a best or tied-best candidate remains at a boundary. This stopping rule
does **not** prove a global optimum or rule out improvement farther away.

Only candidates with finite scores in every fold and a finite mean may win.
The default `--error-score nan` excludes only `IneligibleSelectionError` and
its `IneligibleSubsetError` subclass. Unexpected numerical, input or other
exceptions are recorded and stop the run; they are not silently assigned zero
accuracy or replaced by fallback classifiers. `--error-score raise` stops on
ineligible candidates too. An all-invalid search stops.

The selected parameters and review decision are saved before fitting the
pipeline once on all training rows. The separate test set is scored only
after the search is accepted and this final fit succeeds. A failed final fit
leaves the completed CV records available. The existing
`run_search_workflow` Python helper retains its explicit coarse/local behavior
for historical reproduction; the CLI now calls `run_adaptive_search`.

```shell
python examples/tune_mnist.py
python examples/tune_mnist.py --data mnist.npz --jobs 2 --run
python examples/tune_mnist.py --broad-grid --max-candidates 1000 --no-refine
```

Without `--run`, the example only previews its budget and does not load data.
It accepts local NumPy archives with Keras keys `x_train/y_train/x_test/y_test`,
capitalized `X_train/X_test` equivalents, or `X/y`; it never downloads data.
A supplied test split is preserved. `X/y` creates a separate stratified 25%
test split before CV. No dataset dimensions or sample-count defaults were
changed by the adaptive policy.

Choose a positive `--jobs` count for the available RAM. `pre_dispatch` equals
that count, each worker uses one native thread, and base models fit sequentially.
`--output-dir` must name a fresh directory; the default creates a unique
`results/mnist-*` directory with a shared base cache. `--cache-dir` selects a
trusted cache and `--no-cache` disables reuse. The examples discard base models
after retaining their diagnostics, scores and indices.

`run.json` records folds, data fingerprints, estimator settings and the search
policy. `coarse-cv.json`, `expansion-*-cv.json`, `local-cv.json` when present,
and `accumulated-cv.json` preserve completed stage results. `selection.json`
and `review.json` record the selection before final fitting; `refit.json`,
`status.json`, and `failures/failure-*.json` explain success or failure.
`completed.json` contains the held-out score after success. Reuse of scored
candidates is within one run; automatic recovery of interrupted stages across
new invocations is not implemented.

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
Choose `--jobs` for the available memory. Changes to DWD constructor parameters,
including `prediction_batch_size`, take effect on the next fit. For
`EnsembleKernelSVM`, `prediction_batch_size` can also be changed after fitting:
every prediction call validates it as `None` or a positive integer. Other
SVM fitting parameters take effect on the next fit.

## Results and boundaries

Disjoint sampling applies from version 0.3.2. Earlier validation reports used
base subsets that could overlap. Those reports retain their original meaning;
new sampling requires new evaluation and can change selections and predictions.

After fitting, inspect `subset_indices_`, `selected_indices_`,
`consensus_scores_`, `estimators_`, `final_estimator_`, and `gamma_`.
`n_iter_` and `converged_` describe the `M` base learners followed by the final
learner. An objective stopping event does not itself certify optimality.
`gamma_` is `None` for kernels that do not consume top-level gamma, including
linear and callable kernels. Callable kernels receive only their explicitly
supplied `kernel_kws`; the top-level gamma is not injected into them.

`EnsembleKernelDWD` accepts exactly two classes. Wrap it in `OvOClassifier`
for two or more classes. Precomputed kernels are unsupported.
Base subsets follow Mao's disjoint-subset rule: one uniform permutation
supplies `M` non-overlapping blocks of size `k = max(2, floor(theta * n))`,
requiring `M * k <= n` within every binary fit. A sampled block containing
only one class raises `IneligibleSubsetError` before base fitting; it is not
redrawn. The error subclasses `IneligibleSelectionError`. Final selection keeps exactly
`floor(sv_selection_ratio * n)` rows in stable order of absolute consensus
score. It raises an error if that set has fewer than two rows or only one
class; it does not substitute another classifier or expand the retained set.

See [the method and parameter conventions](docs/method.md) for details.

## Multiclass use

The following example tunes multiclass ensemble DWD. Each candidate uses
one parameter configuration across all
class pairs.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier, mnist_param_grid

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('ovo', OvOClassifier(EnsembleKernelDWD(
        M=5, random_state=42, store_base_estimators=False,
        cache_dir='.ensemble-dwd-cache',
    ), n_jobs=1)),
])
# A single candidate's parameters are shared across all class pairs.
search = GridSearchCV(
    pipeline, mnist_param_grid(prefix='ovo__estimator__'),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy', n_jobs=1, error_score='raise',
)
# search.fit(X_train, y_train)
```

Each class pair trains the same binary selection-and-refit procedure. Only
its final classifier votes. Ties select the smallest sorted original label;
decision-score magnitudes do not break ties. Labels retain
their input dtype, including strings and large integers. `vote_counts(X)`
returns integer counts in `classes_` order, not probabilities.

Numeric `gamma` is shared across all pairs. With `gamma='scale'`, each pair
resolves its own value and shares it within that pair's base and final fits.
Use numeric gamma when a single coefficient across pairs is required.
The configured binary random seed is cloned unchanged for each pair.

`OvOClassifier.n_jobs` controls parallel pair fitting and defaults to 1.
When CV uses multiple workers, leave it at 1 and limit native BLAS threads
inside each worker. For a standalone fit, pair parallelism can be enabled
through `n_jobs`. Prediction streams pair votes in batches controlled by
`prediction_batch_size=1024`. Retaining only final models with
`store_base_estimators=False` reduces memory without changing predictions.

Inspect `pairs_`, `estimators_` and `pair_indices_` after fitting. A child's
sampled/selected indices are local to that pair; map them through its
`pair_indices_` entry to obtain rows of the original multiclass training input.
An ineligible binary selection fails that entire multiclass candidate. No
missing pair or substitute model is silently accepted.

The default 125-candidate coarse stage costs up to 168,750 binary DWD fits
for ten classes and five folds; the broad 405-candidate stage costs up to
546,750. These counts precede adaptive stages and base-cache reuse. Fitting
is not launched automatically.
The multiclass example supports a compact validation grid and stratified subsets
from a local MNIST archive with separate training and test arrays:

```shell
python examples/tune_mnist_multiclass.py --smoke-grid
python examples/tune_mnist_multiclass.py --data mnist.npz --train-size 6000 --jobs 8 --smoke-grid --run
```

The normal multiclass example uses the same adaptive policy and keeps
preprocessing inside each CV fold. `--smoke-grid` overrides the adaptive
budget to exactly four candidates, zero expansions and no refinement;
unresolved winners remain held from final fitting/test scoring. This is an
integration check, not an adequate search for comparative accuracy. See [the multiclass method guide](docs/multiclass.md).

For comparison experiments, the package also includes `EnsembleKernelSVM`,
with binary and multiclass usage documented in [the comparator guide](docs/ensemble-svm.md).

## Attribution

Chang Cheng developed the ensemble kernel DWD method and its multiclass
extension. Kernel DWD fitting is provided by
the separately installed `cheng338/dwd` dependency, derived from
[slicersalt/dwd](https://github.com/slicersalt/dwd), originally authored by Iain
Carmichael with upstream maintenance by David Allemang / Kitware. The
dependency retains its own MIT license and attribution. Its license does not
grant rights to this separate package. Citation information is in
[CITATION.cff](CITATION.cff).

The SVM comparator is Chang Cheng's adaptation of
[Mao et al. (2016), *Fast Kernel SVM Training via Support Vector Identification*](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2016/media/files/0649.pdf).
It retains the label-aware margin-selection principle, with disjoint base subsets and a final hinge-loss `SVC` refit
in place of Mao's squared-loss kernel-coefficient fit. See the
[comparator guide](docs/ensemble-svm.md) for details.
