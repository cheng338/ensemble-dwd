# Overview

`ensemble-dwd` implements ensemble kernel Distance Weighted Discrimination
(DWD) for binary and multiclass classification.
It is based on Chang Cheng's earlier ensemble DWD scripts accompanying his
dissertation.

The package implements:

- **Binary ensemble kernel DWD** (`EnsembleKernelDWD`): fits kernel DWD models
  on disjoint subsets, averages their raw signed decision scores, selects rows
  by the smallest absolute consensus scores, and fits a final DWD model on
  those rows.
- **Multiclass ensemble kernel DWD**
  (`OvOClassifier(EnsembleKernelDWD(...))`): fits one binary ensemble per class
  pair and predicts by the dissertation's one-versus-one maximum-wins vote.
- **Ensemble kernel SVM comparator** (`EnsembleKernelSVM`): averages scores
  from SVMs fitted on disjoint subsets, selects rows whose signed margins are
  closest to one, and fits a final SVC model.

The four-stage ensemble DWD method is described in Chapter 5 of:

> Cheng, Chang (2026). *Ensemble Kernel Distance-Weighted Discrimination and
> Supervised Nonnegative Matrix Factorization for Classification*. Doctoral dissertation
> manuscript, Purdue University.

See [CITATION.cff](CITATION.cff) for citation details.

Copyright 2026 Chang Cheng. **All Rights
Reserved.** A software license has not yet been selected or granted. See
[LICENSE](LICENSE).

# Installation

Python 3.11 or later is required. Version 0.3.14 requires DWD 1.3.18 from the
[DWD release assets](https://github.com/cheng338/dwd/releases/tag/v1.3.18).
From the extracted ensemble-dwd source directory, install that dependency with
hash verification before this package:

```shell
python -m pip install --no-index --find-links https://github.com/cheng338/dwd/releases/expanded_assets/v1.3.18 --no-deps --force-reinstall --require-hashes -r requirements-dwd.txt
python -m pip install .
```

For a local installation, place the released DWD wheels in a `wheels`
directory beside `pyproject.toml`, then use:

```shell
python -m pip install --no-index --find-links wheels --no-deps --force-reinstall --require-hashes -r requirements-dwd.txt
python -m pip install .
```

After installing the locked DWD dependency, build and install a local wheel:

```shell
python -m pip install build
python -m build
python -m pip install dist/ensemble_dwd-0.3.14-py3-none-any.whl
```

The package metadata pins the DWD version; `requirements-dwd.txt` additionally
pins the SHA-256 hashes of both validated wheel variants. The lock installation
uses only the specified release assets or local wheels and reinstalls DWD so an
existing copy does not bypass hash verification. Other numerical dependencies
must already be installed or available from an index. Installing `dwd` by name
from an index instead can retrieve another implementation.

Pip selects a compatible wheel from `wheels`. Both DWD wheel variants require
Python 3.11 or later. The Windows AMD64 `abi3` wheel contains the compiled
helper; the portable wheel contains the optimized Python implementation. To
require the portable build, put only that wheel in `wheels` when installing
the dependency. Windows validation does not establish numerical equivalence
on Linux, a different processor or free-threaded Python.

Python 3.10 is no longer supported. Previously released companion wheels
remain available as historical artifacts and will not receive updates.

DWD 1.3.18 admits later CPython 3 minor releases without a version ceiling.
Its independent compiled residual and dense-score routes accept GIL-enabled
Windows x86-64 CPython 3.11 and later 3.x releases. Its built-in `math.sumprod`
residual route accepts GIL-enabled CPython 3.12 and later 3.x releases under the
existing binary64 and arithmetic guards. Unsupported inputs or runtimes retain
guarded fallback arithmetic. Wheel compatibility does not guarantee that a
compiled scoring path is active. See DWD's
[compiled-helper guide](https://github.com/cheng338/dwd/blob/v1.3.18/docs/compiled_residual.md).

Complete-suite validation targets Python 3.11 and 3.12. Python 3.13 and later
Python 3.x releases are supported without dedicated complete-suite runs.
Recorded numerical checks cover 3.11 through 3.13; forward eligibility does not
claim testing on future interpreters. Earlier 3.13 results retain their
historical scope. This runtime change leaves the ensemble method unchanged.

The dependency also reuses bounded temporary buffers and avoids repeated work
within eligible prediction calls. These changes apply to base and final DWD
estimators without altering disjoint sampling, selection or the fitted objective.
The CI workflow uses the same dependency lock and explicit DWD 1.3.18 release
asset page.

Version 0.3.13 uses automatic affine scoring and residual ordering for new
ensembles and enables eligible equivalent-ratio fold reuse in the examples.
Explicit historical numerical policies and `--no-reuse-equivalent-folds` remain
available. Existing saved models retain their fitted scoring behavior. Direct
RBF computation, extended exact recovery and parallel base fitting remain
optional; sequential base fitting (`base_jobs=1`) and the SVM comparator's RBF
kernel remain the defaults. See
[direct RBF computation](docs/method.md#optional-direct-rbf-computation),
[joint affine scoring](docs/method.md#optional-joint-affine-scoring),
[extended exact recovery](docs/method.md#optional-extended-exact-recovery) and
[parallel base fitting](docs/parallel-base-fitting.md) for their scope.

# Basic use

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

# Defaults

The following defaults apply to `EnsembleKernelDWD`, including each pairwise
DWD ensemble inside `OvOClassifier`.

| Parameter | Default | Meaning |
|---|---:|---|
| `M` | `5` | Number of base learners. |
| `base_jobs` | `1` | Sequential bases; explicit larger values request eligible bounded parallel fitting. |
| `theta` | `0.1` | Base subset fraction; size is `max(2, floor(theta * n))`. |
| `sv_selection_ratio` | `0.3` | Fraction of training rows retained for the final refit. |
| `lambd` | `1e-6` | Shared package DWD regularization parameter. |
| `q` | `1.0` | Ordinary DWD loss exponent. |
| `kernel` | `'rbf'` | Kernel used by every learner. |
| `rbf_computation` | `'standard'` | RBF arithmetic; opt-in `'direct'` uses coordinate differences. |
| `affine_computation` | `'auto'` | Joint affine protection when supported; explicit `'standard'` preserves historical scoring. |
| `residual_check_order` | `'auto'` | Adaptive checks on eligible optimized Cholesky paths; applicable historical ordering elsewhere. |
| `exact_recovery` | `'standard'` | Exact-recovery budget; opt-in `'extended'` permits rank 32 and 8192-bit arithmetic for eligible small kernels. |
| `gamma` | `'scale'` | Resolved once per binary fit and shared by its base and final learners. |
| `max_iter`, `obj_tol` | `100`, `1e-5` | Default objective-change stopping controls. |
| `stopping`, `tol` | `'objective'`, `1e-6` | Stopping policy and optional optimality tolerance. |
| `acceleration` | `'auto'` | Restarted MM on compatible backends; ordinary updates for L-BFGS. Use `None` for ordinary MM explicitly. |
| `linear_action` | `'factor'` | DWD shifted-system action; explicit `'inverse'` trades setup and additional dense storage for repeated matrix-vector actions, with unchanged acceptance/recovery. |
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

To address sensitive RBF distance calculations explicitly, use
`EnsembleKernelDWD(rbf_computation='direct')`. It applies to every base and
final learner and can also be passed through
`OvOClassifier(EnsembleKernelDWD(rbf_computation='direct'))`. The standard
policy remains the default. Direct computation preserves the ensemble method
but can change rounded scores and selected rows and can be substantially
slower. See [its scope, dependency requirement and measured cost](docs/method.md#optional-direct-rbf-computation).

The default `affine_computation='auto'` includes the intercept in joint
evaluation of `K @ alpha + b` for each base and final learner when the
dependency supports it. Explicit `'standard'` preserves historical scoring. It preserves the mathematical
score and selection rule, but corrected rounding can change close rankings,
selected rows and labels. Its cost depends on how many scores need refinement;
see [joint affine scoring](docs/method.md#optional-joint-affine-scoring).

`EnsembleKernelDWD(exact_recovery='extended')` increases the bounded
exact-recovery budget shared by every base and final DWD learner. It can
recover some small difficult fits that exceed the standard rank or integer-bit
limit. It preserves the original kernel and acceptance checks, does not
guarantee convergence, and is not a speed improvement. The entry limit still
excludes kernels with more than 256 rows. See
[extended exact recovery](docs/method.md#optional-extended-exact-recovery).

# Parameter search

Search the complete ensemble and score predictions from its final refitted
model. This MNIST example assumes training and test pixels are already
floating-point values in `[0, 1]`: divide raw `0-255` pixels by 255 once, and
leave already scaled pixels unchanged. The pipeline applies no further
pixel transformation. Any learned preprocessing belongs inside the CV pipeline.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from ensemble_dwd import EnsembleKernelDWD, mnist_param_grid

pipeline = Pipeline([
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
# search.fit(X_train_unit, y_train)
# search.score(X_test_unit, y_test)  # Evaluate only after tuning is complete.
```

This example stops on every fit error, including an expected
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

With the default single seed, five folds and `M=5`, the coarse stage has
625 ensemble CV fits, or up to 3,750 DWD fits before base-cache reuse. The
broad coarse stage has 2,025 ensemble CV fits, or up to 12,150 DWD fits. These
are initial-stage budgets, not measured runtimes or the total cost of adaptive search.

`examples/tune_mnist.py` runs bounded adaptive search through the external
`CheckpointedCV` runner. Every stage uses the same saved stratified five-fold
split, fits preprocessing inside each training fold, and scores accuracy
without an automatic refit. Only new parameter combinations are
proposed; verified completed fold records are reused. Earlier candidates
remain eligible for selection. Changing only retention can also reuse the
package's separate base-stage cache.

The three MNIST tuning commands default to `--preprocessing unit`, dividing
raw pixels by 255 to produce features in `[0, 1]`. Omitting
`--pixel-input-range` declares raw `0-255` input by default; the commands do
not infer the range from observed values. For data already in `[0, 1]`, pass
`--pixel-input-range 0-1` to avoid dividing again. Explicit
`--preprocessing standard` instead fits `StandardScaler` inside each
training-fold pipeline. The commands apply only the chosen transformation.
Retune the coefficients and use a fresh output directory when changing
preprocessing or its declared input range. Neither choice guarantees better
accuracy or speed. See
[MNIST pixel preprocessing](docs/method.md#mnist-pixel-preprocessing).

All three tuning commands optionally accept `--cv-seeds 42 43 44`. Each
candidate uses the same prescribed seeds and folds. Scores are averaged across
seeds within each fold, then across folds; seeds are never competing parameter
choices. Every required seed and fold must have a valid score. This costs one
CV fit per candidate, fold and seed, while `--max-candidates` still counts
parameter sets. The default remains one seed, 42.

The selected parameters receive one separate final fit. With `--cv-seeds`,
`--refit-seed` can prescribe that fit's seed; otherwise it retains the
estimator's configured seed (42 in the commands). Both seed choices are saved
and must remain unchanged on resume. See the
[seed aggregation and records](docs/method.md#multiple-cv-seeds).

The examples automatically reuse an eligible completed CV result
when two retention ratios select exactly the same rows under identical
settings. Each candidate keeps its original ratio and record. This is useful
for closely spaced custom ratios; the default MNIST grids may gain nothing.
Use `--no-reuse-equivalent-folds` to disable this optimization. Pair recovery
remains optional and can be combined with eligible reuse. The selected ratio
still receives a fresh fit on all training rows. See the
[reuse conditions and records](docs/method.md#equivalent-fold-reuse).

The policy expands lambda/gamma endpoints reached by the best or tied-best
fully valid candidates, including cross-combinations when both axes expand.
Each outward step is at most one decade (a factor of 10), or the smaller
adjacent logarithmic spacing when the explored grid is finer. Sampling stays
at 10%; retention stays on the five specified fractions unless the optional
ratio refinement below is enabled. Missing conditional neighbors are tested
at the same values of the other parameters. An invalid
neighbor does not establish an interior optimum.

When logarithm rounding loses a valid outward point, the examples
recover the capped geometric extension with exact arithmetic, as introduced
in version 0.3.8. Previously
successful coordinates stay unchanged; true floating-point limits still stop
expansion. Distinct parameter values are not merged.

Local refinement repeatedly proposes joint logarithmic midpoints around the
top three distinct lambda/gamma pairs, within the explored bounds, with all
fixed fraction choices. The default stopping rule requires valid neighbors
on both sides of the selected candidate in each axis, each within a factor
of two, and two completed local-refinement rounds with no improvement or
change of winning region. Filling gaps at existing coordinates does not earn
stable-refinement credit. An improvement or changed evidence reopens review.
`--no-refine` skips local refinement and factor-two tightening, but still
requires valid conditional neighbors. `--refine` remains accepted.

Defaults remain 125 initial candidates, three complete expansion rounds and
500 unique candidates total (`--broad-grid` starts with 405). Improvement
tolerance is 0 and boundary tie tolerance is `1e-12`. Use `--max-expansions`
and `--max-candidates` to choose an explicit budget. A complete proposed stage
that does not fit the remaining budget is held rather than partly evaluated.
Unresolved boundaries or exhausted limits produce `review.json` with
`needs_review`, without final fitting or test scoring.

`--refine-ratio` adds bounded retention-ratio refinement after the coefficient
search meets its existing acceptance rules. It tests arithmetic ratio
midpoints together with nearby logarithmic coefficient midpoints around
promising complete parameter sets. It does not widen the initial grid or
combine every new ratio with all earlier coefficient choices. Defaults are
three ratio rounds, two stable rounds, and a maximum conditional ratio gap
of `0.01` (one percentage point). The initial ratio range supplies the bounds:
`[0.1, 0.3]` for the MNIST presets.

Use `--ratio-bounds LOW HIGH`, `--max-ratio-rounds N`, and `--ratio-tol X` to
prescribe the range and limits. Bounds must satisfy `0 < LOW < HIGH <= 1` and
contain every initial ratio. These settings require `--refine-ratio`, which
cannot be combined with `--no-refine` or the multiclass `--smoke-grid`.
All added candidates share the existing total `--max-candidates` budget.
An unresolved ratio or coefficient bracket holds final fitting and test
scoring. Acceptance at a ratio endpoint is restricted to the declared range.
See [selection-ratio refinement](docs/method.md#selection-ratio-refinement)
for acceptance, checkpoint and reuse rules.

The full-strip plateau rule can waive further checks at a distant tied outer
edge after two completed nonimproving outward levels, supported by valid
results on the same conditional slice. It does not waive the selected
candidate's own neighbor requirements or treat invalid/missing scores as
plateau evidence. This is empirical evidence within the tested range, not a
global-optimum certificate.

Only candidates with finite scores in every fold and a finite mean may win.
The default `--error-score nan` excludes only `IneligibleSelectionError` and
its `IneligibleSubsetError` subclass. Unexpected numerical, input or other
exceptions are recorded and stop the run; they are not assigned zero accuracy
or replaced by fallback classifiers. `--error-score raise` stops on
ineligible candidates too. An all-invalid search stops.

The binary DWD and multiclass DWD commands optionally accept
`--confirm-max-iter 300` to check a fixed shortlist at a larger iteration cap
after the search is accepted. The cap must exceed the original cap, 100 by
default. `--confirm-top-k` sets the shortlist limit, three by default and at
least two. Confirmation uses the same folds and sampling seeds, with separate
scores, diagnostics and checkpoints. It is disabled unless a cap is supplied;
the SVM command does not provide this DWD diagnostic.

A changed ranking or tie, an invalid confirmation result, or fewer than two
eligible candidates holds final fitting and test scoring for review. An
unchanged ranking allows the original selected parameters to receive their
final fit with the original solver cap. Confirmation neither promotes another
candidate nor certifies convergence. An earlier objective-tolerance stop may
make the larger cap ineffective. The shortlist adds work outside the search's
candidate budget; see [numerical-budget confirmation](docs/method.md#numerical-budget-confirmation)
for cost, records and limits.

The selected parameters and review decision are saved before fitting the
pipeline on all training rows. The separate test set is scored only after
acceptance and successful final fitting. Compact diagnostics are saved for
each completed fold fit and the final refit. Reported objective stopping,
iteration limits and optimality checks are distinct; the search does not
change the default DWD objective tolerance (`1e-5`) or iteration cap (100).
The legacy `run_search_workflow` helper remains a nonresumable coarse/local
workflow. It preserves all parent parameters during local refinement and
retains a better coarse candidate when local results do not improve it.

Without `--run`, the example previews its budget and does not load data.
It accepts local NumPy archives with Keras keys `x_train/y_train/x_test/y_test`,
capitalized `X_train/X_test` equivalents, or `X/y`; it never downloads data.
A supplied test split is preserved. `X/y` creates a separate stratified 25%
test split before CV.

Choose a positive `--jobs` count for the available RAM; the default is 1.
Dispatched CV work is bounded by that count, each worker uses one native thread,
and base models fit sequentially. Optional `--final-native-threads N` sets a
positive integer limit for each detected native BLAS/OpenMP pool during the
selected full-training pipeline fit. Its default is also 1; CV and held-out
prediction stay at one native thread. This is a library-pool limit, not a total
CPU limit or a physical-core assignment. Wider final budgets require sequential
nested estimators and may change numerical rounding. See
[native threads by phase](docs/method.md#native-threads-by-phase) for scope,
runtime records and resume rules.

New runs require a fresh `--output-dir`; the default creates a
unique `results/mnist-*` directory. `--cache-dir` selects a trusted base cache
and `--no-cache` disables it. The examples discard base models after retaining
their diagnostics, scores and indices.

```shell
python examples/tune_mnist.py
python examples/tune_mnist.py --data mnist.npz --jobs 2 --output-dir results/mnist-search --run
python examples/tune_mnist.py --data mnist.npz --jobs 2 --output-dir results/mnist-search --run --resume
python examples/tune_mnist.py --broad-grid --max-candidates 1000 --no-refine
python examples/tune_mnist.py --refine-ratio --max-ratio-rounds 3 --ratio-tol 0.01
python examples/tune_mnist.py --confirm-max-iter 300 --confirm-top-k 3
```

To resume, pass `--resume --run` with the original output directory and the
same data, folds, estimator parameters, policy and budgets, job count, and
numerical runtime. The installation must remain immutable. Identity mismatch
or live ownership prevents reuse. Checksummed completed fold records survive
interruption; unfinished fits restart by default. The multiclass example also
offers [optional completed-pair recovery](docs/multiclass.md#completed-pair-recovery).
This checkpoint is separate from the
optional base-fit cache, so `--no-cache` does not disable completed-fold reuse.

Checkpoint schema 2 preserves parameter types because numerically equal Python
and NumPy values can produce different subset sizes. Checkpoints from earlier
schemas require a new run directory. The checkpoint helper rejects custom
scalar/container subclasses and object, structured, or metadata-bearing array
parameters when their identity cannot be verified safely. These restrictions
apply to checkpointing, not to the estimators' ordinary fit methods.

`run.json` records the identity, settings and search policy. Stage CV files,
including repeated local and conditional-repair stages, and
`accumulated-cv.json` preserve search evidence. `selection.json` and
`review.json` record selection before final fitting; `refit.json`,
`status.json`, and `failures/failure-*.json` describe later outcomes.
`completed.json` mirrors the committed held-out result.

Ensemble failures also retain bounded stage, learner-progress and class-count
details in `ensemble_failure_context`, when available. These survive OvO and
CV failure reporting; bases restored from cache are separate from current fit
work. See [failure evidence](docs/method.md#ensemble-failure-evidence).

The examples also write `work-summary.json` automatically. It separates work
executed during the current invocation from equivalent-ratio reuse and
checkpoint replay, with distinct search, optional confirmation, final-fit
and held-out scoring records. Worker timing sums, stored receipt timings and
elapsed time are separate quantities. See
[work and timing summaries](docs/method.md#work-and-timing-summaries), including
how completed-result replay and interrupted stages are reported.

Before each adaptive stage, `<stage>-plan.json` records its candidate budget
and planned binary, base, final, and full-model learner counts. These describe
one successful pass before reuse or failure; they are not completed work or a
time estimate. Refused coefficient and ratio stages have separate
`*-refused-stage.json` reports. Large proposed grids retain their generating
values and a labeled lower bound when a complete count would be costly.
Reporting preserves candidate identities and search rules. See
[work and timing summaries](docs/method.md#work-and-timing-summaries).

Stage plans also disclose newly introduced close but distinct parameter values,
with bounded detail samples and brief console notices. These observations do
not establish equivalent fits: every candidate keeps its exact values and
checkpoint identity. See [the reporting details](docs/method.md#work-and-timing-summaries).

An interrupted final refit restarts from the beginning; its fitted models are
not serialized. A crash after held-out prediction but before the authoritative
completion receipt is committed can require refitting and rescoring. Once
that receipt is committed, a verified resume returns the saved result without
fitting or scoring again. The Python result then has `model=None` and
`resumed_completed=True`; it does not restore a predictor.

# Reusing work and storing models

Set `cache_dir` to reuse the base stage when only the retention fraction
changes. The key includes the current training data and row order, labels,
subsets, kernel and solver settings, and numerical runtime. It excludes
`sv_selection_ratio`, because selection and the final fit are performed anew.
Entries are written atomically under a per-key operating-system lock so
parallel workers can share the directory. Caching is bypassed for
`random_state=None`, callable kernels or callable kernel keywords, and custom
estimator subclasses. Inspect `cache_hit_`, `cache_key_`, and `cache_status_`
after fitting to see whether the base stage was reused.
`cache_dir` accepts strings and path-like objects. Internal cache files and
locks support long Windows paths, including paths beyond the legacy 260-character limit.

Cache provenance hashes package-owned source/native files, loaded BLAS/OpenMP
libraries, loaded NumPy/SciPy/scikit-learn native extensions, and an already
loaded AOCL bridge's source, manifests and libraries. File contents are hashed
once per process; observed file or native-inventory changes disable reuse until
restart. The runtime identity is checked again before a new cache entry is
saved. The installation must remain unchanged from import. The identity
describes files on disk; it does not verify code already loaded into the
process. Same-version edits to external scientific Python source and
unobserved transitive libraries remain outside this identity.

With `store_base_estimators=False`, `estimators_` is `None` after fitting.
Consensus scores, sampled and selected indices, iteration counts, convergence
flags, and the final predictor remain available. This reduces saved-model and
retained memory without changing selection or prediction. The default keeps
the base models for inspection.

Cache entries use pickle-based local serialization. Use a private directory
and load entries only from trusted sources. Entries without retained base
models do not directly store the training feature matrix, but they still
contain scores and index diagnostics. Entries that retain base models also
contain their training rows.

Training still requires dense kernel matrices. Limiting workers and prediction
batches controls some allocations but does not guarantee a fixed memory limit.
Choose `--jobs` for the available memory. Changes to DWD constructor parameters,
including `prediction_batch_size`, take effect on the next fit. For
`EnsembleKernelSVM`, `prediction_batch_size` can also be changed after fitting:
every prediction call validates it as `None` or a positive integer. Its default
is `None`, using one native call for the supplied rows. The SVM's temporary
kernel-cache capacity target defaults to 1024 MiB per active fit, allocated as needed; concurrent
fits each have their own allowance. Choose `cache_size` and worker counts
together. The SVM example prints capacity guidance, without enforcing memory
admission. Other SVM fitting parameters take effect on the next fit. See the
[SVM memory and prediction notes](docs/ensemble-svm.md#parallel-work-caching-and-diagnostics).

# Multiclass use

`OvOClassifier(EnsembleKernelDWD(...))` fits 45 binary ensembles for ten classes.

The following example tunes multiclass ensemble DWD. Each candidate uses
one parameter configuration across all class pairs. Supply training and test
pixels already in `[0, 1]`, dividing raw `0-255` values by 255 once and leaving
already scaled values unchanged. No further pixel transformation is applied.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier, mnist_param_grid

pipeline = Pipeline([
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
# search.fit(X_train_unit, y_train)
# search.score(X_test_unit, y_test)  # Evaluate only after tuning is complete.
```

Each class pair uses the same binary procedure for selection and refitting. Only
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
budget to exactly four candidates, zero expansions and no refinement. If the
search cannot accept a candidate, it does not fit a final model or score the
test set. This checks that the workflow runs; it is not an adequate search
for comparing accuracy. See [the multiclass method guide](docs/multiclass.md).

`--pair-recovery --no-cache` optionally saves completed binary pairs inside
unfinished CV folds for verified local resume. It is off by default and
does not checkpoint an unfinished binary fit or the final full-training fit.
See [supported settings and storage limits](docs/multiclass.md#completed-pair-recovery).

For binary and multiclass examples of `EnsembleKernelSVM`, see
[the comparator guide](docs/ensemble-svm.md).

# Results and boundaries

See the [0.3.14 changelog](CHANGELOG.md#0314--2026-10-21) for this release's changes.
The [0.3.4 release notes](docs/candidate_changes.md) and
[validation history](VALIDATION.md) retain the evidence and limitations
of earlier implementation changes.

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

The package also offers optional
[`fit_profile()`](docs/method.md#optional-fit-profiling) to measure ensemble-DWD
fit stages and distinguish new base work from cache reuse. It records timing
outside the fitted model, with no change to estimator settings or mathematics.

For explicit concurrent base fitting, see
[`base_jobs` and runtime ownership](docs/parallel-base-fitting.md).
Sequential base fitting remains the default; the final learner is fitted after
the base workers finish.

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

Explicit DWD `linear_action="inverse"` uses a full per-solve shifted-system
inverse action while retaining original-kernel acceptance and recovery. Factor
remains the default. The inverse requires a DWD version providing this option;
released older dependencies raise a clear ImportError rather than substitute
another action. Each active fit adds 8*n*n bytes for B, and construction can
temporarily hold three B-sized arrays plus native workspace; budget every
concurrent learner. The inverse is not persisted in cached models. Requested
and effective policies survive lean/cache diagnostics. This option may help
long repeated solves but is not a universal speedup or memory-saving path.

# Attribution

Kernel DWD fitting is provided by the separately installed `cheng338/dwd`
dependency, derived from
[slicersalt/dwd](https://github.com/slicersalt/dwd), originally authored by Iain
Carmichael with upstream maintenance by David Allemang / Kitware. The
dependency retains its own MIT license and attribution. Its license does not
grant rights to this separate package.

The SVM comparator is Chang Cheng's adaptation of
[Mao et al. (2016), *Fast Kernel SVM Training via Support Vector Identification*](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2016/media/files/0649.pdf).
It retains the label-aware margin-selection principle, with disjoint base subsets and a final hinge-loss `SVC` refit
in place of Mao's squared-loss kernel-coefficient fit. See the
[comparator guide](docs/ensemble-svm.md) for details.
