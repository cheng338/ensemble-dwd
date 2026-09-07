# Binary ensemble kernel DWD

The estimator implements the four stages in Section 5.1.1 of Chang Cheng's
dissertation, *Overcoming the Curses of Dimensionality and Kernelization:
Scalable DWD and Supervised NMF* (Purdue University, August 2026, pp. 46–47).
It supports binary classification; multiclass composition is outside this
version's scope.

## Fit and prediction

For training rows `(X, y)` of size `n`:

1. Generate `M` subsets of size `max(2, floor(theta * n))`, without replacement
   within a subset. Subsets can overlap each other. Each subset contains both
   classes.
2. Fit an optimized kernel DWD model on every subset.
3. Evaluate each base learner's raw decision function on every training row.
   Average across learners and rank rows by the absolute average. Break ties
   by original row order. Keep exactly `floor(sv_selection_ratio * n)` rows.
4. Fit one optimized kernel DWD on these retained rows. Use this final
   classifier to predict new observations.

For base decision functions `f_1, ..., f_M`, the selection score is
`g(x) = sum(f_m(x) for m in range(M)) / M`. The selected rows minimize
`abs(g(x))`. There is no per-learner score normalization or probability
conversion. Selecting by the average of absolute scores would be a different
rule.

The retained rows are a consensus training subset. The term "support vectors"
here describes the selection procedure; it does not imply an exact sparse
support representation of the full-data DWD optimum.

## Sampling and invalid selections

The dissertation specifies uniform sampling without replacement. A binary
learner also requires both labels to be represented. This implementation uses
that distribution conditional on drawing both classes: sample the first
class's count from the feasible, truncated hypergeometric distribution, then
sample the required rows uniformly within each class. This avoids silently
reducing the number of learners after unsuccessful random retries.

The minimum subset size of two is an explicit finite-data rule. Sampling
parameters must still specify a feasible subset of the available rows.

Final consensus selection follows the ranking exactly. A retained count below
two, or a selected set containing only one class, raises an error. No minority
rows are inserted afterward and no full-data or first-base-model fallback is
used. A different retention fraction is a new parameter choice to evaluate
through cross-validation.

## Shared parameters and regularization

Each base model and the final model receives the same kernel, resolved gamma,
`lambd`, `q`, stopping controls, and acceleration choice. Fitted coefficients
and intercepts are learned separately. Every DWD fit explicitly chooses the
optimized implementation of the pinned dependency. Its intercept remains
unregularized.
All DWD models use zero initialization. `random_state` controls subset
sampling; no random coefficient initialization is performed by the learners.

For ordinary DWD, `q=1`, the package objective is

```text
mean(V_1(y * (K @ alpha + b))) + lambd * (alpha.T @ K @ alpha)
```

with `V_1(u) = 1-u` for `u <= 1/2`, and `V_1(u) = 1/(4*u)` otherwise.
Equation 3.11 of the dissertation writes the penalty as
`lambda_dissertation / 2 * ||h||_H**2`. Consequently,
`lambda_dissertation = 2 * lambd` for the same loss and kernel. Both objectives
use the mean loss, so changing a subset's size does not introduce another
sample-count conversion. Search-grid values in this package always mean
literal package `lambd` values.

The theoretical augmented-kernel treatment of the intercept is separate from
this practical free-intercept objective. The implementation does not add a
constant kernel term or an intercept penalty to reproduce that proxy.

The default `q=1` matches the dissertation's stated loss. Generalized positive
`q` values, when requested, describe an extension of that method.

## Kernel and gamma

For the RBF kernel, the convention is

```text
K(x, z) = exp(-gamma * ||x-z||**2)
```

Thus gamma multiplies squared distance; it is not a standard-deviation
bandwidth. If a bandwidth `sigma` is defined by
`exp(-||x-z||**2 / (2*sigma**2))`, then `gamma = 1/(2*sigma**2)`.

`gamma='scale'` is resolved from the complete current training input and reused
for all subset models and the final refit. Numeric gamma values are also
shared. With external preprocessing in a cross-validation pipeline, this is
the transformed training fold, not the held-out fold. The shared value is
available as `gamma_` after fitting.
For kernels that do not use top-level gamma, including linear and callable
kernels, `gamma_` is `None`. A callable receives its explicit `kernel_kws`
only; the estimator does not inject top-level gamma into that callable.

For built-in kernels, gamma belongs in the estimator's `gamma` parameter and
must not also appear in `kernel_kws`. Other supported kernel keywords belong
in `kernel_kws`. A precomputed training kernel is unsupported because subset
fitting and subsequent query evaluation require explicit row correspondence.

## Numerical controls

`stopping='objective'` uses the DWD dependency's successive-objective rule with
`obj_tol=1e-5` and at most `max_iter=100` updates by default. `stopping='fixed'`
and `stopping='optimality'` are explicit alternatives; `tol` controls the latter.
There is no automatic validation split or validation-based early stopping.

`acceleration=None` retains ordinary MM updates. Changing stopping or
acceleration can change raw decision values and therefore the consensus set,
even when base classification signs agree. Evaluate such changes through the
complete estimator. `converged_` reports the dependency's checked numerical
criterion separately from iteration counts and stopping policy.

Prediction is evaluated in batches to limit query-kernel allocations.
Training still uses dense kernel matrices. At `M=5`, an uncached successful fit
consists of six DWD fits; the wrapper trains them sequentially. Parallelize
outer cross-validation jobs within an explicit memory budget. A single largest
training kernel needs at least
`8 * max(floor(theta*n), floor(sv_selection_ratio*n), 2)**2` bytes in float64;
factorization, recovery, feature storage and other workers require additional
memory. This formula is a lower bound, not an estimate of peak RAM.

## Optional base-stage cache and retained models

`cache_dir=None` disables caching by default. A supplied local directory can
reuse the same base stage across candidates with different retention fractions.
The cache identifies the actual current training fold after preprocessing,
including feature values, labels and row order, sampled subsets, kernel and
solver parameters, and numerical runtime settings. Sampling and DWD parameters
remain shared across the base and final learners. The retention fraction is
excluded from the base-stage key; selection and the final DWD fit still run for
every candidate. In particular, retention equal to one preserves the consensus
ordering of the final training rows.

An operating-system lock serializes work for a given key, and completed entries
are committed atomically. Different keys can be computed independently.
`random_state=None`, callable kernels or callable kernel keywords, and custom
estimator subclasses bypass caching because their behavior is not identified
safely for this reuse. `cache_hit_`, `cache_key_`, and `cache_status_` report the
outcome. Cached state is an optimization of external CV execution; it does not
introduce internal parameter tuning.

`store_base_estimators=True` retains the fitted base learners by default. With
False, `estimators_` is None, while the final estimator, consensus scores,
subset/selection indices, update counts and convergence flags remain available.
Dropping the base models after scoring reduces retained feature copies and
serialized model size. It does not alter the selected rows or the final model.
The tuning example uses this lean option and a shared cache within its separate
run output directory.

Cache entries use trusted local pickle-based serialization. Keep the directory
private; loading a modified or untrusted entry is unsafe. Lean cache entries do
not directly serialize the source feature matrix, but their scores and index
diagnostics may still be sensitive. Entries retaining models include those
models' training feature rows. Constructor parameter changes take effect on a
subsequent fit; they do not modify already fitted child models.

## Validation and proposed search space

Use a pipeline containing one scaler followed by `EnsembleKernelDWD`. Each
cross-validation training fold fits its own scaler, constructs its own base
subsets, resolves gamma, selects retained rows, and trains its own final model.
Validation rows are used only for scoring. Scaling each subset independently
would define a different metric across the base learners.

`mnist_param_grid(prefix='')` returns the Cartesian product of nine `lambd`
values, nine gamma values, two sampling fractions, and five retention
fractions documented in the README. It leaves `M=5`, `q=1`, the optimized
implementation, and the selected stopping policy fixed. The prefix is the
literal sklearn parameter prefix, such as `'ensemble__'` inside a pipeline.
It does not tune base models independently or launch a hidden search.

Use five-fold stratified cross-validation with accuracy scoring for the
provided MNIST example. Keep a final test set separate from parameter
selection. The example wraps the whole pipeline for candidate evaluation. Its
`--error-score nan` default handles only `IneligibleSelectionError`, marking an
expected invalid consensus selection ineligible and reporting it. Unexpected
exceptions are recorded with their context and propagate to stop the search;
they are not treated as ordinary parameter choices.
Only candidates with finite scores on every fold and a finite mean score are
eligible for ranking and final refitting. A failure is not converted to zero
accuracy and the estimator does not fall back to another classifier. Select
`--error-score raise` to stop on ineligible selections as well.

The optional `--refine` stage takes up to three highest-scoring, fully valid
coarse candidates. Around each one it tries lambda factors `[0.25, 1, 4]`
and gamma factors `[10**-0.5, 1, 10**0.5]`, retaining that candidate's sampling
and retention fractions. The union removes duplicate parameter combinations
and may extend outside the coarse grid. This adds at most 27 candidates.
Both stages reuse the same five training folds and fit preprocessing within
each fold. Both GridSearchCV stages disable automatic refitting. Each completed
stage is checkpointed with its CV results, exact fold indices, data fingerprint
and selected parameters before fitting the selected bare pipeline on all
training rows. This preserves the model-selection work if the final fit fails.
The test set is scored only after a successful final fit. This is an external
search procedure, not another operation inside `EnsembleKernelDWD.fit`.

The run directory records `run.json` with input fingerprints, exact folds,
settings and the grid; `coarse-cv.json` and optional `local-cv.json` contain
completed stage results. `selection.json` is committed before the explicit
final fit. `refit.json` records its success or failure, `status.json` records
the run state, and individual `failures/failure-*.json` files preserve error
context. `completed.json` includes held-out accuracy only after success.
An unexpected exception during CV can prevent collection of that stage's full
results, while the run identity and failure record survive. Retrying the
selected final fit from saved parameters requires verifying the data hashes,
folds and settings. The example does not automatically resume or select a
different candidate after a final-fit failure.

The command-line example accepts any positive `--jobs` count. Choose it for
the available RAM and the dense kernel size; workers are not selected from
a hardware-specific fixed list. Native BLAS work within each outer worker
uses one thread. `--run` is required to start either search, and a preview
reports the coarse and optional refinement budgets without loading data.
`--output-dir` must name a fresh directory; the default creates a unique
`results/mnist-*` directory. Its `base-cache` subdirectory is shared by the
run's candidates. Set `--cache-dir` to use another trusted cache directory or
`--no-cache` to disable caching. The example stores completed result records,
not an automatically serialized final predictor.

The dissertation uses the symbol theta for both subset sampling and, in its
OVO experiment, a retained fraction of 0.3. This API deliberately exposes
`theta` and `sv_selection_ratio` separately. The numerical learner count and
the broader search grid are software settings; they are not claimed to be
universally optimal or mandated by the dissertation.
