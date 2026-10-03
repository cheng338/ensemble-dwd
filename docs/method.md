# Binary ensemble kernel DWD

The estimator implements the four stages in Section 5.1.1 of Chang Cheng's
dissertation, *Overcoming the Curses of Dimensionality and Kernelization:
Scalable DWD and Supervised NMF* (Purdue University, August 2026, pp. 46–47).
`EnsembleKernelDWD` supports binary classification. `OvOClassifier` composes
it into the multiclass procedure described in [the multiclass method](multiclass.md).

## Fit and prediction

For training rows `(X, y)` of size `n`:

1. Permute the training-row indices uniformly once. Split the first
   `M * max(2, floor(theta * n))` indices into `M` disjoint base subsets of
   equal size. Validate every subset before fitting any learner.
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

Sampling follows the disjoint-subset rule in Mao et al. (2016), Section III-A,
for both ensemble DWD and the SVM comparator. With `N` input rows and
`k = max(2, floor(theta * N))`, the required capacity is `M * k <= N`.
One uniform random permutation supplies all blocks; the remaining rows are
unused by the base learners but remain available for scoring and final selection.
Disjointness applies within one binary fit, including each OvO class pair.
The final selected set is drawn from that fit's complete training set.

The permutation does not use the labels. There is no class conditioning,
stratification, redraw, or replacement of rows after sampling. Both binary
classes must appear in every realized block. `IneligibleSubsetError` is raised
before any learner fit or base-cache lookup when the capacity is insufficient,
a class has fewer than `M` observations, or a sampled block has only one class.
It subclasses `IneligibleSelectionError`, so existing handlers continue to
recognize an ineligible candidate. An external search can report and exclude
it; `--error-score raise` retains the original exception for inspection.

The minimum block size of two is an explicit finite-data rule. The presets
fix the sampling rate at 0.1 (10%) and the examples use `M=5`. Increasing `M` requires checking the capacity on every CV
training fold and every OvO pair. Independent subset draws in older versions
could overlap; their predictions and accuracy results are historical results
for that sampling scheme and do not establish disjoint-sampling performance.

Final consensus selection follows the ranking exactly. A retained count below
two, or a selected set containing only one class, raises an error. No minority
rows are inserted afterward and no full-data or first-base-model fallback is
used. A different retention fraction is a new parameter choice to evaluate
through cross-validation.

## Shared parameters and regularization

Each base model and the final model receives the same kernel, resolved gamma,
`lambd`, `q`, stopping controls, acceleration choice, and backend. Fitted coefficients
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

For MM backends, `acceleration=None` retains ordinary MM updates. Changing stopping or
acceleration can change raw decision values and therefore the consensus set,
even when base classification signs agree. Evaluate such changes through the
complete estimator. `converged_` reports the dependency's checked numerical
criterion separately from iteration counts and stopping policy.

`backend='auto'` retains the optimized DWD dependency's default backend.
Use `backend='spectral'` or `backend='cholesky'` to select an MM representation,
or `backend='lbfgs'` for the dependency's alternative optimizer. The choice is
shared across every base and final fit, including each multiclass pair, and
forms part of the base-cache key. Runtime depends on the data and parameters;
no backend is uniformly faster. Finite-budget scores and selected rows can
differ, so validate an explicit backend through the complete estimator.

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

`mnist_param_grid(prefix='')` returns five lambda values, five gamma values,
one fixed 10% sampling rate and five retention fractions: 125 candidates.
`broad=True` restores the historical nine-by-nine coefficient grid with the
same outer endpoints, for 405 candidates. Exact values are listed in the
[README](../README.md#parameter-search). The prefix is the literal sklearn
parameter prefix, such as `ensemble__`; configure `M=5`, `q=1`, the RBF kernel
and stopping controls on the estimator. The helper only returns fresh lists.

The command-line examples call `run_adaptive_search` from
`examples/adaptive_search.py`. Its external `CheckpointedCV` runner scores
accuracy on fixed stratified training folds, with fold-local preprocessing
and no automatic refit. Only previously unevaluated complete candidates are
proposed. Earlier candidates remain eligible, and verified completed fold
records can be reused across an explicit resume. This is separate from
optional reuse of the classifier's cached base fits.

Best or tied-best lambda/gamma endpoints request outward extensions. When
both axes extend, the stage includes new cross-combinations, with the same
fixed sampling and retention choices. Each outward step is at most one decade
(a factor of 10), or the smaller adjacent logarithmic spacing. Conditional
neighbors must have valid scores at the same values of the other parameters;
invalid or missing neighbors cannot establish an interior optimum. Gaps at
existing coordinates are repaired before additional outward expansion.

Repeated local refinement proposes joint logarithmic midpoints around the
top three distinct lambda/gamma pairs, within the explored bounds, crossed
with all fixed fraction choices. With refinement enabled, the selected
candidate must have a valid lower and upper conditional neighbor in each
axis, each within a factor of two. Two actually completed local-refinement
rounds must then show no improvement or change of winning region. Repairs at
existing coordinates do not earn this credit. Improvement, changed prior
evidence or departure from the established region reopens review. These are
external search rules: the objective, subset sampler and final-refit method
remain unchanged.

`AdaptiveSearchConfig` defaults to `max_expansions=3`, `max_candidates=500`,
`top_k=3`, `plateau_rounds=2`, `improvement_tol=0.0`, `tie_tol=1e-12`,
`refine=True`, `bracket_factor=2.0`, and `stable_refinement_rounds=2`.
The CLI exposes `--max-expansions`, `--max-candidates`, `--no-refine`, and
`--broad-grid`; `--refine` is retained. Disabling refinement skips local
rounds and factor-two tightening, but still requires valid conditional
neighbors. A full proposed stage must fit the remaining budget. Unresolved
expansion, support or refinement limits produce `needs_review` and hold
final fitting and test scoring.

The enabled full-strip plateau rule can waive a distant tied outer edge
only after two completed nonimproving outward levels with valid same-slice
evidence. It never waives the selected candidate's own bracket or uses
invalid/missing neighbors as plateau evidence. This empirical tested-range
stopping rule does not prove global optimality or exclude improvement farther
away.

The default `--error-score nan` excludes only `IneligibleSelectionError`,
including `IneligibleSubsetError`. A candidate needs finite scores on every
fold and a finite mean to be ranked. Unexpected exceptions are recorded and
propagated; an all-invalid search stops. `--error-score raise` also stops on
ineligible candidates. Failed fits receive neither zero accuracy nor a
substitute classifier. Compact diagnostics accompany each completed fold fit
and the final refit. Objective stopping, an iteration cap and an optimality
check are reported separately; default DWD objective tolerance `1e-5` and
`max_iter=100` are unchanged.

`run.json` records data, folds, estimator settings and policy. Stage CV files,
including repeated local and conditional-repair stages, and
`accumulated-cv.json` preserve evidence. `selection.json` and `review.json`
are saved before the explicit final fit; `refit.json`, `status.json` and
`failures/failure-*.json` record later outcomes. An accepted search fits all
training rows, then scores the separate held-out set.

New runs require a fresh `--output-dir`. Explicit `--resume --run` reuses
checksummed completed folds only when data, folds, estimator parameters,
policy and budgets, job count, and numerical runtime match. The installation
must remain immutable; mismatched identity or live ownership blocks reuse.
Unfinished fits restart. Interruptions retain completed evidence and record
an interrupted status when possible. The legacy `run_search_workflow` helper
remains nonresumable and retains its explicit coarse/local procedure. Its
local stage preserves parent parameters and cannot discard a better coarse
candidate.

No fitted predictor or partial optimizer state is serialized. An interrupted
final refit starts again. A crash after held-out prediction but before the
authoritative completion receipt commits can repeat the fit and scoring.
Once committed, a verified completed resume restores saved summaries without
fitting or scoring and returns `model=None`, `resumed_completed=True`.
`completed.json` is the public mirror of that saved outcome.

No data are loaded without `--run`. The preview reports initial costs and
total candidate budget separately. The default output is a unique
`results/mnist-*` directory. `--cache-dir` selects a trusted base cache and
`--no-cache` disables that cache, not completed-fold checkpoints. Choose
`--jobs` for the available RAM; dispatched work is bounded by that count and
each outer worker uses one native thread.

The dissertation uses the symbol theta for both subset sampling and, in its
OVO experiment, a retained fraction of 0.3. This API deliberately exposes
`theta` and `sv_selection_ratio` separately. The numerical learner count and
the broader search grid are software settings; they are not claimed to be
universally optimal or mandated by the dissertation.
