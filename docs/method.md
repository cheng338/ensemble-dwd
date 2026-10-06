# Binary ensemble kernel DWD

The estimator implements the four stages in Section 5.1.1 of Chang Cheng's
dissertation, *Overcoming the Curses of Dimensionality and Kernelization:
Scalable DWD and Supervised NMF* (Purdue University, 2026).
`EnsembleKernelDWD` supports binary classification. `OvOClassifier` uses it
for each class pair, as described in [the multiclass method](multiclass.md).

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
classes must appear in every sampled block. `IneligibleSubsetError` is raised
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

The explicit option `residual_check_order='adaptive'` changes the order of
existing residual checks in unaccelerated optimized Cholesky MM. It requires
`backend='auto'` or `'cholesky'` and `acceleration=None`; the default remains
`'refinement_first'`. Automatic spectral recovery uses the original check
order, and anchor recovery retains its original checks. The option does not
loosen tolerances or recovery limits, but a different accepted coefficient
state can change finite-iteration scores, selected rows and predictions.
Validate it through the complete ensemble before relying on its results.

Prediction is evaluated in batches to limit query-kernel allocations.
Training still uses dense kernel matrices. At `M=5`, an uncached successful fit
consists of six DWD fits; the wrapper trains them sequentially. Parallelize
outer cross-validation jobs within an explicit memory budget. A single largest
training kernel needs at least
`8 * max(floor(theta*n), floor(sv_selection_ratio*n), 2)**2` bytes in float64;
factorization, recovery, feature storage and other workers require additional
memory. This formula is a lower bound, not an estimate of peak RAM.

### Optional direct RBF computation

`EnsembleKernelDWD(rbf_computation='direct')` selects the DWD dependency's
direct coordinate-difference calculation for every base model and the final
refit. The default is `'standard'`, which retains the existing norm/dot
construction and its documented extreme-input limitations. This unreleased
option requires the updated local DWD and ensemble-DWD checkouts; the pinned
published DWD 1.3.10 wheel does not include it.

Direct computation supports the named `kernel='rbf'` with the corrected Schur
solver used by this estimator. Set gamma through the usual top-level `gamma`
parameter and leave `kernel_kws` empty or `None`; extra kernel keywords are
unsupported for this option. The resolved gamma is shared by every learner.
Other kernels retain their existing behavior with the standard policy; asking
for direct computation with an unsupported configuration raises.

The option changes finite-precision evaluation of
`exp(-gamma * ||x-z||**2)`, with range-safe handling of exceptional distances.
It addresses the demonstrated RBF self/equal-copy discrepancy and cancellation
between nearby large vectors. It leaves the DWD objective, free intercept,
solver acceptance checks, disjoint sampling, averaging of raw base scores,
ranking by absolute consensus score and fresh final refit unchanged. More
accurate rounded kernel entries can nevertheless change close score rankings,
selected rows and predictions. Sequential distance sums are not universally
correctly rounded, and ordinary score/intercept rounding remains possible
with the default affine computation policy.

This option improves numerical accuracy at a cost. In ten paired MNIST 2-vs-3
prediction measurements per estimator with one native thread, direct
computation took about 6 times as long for the complete ensemble prediction
pipeline and 18 times as long for standalone full-kernel DWD. Four pairs had
background-CPU flags; the unflagged results retained the same conclusion.
These ratios are specific to those workloads. Six saved MNIST fits retained
test labels and accuracies and the three ensembles retained selected rows;
sensitive synthetic cases did change selections and labels.

The option is included in cache and checkpoint identity. Old fitted models
retain their original kernel policy; changing the constructor parameter
affects the next fit. Configure `ensemble__rbf_computation='direct'` in the
binary pipeline shown in the README, or
`ovo__estimator__rbf_computation='direct'` in its multiclass pipeline. The
existing parameter grids and command-line defaults remain unchanged. This
option applies to ensemble DWD, not `EnsembleKernelSVM`.

### Optional joint affine scoring

`EnsembleKernelDWD(affine_computation='joint')` forwards the same numerical
policy to every base learner and the fresh final DWD fit, including each
pair inside `OvOClassifier`. The default is `'standard'`. This unreleased
option requires the updated local DWD and ensemble-DWD sources; the pinned
published DWD 1.3.10 wheel does not accept it.

The policy evaluates the unchanged decision expression `K @ alpha + b`,
including the intercept in the rounding check and, when needed, in joint
compensated summation. Exceptional overflow or unresolved near-zero cases
use exact arithmetic on the represented floating-point inputs before
rounding the result to float64. It addresses cases where rounding the dot
product before adding the intercept loses a small residual. It does not
change the kernel or modify coefficients while scoring. The free-intercept
objective, solver acceptance tests and strictly positive decision rule are
unchanged; a represented zero
still selects the first class. The ordinary accepted path is not universally
correctly rounded or bitwise invariant across numerical backends and batches.

Disjoint sampling, averaging raw base scores, stable ranking by absolute
consensus score and fresh final fitting are unchanged. Corrected scores can
nevertheless change close rankings, retained rows and predictions. The
consensus mean itself uses the existing arithmetic. Joint scoring can be
combined with `rbf_computation='direct'`: the two options address different
rounding stages. The ensemble does not provide per-learner validation
stopping, so that separate DWD capability remains unsupported here.

This is a precision option with conditional cost. Ten paired saved MNIST
2-vs-3 ensemble prediction measurements, with one native thread, took about
9% longer using joint scoring. Two pairs had background-CPU flags; the
eight unflagged pairs showed about the same overhead. Standalone full-kernel
prediction in the corresponding test was approximately unchanged. Constructed
exact-tie inputs were substantially more expensive because many rows entered
exact arithmetic. These measurements do not establish a universal cost or
training-speed improvement.

Fresh ensemble refits on three saved MNIST cases, each with standard and
direct RBF, retained all selection rankings, selected rows, test labels and
accuracies. Two cases had small score differences. A separate two-row fitted
linear example changed an incorrectly rounded zero into a positive residual
and corrected its predicted label. These checks show why the option is useful
for sensitive cases without establishing a general accuracy improvement.

The fitted policy is recorded as `affine_computation_`, including in the
fitted child models. Changing the constructor parameter affects the next
fit; it does not alter an existing predictor. Old serialized models retain
standard scoring. Base-cache and example CV identities distinguish the two
policies. In the README's binary pipeline use
`ensemble__affine_computation='joint'`, or
`ovo__estimator__affine_computation='joint'` in its multiclass pipeline.
The command-line defaults and search grids remain unchanged. This option
applies to ensemble DWD, not `EnsembleKernelSVM`.

### Optional extended exact recovery

`EnsembleKernelDWD(exact_recovery='extended')` gives every base learner and
the fresh final model the DWD dependency's extended exact-recovery budget.
`OvOClassifier` clones the setting into every class pair. The default remains
`'standard'`. The option requires the updated local DWD and ensemble-DWD
sources; the pinned published DWD 1.3.10 wheel does not accept it.

The standard limits are exact rank 16 and 4096-bit rational arithmetic.
Extended recovery raises those limits to 32 and 8192 bits for both exact
kernel-factor preparation and the associated MM recovery. Both profiles
retain the 65,536-entry and one-million-operation limits. The entry limit
restricts a square kernel to at most 256 rows; the other bounds may still
refuse a smaller matrix. Each later MM recovery step has a separate operation
budget. These bounds do not guarantee whole-fit wall time or peak memory.

The option changes capacity, not the mathematical method. It retains the
original represented kernel, exact PSD/rank certificate, MM equations, free
intercept, DWD objective and acceptance tolerances. It adds no jitter or rank
truncation and does not change the sampler, consensus averaging, row-selection
rule or final-refit procedure. Healthy fits continue through their existing
route. A fit that becomes possible can naturally supply scores and selections
where the standard profile raised an error. The budget governs MM recovery;
`backend='lbfgs'` does not use it.

In the isolated trial, three of nine saved historical difficult RBF fits
completed five fixed updates with the extended profile. The same three also
completed 100 updates with objective stopping, but reached the iteration cap
with `converged=False` and no reached stopping criterion. This demonstrates
additional recovery coverage, not convergence or improved prediction accuracy.
The other six fits still exhausted the rank-32 budget. Further independent
exact checks found that their two stored kernel matrices were indefinite:
larger resources cannot supply an exact PSD certificate for those unchanged
matrices. That diagnosis concerns rounded stored entries, not the mathematical
Gaussian kernel. The selected profile's actual refusal remains the rank limit;
it does not reach those later negative witnesses itself.

Extended recovery can cost more and is not intended as a speed optimization
or a solution for large full-MNIST kernels. It is independent of the direct-RBF
and joint-affine options. Different requested profiles have different base-cache
and example CV identities. `exact_recovery_` records fitted provenance on the
ensemble and its learners; compact base diagnostics retain it when base models
are discarded. Changing the constructor setting affects the next fit and does
not modify an existing predictor. Old serialized models retain the standard
constructor setting.

In the README's pipelines use `ensemble__exact_recovery='extended'` or
`ovo__estimator__exact_recovery='extended'`. Search grids and command-line
defaults remain unchanged. This option applies to ensemble DWD, not
`EnsembleKernelSVM`.

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
The tuning example discards these base models and uses a shared cache within
its separate run output directory.

Cache entries use trusted local pickle-based serialization. Keep the directory
private; loading a modified or untrusted entry is unsafe. Entries without
retained base models do not directly serialize the source feature matrix, but
their scores and index diagnostics may still be sensitive. Entries retaining
models include those models' training feature rows. Constructor parameter changes take effect on a
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
and no automatic refit. Only parameter combinations not yet evaluated are
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
axis, each within a factor of two. Two completed local-refinement rounds
must then show no improvement or change of winning region. Repairs at
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
invalid/missing neighbors as plateau evidence. This stopping rule is based
on the evaluated range; it does not prove global optimality or exclude
improvement farther away.

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

No data are loaded without `--run`. The preview reports initial costs and
total candidate budget separately. The default output is a unique
`results/mnist-*` directory. `--cache-dir` selects a trusted base cache and
`--no-cache` disables that cache, not completed-fold checkpoints. Choose
`--jobs` for the available RAM; it defaults to 1. Dispatched work is bounded by that count and
each outer worker uses one native thread.

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
Once that receipt is committed, a verified resume restores saved summaries
without fitting or scoring. Its result has `model=None`, `resumed_completed=True`.
`completed.json` is the public mirror of that saved outcome.

The dissertation uses the symbol theta for both subset sampling and, in its
OVO experiment, a retained fraction of 0.3. This API deliberately exposes
`theta` and `sv_selection_ratio` separately. The numerical learner count and
the broader search grid are software settings; they are not claimed to be
universally optimal or mandated by the dissertation.

### Native threads by phase

All three tuning commands accept optional `--final-native-threads N`, a positive
integer with default 1. The example APIs expose the same setting as
`run_adaptive_search(..., final_native_threads=N)` and
`run_search_workflow(..., final_native_threads=N)`. It applies to the complete
selected full-training pipeline fit: preprocessing, ensemble base fits,
consensus scoring and the fresh final learner. For sequential OvO it applies
to every class pair. CV, ratio refinement, numerical confirmation and held-out
prediction retain their existing one-native-thread scopes.

`--jobs` controls concurrent CV candidate/fold tasks or equivalent-fold groups.
The final-thread setting instead limits each detected native BLAS/OpenMP pool;
it is not a total machine CPU limit, a worker count or physical-core affinity.
There is no automatic CPU allocation policy. A nondefault final budget requires
every exposed nested estimator `n_jobs` setting to be explicitly 1. That check
does not control workers hidden inside custom estimators or callable kernels.
Native limits act across the Python process, so do not run unrelated numerical
work in another Python thread while these workflow scopes are active.

The objective and ensemble procedure remain unchanged, but native reduction
rounding can change scores, stopping decisions, selected rows or labels. More
threads do not guarantee a faster fit. The SVM comparator accepts the shared
workflow option; additional BLAS threads do not parallelize its LIBSVM solver.

The requested final budget and helper source are part of workflow identity.
Nondefault final-fit receipts record observed native runtime before and after
fitting and check restoration to the CV runtime, including on failure or
interruption. The default adds no separate final runtime observation or
inventory scan: its receipt labels the existing workflow scope and references
frozen CV provenance where available. Base-cache keys already include observed
thread counts, keeping entries from different budgets separate.

Changing the final budget rejects resume under the current strict workflow
identity. Changing the helper source also invalidates reuse; use a fresh run
directory after this source change. Interrupted final fitting restarts with
the same budget, while verified completed replay performs no new fit or test
prediction. These controls do not alter estimator defaults or search grids.

### Work and timing summaries

All three adaptive tuning commands write `work-summary.json` automatically;
there is no additional option or model evaluation. The summary describes the
current invocation, with separate original-search stages, optional numerical
confirmation, final refitting and held-out scoring. Numerical confirmation
itself remains disabled unless requested. Existing timing fields keep their
names and meanings.

For each completed CV stage, fold-and-seed evaluations fall into three
separate groups: newly executed evaluations (including recorded ineligible
results), newly created equivalent-ratio receipts, and previously existing
receipts read from checkpoints. An equivalent-ratio receipt read on a later
invocation belongs to checkpoint replay. Reuse preserves the candidate's
score; it does not show that model training became faster.

CV base-cache outcomes count only observations from newly executed
evaluations. Multiclass records count the observed binary pairs. The summary
distinguishes hits, misses, disabled or bypassed caching, and non-hits with
an unknown reason. A false cache-hit flag alone does not establish a miss.
Missing cache metadata is unknown; replayed diagnostics do not count as new
cache activity. These counts describe the base stage and do not count the
separate final model fitted on the selected rows. Full-training refitting
reports its own base-cache observations separately.

Current worker fit and score times are sums of measured worker intervals.
They are neither CPU time nor elapsed parallel-run time. Timings stored in
replayed receipts are reported separately; they can describe work from an
earlier invocation. Alias references to source fit and score timings are not
added to totals. Do not add overlapping worker, stage and whole-invocation
timers, or interpret reuse as a measured training speedup. Equivalent-ratio
reuse timing retains the existing convention of excluding atomic receipt
publication. Invocation elapsed time starts at entry to the adaptive workflow
and ends at the named report snapshot. It includes checkpoint verification
and earlier report writes, but excludes command-line data loading and the
summary's own publication and final context cleanup. Replay verification is
not a new fit.

Confirmation has its own work summary. Copies of original-search scores kept
for comparison do not add original work a second time. Final fitting and
held-out scoring have separate elapsed intervals rather than being folded
into CV costs.

The authoritative completion artifact preserves `completion_work_summary`,
a snapshot taken before that artifact is published. The returned result and
`work-summary.json` describe the latest invocation, using a later snapshot
after the ordinary result files have been written. A verified replay of an
already completed result therefore reports zero new CV, final fitting and
held-out scoring, while still spending time verifying and restoring records.
Failure to publish this auxiliary report does not replace a fit outcome or
mask an exception from the workflow.

An interrupted stage may have work that was not measured completely. The
summary marks that limitation and treats its completed-stage measurements
as a lower bound, rather than reconstructing exact cumulative work or time
across crashes and downtime. A missing stage summary also leaves totals
incomplete; it is not evidence of zero work. The report changes no solver
mathematics, sampling, candidate selection or numerical defaults, and does
not establish a speed improvement or zero reporting overhead.

### Selection-ratio refinement

The initial presets and the default adaptive search keep their five retention
fractions. Enable `--refine-ratio` in any of the three tuning commands to
search between nearby fractions after the coefficient search meets its
existing acceptance requirements. Sampling fraction `theta`, learner count
`M`, the scoring and row-selection rules, and each estimator's numerical
controls are unchanged.

The extra search targets complete parameter sets among the best valid results,
using the existing `top_k` limit (three by default). It proposes local joint
combinations of arithmetic ratio midpoints and logarithmic coefficient
midpoints. A new ratio is not crossed with every earlier coefficient choice.
Previously evaluated candidates remain eligible, and only new complete
parameter sets use the remaining candidate budget. The initial grid is not
expanded by this option. If a parent lacks a conditional coefficient neighbor,
the search first tests a nearby coordinate already explored in the same fixed
parameter context. If none exists on that side, it can extend that coefficient
by at most a factor of 10, using the earlier search's outward-step rule. These
local support checks count toward the ratio rounds and total candidate budget;
`--max-expansions` controls only the preceding coefficient-search stage.

`--ratio-bounds LOW HIGH` declares the permitted range. It must satisfy
`0 < LOW < HIGH <= 1` and contain every initial ratio. Without explicit
bounds, the minimum and maximum initial ratios define the range, `[0.1, 0.3]`
for the MNIST presets. At least two distinct initial ratios are required.
For this optional policy, initial ratios and bounds must be ordinary Python
`int` or `float` values, excluding booleans. The resolved bounds are reported
and saved. If a parent has no ratio neighbor on a required side, the declared
endpoint is tested before narrowing that gap.
Refinement stays inside the bounds; it does not claim to find an optimum
outside that range.

`--max-ratio-rounds` defaults to three complete rounds. `--ratio-tol` defaults
to `0.01`, an absolute ratio difference of one percentage point. Acceptance
requires two completed stable rounds and valid conditional neighbors around
the selected candidate: coefficient neighbors within the existing factor-two
limit and ratio neighbors within the configured absolute tolerance. A
candidate exactly at a declared ratio bound needs support only toward the
interior on that axis; the review records this range restriction. Stable
rounds must preserve the winning complete parameter set and score, with valid
results for every proposed candidate. A changed winner or score restarts the
stability count. Missing or invalid support does not count as evidence of a
bracket.

All added parameter sets count toward the same `--max-candidates` total as
the earlier coefficient search. The controller holds a proposed stage when
it cannot fit the remaining budget; it does not evaluate part of that stage.
Unresolved support or exhausted round/candidate limits produce `needs_review`,
with no final fit or held-out score. This is a bounded search procedure, not
a guarantee of higher accuracy or of a global optimum. Finer search does not
increase a learner's iteration budget or establish solver convergence.

`--ratio-bounds`, `--max-ratio-rounds` and `--ratio-tol` require
`--refine-ratio`. Ratio refinement cannot be combined with `--no-refine` or
multiclass `--smoke-grid`. The Python helper accepts
`run_adaptive_search(..., ratio_refinement=RatioRefinementConfig(...))`;
`RatioRefinementConfig` is available from `examples/adaptive_search.py`, with
`bounds=None`, `max_rounds=3`, `ratio_tol=0.01`, and `stable_rounds=2` by
default. Omit `ratio_refinement` to preserve the original workflow. The saved
policy includes these settings, so changing them requires a fresh run.

Every stage uses the same folds and prescribed CV seeds. Each candidate
keeps its exact ratio; the estimator's existing `int(n * ratio)` retained-row
count is unchanged. Optional equivalent-fold reuse still requires the same
seed and all existing identity and selected-row checks. It preserves each
candidate label rather than rounding ratios together. The selected candidate
receives a separate fresh fit on all training rows with the declared final
seed. Completed fold-seed records can be reused on a verified resume;
unfinished evaluations restart.

### Numerical-budget confirmation

The binary and multiclass DWD tuning commands offer a separate diagnostic
through `--confirm-max-iter N`. It is disabled by default. Supply a positive
integer greater than the original DWD iteration cap, 100 in the commands.
`--confirm-top-k K` limits the shortlist to `K` candidates, with a default of
three and a minimum of two; it requires `--confirm-max-iter`. The SVM tuning
command does not expose this DWD option.

Confirmation starts only after the coefficient search and any requested
ratio refinement meet their acceptance rules. It freezes the original winner
and the highest-ranked eligible competitors, up to the shortlist limit, from
the completed search. At least two eligible candidates are required. Every
shortlisted candidate is then evaluated at the declared larger cap on the
same training folds and prescribed sampling seeds. Preprocessing remains
fold-local. The held-out test set plays no role in this diagnostic.

Only the iteration cap changes. The objective, stopping mode, tolerances,
backend, subset sampler, row-selection rule and predictions are computed by
the same estimator. Allowing more updates can change fitted scores and the
selected training rows, so mathematical equality between the two budgets is
not assumed. In particular, an objective-tolerance stop before the original
cap may leave no additional updates to run. The report retains learner
stopping diagnostics so this case is distinguished from use of the larger
budget. Stable CV scores do not establish solver convergence or optimality.

The original and confirmation scores remain separate. Confirmation compares
the relative ranking and exact score ties among the frozen candidates, matching
the original candidate-ranking rule. The search's boundary tie tolerance does
not merge ranking ties. An altered ranking or exact tie requires review before
any final fit or held-out scoring. An invalid confirmation result or fewer than two eligible
candidates also produces `needs_review`. Unexpected fit errors continue to
stop the run. There is no automatic replacement of the original winner by a
confirmation winner, and this bounded shortlist does not repeat the original
search or certify its parameter brackets at the larger budget.

When the ranking is unchanged and every confirmation result is valid, final
fitting uses the original selected parameter set, original solver cap and
prescribed final seed. The larger cap is confined to the diagnostic. Both
the default search and ordinary estimator fits remain unchanged when the
option is omitted.

The shortlist has a separate, explicit cost. With `K` shortlisted candidates,
`F` folds and `S` CV seeds, confirmation requires `K * F * S` complete ensemble
fits before reuse. Each binary fit has `M` base learners and one final learner;
multiclass confirmation repeats that work for every class pair. The original
`--max-candidates` limit still bounds the search's distinct parameter sets;
these declared confirmation repetitions are additional work. No runtime or
accuracy improvement is implied.

Confirmation settings are part of the saved run identity and must remain
unchanged on resume. `numerical-confirmation.json` preserves the shortlist,
original and confirmation scores, ranking changes, timing and stopping
diagnostics, including multiclass pair details. A separate checkpoint context
under `numerical-confirmation/` stores its completed fold-seed evaluations. Verified completed work can be reused after interruption;
unfinished fits restart. Original-budget records cannot substitute for
larger-budget evaluations. Optional equivalent-ratio reuse still requires all
existing identity and row-equivalence checks within the confirmation context.

The Python example API accepts
`run_adaptive_search(..., numerical_confirmation=NumericalConfirmationConfig(max_iter=300, top_k=3))`.
`NumericalConfirmationConfig` is available from `examples/adaptive_search.py`.
Its `max_iter` is required; `top_k` defaults to three. Omit
`numerical_confirmation` to retain the workflow without this diagnostic.

### Multiple CV seeds

The tuning examples use one ensemble sampling seed by default. To compare
candidates over several prescribed draws, pass `--cv-seeds 42 43 44` to any
of the three commands. The folds stay fixed, and every candidate uses the
same ordered seed list and numerical budgets. Seeds must be distinct integers
from 0 through 4294967295. Average the seed accuracies
within each fold, then average those fold scores for candidate selection.
Seeds are repetitions of a candidate, not hyperparameters to select. A
candidate is eligible only when every required seed and fold has a finite
score. Unexpected failures still stop the run.

`--max-candidates` continues to count parameter sets. With `S` seeds and `F`
folds, each candidate requires `S * F` ensemble fits before reuse. Seeds
change the ensemble's sampling, not the folds or the number of base learners.
For multiclass DWD, each seed evaluates a complete OvO model, and the existing
pairwise sampling and voting rules remain unchanged. Repeating sampling on
the same folds does not create independent validation folds; the reported
fold variation is not a standard error based on `S * F` independent observations.

Each completed fold and seed has its own checkpoint, score and diagnostics.
Interrupted work resumes at that level; a partial collection cannot enter
candidate ranking. Equivalent-ratio reuse, when enabled, requires the same
seed as well as the other existing identity checks. The ordered CV seed list
and final-refit seed are part of the saved policy. Changing either requires
a fresh output directory.

After selection, the workflow fits once on all training rows using a
separately prescribed seed. `--refit-seed` requires `--cv-seeds`; when omitted,
the final fit retains the estimator's configured seed, 42 in the commands.
It does not choose the best-performing CV seed or combine the fitted models.
The effective final seed is recorded before fitting.

The Python example API accepts
`run_adaptive_search(..., cv_seeds=[42, 43, 44], refit_seed=42)`. It identifies
the seed parameter from the configured estimator prefix. Direct use of
`CheckpointedCV` additionally specifies `seed_parameter`, such as
`'ensemble__random_state'` or `'ovo__estimator__random_state'`. Leave
`cv_seeds=None` to retain the existing single-seed workflow. With multiple
seeds, the estimator's configured seed must also be an integer, even when
an explicit final-refit seed is supplied.

For multiple seeds, candidate fold-fit and score times sum the seed work
within each fold, then average across folds. Per-seed records preserve their
individual times, and the stage reports elapsed time separately. These
quantities differ when work runs in parallel or a completed result is reused.

### Equivalent-fold reuse

All three tuning commands accept `--reuse-equivalent-folds`, disabled by
default. The example APIs expose the same option as
`CheckpointedCV(..., reuse_equivalent_folds=True)` and
`run_adaptive_search(..., reuse_equivalent_folds=True)`. Keep the option
unchanged when resuming a run.

Two retention ratios can produce the same integer number of selected rows.
When every other setting and the training fold match, the built-in ensemble
has the same base fits, deterministic ranking and selected rows. The option
reuses one completed fold's score and diagnostics for the other ratio. For
OvO, the retained count must match separately in every original class pair.
Equal accuracies alone never establish equivalence.

Reuse is restricted to supported built-in ensembles with fixed integer seeds
and known deterministic preprocessing. Supported Python parameter types are
required throughout the estimator, including nested kernel and preprocessing
options. NumPy scalars or arrays and custom estimators or transformers do not
gain equivalent-fold reuse; they use ordinary fitting when their exact
checkpoint identity is supported. Custom scalar/container subclasses and
object, structured, or metadata-bearing array parameters are rejected by the
checkpoint helper. Schema 2 distinguishes parameter types, and earlier
checkpoint formats require a new run directory.
Invalid options still fail validation. A failed or ineligible fold does not
supply a reusable result. Existing completed-fold identity and integrity checks
remain required.

Each candidate retains its parameter label, score, rank and place in the
candidate budget. Reuse records identify the original completed source and
distinguish its fitting and scoring times from the work performed for reuse.
`reuse_seconds` includes verification before publishing the record; the atomic
write is included in the enclosing stage's wall time.
Interrupted searches retain committed sources; unfinished work is evaluated
on resume. Full-training fitting remains separate: two ratios with equal CV
fold counts can retain different numbers of rows on the complete training
set, so the selected original ratio is always fitted there.

This option complements the base-stage cache, which already reuses base
models across ratios. It can avoid the remaining duplicate final fold fits
and validation predictions when supplied ratios are sufficiently close.
The default MNIST grids keep five well-spaced retention fractions, so this
option does not imply a speed improvement for their usual training sizes.
