# Multiclass ensemble kernel DWD

`OvOClassifier(EnsembleKernelDWD(...))` provides one-versus-one multiclass
classification using the binary ensemble DWD estimator for each class pair.

## Method

Chapter 5's multiclass experiment in Chang Cheng's dissertation trains one
binary classifier for each unordered class pair. With ten classes there are
45 pairs. Each pair uses its own training rows and the four-stage binary
ensemble: sample `M` subsets, fit their kernel DWD models, select the rows
with smallest absolute mean raw score, and fit one final kernel DWD.
The final classifier of each pair casts one vote. The class with the most
votes is predicted. There is no vote among the base learners at test time.

`OvOClassifier(EnsembleKernelDWD(...))` preserves the binary estimator's
objective, including its unregularized intercept, sampling distribution,
selection rule and stopping controls.
The wrapper does not change or reimplement the DWD solver. For `M=5`, a
successful uncached ten-class fit comprises 45 ensembles and 270 DWD fits.

For each sorted pair `(a, b)`, encode `a` as `+1` and `b` as `-1`. DWD predicts
its negative encoded class at an exactly zero margin, so an exact zero votes
for `b`. Equal total win counts select the smallest sorted original class label.
This makes the dissertation's maximum-wins rule explicit. scikit-learn's standard
`OneVsOneClassifier` uses confidence to break vote ties and therefore has
different tie behavior.

## Shared parameters and preprocessing

One external CV candidate supplies shared binary parameters to
all class pairs. It is scored on the complete multiclass prediction. Individual
pairs do not run separate searches. Within each pair, all `M` base models
and the final refit share their DWD parameters. The parameters `theta` and
`sv_selection_ratio` denote the base sampling fraction and final retention
fraction respectively; both fractions are applied to that pair's row count.
For each pair, one permutation supplies `M` disjoint blocks of size
`k = max(2, floor(theta * N_pair))`, requiring `M * k <= N_pair`.
Every block must contain both pair labels; the sampler does not redraw an
ineligible allocation. Disjointness is enforced within each pair. The same
observation can participate in other class pairs and in the final selected set.

If a binary ensemble fit fails, bounded stage and learner-progress details
survive OvO wrapping and example checkpoint reporting. The failure record's
class counts use the pair's encoded labels, while the existing pair message
or note identifies the original classes. Restored bases are reported separately
from current work; see [failure evidence](method.md#ensemble-failure-evidence).

Put a `StandardScaler` before the OVO estimator inside a sklearn `Pipeline`.
Each CV training fold then fits one scaler on that fold's multiclass rows.
All pairs within the fold share that feature transformation. Validation and
test rows do not participate in fitting the scaler or choosing subsets.

Use a numeric `gamma` for a common RBF coefficient across pairs. With
`gamma='scale'`, the binary estimator resolves it on each pair's transformed
rows, then shares it within that pair. The configured
binary random state is cloned for each pair without using global randomness.
As in the binary API, `lambda_dissertation = 2 * lambd` under the practical
mean-loss objective with the dissertation's one-half penalty convention.

The optional RBF calculation is also shared through cloning:
`OvOClassifier(EnsembleKernelDWD(rbf_computation='direct'))`. In the README's
pipeline its parameter name is `ovo__estimator__rbf_computation`. The default
remains `'standard'`; direct computation requires the paired DWD 1.3.14
dependency, can change rounded scores and selected rows, and can cost more.
See [supported settings and costs](method.md#optional-direct-rbf-computation).

The default grid is `mnist_param_grid(prefix='ovo__estimator__')`, with
125 candidates. Passing `broad=True` restores the historical 405 candidates.
The four-candidate `--smoke-grid` remains an execution and integration check,
not a claim of optimal parameters.
The theoretical augmented-norm treatment in the dissertation remains distinct
from its practical unregularized-intercept implementation; this integration
does not supply a new justification for applying those bounds.

For an optional SVM comparison, the same wrapper accepts
`EnsembleKernelSVM`; use `mnist_svm_param_grid` with the
`ovo__estimator__` prefix. Its distinct method, attribution and usage example
are documented in [the SVM comparator guide](ensemble-svm.md#multiclass-use).

## Validation behavior

Preprocessing is fitted within each CV training fold. Each class pair trains
exactly `M` base learners, including when `M=1`. Selected sets must be large
enough and contain both classes; an ineligible pair fails the complete candidate.
The wrapper preserves original labels, clears fitted state after a failed fit,
and accumulates votes in batches. The supplied tuning example preserves separate
training and test arrays, limits dispatched work, and saves completed CV selection
before one explicit final refit. Compare historical dissertation results only
after matching data, parameter search, selection rules, and stopping settings.

## Adaptive command-line search

`examples/tune_mnist_multiclass.py` uses the same external adaptive controller
as the binary examples, with prefix `ovo__estimator__` and regularization
`lambd`. One complete candidate is shared across all 45 pairs. Every stage
uses the same stratified folds and evaluates only new complete candidates.
The defaults are five folds, a stratified 6,000-row training subset and
all supplied test rows.

Normal search starts at 125 candidates, expands best/tied-best lambda/gamma
boundaries with all cross-combinations, and repeatedly refines joint
logarithmic midpoints around the top three distinct coefficient pairs.
Each outward step is at most a factor of 10 or the smaller adjacent
logarithmic spacing. Sampling stays at 10% and M at five. Retention stays on
its five choices unless optional ratio refinement is enabled. Defaults remain
three expansion rounds and 500 unique candidates total, zero improvement
tolerance and tie tolerance `1e-12`.
With refinement enabled, the selected candidate needs valid bilateral
conditional neighbors within a factor of two and two completed stable local
rounds. Existing-coordinate repairs do not earn stable-round credit.

The full-strip plateau rule may waive a distant tied outer edge after two
completed nonimproving outward levels with valid same-slice evidence. It
cannot waive the selected candidate's bracket or treat invalid/missing
support as interior; it is empirical evidence, not proof of global optimality.
`--broad-grid` starts at 405 candidates. `--no-refine` skips local refinement
and factor-two tightening but still requires valid conditional neighbors.
`--max-expansions`/`--max-candidates` set explicit limits, and `--refine`
remains accepted. The DWD objective tolerance and iteration cap do not change.

`--refine-ratio` optionally adds bounded local ratio refinement after the
coefficient search meets its existing acceptance rules. A proposed ratio is
shared by every class pair in the complete OvO candidate; it is not chosen
separately for each pair. Defaults are three ratio rounds, two stable rounds,
an absolute ratio tolerance of `0.01`, and the initial range `[0.1, 0.3]`.
`--ratio-bounds LOW HIGH`, `--max-ratio-rounds N`, and `--ratio-tol X` require
this option. All candidates count toward the existing total budget. See
[selection-ratio refinement](method.md#selection-ratio-refinement) for the
conditional-neighbor requirements and restricted-range endpoint acceptance.
The option cannot be combined with `--no-refine` or `--smoke-grid`.

`--confirm-max-iter N` optionally repeats a fixed shortlist of complete OvO
candidates at a larger DWD iteration cap after search acceptance. It is
disabled by default; the cap must exceed the original cap, 100 by default.
`--confirm-top-k K` requires that cap and limits the shortlist to at least two
candidates, three by default. All pairs share the candidate's declared cap,
and every repetition uses the original folds and prescribed sampling seeds.
The shortlist adds work outside `--max-candidates`: up to `K * folds * seeds`
complete OvO fits, each comprising all class pairs and their base and final
learners before reuse.

Confirmation records scores and learner diagnostics separately. A changed
ranking or tie, invalid result, or fewer than two eligible candidates holds
final fitting and test scoring for review. Otherwise the original winner is
fitted with its original solver cap and prescribed final seed. A larger cap
may have no effect when objective-tolerance stopping occurs earlier; stable
scores do not establish convergence. See
[numerical-budget confirmation](method.md#numerical-budget-confirmation)
for shortlist, stopping, checkpoint and resume rules.

With one seed, ten classes and five folds, the initial 125-candidate stage
permits up to 168,750 DWD learner fits before caching; the 405-candidate stage
permits up to 546,750. Adaptive stages are additional work within the total
candidate budget.
The preview prints these separately. Optional `--final-native-threads N`
sets the native BLAS/OpenMP pool limit for the selected full-training pipeline
fit, including all sequential class pairs. It defaults to 1; outer CV `--jobs`
also defaults to 1 and can be adjusted independently. CV workers and held-out
prediction keep one native thread. A wider final budget requires exposed nested
`n_jobs` settings to remain explicitly 1 and is not a total CPU or core-affinity
limit. See [native threads by phase](method.md#native-threads-by-phase) for
rounding, runtime records and resume restrictions.

Pair fits stay sequential within each outer CV worker. No data are loaded
without `--run`.

`--smoke-grid` overrides the adaptive budget to four total candidates, zero
expansions and no refinement, even if other limit/refinement flags were supplied.
It cannot be combined with `--broad-grid` or `--refine-ratio`. All four smoke
candidates lie at endpoints of their two-point coefficient axes, so the search remains
unresolved and does not proceed to final fitting or test scoring. Use it to
check CV execution and records; it is not a completed accuracy search. Normal
searches also hold when a complete required stage cannot fit the budget. The
review decision is saved before the final refit and separate test evaluation,
which proceed only when the neighbor and configured refinement requirements
are met.

Completed fold fits and the final refit save compact diagnostics. To resume,
use `--resume --run` with the original output directory and identical data,
folds, estimator settings, policy/budgets, jobs and immutable numerical
runtime. Verified completed folds are reused; unfinished fits restart by
default. The option below can retain completed pairs inside those folds. An
interrupted final fit has no optimizer checkpoint. If the completion receipt
has already been committed, a verified resume performs no fitting or scoring
and returns `model=None`. A crash before that
receipt is committed can require final fitting and scoring again. See the
[shared workflow](method.md#validation-and-proposed-search-space) for records,
ownership checks and failure handling.

`work-summary.json` separates this invocation's new CV evaluations,
equivalent-ratio reuse and checkpoint replay, optional confirmation, final
fitting and scoring. Cache observations count reported binary pairs; missing
metadata is unknown. Worker timing sums, elapsed time and stored receipt
timings remain distinct. See
[work and timing summaries](method.md#work-and-timing-summaries) for the
completed-replay and interruption rules.

Each adaptive stage also writes `<stage>-plan.json` before dispatch. Counts
use the classes present in each training fold and each candidate's `M`, then
include the requested CV seeds. They separate binary models, ensemble bases,
ensemble finals, and full-model learners. These are planned counts before
reuse or failure, not observed fitting or an ETA. Unknown estimator structures
have unknown learner counts. Refused stages preserve any generated proposal and
stopping reason in `coefficient-refused-stage.json` or `ratio-refused-stage.json`;
no partial stage is launched to fill the remaining allowance.

The same stage plans disclose newly introduced close but distinct coordinates
with bounded detail samples. Coordinate closeness alone does not establish
equivalent multiclass fits. All candidates and checkpoint identities remain
separate; see [reporting details](method.md#work-and-timing-summaries).

`--reuse-equivalent-folds` optionally shares completed CV results between
retention ratios only when the retained count agrees for every class pair.
Candidate labels and full-training refitting remain unchanged. See the
[reuse conditions](method.md#equivalent-fold-reuse).

Use `--cv-seeds 42 43 44` to repeat every complete OvO candidate with the same
prescribed sampling seeds and fixed folds. Average seed scores within each
fold, then average folds. Every seed must yield a valid complete multiclass
result; an ineligible class pair invalidates that candidate. The seed count
multiplies CV work, while the candidate budget still counts parameter sets.
Final fitting uses one separate prescribed seed, set by `--refit-seed` or the
configured default, 42. The default remains a single seed. See
[multiple CV seeds](method.md#multiple-cv-seeds) for checkpoint, timing and
resume rules.

## Completed-pair recovery

`--pair-recovery --no-cache` enables optional recovery of completed binary
pairs inside an unfinished CV fold. It is off by default. For example:

```shell
python examples/tune_mnist_multiclass.py --data mnist.npz --output-dir results/pair-search --pair-recovery --no-cache --run
python examples/tune_mnist_multiclass.py --data mnist.npz --output-dir results/pair-search --pair-recovery --no-cache --run --resume
```

The example API exposes `CheckpointedCV(..., pair_recovery=True,
pair_recovery_max_bytes=None)`. The adaptive controller and optional
numerical confirmation forward the same settings. Confirmation has a
separate checkpoint identity; original-budget pairs cannot replace its fits.

Recovery currently supports an exact two-step `Pipeline` containing
`StandardScaler` and `OvOClassifier`, with exact `KernGDWD` or
`EnsembleKernelDWD` children. The Pipeline must have no memory cache or
metadata routing and must use `verbose=False`. Inputs and labels must be
dense, finite numeric arrays. Each evaluated candidate must use a named RBF kernel, a positive
gamma supplied as a plain Python `int` or `float`, a Python integer random
seed, and `OvOClassifier.n_jobs=1`.
Both retained and discarded ensemble base models are supported. Custom
subclasses, mutable random generators, the separate ensemble base-fit cache,
multiple CV seeds and equivalent-fold reuse are not supported together with
this option. Ordinary fitting retains its existing supported settings.

Completed models are saved as trusted local pickle files. Only reuse files
created by a trusted run: checksums detect corruption but do not make an
untrusted pickle safe. Resume verifies the data and row order, scaler,
candidate, pair labels, fitted state, source files and numerical runtime.
A changed identity requires a new run directory. The existing fold lease
and coordinator generation guard also protect the pair files.

`--pair-recovery-max-bytes N` limits committed pair payloads and manifests
per candidate/fold store. It is not a total disk or temporary-write limit.
The default has no byte limit; zero skips saving new models. Quota or
optional storage failures leave successful computations usable, while
identity or ownership failures stop reuse. An unfinished binary fit still
restarts from the beginning. Once the authoritative fold receipt is
committed and verified, pair payloads can be removed; interrupted cleanup
can finish on resume without refitting the completed fold. Cleanup status
is recorded separately without rewriting the authoritative receipt.

Recovery uses the existing pair fitting, candidate failure handling and
multiclass prediction. It does not change the solver, sampling, row
selection, CV candidate ranking or votes. It does not save a final
full-training refit or checkpoint the binary optimizer. Saving, checking
and loading models costs time and disk space; enable it when avoiding
repeated completed pairs after interruption is worth those costs.

Reports distinguish newly fitted pairs, restored pairs and completed-fold
replay. Current pair fit/load/save times are components of the enclosing
fold fit time, not additional time to add to it. Historical fit times and
cache observations of restored children are not counted as new work. See
[work and timing summaries](method.md#work-and-timing-summaries).

## Resource use and diagnostics

Pair fitting is sequential by default. Set `OvOClassifier.n_jobs` for a
standalone parallel fit, or leave it at 1 while parallelizing CV candidates.
Do not multiply both worker counts without an explicit resource budget.
The tuning example bounds dispatched jobs and limits native library work to
one thread per outer CV worker. Pair feature matrices are formed inside their
worker, rather than constructing all pair matrices before dispatch.

`predict` accumulates votes in batches. `vote_counts` returns all class vote
counts when requested. These counts are not probabilities or decision margins.
To discard fitted base models, set the binary estimator's
`store_base_estimators=False`. Training still uses dense kernel matrices;
these controls do not guarantee that a fit will stay within available memory.

After fitting, `pairs_` identifies each child in `estimators_`. To locate a
child's selected rows in the original multiclass training input, use
`model.pair_indices_[p][model.estimators_[p].selected_indices_]`.
Each child's `n_iter_` and `converged_` cover its `M` base fits followed by
its final fit. An objective-change stopping event does not certify optimality.
Ineligible base subsets or final selections invalidate the complete candidate and are reported with
the responsible pair. Changing the retention ratio is a new CV candidate,
not an automatic recovery action.
