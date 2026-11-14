# Choosing Ensemble options

Use the defaults unless a specific numerical, execution or search requirement
calls for an override. This guide explains those requirements and the tradeoffs;
it does not change any default. It describes the current source, including the
explicit `affine_computation='joint'` default. Installation examples for the last
published release do not establish that every current-source option is available
in an older package pair. See the [upgrade boundary](method.md#public-solver-api-upgrade).

Constructor options below apply to the estimators. Flags such as `--resume` and
`--confirm-max-iter` belong to the supplied tuning examples, not to `fit`.
The example checkpoint/search helpers are separate from the core estimators.
Ordinary statistical choices such as kernel, regularization, `M`, `theta` and
retention fraction still require training-fold validation; this is not a claim
that one setting is best for every dataset.

For DWD, a numerical override reaches **every base learner and the final
learner**, and every binary ensemble inside OvO. Even when two routes preserve
class labels, changed raw scores can change consensus ordering, selected rows
and the final model. Compare the complete estimator with the same data,
preprocessing, folds, seeds, hyperparameters and resource budget. Inspect
convergence and scores as well as wall time; use the heldout test set only after
selection. A small speed difference alone does not decide a numerical policy.

## DWD numerical choices

### Stopping and iteration budgets

| Choice | When to use it | Cost and limit |
|---|---|---|
| Default `stopping='objective'` | Practical fitting when the returned-model quality and reported residuals are adequate. | A small successive objective change is not a stationarity certificate. Inspect all learners, not only the final one. |
| `stopping='optimality'` | A numerical stationarity target is required, default residuals are too large for downstream use, or methods must be compared at a common achieved optimization tolerance. | Can materially increase full ensemble and CV time. `tol=1e-6` with the unchanged `max_iter=100` can still miss the target; tighter optimization does not guarantee better predictions. |
| `stopping='fixed'` | A controlled update-budget comparison or a deliberately capped trajectory is the question, rather than objective-change early termination. | Does not promise convergence or equal wall time. Kernel recovery can add discarded work; the cap is per attempt, and numerical failures still raise. |

Choose `max_iter`, `obj_tol` and `tol` for the declared numerical requirement;
do not relax acceptance merely to make a preferred route appear faster.
Inspect `n_iter_`, `converged_`, `base_diagnostics_` and the fitted final learner.
The [optimality guide](method.md#choosing-optimality-stopping) explains the full
cost and diagnostic scope. Ensemble does not expose per-learner validation
stopping or random initialization: its DWD learners start from zero, while
`random_state` controls subset sampling.

### Optimizer, acceleration and residual ordering

| Choice | When to use it | Cost and limit |
|---|---|---|
| Default `backend='auto'` | Ordinary use when no particular solver representation is required. | Retains the dependency's guarded selection and eligible automatic recovery; an additional recovery attempt can add work. |
| `backend='cholesky'` | Isolate the shifted-system MM route in a controlled investigation, or deploy a workload for which that explicit route has been validated. | Fixing the route also forgoes auto's eligible switch to a spectral attempt. It is not a remedy for an unresolved Cholesky failure. |
| `backend='spectral'` | Isolate eigenbasis MM to investigate route-sensitive behavior, or use a workload where its achieved quality and complete cost have been validated. | Includes eigenbasis preparation. It is not universally faster or more accurate, and finite-budget scores can differ. |
| `backend='lbfgs'` | Compare a different optimizer on the same DWD problem when MM's progress or cost warrants investigation. | Keep numerical-quality checks explicit; an equal iteration cap is not equal work or equal quality. Explicit restart acceleration and adaptive residual ordering are unsupported. |
| Default `acceleration='auto'` | Keep route-appropriate acceleration: restarted MM on compatible backends, no acceleration for L-BFGS. | Resolves from the backend, not from measured progress; finite-iteration behavior differs from ordinary MM. |
| `acceleration=None` | Establish an ordinary-MM control, diagnose restart-sensitive trajectories, or retain a present workload's validated ordinary-MM quality/cost tradeoff. | May need more updates. Conversely, objective-change termination under restart need not deliver the same quality at the same cap. Compare achieved quality and the full ensemble. |
| `acceleration='restart'` | Require restarted MM explicitly, so an incompatible backend request fails rather than dispatching to ordinary updates. | It does not add another acceleration method beyond auto on eligible MM routes. Do not use it with L-BFGS. |
| Default `residual_check_order='auto'` | Use adaptive scheduling for eligible fresh Cholesky fits and the applicable refinement-first route elsewhere. | Acceptance checks and recovery budgets remain in force; scheduling can change the accepted finite-precision state. |
| `residual_check_order='refinement_first'` | Compare or diagnose correction-first behavior separately from acceleration, or use a workload for which that order is specifically validated. | May spend time on corrections subsequently rejected by accurate checks. It is not a stronger acceptance tolerance. |
| `residual_check_order='adaptive'` | Explicitly require the eligible Cholesky check-scheduling optimization, including in a controlled one-factor comparison. | It can avoid repeated unsuccessful speculative corrections; it does not guarantee a speedup. Explicit spectral/L-BFGS requests are incompatible. |

These explicit routes are useful controls, not promises of universal superiority.
Read [numerical controls](method.md#numerical-controls) for forwarding and fitted
diagnostics. Change one factor at a time before investigating interactions.
Different requested policies have distinct cache identities; do not reuse a
result from another route as its measurement.

### Kernel construction, score arithmetic and recovery

| Choice | When to use it | Cost and limit |
|---|---|---|
| `rbf_computation='direct'` | Investigate sensitive distances between nearby large vectors, or other represented inputs where current selective standard repair is insufficient for the needed precision. | Computes coordinate differences for every pair, including ordinary pairs; can be much slower. Standard already repairs eligible cancellation-sensitive pairs. Direct is not universally correctly rounded or entry-wise more accurate. Validate kernel/score accuracy and complete fit/prediction cost. |
| Default `affine_computation='joint'` | Protect the combined `K @ alpha + intercept` expression, especially when cancellation makes a small score or its sign important. | Refinement has conditional cost. Joint is not a universal correct-rounding guarantee for every ordinary accepted score. |
| `affine_computation='standard'` | A controlled study of separately rounded score arithmetic, or a current workload with a validated accuracy/cost tradeoff that justifies its weaker cancellation protection. | A faster wrong sign is not an equivalent speedup. Check small margins, score errors, selection ordering and final predictions, not just average accuracy. Keep joint as the accuracy-oriented default; old records alone are not a reason to choose standard. |
| `exact_recovery='extended'` | A small difficult fit reaches the standard exact-recovery rank or integer-bit limit and unchanged-input recovery is needed. | Raises rank 16 to 32 and bit limit 4096 to 8192, but preserves entry/operation limits. At most 256 kernel rows can fit the entry bound; smaller cases can still refuse. Not an indefinite-kernel repair, a speed option or a convergence guarantee. L-BFGS does not use this MM recovery budget. |
| `linear_action='inverse'` | Repeated shifted-system solves are a material part of an eligible Cholesky workload and measured full-fit time benefits from paying setup once. | Adds a dense `8*n*n`-byte action per active fit plus setup temporaries/workspace. Short fits may not amortize setup. It is not supported on explicit spectral/L-BFGS routes and is not a memory-saving path. |

See [direct RBF](method.md#optional-direct-rbf-computation),
[joint affine scoring](method.md#joint-affine-scoring),
[extended recovery](method.md#optional-extended-exact-recovery) and the
[inverse-action note](../README.md#results-and-boundaries).
Direct RBF and joint affine scoring address different rounding stages; one
does not replace the other. Earlier timing examples in the detailed method
notes retain their explicitly historical scope, rather than predicting current
performance. Changing fitted numerical policies requires a new fit; changing
constructor attributes does not reinterpret an existing child's fitted policy.

## Parallel work and prediction allocation

Choose one main level of parallel work first. The memory required by a training
kernel is only a lower bound on a fit: factorization, recovery, feature copies,
scoring and other active processes also count. Set concurrency from the actual
learner sizes, and account for workers multiplied by native threads. More
workers are not useful when the final serial fit dominates or startup costs
outweigh the base work.

| Choice | When to use it | Cost and limit |
|---|---|---|
| `base_jobs > 1` on DWD | A standalone binary ensemble has enough independent base work to amortize scheduling, and the caller owns the numerical runtime and has memory for concurrent learners. | At most `M` base workers, one native numerical thread each, then a separate final fit. Leave at 1 under already-parallel CV unless the combined budget is explicit. Custom/unsupported cases can remain sequential; inspect `base_execution_`. |
| `OvOClassifier.n_jobs > 1` | Pair fitting is the selected parallel level in a standalone multiclass fit. | Leave at 1 when outer CV supplies concurrency. Pair data/models and any inner base workers multiply resource use. Prediction remains separate from pair-fit scheduling. |
| Example `--jobs N` | Enough independent CV work exists to use multiple processes within the fit-specific memory budget. | Examples use one native thread per outer worker. Do not multiply it by unrestricted pair/base workers; higher concurrency is not a speed guarantee. |
| Example `--final-native-threads N` | The selected full-training pipeline fit dominates and a measured comparison shows useful native-library scaling while all nested fitting stays sequential. | Defaults to 1. Does not change CV, refinement, confirmation or heldout prediction. Limits detected pools, not total machine CPU or core placement. It does not parallelize a single LIBSVM SVC. |

For `final_native_threads > 1`, the nonresumable `run_search_workflow`
checks native resources and restoration, assuming installed code remains
unchanged. It records `source_installation_attestation=false`; it does not
provide a complete source or installation attestation. Strict phase helpers,
adaptive search and checkpoint/recovery retain their complete/frozen checks.
Default 1 does not initialize the additional authority. The measured ordinary
workflow benefit does not establish larger-workload speed or native-thread
scaling; see the [policy and timing limits](method.md#native-threads-by-phase).

The caller must not let unrelated Python threads concurrently use or change the
same numerical runtime during the controlled native-thread phases. See
[parallel base fitting](parallel-base-fitting.md),
[multiclass resources](multiclass.md#resource-use-and-diagnostics) and
[phase-specific native threads](method.md#native-threads-by-phase), including
runtime-identity and resume constraints.

`EnsembleKernelDWD.prediction_batch_size` defaults to 1024. `None` permits a
whole-query call to the dependency; a positive value limits query rows per
kernel allocation. OvO has its own positive vote-batch size, also defaulting to
1024. Its outer batches and the child model's batches are distinct. Consider
larger batches or `None` when repeated small-call overhead matters and the
complete allocation is admissible; compare smaller batches when characterizing
that workload's allocation/throughput boundary. Measure representative end-to-end
prediction time and preserve or improve speed when selecting an allocation
change. Lower memory use alone does not justify a slower setting. Batching does
not make dense training kernels smaller or impose a whole-process memory cap.

Both DWD and SVM ensemble prediction-batch changes take effect on subsequent
predictions without refitting. DWD passes a supported per-call override without
mutating the child; older/custom children may keep a tighter internal limit.
SVM defaults to `None` and does not form a full query-by-support Gram matrix for
its supported named kernels. Prefer C-contiguous float64 inputs to avoid a
layout copy when possible. SVM batching can bound that per-call copy but adds
native-call overhead and does not avoid full-query validation or result storage.
See [SVM prediction details](ensemble-svm.md#parallel-work-caching-and-diagnostics).

## Storage, reuse and inspection

| Choice | When to use it | Cost and limit |
|---|---|---|
| `cache_dir=...` | Several candidates share the exact base stage, notably retention-ratio variants on the same training fold. | Hashing, locking, serialization and storage have costs. Selection and the final fit still run. Use trusted private storage; callable/custom/unidentified cases can bypass caching. |
| `cache_dir=None` / example `--no-cache` | Measure uncached base work, expect little repeated base-stage identity, or do not want persistent fitted-base payloads. Pair recovery also requires `--no-cache`. | Repeated base fits can cost more. This disables only the base cache, not completed-fold checkpoints or SVC's in-memory kernel cache. The estimator default is off; examples normally enable a run-local cache. |
| `store_base_estimators=False` | Only the final predictor and retained scalar/array diagnostics are needed after fitting. | Base models cannot then be inspected individually. Subset/selection indices, consensus, convergence diagnostics and final predictor remain; reduced storage does not imply faster fitting. |
| `fit_profile()` | Determine which DWD fit stages or cache operations dominate before choosing a scheduling or numerical optimization. | Optional instrumentation adds overhead; inclusive and self times must not be double-counted. The supported exact estimator scope and runtime ownership rules apply. |

See [base-cache and retained-model contracts](method.md#optional-base-stage-cache-and-retained-models)
and [profiling](method.md#optional-fit-profiling). Cached pickle payloads must be
trusted; a checksum is not protection against malicious serialized code.
Retained scores/indices can still be sensitive even without base-model objects.

## Search, interruption and recovery

The following controls belong to the supplied examples. A fresh run without
`--run` prints a preview; inspect its candidate/fit counts and resource choices
before launching. They do not introduce internal tuning into the estimators.

| Choice | When to use it | Cost and limit |
|---|---|---|
| Multiclass `--smoke-grid` | Check that data, CV execution and records work before a full search. | Four endpoint candidates are not an accuracy search and hold final fitting. Incompatible with broad-grid and ratio refinement. |
| `--broad-grid` | Prior training evidence justifies wider initial coefficient coverage, and the larger complete initial stage fits the declared budget. | Starts with 405 rather than 125 candidates. It is not a cheaper substitute for adaptive expansion or evidence of a global optimum. Keep the test set out of the choice. |
| `--no-refine` | A deliberately coarser, bracket-supported search is sufficient, or a controlled study needs to isolate the cost/value of local refinement. | Skips local midpoint rounds and factor-two tightening, but still repairs missing conditional neighbors and expands unresolved boundaries within budget. It does not force acceptance. Incompatible with ratio refinement. |
| `--max-expansions`, `--max-candidates`, bracket/stability settings | Prespecify affordable search work and the desired local resolution/stability evidence. | Complete proposed stages must fit the remaining budget. Exhaustion can leave `needs_review` and no final fit/test score. Lowering requirements changes the search evidence, not just scheduling. |
| `--refine-ratio` | Coefficient search is accepted and intermediate retention fractions are a relevant unresolved training-CV question. | Adds bounded local ratio/midpoint work; does not expand the original ratio bounds or prove global optimality. |
| `--confirm-max-iter` with `--confirm-top-k` | Test whether ranking of a fixed shortlist is sensitive to a larger numerical update budget. | Extra shortlist-by-fold-by-seed work is outside the original candidate budget. Objective stopping can still stop early. Changed/tied rankings hold the workflow; confirmation does not automatically promote a model or change the original final-fit budget. |
| `--cv-seeds` and `--refit-seed` | Compare candidates over several prescribed subset samplings and specify a separate final-fit seed. | Multiplies CV work by seed count. Use all declared seeds; do not select the best seed. Repeated seeds on the same folds are not independent generalization samples. |
| Equivalent-fold reuse (automatic or explicit enablement) | Close retention ratios select the same integer row set under the qualified deterministic pipeline. | Reuses verified completed fold results, not merely equal accuracies. Usual well-spaced grids may gain nothing. Final full-training fitting remains fresh. |
| `--no-reuse-equivalent-folds` | Independently execute candidate/fold work for a controlled cost or implementation check, rather than sharing an equivalent result. | Can repeat final fold fits and predictions; base caching is a separate layer. For an uncached-cost study also disable the base cache and use a fresh run directory. |
| `--error-score raise` | Fail immediately even for a known ineligible subset/selection, for example while checking a proposed search space. | Default `nan` excludes only these declared ineligibilities. Unexpected exceptions already propagate; neither mode fabricates a score or replacement model. |
| `--resume` | Continue an interrupted example run while preserving its completed folds and declared experiment. | Requires the original directory and matching data, settings, budgets, source/runtime and ownership checks. An unfinished fit may restart; ambiguous scoring cannot be silently repeated. |
| `--pair-recovery --no-cache` | Multiclass CV has expensive binary pairs worth preserving inside an unfinished fold if interrupted. | Saves/loads trusted model payloads, costing time/disk. Restricted supported pipelines, fixed seeds and serial pair fitting; no partial binary-optimizer checkpoint or final-fit recovery. Multiple CV seeds and base caching are incompatible. |
| `--pair-recovery-max-bytes N` | Bound committed pair-store payloads while keeping optional recovery. | Per candidate/fold store, not total disk or temporary writes. Zero skips new saves. Quota/storage failures need not invalidate completed computation, but identity/ownership failures refuse reuse. |
| `recover_final_fit` / explicit final-only recovery | An accepted search needs a separately recorded final-fit attempt, such as after an unscored failure or an approved final-phase resource correction. | Preserves selected pipeline, data and search result. Uses a distinct attempt identity; it does not replay CV or permit ambiguous scoring retries. |

Detailed contracts: [adaptive search](method.md#validation-and-proposed-search-space),
[ratio refinement](method.md#selection-ratio-refinement),
[confirmation](method.md#numerical-budget-confirmation),
[multiple seeds](method.md#multiple-cv-seeds),
[equivalent folds](method.md#equivalent-fold-reuse),
[checkpoint identity](method.md#checkpoint-runtime-identity),
[pair recovery](multiclass.md#completed-pair-recovery) and
[final-only recovery](method.md#explicit-final-only-recovery).
Changes to these recorded policies generally require a fresh run directory;
do not relabel old receipts as a newly configured experiment.

For MNIST examples, use `--pixel-input-range 0-1` only when pixels are already
scaled; otherwise the default unit preprocessing divides declared raw 0–255
pixels by 255. Explicit `--preprocessing standard` is a different feature
geometry: use it when variance-standardized features are the intended model,
fit the scaler inside each training fold, and retune in a new run. It is not an
extra step after unit scaling in these commands. See
[preprocessing](method.md#mnist-pixel-preprocessing).

## SVM-specific choices

The SVM comparator has its own solver; DWD backend, acceleration, residual,
direct-RBF, affine and exact-recovery options do not apply to it.

| Choice | When to use it | Cost and limit |
|---|---|---|
| `cache_size` / `--cache-size MIB` | Repeated LIBSVM kernel access is significant and memory permits testing a larger cache, or concurrency needs an explicitly budgeted capacity. | Default 1024 MiB per active fit, allocated as needed; not peak RAM or a strict byte cap. Larger is not always faster and can reduce feasible concurrency. Bases and final fit are sequential, so multiply by concurrent ensembles, not by `M+1`. |
| `shrinking=False` | Compare whether shrinking bookkeeping is worthwhile on a short/loosely solved SVC optimization. | Upstream guidance says shrinking can help long optimizations and disabling it can help short ones. This is a reason to measure, not a package-specific speed guarantee. Keep tolerances and other settings fixed. |
| Positive `max_iter` instead of `-1` | An explicit iteration ceiling is required for a bounded solver experiment or operational budget. | May return unconverged base/final SVCs; inspect `fit_status_`, `n_iter_` and warnings. The cap is not a wall-time bound. |
| Changing `tol` | The required optimization accuracy differs from the default and its effect on the complete ensemble is being validated. | Looser termination can alter margins, selected rows and final predictions; do not loosen it solely to manufacture a favorable speed comparison. |

The shrinking rationale comes from scikit-learn's
[SVM practical tips](https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use),
not an Ensemble-specific timing result. Consult the
[SVC parameter contract](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
and [comparator resource notes](ensemble-svm.md#parallel-work-caching-and-diagnostics).
When comparing shrinking, keep `tol`, `max_iter`, kernel, `C`, `gamma`, data,
subsets and resources fixed and time the full ensemble. No documentation here
establishes a universal threshold at which that toggle or a numerical override
becomes faster; that remaining decision is workload-specific.
