# Parallel base fitting

For choosing between base, pair and outer-CV parallelism, see
[Choosing Ensemble options](choosing_options.md#parallel-work-and-prediction-allocation).

`EnsembleKernelDWD(base_jobs=4)` fits up to four base learners concurrently.
The default, `base_jobs=1`, keeps sequential base fitting. The actual worker
count cannot exceed `M`. Each parallel worker uses one native numerical thread;
the caller's native limits are restored before the fresh final fit.

This is a scheduling option. It preserves the sampled disjoint subsets, learner
parameters, consensus accumulation in original base order, stable row-selection
order and final-refit procedure. Numerical acceptance criteria are unchanged.
Different BLAS thread counts can nevertheless change floating-point results;
bitwise equivalence is not promised for arbitrary inputs or runtimes.

Each eligible internal base task uses a snapshot of the calling thread's four
NumPy error modes and its registered callback or logger, including an absent
handler. Prior worker state is restored when the task returns or raises.
Base scoring retains the sequential path's local overflow/invalid ignore mask;
other error modes continue to follow the caller's settings.

Callbacks and loggers are shared by reference and may be invoked concurrently.
They must tolerate concurrent calls; serial invocation order and identical
event counts are not guaranteed. This propagation applies to internal DWD base
tasks, not the separate `OvOClassifier` joblib dispatch or arbitrary caller
context variables.

## Runtime ownership and resource planning

The caller must own native-thread configuration throughout the fit. Unrelated
Python threads must not concurrently use or change the same numerical runtime.
Calling from the main Python thread alone does not establish this ownership:
native thread limits can affect other code in the same process.

Calls from a Python worker thread, custom learner factories, callable kernels
and callable kernel arguments keep sequential execution. `base_execution_`
describes the policy of the fitted base result; read it together with
`cache_hit_`, as detailed below. These fallbacks
preserve the caller's existing behavior; they do not repair an independently
unsafe external numerical-thread configuration.

If CV already runs parallel processes, budget `base_jobs` across those processes
or leave it at one. Concurrent base fits can increase peak memory. Choose the
worker count using the subset and final-fit sizes, available physical and commit
memory, and measured costs on the intended runtime. CPU count alone does not
determine a safe or useful setting. The final fit can dominate complete-fit time,
so parallel bases do not guarantee a speedup.

Workers intentionally use one native thread. On the qualified Windows AOCL
runtime, a SciPy-only control reproduced a native failure after a Python child
thread performed multithreaded numerical work. Increasing native threads inside
these workers is therefore not an interchangeable scheduling choice.

## Caching, diagnostics and interruption

The requested worker setting, effective execution policy, package source and
numerical runtime participate in base-cache identity. A parallel request and its
background-thread sequential fallback have separate entries, even when their
requested worker counts match. Default sequential requests retain the caller's
numerical runtime. Custom learner factories bypass the cache.
A cache hit reuses completed base work but still fits a fresh final learner.
Failed base work is not published as a completed cache entry.

`base_execution_` describes how the fitted base result was produced. On a
miss, or with caching disabled or bypassed, it describes the newly computed
base stage. On a hit, it retains the producing stage's policy rather than
claiming that the current fit ran those base workers. In particular:

- `effective_jobs` is the producing stage's worker limit, not the number
  of base fits performed by the current call.
- For parallel bases, `worker_runtimes` contains one observation per base
  task. Four bases run with three workers therefore have four entries,
  not four distinct-worker identities.
- On a parallel cache hit, `worker_runtimes` and `restored` are historical
  producer observations, not a fresh check of the consuming process.
- Sequential results have a synthesized policy summary; they do not
  contain a captured sequential worker-runtime inventory.

After a successful `EnsembleKernelDWD.fit` using the built-in base stage,
the cache outcome and the number of subsets tell you how many base learners
this call fitted or reused.

```python
fitted = estimator.fit(X, y)
bases = len(fitted.subset_indices_)
print("Base learners fitted", 0 if fitted.cache_hit_ else bases)
print("Base learners reused", bases if fitted.cache_hit_ else 0)
print(fitted.base_execution_)  # policy associated with the fitted bases
```

Every successful call fits one fresh final learner. On a cache hit the call
also verifies and reads the cache and performs selection. For partial work
on an interrupted or failed fit, inspect the exception's
`ensemble_failure_context` when available. See [ensemble failure evidence](method.md#ensemble-failure-evidence).

For elapsed fit-stage measurements, use one estimator worker and one native
thread throughout the fit. See the [serial profiling example](method.md#optional-fit-profiling).
When interpreting existing parallel profiles, worker intervals can overlap
with each other and with the parent's elapsed time. Completed-fit and scoring
counts include completed workers even when the caller was interrupted before
receiving their results.

While waiting, the caller periodically returns to Python so pending interrupts
can be processed on Windows. An interruption cancels work that has not started,
then joins started workers before restoring native limits and propagating the
original exception. It does not forcibly terminate an active numerical call;
cleanup can therefore wait for already started base fits. No fitted state is
published after a failed or interrupted fit. Error context records the affected
base position when known and distinguishes completed work from queued work.

## Checkpoint entry

Example workflows using `CheckpointedCV` have a separate process-startup path.
Fresh parallel entry dispatches its existing `jobs - 1` import-only tasks while
the parent computes its first full runtime fingerprint. After parent
qualification and import completion, it submits the single estimator-bearing
worker authority probe on the same managed pool. Parent/worker compatibility,
the full post-probe parent check, and source, native, receipt and lease guards
remain in place. There are no additional tasks or fits, and these import tasks
do not guarantee that every worker is warm. Serial, resume and existing-manifest
paths bypass this preparation.

Imports now start earlier, so their side effects can occur before a later parent
refusal. A preparation failure prevents submission of the estimator-bearing
probe. Parent and manifest errors retain precedence; cleanup preserves the
primary exception and attempts lease release. Preparation is not retried after
it starts. This private scheduling change adds no public option and does not
change the statistical or numerical method. Measured benefits depend on the
workload and runtime; see the [qualification and timing record](../VALIDATION.md#2026-11-11-fresh-parallel-checkpoint-entry-overlap).
