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
records the effective worker count and runtime information. These fallbacks
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

Profiling records the parent parallel stage as elapsed wall time and stores
individual worker profiles separately. Worker times overlap and must not be
added to parent wall time. Completed-fit and scoring counts include completed
workers even when the caller is interrupted before receiving their results.

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
