# Parallel base fitting

`EnsembleKernelDWD(base_jobs=4)` fits up to four base learners concurrently.
The default, `base_jobs=1`, keeps sequential base fitting. The actual worker
count cannot exceed `M`. Each parallel worker uses one native numerical thread;
the caller's native limits are restored before the fresh final fit.

This is a scheduling option. It preserves the sampled disjoint subsets, learner
parameters, consensus accumulation in original base order, stable row-selection
order and final-refit procedure. Numerical acceptance criteria are unchanged.
Different BLAS thread counts can nevertheless change floating-point results;
bitwise equivalence is not promised for arbitrary inputs or runtimes.

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
