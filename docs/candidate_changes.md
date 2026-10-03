# Ensemble DWD 0.3.4 release changes

Version 0.3.4, released on October 3, 2026, fixes persistent base-cache identity
and retains small base-fit diagnostic records. It also forwards the optional
DWD residual ordering setting. The statistical ensemble procedure is unchanged:
one uniform permutation
provides mutually disjoint base subsets; raw signed decision scores are averaged;
rows with the smallest absolute consensus scores are selected; a final DWD is
fitted to those rows. All original training rows remain eligible for selection.
The intercept remains unregularized, and the dissertation-to-package lambda
conversion remains unchanged.

## Cache identity

The separate `base-stage-v2` namespace and format include package-owned Python,
C/header and native artifacts, accelerator availability, Python implementation,
and existing dependency/threadpool provenance. Old v1 payloads are neither
promoted nor deleted. Changing retention alone still reuses the same base stage.

Package file metadata is recorded when the cache identity module imports. The
first cache lookup checks that inventory before hashing content. Later lookups
use metadata checks rather than rereading binaries. An unreadable, unknown or
observably changed identity bypasses persistent caching with an explicit status.
Binary content is identified anew in every fresh process. A file present on disk
and an accelerator that actually loaded are represented separately.

This assumes immutable installations. The identity describes package artifacts
observed since the cache module imported; it cannot verify code loaded
earlier, detect every same-size timestamp-preserving mutation, or identify
mapped machine code from disk bytes alone. Loaded BLAS/OpenMP libraries and
NumPy/SciPy/scikit-learn native extensions have a separate content snapshot,
including an already loaded AOCL bridge's source, manifests and libraries.
These files are hashed once per process after known backends are initialized;
later uses check their metadata and loaded inventory. Discovery uncertainty
disables reuse until restart. Identity is rechecked before cache reuse and
publication. Same-version edits to other external scientific Python sources and
unobserved transitive libraries remain outside this bounded snapshot.
Some operations can still fall back from native code to the portable path.

## Diagnostics and numerical controls

`base_diagnostics_` copies each original base fit's reported stopping flags,
iteration counts and backend, together with scalar optimality and residual
diagnostics. A cache hit preserves those original records. Objective stopping,
reaching an iteration cap, and satisfying an optimality criterion are distinct.
These records do not include timings for individual base fits.

`EnsembleKernelDWD(residual_check_order='adaptive')` explicitly selects DWD's
experimental order of residual checks for every base and final model. The
unchanged default is `'refinement_first'`. The setting is in the base-cache
key, so results from the two policies cannot be mixed. It changes the order
of numerical checks, not the ensemble sampling, scoring, selection,
regularization or final-refit definition. It requires `backend='auto'` or
`'cholesky'` with `acceleration=None`. A different accepted coefficient state
can change finite-iteration scores, selected rows and predictions despite
the unchanged objective and numerical tolerances. See the method guide's
numerical controls for its recovery behavior.

## Parameter search and resuming

The tuning examples now require valid conditional neighbors at boundary
candidates and repeat joint local refinement until the selected candidate
has a factor-two bracket and two completed stable local rounds. Filling gaps
at existing coordinates does not count as a stable refinement round;
improvements reopen review. An empirical full-strip plateau can waive a
distant tied edge only with valid tested evidence, never the
selected candidate's own bracket. Initial grids, the default 500-candidate
and three-expansion limits, solver tolerances and iteration caps are unchanged.
`--no-refine` skips local rounds but retains valid-neighbor requirements.

Explicit `--resume --run` reuses checksummed completed folds under identical
data, folds, estimator parameters, policy/budgets, jobs and immutable runtime
identity. Compact diagnostics are saved per fold and final refit. Unfinished
fits and interrupted final refits restart; no predictor is serialized. A
crash before the authoritative completion receipt commits can repeat final
fitting and held-out scoring. Once that receipt is committed, a verified
resume returns saved results with `model=None`, without fitting or scoring
again. The legacy
coarse/local helper remains nonresumable; it now preserves all parent
parameters and any better candidate from the coarse stage, and reports
final-stage flags and interruptions accurately. Binary data helpers reject
complex arrays.

## Dependency installation

This release requires DWD 1.3.9. Its dependency lock records the
SHA-256 hashes of the validated native and portable wheels through relocatable
artifact references. Install the locked dependency from the DWD 1.3.9 GitHub
release assets or downloaded local wheels before installing this package.
See the README installation instructions.
