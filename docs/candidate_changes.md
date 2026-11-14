# Ensemble DWD change notes

## Unreleased base-cache observations

Persistent base-stage caching still checks the current installation and
numerical runtime before admitting reuse. The observation helpers reduce
repeated filesystem and Python path work in three ways:

- On the normal successful path, one fresh external-artifact `stat` supplies
  both regular-file classification and its identity stamp. No observation is
  retained across provenance boundaries.
- Exact built-in string origins ending in `.py` or `.pyc` avoid path
  construction when they cannot denote native code. Extension loaders, custom
  path-like objects, string subclasses and versioned `.so.` names retain the
  full path checks.
- A package inventory constructs its directory path and relative-key prefix
  once per visited folder. It still visits the same files in the same order,
  observes every included file, and retains error handling and content hashes.

A successful cache hit retains three complete provenance observations: initial
key construction, verification under the cache lock, and verification after
deserialization. Cold computation and publication retain their existing checks,
including checks after content hashing and before publishing the entry. Source
identity, native-library and loaded-module identity, NumPy error policy,
accelerator qualification, file-change detection and sticky cache-bypass
behavior remain. This does not cache a provenance result across those checks.

The installation must remain immutable from import. Sharing a successful
`stat` removes an adjacent second observation: a metadata-visible change just
after that observation can escape the current check and be rejected at a later
check if it persists. A transient failure of the first `stat` can also cause a
conservative cache bypass even if a later file-type query succeeds. This is
not exact preservation of every concurrent-mutation detection point, atomic
observation of the filesystem, verification of previously loaded machine code,
or detection of every timestamp-preserving edit. Do not modify or upgrade
packages during use.

The bounded comparison used a synthetic 256-row, 16-feature RBF ensemble with
four cached base learners and a new final fit on each warm call. Both ordinary
and strict NumPy error policies were tested with unchanged numerical settings,
16 native threads on Windows AOCL and 20 on Linux MKL. In the final fixed panel,
median whole-fit time for each of the three warm positions improved by
31-34% locally and 26-34% on Robust relative to the unchanged development
baseline. Adding the directory-prefix change to the other two changes improved
those medians by a further 8-11% locally and 10-33% on Robust. These are
within-host comparisons, not AOCL-versus-MKL or library-version claims.

All 12 local timing blocks qualified under the original CPU screen; 11 of 12
Robust blocks qualified, with the excluded block retained separately. Direct
warm comparisons favored the combined change in all 12 local blocks and
10-11 of the 11 eligible Robust blocks at each position. Some incremental
comparisons and order subgroups were slower. The workload does not establish a
universal speedup, and repeated warm calls are not independent replications.
Separate instrumented profiles locate the reduction in cache identity work;
their elapsed times are not substituted for the clean whole-fit measurements.
Saved within-host scientific arrays matched exactly across compared arms.

No public option, default, numerical algorithm, acceptance criterion or cache
format changes. Existing source hashes still distinguish an updated
installation from earlier cache identities. This change concerns cached full
fitting: it does not change standalone warm prediction or the separately
documented example checkpoint inventory and metadata-discovery paths.

## Unreleased checkpoint metadata discovery

The checkpoint helper reads distribution metadata to bind installed package
versions and selected `.dist-info/METADATA` files to runtime identity. For an
ordinary filesystem-backed standard-library distribution on CPython 3.12 and
later, it parses the complete RECORD while avoiding unused file-hash objects and
existence checks for unrelated members. Row shape and integer-size validation
still finish before any selected metadata hash is published. Selected metadata
order, duplicates, normalized aliases, content hashing and drift checks remain.

This intentionally narrows ancillary error behavior: filesystem errors caused
only by an unrelated RECORD member are no longer raised during metadata
discovery. This is not a general relaxation of runtime verification. The
separate source/native inventories, loaded-artifact checks, worker qualification
and checkpoint-boundary checks are unchanged. Installations must still remain
immutable while the workflow runs.

Python 3.11, non-CPython interpreters, custom distribution providers, non-native
filesystem roots and absent or empty RECORDs use the original
`Distribution.files` fallback. Focused checks ran on actual CPython 3.11 and
3.12; routing later CPython versions to the fast path is forward compatibility,
not a claim of validation on every future standard-library version.

The bounded Windows AOCL and Linux MKL comparisons used identical numerical
work. The six original Linux pairs, reported descriptively across all scheduled
pairs with the original conservative CPU-screen exclusions preserved, favored
the initial checkpointed workflow
by a median 6.8%, and initial workflow plus one completed replay by 5.7%.
Completed replay was initially 6.2% slower at the paired median, but controlled
follow-ups reversed its direction. The precise cause remains unresolved.
These observations do not establish a universal speedup, a fixed replay
penalty, or an AOCL/MKL library defect. Completed replay adds no new model fits.

No public option, numerical algorithm, scheduling rule or acceptance criterion
changes. As with other helper-source changes, strict runtime identity refuses
reuse across the old and new sources. Use the original frozen workflow for an
old checkpoint, or start a new run directory; do not migrate historical records.

## Unreleased checkpoint inventory

`examples/cv_checkpoint.py` reduces repeated filesystem work while retaining
full checkpoint runtime inventories and checks. Where `Path.walk` is available,
it scans each visited directory once and prunes excluded `__pycache__` trees
before descent. On Python 3.11 it delegates to the original `rglob` traversal,
preserving that interpreter's error behavior. Regular-file classification and
the hash/cache precondition share one fresh lexical-path `stat` result;
`DirEntry` metadata is not used as content authority. Focused compatibility
checks covered actual CPython 3.11 and 3.12, rather than simulated version labels.

Installed packages must remain unchanged while the workflow runs. Full source,
metadata, native-library, alias and checkpoint-boundary checks remain, as do
worker qualification and checks after uncached content reads. Sharing a stat
removes an adjacent second observation: if a previously cached, unloaded file
changes immediately after the retained observation, that check can return the
previous digest. A persistent metadata-visible change is rejected at the next
full boundary. This does not preserve every concurrent-mutation detection point,
detect every metadata-preserving edit, or provide an atomic snapshot of loaded
code. Do not modify or upgrade packages during a run.

The bounded Windows AOCL and Linux MKL comparisons covered initial checkpoint
contexts and authenticated replay with exact matching scientific work. Relative
to the stat-only alternative, the combined change reduced local replay medians
by about 7-8%. Relative to traversal alone, Linux results were mixed: the larger
case added about 1.6% (0.197 s) to the initial context and 0.8% (0.007 s) to
replay in paired medians. These are direct alternative comparisons, not a
combined-versus-original estimate or a universal speedup. Worker startup,
execution probes and required integrity checks still have costs.

No estimator mathematics, scheduling, public option or numerical default
changes. Source identity continues to refuse old checkpoint reuse across this
helper change: use its original frozen workflow or a new run directory. The
separate nonresumable native-resource scope is unchanged.

## Historical release 0.3.4

Version 0.3.4, released on October 3, 2026, fixes persistent base-cache identity
and retains small base-fit diagnostic records. It also forwards the optional
DWD residual ordering setting. The statistical ensemble procedure is unchanged:
one uniform permutation
provides mutually disjoint base subsets; raw signed decision scores are averaged;
rows with the smallest absolute consensus scores are selected; a final DWD is
fitted to those rows. All original training rows remain eligible for selection.
The intercept remains unregularized, and the dissertation-to-package lambda
conversion remains unchanged.

### Cache identity

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

### Diagnostics and numerical controls

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

### Parameter search and resuming

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

### Dependency installation

This release requires DWD 1.3.9. Its dependency lock records the
SHA-256 hashes of the validated native and portable wheels through relocatable
artifact references. Install the locked dependency from the DWD 1.3.9 GitHub
release assets or downloaded local wheels before installing this package.
See the README installation instructions.
