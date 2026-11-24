# ensemble-dwd validation

The [package metadata](pyproject.toml) pins Ensemble DWD 0.4.1 to DWD 2.0.2.
The [wheel hash lock](requirements-dwd.txt) identifies the native Windows and
portable DWD 2.0.2 wheels for this release. Results below
apply to the artifacts and workloads named in each entry.

## 2026-11-19 paired candidate comparison

The full native-wheel regression suites used CPython 3.12.15. The baseline was
Ensemble DWD 0.3.15 with DWD 1.3.19. The candidates were Ensemble DWD 0.3.16
with DWD 1.3.20. Linux native DWD wheels were built from their respective source
distributions. The table counts individual pytest cases from each version's
own suite. These runs preceded the 0.4.0 and 2.0.0 version and packaging edits.
The Windows environment loaded AOCL-BLAS 5.3.0 and MKL 2026.2. The Linux
environment used MKL 2026.2. Attribution of a particular operation requires
evidence from its solver path.

| Environment | Package | Baseline passed / failed / skipped | Candidate passed / failed / skipped |
|---|---|---:|---:|
| Windows | Ensemble DWD | 962 / 4 / 2 | 2,226 / 0 / 17 |
| Linux | Ensemble DWD | 959 / 0 / 9 | 2,233 / 0 / 10 |
| Windows | DWD | 1,229 / 4 / 12 | 6,184 / 0 / 71 |
| Linux | DWD | 1,225 / 0 / 20 | 6,231 / 0 / 24 |

Across the separate common panel, the candidate pair met all 106 case
expectations per host. All 352 labels in 17 ordinary comparisons agreed with
the baseline on each host. All 2,208 timing jobs passed their numerical checks.
The 92 timing phases covered different workloads. In paired medians, the larger
checkpointed search added 2.29 s on Windows and 1.84 s on Linux for the initial
run and saved 0.37 s and 0.97 s on completed replay, respectively. The larger
complete CV, final-fit and scoring workflow saved 1.30 s and 3.39 s. The separate
256-row, 2,048-feature DWD L-BFGS measurement added 137 ms and 296 ms to a
second fresh fit. Workload, build and stage matter when applying these comparisons.
Ordinary scikit-learn `GridSearchCV` remains available. The measured
shared-fold opt-in uses DWD's standalone `KernMD`, outside the ensemble
workflow.

## 2026-11-11: fresh parallel checkpoint entry overlap

The isolated import-overlap candidate was qualified on Windows with AMD AOCL and
Linux with Intel MKL, both using CPython 3.12.15. Per host, 28 focused controls and
189 existing regression cases passed, with 47 subtests and one expected
version-specific skip. These are unique cases after documented fixture
corrections, not summed attempts. Coverage includes real worker cancellation,
cleanup and exception precedence, estimator serialization after parent proof,
managed-pool reuse, effective sequential execution, source/runtime guards, and
serial/resume bypass. The [behavior and failure-order tradeoff](docs/parallel-base-fitting.md#checkpoint-entry)
is independent of estimator mathematics.

Six fixed paired blocks per workload and host compared the candidate with the
preceding development helper, using fixed saved inputs and settings. Ordinary
and partial cases used 384-by-32 data; the larger case used 1536-by-32 data, with
three folds and a solver iteration limit of 30. Checkpoint jobs were 12, 6 and 6,
respectively, with one native thread per worker. Each host completed 36 original
children, 72 contexts, 360 ensemble fits and 1080 DWD learner fits. All 18 paired
scientific states agreed per host.

Initial time covers constructor, entry, search/fitting and normal cleanup.
Combined time adds the same child's immediate replay or partial continuation;
it is not total interpreter lifetime. Savings below compare mean costs within
one host, using only complete timing-qualified pairs.

| Host | Workload | Eligible pairs | Initial mean saving | Combined mean saving |
| --- | --- | ---: | ---: | ---: |
| Windows AOCL | Ordinary | 3 | 4.80% / 1.192 s | 4.69% / 1.208 s |
| Windows AOCL | Partial | 5 | 2.96% / 0.418 s | 2.76% / 0.419 s |
| Windows AOCL | Larger | 4 | 3.16% / 0.453 s | 3.45% / 0.526 s |
| Linux MKL | Ordinary | 6 | 2.37% / 0.283 s | 2.37% / 0.307 s |
| Linux MKL | Partial | 6 | 1.54% / 0.177 s | 1.06% / 0.135 s |
| Linux MKL | Larger | 6 | 1.03% / 0.121 s | 1.15% / 0.146 s |

Six Windows pairs were excluded from timing because lifecycle evidence was
insufficient; their scientific results and original failure flags remain
retained. There were no replacements or CPU-load filters. Windows run-order balance
is consequently incomplete. Excluding the qualified but interrupted first
ordinary pair still gives 4.47% initial and 4.25% combined mean savings across
two opposite-order pairs. Linux partial/larger initial gains depend on order,
and individual pairs sometimes favor the preceding helper.

Linux partial resume increased by 41.5 ms on average (3.16%; five of six pairs),
although its initial-plus-resume mean remained lower. Resume bypasses the new
preparation; these observations do not establish a recurring penalty or a
universal no-slowdown result. Source prechecks warm file caches, so this is not
cold-filesystem evidence. The results are descriptive and workload-specific,
not significance, standalone-fit or release-gap-elimination claims. Integration
verification is separate from this pre-integration evidence; this is not a
complete package suite or release qualification.

Integration verification on both hosts initially passed 87 selected tests and
failed one bytecode-fidelity assertion in the new test fixture. Independent
compilation identified missing module-import context in the fixture's extracted
methods, which changed CPython call flags; the production helper was unchanged.
The corrected fixture uses the imported production code with isolated test
globals and checks fidelity against compilation of the full source. All 28
controls then passed on each host. Together with the 60 retained passing tests
from the other selected modules, this gives 88 unique passing checks per host;
the initial failure and all attempts remain recorded. One fresh partial-checkpoint
smoke check per host also matched the qualified scientific state exactly: two
contexts, 12 ensemble fits, 36 DWD learner fits and six reused folds on resume,
using six workers with one native thread each. Both smoke children exited zero,
all captured process identities were closed, and no unproven discoveries were
recorded. These integration checks add no performance claim.

## 2026-11-09: interruption status test expectation

The interruption fixture now expects the two existing noncheckpoint runtime
metadata fields: `runtime_policy='noncheckpoint-native-scope-v1'` and
`source_installation_attestation=False`. Full status-dictionary equality,
exception identity, completed CV/selection records, and the absence of ordinary
failure or completion records are still checked. No runtime code changed.

The complete affected module, `test_legacy_example_repairs.py`, passed 15 tests
and 28 subtests on both Windows AOCL and Linux MKL, with no failures
or skips. Both KeyboardInterrupt and SystemExit cases passed. Validation used
isolated current source copies, CPython 3.12.15, pytest 9.1.1 and psutil 7.2.2,
with initial native-thread limits of 16 on Windows and 20 on Linux. The monitored
children and parent drivers exited zero; source hashes and dependency
inventories remained unchanged.

This is focused validation of the integrated fixture correction. The original
accumulated-comparison full-suite failures remain preserved; this result does
not relabel those suites as passed or constitute a new full-suite run.

## 2026-11-09: integrated base-cache observation optimization

The combined cache-observation change is integrated: a fresh external-artifact
stat supplies both file classification and its identity stamp; ordinary Python
module origins avoid unnecessary path construction; and package inventories
reuse each directory path and relative-key prefix. The two implementation files
exactly match the candidate measured in the completed comparison. Numerical
algorithms, public options/defaults, cache format and provenance boundaries are
unchanged. The immutable-installation contract and the removed adjacent-stat
observation limit are described in [the change notes](docs/candidate_changes.md#040-base-cache-observations).

Focused integration validation used isolated copies of the current paired
package sources on CPython 3.12.15, with pytest 9.1.1 and psutil 7.2.2:

| Target | Passed pytest items | Passed subtests | Skipped |
| --- | ---: | ---: | ---: |
| Windows AOCL | 202 | 40 | 4 |
| Linux MKL | 206 | 40 | 0 |

The nine selected modules include the two new permanent regression modules
(66 cases), existing external/package identity and NumPy policy checks,
selection-source checks, real cache reuse and invalidation, concurrent cache
locking, and effective parallel-execution identity. The four Windows skips are
one POSIX-only filename case and three real symlink fixtures unavailable with
the current permissions. Those four cases passed on Linux; this does not
claim Windows symlink coverage. No machine setting was changed.

Both monitored test processes and their parent drivers exited zero. Initial
native thread limits were 16 on Windows and 20 on Linux; existing test-specific
limits and subprocess thread-environment requests were retained. The
517-file source snapshots passed hash checks
before and after the tests, recorded loaded modules came from those snapshots,
and dependency inventories were unchanged. Known admitted process identities
were closed on both hosts. Three transient Windows process observations remained unproven
and are not claimed as owned or independently closed.

The earlier clean performance panel remains the timing evidence: on the
studied 256-by-16 cached RBF ensemble workload, warm whole-fit medians improved
31-34% on Windows and 26-34% on Linux against the preceding development version.
These are within-host, workload-specific observations, not a universal speedup,
a comparison of library versions, or a standalone-prediction result. No new
timing claims are drawn from pytest durations. This focused integration check
is not a new complete package suite or a built-wheel/release qualification.

<a id="0315-current-source-coverage-and-acceleration-policy-test"></a>

## 0.3.15 source coverage and acceleration-policy test

For the 0.3.15 production source and filesystem/test-support changes, the
complete Python 3.12 suites recorded 966 passes and two platform skips on
Windows AOCL 5.3, and 959 passes and nine platform skips on Linux MKL 2026.2. Each suite collected 968 unique test items across 77 modules,
with no failures or unresolved items and direct exit zero. The matched source
snapshots were unchanged. Platform-specific errno parameters were matched by
their named error conditions; skips remain excluded coverage. These suites
used an initial native-thread budget of 16 while preserving test-specific
thread clamps.

Those complete suites contained the preceding acceleration-policy test. Its
replacement was then validated separately on both targets: 25 pytest items
and 33 subtests passed on each, with direct exit zero, no monitor error and
unchanged tested sources. This focused selection included the replacement,
existing backend and selected-objective tests, and validation-only controls.
It used a portable DWD source snapshot and does not newly qualify a compiled
DWD route.

The replacement observes each real learner's fit call and checks exact policy,
parameters and training-data forwarding for auto, spectral, Cholesky and
L-BFGS backends. Each final fit is checked against its own selected rows.
It does not replay eigensystems or compare independently solved coefficients,
histories or scores bitwise. Existing cache and pickle checks are unchanged.
Synthetic accepted eigenpair variation reproduced the old assertion failure;
the replacement passed while rejecting wrong policies, penalties, data and
missing or reused learners. Injected exceptions and malformed provider tuples
confirmed that the original eigensolver fallback remained in charge. These
controls do not claim a spontaneous AOCL failure or alter scientific acceptance.

The saved results establish source regression and focused test coverage, not
performance, universal necessity of historical changes, or qualification of
the final built wheels and minimum-dependency installation. Release-artifact
checks are separate from these source records.

<a id="2026-10-26-follow-up-actual-robust-nfs-contention"></a>

## 2026-10-26 follow-up on Linux NFS contention

The original blocking-lock code reproduced two failures on the Linux host's NFS-mounted
home filesystem: a four-process timeout and a three-process ENOLCK error. The other
20 selected items passed. Simple lock acquisition, exclusion and owner-exit
checks had passed, so those alone did not establish reliable cache contention.
The failed receipts are preserved.

The revised POSIX path requests the same exclusive OS lock nonblockingly and
retries EACCES/EAGAIN every 50 ms. All 34 focused cache items then passed on the
same NFS mount, including the unchanged three-/four-process compute-once tests
with their original 20-second deadlines, and a real killed-owner/reacquisition
test. Windows AOCL checks passed 33 items plus 26 subtests; only the real POSIX
owner-exit test was skipped on Windows. Injected non-contention failures and
cancellation checks also passed. Psutil observed owned test processes. The Linux run
recorded direct exit zero, no cleanup errors and unchanged tested sources.

This resolves the observed package contention failure without moving the cache,
extending deadlines or changing server configuration. It does not establish
general NFS server health, cross-host test coverage, fairness or bounded RPC
duration. ENOLCK and other unsupported/error conditions still fail before cache
computation. No performance claim is inferred from these correctness tests.

## 2026-10-26: filesystem locks and temporary test paths

The final isolated source artifact passed 130 pytest test items on Windows
Python 3.12.15/AOCL 5.3 (one POSIX-only skip), and 122 on Linux
Python 3.12.15/MKL 2026.2 (nine Windows-only skips). Neither run had failures
or errors. Pytest 9.1.1 ran existing unittest cases as well as new pytest tests;
JUnit's extra subtest records are not additional independent test items.
Psutil 7.2.2 tracked validation processes and observed CPU/RSS samples, while
direct waits supplied actual exit codes. Source hashes remained unchanged.

Four independent cache workers proved compute-once and shared-key behavior.
Injected unavailable-lock errors proved that computation and publication remain
blocked and that errno/cause are retained. The Linux test used a private scratch directory under `/tmp`. The home NFS
lock service lay outside this test. No lock bypass, automatic cache relocation or timeout extension was
introduced. The chosen cache filesystem must still provide working locks.

Windows checks include a fresh process with a requested temporary root longer
than 260 characters, helper reads and cleanup, environment restoration, and all
39 workflow methods previously affected by deep paths. No OS path setting was
changed. The paired DWD selection passed 304 Windows and 295 Linux test items.
Its nine Linux skips are Windows-specific. These are focused correctness and
integration checks, not a complete suite, performance comparison or new wheel
release. The remaining timings in the separate resource benchmark were canceled.

## 0.3.14: automatic acceleration and SVM defaults

The applied changes passed 947 ensemble-DWD test methods under Windows CPython
3.12 with AOCL, with zero failures or errors and one POSIX-only path skip. The
paired DWD suite collected 1,069 methods, with zero failures or errors and five
runtime, floating-point-format or file-symlink privilege skips. These checks
precede the final version metadata and forward-runtime changes; they are not
new full-suite results for the release artifacts.

Coverage includes automatic and explicit DWD acceleration, dependency dispatch,
cache identity, lean diagnostics and optional inverse action. Disjoint subset
sampling, absolute-consensus selection and the final DWD objective are unchanged.
Automatic restarted MM can change a finite stopping result relative to ordinary
MM; explicit `acceleration=None` remains available. `linear_action='factor'`
remains the default, and inverse action remains explicit.

The SVM comparator now defaults to unbatched native prediction and a 1024 MiB
temporary LIBSVM kernel-cache capacity per active fit. Explicit and saved batch
or cache settings remain honored. Tests preserve the comparator's estimator,
signed-margin selection and final refit; the cache capacity is not a measured
memory peak. Documented input-conversion and concurrent-fit memory costs still
apply. These changes do not claim a universal speedup. Release artifact checks
and hosted CI are recorded separately.

## 0.3.14: forward Python 3 runtime support

The paired DWD change passed a focused 48-method selection in four Windows
configurations, each including one 48-row ensemble integration fit. Passed/skipped
counts were 48/0 for Python 3.12 with the compiled helper, 46/2 for Python 3.11
with the helper, 32/16 for portable Python 3.12 and 27/21 for portable Python
3.11. All configurations had zero failures or errors. Skips concern unavailable
compiled arithmetic and Python 3.11's absent `math.sumprod`.

Checks used isolated source copies and the unchanged compiled helper, with
current C and build-source hashes verified. Simulated 3.14, 3.15 and 3.99
identities tested dispatch on actual 3.11/3.12 interpreters, not future arithmetic.
These were focused checks, not complete suites or large benchmarks; that check
performed no new 3.13 run, installation, build or release. Routine complete-suite
validation targets 3.11/3.12, while Python 3.13 and later Python 3.x releases remain
supported. Ensemble sampling,
selection and objectives are unchanged.

## 0.3.14: DWD dependency integration

The integration tests recognize DWD's new fitted RBF construction modes.
The ensemble estimator implementation, disjoint sampling, selection rule and
objectives are unchanged. Corrected distance calculations in DWD can change
scores and selected observations on affected inputs, so historical fitted
results are not rewritten. Old fitted construction policies remain identifiable.

Local paired checks against released 0.3.13 preserve coefficients, predictions,
selected rows and subset identities on the ordinary control problems. Both
packages are checked with isolated native and portable artifacts; Python 3.12
is the primary runtime, with focused 3.11/3.13 checks. Dependency corrections
and their numerical/resource limits are documented in DWD's kernel guide and
validation record. These checks do not change the ensemble methodology or
establish a general accuracy or speed advantage.

A subsequent local Python 3.12/AOCL comparison kept those correctness fixes in
both arms and changed only DWD's guard scanning and sparse prediction reuse.
Ensemble input remains dense; it benefits from the guard-scan change, not the
sparse-input path. Six alternating fresh-process pairs per ensemble case
preserved all saved numerical arrays, including consensus, disjoint subsets,
selected rows, coefficients and predictions. The median paired fit-time change
was -2.55% in the base-heavy case, with ordinary and larger ensemble controls
approximately unchanged. Each trial ran alone with 16 native threads. These
modest measurements do not establish a general speed advantage or remove all
correctness-check overhead.

An additional untimed replay of the frozen MNIST case preserved its original
preprocessing and settings and matched all saved numerical arrays exactly.
The final dependency payload passed the full ensemble suite: 938 methods collected,
zero failures or errors and one POSIX path behavior skip on Windows. DWD's
full and focused runtime checks, including their explicit skip limitations,
are documented in its validation record. No ensemble estimator changes,
environment installation or release were made by this follow-up.

## 0.3.13: portable selection-order regression tests

Version 0.3.13 includes the accumulated changes documented for the unpublished
0.3.12 candidate below and remains paired with DWD 1.3.17. The 0.3.12 hosted CI
found a platform-sensitive expected row in a selection-metadata test: two
almost equal computed scores did not have the assumed order on every runtime.
The corrected test independently derives the expected selection from the actual
decision scores. A separate controlled-score fixture checks exact tie ordering.
These test-only changes preserve the estimator implementation, selection rule,
objectives and numerical acceptance. They do not represent a correction to
historical experimental results or establish a performance gain.

The dated application checks below retain their original scope. Validation of
the corrected release artifacts and hosted CI is recorded separately; the
unpublished candidate's failed test is not counted as a passing release check.

## 0.3.12: optional parallel bases and named SVM kernels

`EnsembleKernelDWD(base_jobs=1)` retains sequential base fitting. Explicit
parallel requests use one native thread per eligible base worker, accumulate
consensus in original base order and restore the caller's native-thread limits
before the final fit. The caller must own the process-wide numerical runtime
settings during fitting. Custom factories, callable kernels and execution
from a Python worker thread retain sequential behavior. Requested scheduling
and effective execution eligibility participate in cache identity. Sampling,
selection, learner objectives and numerical acceptance are unchanged; different
native-thread allocations can change floating-point results. See the
[parallel fitting guide](docs/parallel-base-fitting.md).

`EnsembleKernelSVM` now supports explicit linear and polynomial kernels while
retaining RBF as the default. The polynomial option requires positive gamma,
a positive integer degree and nonnegative `coef0`. Base and final learners
use the same resolved settings. Named-kernel settings enter cache/checkpoint
identity; old saved RBF predictors remain readable. Signed-margin selection
and the final SVC refit are unchanged. Callable, precomputed and sigmoid
kernels remain unsupported. See the [SVM guide](docs/ensemble-svm.md).

Application checks on 2026-10-19 ran 937 test methods with the isolated native
wheel under Windows CPython 3.12 with AOCL: zero failures, zero errors and one
POSIX path behavior skip. Another 28 focused tests passed on each of Python
3.11, 3.12 and 3.13 using both native and portable DWD wheels, with zero failures, errors or
skips in every configuration. These overlapping runs are not distinct counts
of unique tests. The exact installed runtime payloads were verified; focused
Python 3.11/3.13 environments were isolated from the working environments.

The checks cover numerical parity on the tested fixtures, base ordering,
profiling, interruption cleanup, effective cache identity, custom-factory
bypass, named-kernel forwarding, serialization and independent SVC
reconstruction. They do not establish universal bitwise equality or a speedup.
No new performance benchmark or hosted CI run is claimed by these application
checks.

## 0.3.12: automatic policies and equivalent-fold reuse

New ensembles request automatic affine scoring and residual ordering, passing
the concrete supported policy to every DWD learner. Explicit choices and old
fitted behavior remain supported. The example searches now request eligible
equivalent-fold reuse automatically, with `--no-reuse-equivalent-folds` as the
opt-out. Completed-pair recovery remains optional. Earlier sections record
historical validation and do not override these current defaults.

Windows CPython 3.12/AOCL checks passed 60 focused ratio/recovery tests and 13
multi-seed compatibility tests, with no skips. Coverage includes exact retained
counts for every class pair, candidate and runtime identity, interrupted pair
and orphan-fold adoption, parallel publication, higher-budget confirmation
separation, zero new fit/score times for aliases, and a fresh final refit.
Injected sidecar directory failures preserve the original interruption.
Automatic and explicitly enabled reuse remain distinct resume settings.

Twenty-one policy and dependency tests also passed against the released DWD
1.3.15 artifact, with no skips. New sign-orientation regressions passed on
actual Python 3.11, 3.12 and 3.13. They protect DWD absolute-consensus and SVM
signed-margin selection under global class reversal; they do not establish
that an earlier production sign bug existed. Sampling remains disjoint, and
selection, objectives and numerical acceptance rules are unchanged.

These are correctness checks using small fixtures, generally one native
thread and two workers for designated parallel tests. Concurrent checks are
not performance benchmarks. These dated checks precede the release version
and dependency-lock changes; release-artifact verification and hosted CI have
separate records. The corrected release, version 0.3.13, is paired with DWD 1.3.17.

## 0.3.11: sparse-input validation on older dependencies

The public validation path was checked on local Windows Python 3.12 with
NumPy 1.26.4, SciPy 1.12.0 and scikit-learn 1.6.1, and separately with NumPy
2.5.4, SciPy 1.18.1 and scikit-learn 1.9.1. Each environment passed the same
20 focused tests with no failures, errors or skips. These overlapping suites
are not 40 distinct tests.

Before the correction, a real-valued DOK sparse array reproduced the conversion
failure in the shared guard, ensemble DWD, ensemble SVM and the multiclass
wrapper. The guard now recognizes sparse containers before NumPy conversion.
Unsupported real sparse inputs receive an explicit dense-input error; complex
sparse inputs retain the complex-data error without discarding imaginary values.
Regression coverage includes all seven SciPy sparse formats as matrices and
arrays, failed refitting, prediction, score validation and the three estimator
contracts that exposed the problem. Supported dense-input calculations, fitting
objectives, subset sampling and numerical acceptance are unchanged.

The 0.3.10 candidate was not published because its minimum-dependency CI job
found this issue. Version 0.3.11 includes its other changes below and remains
paired with DWD 1.3.15. Release CI runs on the branch; the tag is verified to
reference the same tested commit rather than launching a duplicate matrix.

## 0.3.10: real-valued input validation and MNIST preprocessing

Version 0.3.10 is paired with DWD 1.3.15 and requires Python 3.11 or later.
The checks below cover development sources on local Windows environments;
they do not establish validation of every release artifact or platform.

The complex-input changes passed 121 focused and estimator-contract tests and
three integration tests using the real DWD implementation, with no failures,
errors or skips. The checks cover complex arrays, complex scalars stored in
object arrays, MNIST labels, learner scores and predicted labels. Unsupported
complex values are rejected before conversion can discard their imaginary
parts. Ordinary real-input comparisons preserved scores, predictions, consensus,
subsets and selected rows exactly. No evidence was found that the historical
experiments supplied complex input, and no speed improvement is claimed.

The final unit-pixel default passed 120 relevant tests: 78 preprocessing,
checkpoint, search and recovery checks plus 42 existing command-line checks.
They cover raw pixels divided by 255, explicitly declared inputs already in
`[0,1]`, real fits, parallel execution, equivalent-ratio reuse and partial-pair
recovery. Checkpoints created with standardization are rejected under the new
unit-pixel default before fitting, with their prior receipts preserved.
Fold-local `StandardScaler` remains available explicitly. These tests are
correctness checks, not a new accuracy or timing study; historical experiments
were not rerun or rewritten.

Three release-metadata checks passed for the installation examples, dependency
lock structure and All Rights Reserved notice before the version update.
The suites above overlap and must not be summed as unique tests. The estimator
objectives, disjoint sampling, selection rules and numerical acceptance checks
remain unchanged. See the [changelog](CHANGELOG.md#0310--2026-10-17).

## 0.3.9: adaptive restart ordering and example reporting

Version 0.3.9 is paired with DWD 1.3.14 and requires Python 3.11 or later.
The Python 3.10 companion is no longer maintained; older records below retain
their original scope.

The development source passed 90 focused ensemble tests and 59 paired DWD
tests on Windows CPython 3.12 with AOCL, with no failures, errors or skips.
Sources were held unchanged during each suite. The exact-candidate correction
also passed 32 helper and workflow checks; these overlap earlier checks and
must not be added as unique tests. This evidence is not a full cross-version
or operating-system release matrix.

Coverage includes the optional adaptive/restart combination and unchanged
eligibility restrictions; all-learner diagnostic coverage; current versus
cached/recovered work; missing historical counters; non-overlapping timing
subtotals; class composition and secondary metrics; immutable checkpoint-owned
labels; exact adjacent candidate identity; and SVM cache-size validation,
forwarding and checkpoint identity. Existing numerical acceptance, sampling,
selection formulas and ordinary-accuracy tuning remain unchanged. Restoring
formerly merged parameter candidates can change a future search's winner.

Adaptive ordering was compared with the default ordering in 32 fresh fits:
three saved TCGA ensemble cases and one saved full-DWD case, at one and 16 native
threads, with two balanced pairs per case and thread count. The two difficult
ensemble cases improved complete fitting time by approximately 28–29%; the
full-DWD case improved by about 14.6%. The healthy ensemble case was essentially
unchanged at one thread and about 5.4% (7 ms) slower at 16 threads. The default
therefore remains `residual_check_order='refinement_first'`.

The compared saved final coefficients, intercepts, histories, scores,
predictions, subsets and selected rows were identical, and all learners matched
iteration counts and final objectives under the unchanged acceptance checks.
This does not establish universal bitwise equality or direct comparison of
every base coefficient array. The monitored confirmation runs had no competing
Python process in 50 ms observations; other operating-system activity cannot
be excluded. Earlier contended timings were excluded from these timing claims.

The reporting changes add no fit or prediction. Their own processing cost is
reported separately; zero overhead is not claimed. Completed studies and their
tables were not rewritten. See the [method guide](docs/method.md) for control
scope and the [changelog](CHANGELOG.md#039--2026-10-15) for the release changes.

## 0.3.8: boundary recovery and optional fit profiling

The boundary fallback was checked on 8,158 cases: 5,304 previously successful
coordinates were preserved bit for bit, 2,838 false refusals were recovered,
and 16 genuine arithmetic limits remained refused. All 2,854 fallback outcomes
agreed with independent 1,000-digit Decimal calculations. The boundary and
surrounding search workflows passed 144 distinct test methods, including
adaptive and ratio-policy replay, candidate budgets and interrupted-fold
identity. Two methods initially detected concurrent source edits and passed
when replayed with stable sources. This correction can extend future searches;
it does not revise completed study results or merge distinct candidates.

The optional `fit_profile()` context passed 19 focused tests and 67 existing
estimator, cache, diagnostic and interruption regressions. Checks include
exact numerical parity, overflow recovery, real cache hits, aborted final fits,
nested scopes, thread/task isolation, and copied contexts after scope exit.
A separate integration check used the real source and native-library provenance:
a cache created without profiling was reused with profiling, preserving its
key, selected rows and numerical result. Disabled instrumentation read no clocks.

Two complete MNIST training-draw fits, with profiling off and on, each fitted
all 45 class pairs and 270 binary DWD learners. Both reproduced the saved
scientific state, and their checked training scores and predictions matched
bit for bit. The comparison covered 4,779 saved attributes; the original lean
models did not retain base coefficients, so base diagnostic facts, consensus,
subsets and selections supplied the available base-stage evidence.
The profile reported 225 base fits, 225 initial scoring passes
and 45 final fits. No official test arrays were decoded. These checks preserve
the original finite iteration budget; they do not establish new convergence.

The sequential Windows CPython 3.12/AOCL fits used one native thread and took
178.109 and 177.406 seconds. These two observations do not establish zero
profiling overhead or a speed improvement. Whole-interval background CPU
observations were retained, without certifying the absence of short bursts.
The objective, sampling, selection and solver settings are unchanged.
See the [profiling scope and usage](docs/method.md#optional-fit-profiling).

## 0.3.7: paired dependency validation

Version 0.3.7 updates the paired DWD dependency to 1.3.12. All ensemble
runtime files are byte-identical to 0.3.6 except the version value, and
all examples are unchanged. The separate CPython 3.10 companion retains
the previously validated exception-note compatibility overlay.

On Windows CPython 3.12 with AOCL, the installed standard ensemble wheel
and portable DWD wheel passed 55 focused tests.
On Windows CPython 3.10 with OpenBLAS, the paired companion wheels passed
64 focused tests. The focused suites had no failures or skips.
Each configuration also passed three public-fit checks for standard
binary scoring, joint/direct binary scoring and sequential three-pair
OvO fitting, including serialization and retained-row checks.

These completed checks cover the portable dependency variants; separate
full-suite results for native DWD and release CI belong to the release
validation report.

Actual pip installations selected the intended interpreter-specific
wheels. Package origins, installed source bytes, wheel records and
`pip check` were verified; workers from these checks exited. The compatibility
wheel rebuilt identically from its bundled source and recipe. These
checks validate package integration, not convergence, runtime speed or
a new MNIST search. Correctness suites could run concurrently. They do
not establish Linux/Intel numerical or performance equivalence.

DWD's new dense compensated-scoring path is qualified for eligible
Windows CPython 3.12 x86-64 inputs. Other runtimes and unsupported inputs
retain the Python calculation and its numerical checks. See the paired
[DWD validation record](https://github.com/cheng338/dwd/blob/v1.3.12/VALIDATION.md)
for the numerical and timing scope of that dependency change.

## 0.3.6: installed release artifacts

The standard wheel retains Python 3.11 or later. All runtime source files match
the preceding development source byte for byte, except the release
version value. Examples are unchanged. The separate legacy
interruption-test correction checks the final-runtime metadata already returned
by the phase-thread workflow; it does not change runtime behavior.

On Windows CPython 3.12 with AOCL, the installed standard wheel and paired
DWD 1.3.11 wheel passed 19 focused metadata, error-handling and legacy-workflow
tests, with no failures or skips. Three additional public-fit checks covered
standard binary scoring, joint/direct binary scoring and sequential three-pair
OvO fitting, including finite scores, serialization and retained-row checks.
These used five fixed iterations and are integration checks, not convergence or
timing studies. Import locations, package bytes and `pip check` were verified.

With both standard and companion wheels available, actual pip installations on
CPython 3.12 selected the standard pair; CPython 3.10 selected the separately
tagged companion pair. The Python 3.10 error-note fallback is absent from the
standard package. These Windows checks do not establish Linux/Intel equivalence.

## Historical release pairing and validation records

The 0.3.9 release was paired with DWD 1.3.14. See the
[0.3.9 changelog](CHANGELOG.md#039--2026-10-15) for its changes.
The records below retain their original workload and version scope; they are
not new full MNIST searches for version 0.3.9. DWD's
[1.3.10 validation scope and timing follow-up](https://github.com/cheng338/dwd/blob/f6db6e45463554db52385caba8c6c1c1f63fcd8b/docs/validation-1.3.10.md)
remains historical evidence for the previous dependency release.

## 0.3.6: final-fit native threads

The optional example setting `final_native_threads` defaults to 1 and controls
the selected full-training pipeline fit. CV and held-out prediction retain
one native thread; the core estimators and their defaults are unchanged.

The isolated prototype passed 23 focused tests and 99 existing workflow,
checkpoint, interruption and example tests. Coverage included default behavior,
scope and caller restoration, failure and interruption, resume and completed
replay, changed-budget/helper rejection, invalid inputs and all three CLI
entry points. Actual base-cache checks at 1, 2, 2 and 1 threads produced
miss, miss, hit and hit, with separate keys and equal state when repeating
the same budget. A real sequential three-pair OvO fit checked the wider scope
and restoration. An isolated helper-file replacement verified the live guard.

After adoption, the canonical local source passed the same 23 focused and
99 existing tests with no failures or skips. Imported DWD and ensemble modules
came from the canonical source directories; the example and test
files matched the validated candidate, and core estimator source was unchanged.

These were correctness checks conducted alongside other validation work, not
timing benchmarks or evidence that different thread counts always return the
same bytes. Native rounding can affect stopping and selected rows. More threads
can help some fits and slow others; no automatic increase is made. See
[native threads by phase](docs/method.md#native-threads-by-phase) for scope and
the distinction between a library-pool limit and total CPU use.

## 0.3.6: extended exact recovery

The optional `exact_recovery='extended'` profile raises exact-rank and
integer-bit limits to 32 and 8192 while preserving the standard default,
entry/operation caps, represented kernel, recovery equations and acceptance
checks. The dependency's isolated trial recovered three of nine saved difficult
RBF fits for five and 100 updates; the longer runs reached the iteration cap
without convergence. The remaining six still reached the rank limit. Separate
exact witnesses establish that their two stored kernels are indefinite, which
additional resources cannot turn into exact PSD certificates.

The ensemble integration checks option validation, real base/final and OvO
forwarding, ordinary-fit parity, cloning, fitted and compact diagnostic
provenance, old/current serialization, older-dependency compatibility, cache
separation and equivalent-fold identity. This is recovery coverage for small
kernels, not evidence of a speed or accuracy improvement.
See [the method guide](docs/method.md#optional-extended-exact-recovery) for
the limits and the distinction between successful fitting and convergence.

## 0.3.6: joint affine scoring

The optional `affine_computation='joint'` route includes the intercept in
cancellation-aware evaluation of `K @ alpha + b`. Standard scoring remains
the default. The isolated numerical trial included exact rational checks,
near-zero fitted decisions, dense and sparse kernel readouts, overflow and
underflow controls, and explicit validation stopping in DWD. The public
ensemble integration separately covers base/final propagation, OvO cloning,
failed-refit cleanup, old serialization, fitted policy retention, dependency
compatibility, base-cache separation and equivalent-fold identities.

Six paired fresh MNIST ensemble configurations (three digit pairs with
standard and direct RBF) retained rankings, selected rows, labels and
accuracies; two had small decision-score differences. This is bounded
validation on fixed saved data and settings, not a new parameter search.
The two-row linear fitted example demonstrates a corrected near-zero label.
Ten paired prediction observations on the difficult saved MNIST ensemble
showed about 9% additional time; background CPU activity flagged two pairs,
and the other eight retained that conclusion. Constructed exact ties cost
substantially more. See [the method guide](docs/method.md#optional-joint-affine-scoring)
for scope and limitations. The option requires the paired DWD 1.3.11 dependency.

## 0.3.4 validation scope

Version 0.3.4 is paired with DWD 1.3.9. See the
[0.3.4 release changes](docs/candidate_changes.md) for its cache identity,
diagnostics, search and recovery changes and their limitations. The
[0.3.3 change-by-change validation](docs/validation-0.3.3.md) remains historical
evidence, including the earlier DWD 1.3.8 dependency checks; its numerical
results should not be relabeled as a new full MNIST run for version 0.3.4.

## 0.3.2: disjoint base subsets and fixed 10% sampling

Both ensemble classifiers use one label-independent uniform permutation to
form disjoint base subsets, following Mao et al. (2016), Section III-A.
The MNIST search presets fix `theta=0.1`, with `M=5` in the examples, and
retain nine regularization values, nine gamma values and five retention
fractions: 405 candidates. CV boundary review still applies only to
regularization and gamma. Selection fractions keep their specified range.

All 180 package tests passed against each released DWD 1.3.7 wheel (native
and portable) under Anaconda CPython 3.12, with no failures or skips. Coverage
includes exhaustive small-population allocation checks, global uniqueness,
unused remainders, block-size/capacity constraints, label-independent draws,
no redraw after an ineligible proposal, pre-fit failure cleanup, shared DWD/SVM
sampling, cache separation, OvO context, and external CV eligibility reporting.
The usual solver, selection, prediction, sklearn and concurrency checks remain
in the suite. Distribution hashes and installed-wheel verification accompany
the release artifacts.

`IneligibleSubsetError` reports an impossible disjoint allocation or a sampled
single-class block. It subclasses `IneligibleSelectionError`; unexpected
numerical and memory failures still propagate. Neither classifier changes its
sampling or numerical parameters automatically in response to a failure.
The DWD backend option is shared by all base and final fits and distinguished
in cache keys; the default remains `auto`.

The prior independent draws allowed overlap. Their validation results below
and in historical reports describe that earlier sampling scheme. They are
not disjoint-sampling accuracy or timing results.

---

## ensemble-dwd 0.3.0 validation

This record describes validation of the `v0.3.0` release of ensemble-dwd,
including its binary and multiclass ensemble DWD interfaces and optional
SVM comparator. It covers prediction-batch validation and the external
binary examples' review of regularization and gamma boundary winners.
Neither estimator performs cross-validation internally.

### Environment and dependency

Validation used Windows, Anaconda CPython 3.12.14, NumPy
2.5.2, SciPy 1.18.0, scikit-learn 1.9.0, joblib 1.5.3, and the published
optimized cheng338/dwd 1.3.5 dependency with its compiled Windows helper.
Imports were isolated from older ensemble installations in that environment.
Native numerical threads were limited to one for the functional tests.

The metadata pins these optimized DWD artifacts:

| Artifact | SHA-256 |
|---|---|
| `dwd-1.3.5-cp311-abi3-win_amd64.whl` | `f792d5fa251c57526990221c7d573274e50a99da60e5f508f51515f8f622435e` |
| `dwd-1.3.5-py3-none-any.whl` | `45ec234a373f6080d805089bb24fa869e6de57a1660c3988aeba6ae2c529cb13` |

In the validated `v0.3.0` source, the DWD dependency, objective, unregularized
intercept, and stopping defaults are unchanged by the prediction-batch repair.
The DWD estimator, cache, multiclass wrapper, and parameter-grid implementation
files matched the candidate used for the completed three-pair study.

### Regression coverage

All **159 source tests passed** in 38.747 seconds, with no failures or skips.
The three-pair source prediction check also passed, preserving all labels
and scores exactly and verifying 171 frozen-file identities before and after.
Installed-wheel test receipts and distribution hashes accompany the built
release artifacts.

The original candidate reproduced the post-fit batching defect. Regression
checks cover negative, zero, boolean, floating-point, nonfinite and string
batch sizes; both prediction methods must reject them before calling SVC.
Restoring a valid size preserves the fitted state and resumes prediction.
Positive Python/NumPy integers and `None` retain the same labels and scores,
including when changed through a scikit-learn Pipeline.

The full suite also covers independently reconstructed real SVC base and
final fits, signed selection, arbitrary labels, exact ties and rounding,
shared C/gamma, cold/warm cache reuse and warnings, failed-refit cleanup,
real multi-worker GridSearchCV with fold-local scaling, all 45 ten-class OvO
pairs, scikit-learn estimator checks, and Windows cache concurrency.
Workflow checks verify that a boundary hold saves CV/selection records and
performs neither final fitting nor held-out scoring; valid review and default
workflow paths still complete. Unexpected failures remain explicit.

### Completed MNIST study and prediction checks

The five-classifier study used 20% of each pair's official MNIST training
partition, fivefold training-only CV and the full corresponding official test
partition. Its detailed record is in
[the three-pair validation summary](docs/mnist-validation-0.3.0.md).
The frozen pre-repair source, models, outputs and protocol are preserved.

The batching regression reuses the three already fitted SVM ensembles. It
compares labels and float64 decision values byte for byte with the saved
results at several valid batch sizes, checks invalid values, and checks exact
recovery after restoring a valid value. This is an inference regression,
without fitting or parameter selection. Source and installed-wheel checks
also verify the frozen source/data and all 15 model/output identities.

All final SVC fits completed. The study's DWD fits reached their configured
100-iteration cap. The reported results describe this study's accuracy and
runtime without establishing numerical convergence for those DWD fits.

**Attribution clarified November 19, 2026.** The original 0.3.0 validation
text referred to unsigned ranking, `abs(abs(g) - 1)`, in earlier dissertation
wording. The October 2026 deposited dissertation (version 1) describes the
signed condition in §4.2.1 (p. 51) and ranks SVM observations by
`abs(y_signed * g - 1)` in §5.2.2 (p. 58). That section ranks DWD observations
by `abs(g)`. The saved 0.3.0 MNIST study used the signed SVM rule. Its results
and the separate historical unsigned criterion retain their original scope.
[Deposited dissertation, version 1](https://doi.org/10.25394/PGS.34548831.v1) ·
[0.3.0 study record](docs/mnist-validation-0.3.0.md).

### Performance review and distribution

A bounded profile on the three saved models' training rows found roughly
1–2 ms of prediction-wrapper overhead. Removing batching saved less than
1 ms on those cases, so the default stays at 1024. Base fits are already
reused across selection ratios. No additional solver rewrite or nested
parallelism was justified by these measurements.

The wheel and source distribution are checked against the release source;
the source distribution includes the changelog, tests, examples and method
documentation. Local results do not substitute for the configured Linux/
Windows and Python-version GitHub Actions matrix. Free-threaded Python has
not been runtime-tested locally.

An observed NumPy shape-assignment deprecation warning originates in
joblib 1.5.3 array deserialization and also reproduces without either DWD
package. It does not change the verified loaded arrays or predictions.

Historical validation remains in the [0.2.1](docs/validation-0.2.1.md),
[0.2.0](docs/validation-0.2.0.md) and [0.1.1](docs/validation-0.1.1.md) records.
At the time of the 0.3.0 validation, Chang Cheng maintained the project as
private research and had not selected a software license.
