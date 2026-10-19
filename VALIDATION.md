# ensemble-dwd validation

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
wheel in local `py12_df_aocl`: zero failures, zero errors and one POSIX path
behavior skip. Another 28 focused tests passed on each of Python 3.11, 3.12 and
3.13 using both native and portable DWD wheels, with zero failures, errors or
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

Local `py12_df_aocl` checks passed 60 focused ratio/recovery tests and 13
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
the preceding approved development source byte for byte, except the release
version value. Examples are unchanged. The separately approved legacy
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

## Current release and validation history

Version 0.3.9 is paired with DWD 1.3.14. See the
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
came from the canonical source directories; the approved example and test
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
and portable) under Anaconda `py12_df`, with no failures or skips. Coverage
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

Local validation uses Windows, Anaconda CPython 3.12.14 in `py12_df`, NumPy
2.5.2, SciPy 1.18.0, scikit-learn 1.9.0, joblib 1.5.3, and the published
optimized cheng338/dwd 1.3.5 dependency with its compiled Windows helper.
Imports are isolated from older ensemble installations in that environment.
Native numerical threads are limited to one for the functional tests.

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

This does not establish a universal accuracy or speed advantage. The SVM
variant differs from the dissertation's unsigned selection rule; those
historical results are not interchangeable. All final SVC fits completed;
the study's DWD fits reached their configured 100-iteration cap and did not
claim numerical convergence.

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
Chang Cheng's authorship, private research status and undecided software
license remain unchanged.
