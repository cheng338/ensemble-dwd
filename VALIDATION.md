# ensemble-dwd validation

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

Version 0.3.7 is paired with DWD 1.3.12. See the
[0.3.7 changelog](CHANGELOG.md#037--2026-10-06) for its changes.
The records below retain their original workload and version scope; they are
not new full MNIST searches for version 0.3.7. DWD's
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
