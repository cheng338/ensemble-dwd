# ensemble-dwd 0.3.0 validation

This record describes validation of the `v0.3.0` release of ensemble-dwd,
including its binary and multiclass ensemble DWD interfaces and optional
SVM comparator. It covers prediction-batch validation and the external
binary examples' review of regularization and gamma boundary winners.
Neither estimator performs cross-validation internally.

## Environment and dependency

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

## Regression coverage

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

## Completed MNIST study and prediction checks

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

## Performance review and distribution

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
