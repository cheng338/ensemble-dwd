# Private ensemble-dwd 0.2.0 validation

The multiclass extension of Chang Cheng's ensemble kernel DWD is implemented
and tested under `C:\Users\manea\.conda\envs\py12_df\python.exe`, using
the installed optimized `cheng338/dwd` 1.3.4 compiled Windows wheel.
The new wrapper composes the existing binary estimator; the binary estimator,
cache implementation and parameter-grid helper are byte-identical to 0.1.1.
The source reference files `multi.py` and `formal2.tex` were not modified.

## Changes and audit

- Added `OvOClassifier`, cloning the binary estimator once per class pair.
- Preserved the dissertation's maximum-wins voting and the script's smallest
  sorted label tie rule, including its explicit pair-label orientation.
- Kept the practical kernel DWD objective and its unregularized intercept.
- Corrected multiclass sklearn integration, retained original label dtypes,
  bounded pair dispatch, batched vote accumulation, and cleared failed fits.
- Reused the binary package's reproducible sampling, valid M=1 handling,
  optional cached base stage, lean storage and explicit ineligible selections.
- Added a local-archive MNIST example with fold-local preprocessing and
  external GridSearchCV, shared pair parameters and durable result records.
- Updated dependency metadata from an older fork commit to hash-pinned
  optimized cheng338/dwd 1.3.4 release wheels.

See [the detailed method and original-script audit](multiclass.md).
Private status, Chang Cheng's authorship and the undecided license remain.

## Tests

All 117 unittest methods passed against the isolated installed wheel in py12_df:
the existing 85 tests plus 19 multiclass estimator tests and 13 example tests. These cover
45 pairwise fits, pair isolation, binary label orientation, zero margins,
cyclic vote ties, string and large-integer labels, feature validation,
failed-refit cleanup, serial/parallel parity, nested GridSearchCV parameters,
stratification, fold-local scaling, caching across retention ratios and gamma
resolution and near-full MNIST subsetting. A test runs sklearn's estimator
contract checks as well. The original source test run passed 114 tests;
the final wheel run includes three additional near-full subsetting regressions.

Wheel checks and their log are recorded separately with the built artifacts.
Cross-platform CI is configured, but was not remotely run for this local build.
Historical 0.1.1 validation is retained in docs/validation-0.1.1.md.

## All-ten-digit MNIST validation

Used the official local MNIST archive, SHA-256
`731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
Training used a stratified 6,000-row subset of the 60,000 official training
images (10%, seed 42); evaluation used all 10,000 official test images.
The splits were never pooled. All ten digits are present in both subsets.
Images are flattened to 784 float64 values. StandardScaler is fitted within
each training fold, then once on the selected full training subset for refit.

The explicit validation search used three shuffled stratified folds, seed 42,
and accuracy scoring through external GridSearchCV. The eight combinations were:

| Parameter | Values searched |
|---|---|
| Package `lambd` | 2^-24, 2^-18 |
| RBF `gamma` | 0.0001, 0.001 |
| Base sampling `theta` | 0.1, 0.2 |
| Final retention `sv_selection_ratio` | 0.3 (fixed) |

Fixed settings: M=5, q=1, kernel=rbf, random_state=42, max_iter=100,
stopping=objective, obj_tol=1e-5, tol=1e-6, acceleration=None,
prediction_batch_size=1024, store_base_estimators=False. All pairs and all
their base/final fits share the candidate's numeric gamma and DWD settings.
The default broader 810-candidate preset is still available; this compact
validation did not determine globally optimal hyperparameters.

All 24 CV fits were successful and finite. The selected parameters were
`lambd=2^-18`, `gamma=0.001`, `theta=0.2`, `sv_selection_ratio=0.3`.
Mean CV accuracy was 92.0667%. Held-out accuracy was **94.11%**, or
9,411 correct predictions out of 10,000. The final model contains 45 binary
ensembles, comprising 225 base fits and 45 selected-set final fits.

A direct transcription of the script's stored-vote/mode prediction rule
matched the new implementation on 1,000 test rows using the same fitted
children. A fresh uncached eight-worker refit reproduced every pair's sampled
rows, consensus scores and selected rows, and all 10,000 test predictions.
No missing pair, numerical rejection or ineligible selection occurred in
these runs. This does not establish that every possible grid input is eligible.

All 270 DWD fits in the selected model reached 100 updates and reported
`converged_=False`. The agreed stopping defaults were preserved. These results
validate classification and integration, not convergence or optimality of the
underlying DWD fits. They also do not establish a speed or accuracy improvement
over the historical dissertation table or the original script under identical
conditions; no such paired comparison was performed here.

## Runtime and resource settings

Python 3.12.14; NumPy 2.5.2; SciPy 1.18.0; scikit-learn 1.9.0;
MKL runtime 2025.3-Product. The host exposes 32 logical processors and about
92.7 GiB physical RAM. CV used 16 outer workers, one pair at a time per
worker and one native thread per worker. The selected refit initially used
one pair worker and one native thread. The parity refit used eight pair
workers with one native thread each. Prediction was measured with one native
thread. These are allocation limits, not CPU-utilization measurements.

Observed single-run durations:

| Operation | Seconds |
|---|---:|
| Eight-candidate, three-fold CV | 20.305 |
| Selected sequential refit | 7.481 |
| Fresh eight-pair-worker refit | 2.220 |
| Prediction of all 10,000 test images | 5.527 |

Timings are validation-run observations, not a controlled repeated speedup
benchmark. Tests may overlap part of the search/refit execution. The sequential
refit enables the cache (its 45 full-training keys miss); the parity refit
disables it. Resource receipts contain available-memory snapshots and thread
limits, not continuous peak-memory or CPU-percent telemetry. No OOM occurred.

Complete CV scores, exact train/fold indices, confusion matrix, predictions,
runtime configuration and each pair's solver diagnostics are saved alongside
the release artifacts. Full 60,000-row training and broad multiclass tuning
were not run for this integration validation.
