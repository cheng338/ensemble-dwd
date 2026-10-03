# Three-pair MNIST validation

This completed comparison used 20% of the official training rows for each
pair, selected reproducibly with seed 42. It used the full official test rows
for those digits only after training-only CV and parameter selection.
Images were flattened to 784 float64 features, and StandardScaler was fitted
inside each training fold. Five stratified folds were shared across classifiers.
The official test set was excluded from this run's tuning, but had been
evaluated in earlier development experiments.

| Pair | Training rows | Official test rows |
|---|---:|---:|
| 2 vs 3 | 2,417 | 2,042 |
| 4 vs 9 | 2,358 | 1,991 |
| 3 vs 8 | 2,396 | 1,984 |

The ensemble SVM used the signed selection criterion documented in
[the SVM guide](ensemble-svm.md). M was fixed at 5 for both ensembles,
sampling fractions were 0.10 and 0.20, and retention fractions were 0.10,
0.15, 0.20, 0.25 and 0.30. Each classifier's regularization and kernel
coefficient were tuned independently. Both ensemble presets started with
nine regularization and nine gamma values, as listed in the package guides.

The external runner extended any winning C, lambda or gamma endpoint and
evaluated all new combinations in the enlarged grid. C/gamma expanded by
a factor of ten; lambda used the adjacent log2 interval. This rule applied
to all five classifiers. Only ensemble SVM required expansion: C reached
1e5 in the 2-vs-3 grid and 1e6 in the 3-vs-8 grid. Every final C/lambda/gamma
winner was interior. Sampling and retention ranges stayed fixed throughout the
search. Interior status is not proof of a global optimum between grid points.

Linear SVM used `SVC(kernel='linear')`, with C values `10**e` for e = -6, ..., 2.
Kernel SVM used `SVC(kernel='rbf')` with the same initial C and gamma grids as
ensemble SVM. Kernel DWD used the same initial `lambd` and gamma grids as
ensemble DWD.

## Final test accuracy and training time

Each cell below shows accuracy (%) followed by final training time (seconds).
Times are uncached, sequential final fits with one native thread, excluding
CV and test scoring. Ensemble times include all five base fits, selection
and the final selected-set fit. Final-fit times include fitting StandardScaler
and were measured once per selected model; they are not repeated benchmark means.

| Classifier | 2 vs 3 | 4 vs 9 | 3 vs 8 |
|---|---:|---:|---:|
| Linear SVM | 97.307 / 0.127 | 96.585 / 0.121 | 96.976 / 0.104 |
| Kernel SVM | 98.139 / 0.133 | 97.087 / 0.146 | 97.429 / 0.154 |
| Ensemble kernel SVM | 97.111 / 0.276 | 95.279 / 0.321 | 96.421 / 0.424 |
| Kernel DWD | 98.629 / 0.575 | 97.388 / 0.534 | 98.034 / 0.552 |
| Ensemble kernel DWD | 98.286 / 0.389 | 97.087 / 0.377 | 97.581 / 0.382 |

CV used 16 external workers with one native thread each. The monitored run
took 49.08 minutes, with no out-of-memory event or runtime pause. Recorded
peak process-tree private memory was 3.49 GiB and resident memory 4.14 GiB;
minimum available system memory was 73.10 GiB. These are observations for
this workload, not worker-memory guarantees for larger datasets.

Across all searches, 5,643 candidates produced 28,215 CV fold outcomes:
20,144 finite scores and 8,071 ineligible selected sets. Only candidates with
five valid fold scores could win. Those ineligible selections contained one
class and were excluded explicitly; they were not numerical solver failures.
No unexpected fit exception occurred. All 24 final SVC leaf fits completed.
All 21 final DWD leaf fits reached 100 MM iterations with `converged_=False`;
no numerical-convergence claim is made. DWD used `stopping='objective'`,
`obj_tol=1e-5`, `tol=1e-6`, no acceleration, and the optimized Cholesky path.

The study used valid prediction batches of 1024, so the post-fit invalid-batch
bug did not affect its results. Release validation separately checks exact
saved-model prediction parity at multiple valid sizes. The full original
protocol and model/result hashes are retained with the experiment artifacts;
protocol SHA-256:
`4920e9c35f56430a523207145e2eada0db8c443dc77aebdccf6689f98c19f3df`.

The ensemble DWD was faster but slightly less accurate than full kernel DWD
on these three samples. Ensemble SVM was slower and less accurate than full
kernel SVM here. Different selected parameters and the additional base-stage
work preclude interpreting these times as isolated solver speed ratios.
