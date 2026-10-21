# Private ensemble-dwd 0.1.1 validation — 2026-09-07

The audit improvements are implemented and validated using the optimized
cheng338/dwd 1.3.2 dependency. All 85 unittest methods passed against
the installed wheel under CPython 3.12 with MKL, with no failed or skipped
unittest methods.
The original 0.1.0 release artifacts remain unchanged. Validation preceded publication to the private repository.

## Changes

- Optional process-safe base-stage caching reuses the M base learners' work
  across selection ratios. Training data/order, actual subsets, effective
  kernel parameters, solver settings, source and numerical runtime identify
  the entry; final selection and fitting always run afresh.
- Per-key operating-system locks prevent duplicate concurrent computation.
  Checksums and atomic writes prevent incomplete entries from becoming hits.
  Tiny BLAS initialization stabilizes fingerprints in fresh MKL workers;
  a fresh-process regression covers the lazy-runtime-loading case.
- Optional storage without base models retains the final
  predictor, consensus, sampled/selected rows and convergence diagnostics.
- Both external GridSearchCV stages use refit=False and save completed CV
  results, exact folds and selected parameters before the explicit final refit.
- Structured failure records distinguish ineligible selections from numerical,
  memory and other unexpected failures. Only expected ineligibility can yield
  NaN; unexpected failures stop the example. Logging failures preserve the
  original fit exception. A failed winner does not trigger another classifier.
- Explicit joblib>=1.3, updated documentation and a CI job for older compatible
  dependencies. The CI configuration was parsed but not remotely executed.
- Nonconstant feature variance underflow now requests rescaling or explicit
  gamma. An exceptional finite-mean recovery handles intermediate score-sum
  overflow; ordinary averaging arithmetic is unchanged.

The sampling/selection method, objective, unregularized intercept and
stopping defaults are unchanged: objective stopping, obj_tol=1e-5,
max_iter=100 and acceleration=None. This release does not change DWD itself.

## Dependency and artifact identity

Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0,
joblib 1.5.3 and MKL 2025.3 were used for validation.
The wheel was installed in an isolated validation directory; the environment's
installed packages were not modified.

All 25 installed DWD package files match the published
cheng338/dwd 1.3.2 wheel byte for byte. Its SHA256 is
`0d181990209ef8f011559299c289c13cf92552ec593e96d5225764f51fbef42d`.
The ensemble dependency remains pinned to commit
`9ed1892d74c4eda5ed2e058df239ffb7973c120a`. Base and final learners explicitly set
implementation='optimized' and solver_mode='schur'; tests check those settings.

Artifact checksums, every wheel RECORD entry and all 21 source
files were verified. Rebuilding the source distribution produced an identical
wheel. Archives contain no interpreter caches or local workspace paths.
Authorship is Chang Cheng, with dependency attribution preserved. Licensing remains undecided and the build is private.

## MNIST result preservation and timing

Each pair used 3,000 fitting rows and 500 disjoint query rows from MNIST's
original training pool, sampled with seed 73. Official test arrays were unused.
StandardScaler was fitted only on fitting rows. M=5, theta=0.1, lambd=1e-6,
gamma=0.001, q=1 and random_state=42; five selection ratios
0.10/0.15/0.20/0.25/0.30 were evaluated. These are equivalence and timing checks,
not a new broad accuracy search.

Times below are the median over three repetitions for fitting all five ratios,
using one process and four BLAS threads. Baseline/cached ordering alternated.
Hashing, checksums, disk I/O, base scoring and final fitting are included;
loading, preprocessing and query prediction are outside the timer. Each cold
group begins with an empty cache, so its first candidate pays the cache setup.

| Pair | 0.1.0 seconds | 0.1.1 uncached seconds | 0.1.1 cold-cache seconds | Cold-cache speedup | Accuracy |
|---|---:|---:|---:|---:|---|
| 7 vs 9 | 1.002 | 0.999 | 0.445 | 2.25x | Exact match |
| 2 vs 3 | 0.988 | 0.992 | 0.443 | 2.23x | Exact match |
| 4 vs 9 | 0.988 | 1.001 | 0.452 | 2.19x | Exact match |
| 3 vs 8 | 0.979 | 0.987 | 0.437 | 2.24x | Exact match |

Every sampled subset, consensus score, selected row, query score, predicted
label, update count and convergence flag matched exactly across the compared
implementations. Accuracy was unchanged on these checks; no accuracy increase
or universal speedup is claimed. A five-ratio cold group requires 10 DWD fits
instead of 30; a fully warm group requires five fresh final fits.

Models stored without base learners were about 62.3% smaller when serialized
(approximately 15.13 MB to 5.70 MB at final ratio 0.3). Predictions after
saving and loading matched exactly. Base cache entries without retained
models were about 49 KB. These are storage measurements, not promises about
peak training RAM.

## Parallel CV and failure handling

A separate 7-vs-9 check used 2,000 training-pool rows, five stratified folds,
the same five selection ratios, two outer workers and one BLAS thread per
worker. Pipeline preprocessing remained inside each fold. Workers were started
before timing; cold/uncached order alternated over three repetitions.
All 25 validation scores matched exactly in every mode. Median times:
uncached 2.049 s; cold shared cache 1.148 s;
warm shared cache 0.848 s. Exactly five base-stage entries
were produced per cold search, one for each fold. This compares 0.1.1 with
and without caching; the four-pair table separately uses the 0.1.0 wheel.
The first parallel cold-cache run took 2.277 s; these medians do not imply
that every small run is faster.

The basic example passed. The real two-stage example completed with two
workers, one coarse and nine refined candidates over five folds on 60 synthetic
observations (45 training, 15 held out). This was a workflow smoke test.
Injected-failure tests separately verified five successful CV folds followed
by a failed full-training refit: CV/selection records survived and the test
set was not scored. All-ineligible CV results also survived. Numerical,
memory and prediction failures propagated with structured causes. Three
independent processes sharing one cache key computed its value exactly once.

## Limits

The full 810-candidate search and remote Linux/Python-version CI matrix were
not run for this release. No new claim about optimal parameters, improved
accuracy, or universal solver convergence follows from these checks. Binary
dense features remain the scope. Caching is opt-in on the estimator and enabled
in the example; custom/callable or unseeded fits bypass it. Cache entries are
trusted local pickle-based files. Worker counts remain caller-controlled;
there is no automatic RAM cap or automatic interrupted-search resume.
