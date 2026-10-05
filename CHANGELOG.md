# Changelog

This package originated in Chang Cheng's ensemble DWD research scripts
accompanying his dissertation.

## 0.3.5 — 2026-10-05

- Pair this release with DWD 1.3.10 and update installation and dependency
  metadata. Preserve the ensemble method and estimator defaults.
- Add a current-invocation work summary to the three adaptive tuning examples.
  Separate executed fold evaluations, equivalent-ratio reuse and checkpoint
  replay, observed base-cache outcomes, stored receipt timings and elapsed
  time. Report search, optional numerical confirmation and final fitting
  separately without changing existing timing fields or fitting behavior.
- Add optional confirmation at a larger DWD iteration cap to the binary and
  multiclass tuning examples. Repeat a fixed shortlist on the same folds and
  seeds, retain separate checkpoints and diagnostics, and hold final fitting
  if confirmation changes the ranking or is incomplete. Keep the original
  search and final-fit settings unchanged.
- Add optional bounded retention-ratio refinement to the three tuning examples.
  Refine nearby ratios and coefficient combinations in complete rounds, retain
  the shared candidate budget, and hold final fitting on unresolved searches.
  Keep the initial grids, default search behavior and estimator methods unchanged.
- Add optional multi-seed CV to the three tuning examples, averaging seeds
  within each fold and preserving individual checkpoints and diagnostics.
  Keep the single-seed default and use one prescribed seed for final fitting.
- Preserve parameter types in example CV checkpoint identities, including NumPy
  scalars and list/tuple distinctions, to prevent reuse across different fits.
  Version the checkpoint format; old run directories require a fresh run.
- Support long Windows paths for example run directories and public status
  and result files, including atomic temporary files.
- Include class-pair labels and each binary estimator's diagnostics in
  multiclass cross-validation and final-fit records.
- Add optional reuse of completed example CV folds for retention ratios
  that select identical rows, preserving candidate records and final refitting.
- Correct the SVM documentation to reflect the latest dissertation
  methodology: signed-margin selection followed by a fresh hinge-loss SVC.
  Distinguish that practical comparator from Mao's squared-loss final fit;
  the implemented selection rule and refit are unchanged.

## 0.3.4 — 2026-10-03

- Release the validated development changes as version 0.3.4, paired with
  DWD 1.3.9. Retain the ensemble method and the unchanged estimator defaults.
- Correct the cache-verification callback test and canonicalize fixture paths
  so cache-identity tests exercise the intended conditions on Windows and Linux.
- Clarify the introduction's origin in the earlier dissertation scripts and
  use a release-specific, hash-locked DWD wheel source in CI.

- Add the `base-stage-v2` cache identity, including package source
  and native artifact hashes, actual accelerator availability and numerical
  runtime provenance. Keep old cache entries separate and bypass reuse when
  identity is unavailable or observably changed.
- Include content identity for loaded BLAS/OpenMP and scientific native
  extensions, plus an already loaded AOCL bridge. Recheck runtime identity
  before cache reuse and publication; uncertain results remain uncached.
- Retain compact `base_diagnostics_` records through cache hits and after base
  models are discarded, distinguishing reported stopping and optimality facts.
- Forward opt-in DWD `residual_check_order='adaptive'` to every base and final
  fit and include it in cache identity. Keep `'refinement_first'` as the default;
  ensemble sampling, scoring, selection and final-refit rules are unchanged.
- Require valid conditional neighbors for adaptive boundary review, repeat
  joint local refinement, and require a factor-two winner bracket plus two
  completed stable local rounds by default. Scope the empirical full-strip
  plateau rule to tested evidence; hold final fitting on unresolved budgets.
- Add explicit completed-fold resume to the tuning CLIs under matching data,
  folds, settings, policy/budgets and runtime identity. Save compact fold and
  final-fit diagnostics. Interrupted final fits restart; committed completed
  results resume without fitting or scoring and do not restore a predictor.
- Support long Windows paths for internal checkpoint files and locks.
- Support long Windows paths for reusable base-cache payloads, metadata and
  locks, including atomic temporary files.
- Clear partial learned state after `KeyboardInterrupt` and `SystemExit` in
  binary and one-versus-one fits, then propagate the original interruption.
- Preserve all parent parameters and any better candidate from the coarse stage
  in the legacy nonresumable coarse/local helper. Keep review flags consistent
  with final fitting and scoring, and record interruption separately from failure.
- Reject complex feature and label arrays, nonfinite features and empty feature
  dimensions in binary MNIST data helpers before CV, including supplied test splits.
- Use an exact DWD version requirement
  and a relocatable SHA-256 lock for the validated native and portable wheels.

## 0.3.3

- Pin the validated cheng338/dwd 1.3.8 native and portable builds by hash.
- Provide smaller initial parameter grids and explicit boundary expansion for
  regularization and gamma. Keep M=5, 10% disjoint sampling and retention choices fixed.
- Preserve fold-local selection and shared learner parameters in adaptive CV.
- Document the justification, tests and limits of each major change in the
  [release validation](docs/validation-0.3.3.md).


## 0.3.2

- Use one uniform permutation to form disjoint base subsets for both ensemble
  classifiers. Validate capacity and binary-class eligibility before fitting;
  report ineligible subsets through `IneligibleSubsetError` without redraws.
  Existing overlapping-sampling results remain historical comparisons.
- Added a shared `backend` parameter to `EnsembleKernelDWD`, forwarded to
  every base and final DWD fit, including each multiclass pair. Supported
  choices are `auto`, `spectral`, `cholesky`, and `lbfgs`; `auto` remains the
  default. Different backend choices use separate base-cache entries.
- Fix the MNIST presets at 10% base sampling, retaining nine regularization
  values, nine gamma values and five retention fractions (405 candidates).
- Updated the hash-pinned optimized DWD dependency to cheng338/dwd 1.3.7.

## 0.3.0

- Added `EnsembleKernelSVM`, using optimized scikit-learn SVC base learners,
  label-aware consensus selection `abs(y * mean_score - 1)`, and a final SVC
  fit on the selected observations. Both stages retain a free intercept.
- Validate SVM prediction batch sizes on each prediction call, including
  after changing parameters on a fitted estimator. Invalid sizes raise an
  error instead of allowing an empty loop to return uninitialized output.
- Aligned M, sampling, retention, random-state and storage defaults with
  ensemble DWD. Native SVC C and the resolved RBF coefficient are shared by
  the base and final fits.
- Added safe reuse of SVM base fits across selection ratios, batched scoring,
  convergence diagnostics, and support for the existing OvO wrapper and
  external GridSearchCV.
- Added `mnist_svm_param_grid`: nine C values, nine gamma values, two sampling
  rates and five selection ratios, matching the DWD preset's 810 candidates.
- Added a review before final fitting and held-out scoring in the binary
  tuning examples when the CV winner is at a regularization or gamma boundary.
  CV and selection records remain available for an expanded training-only search.
- Clarified expected ineligible selections in the tuning documentation and
  included the changelog in the source distribution.
- Documented the SVI-inspired comparator, including its differences from
  the unsigned margin-selection criterion and Mao's squared-loss refit.

The DWD estimator, its search preset and the pinned cheng338/dwd 1.3.5
dependency retain their existing definitions.

## 0.2.1

- Fixed a Windows race when concurrent workers first opened the same cache
  lock file. Lock files now remain empty; the operating-system lock is acquired
  before any cache data is read or written. This avoids an initialization write
  conflicting with another worker's already acquired byte-range lock.
- Existing one-byte lock files remain compatible. Process exit continues to
  release the operating-system lock, including after an interrupted worker.
- Added real Windows multiprocessing regressions, including 32 concurrent
  workers sharing a new cache entry.
- Updated the hash-pinned optimized DWD dependency to cheng338/dwd 1.3.5,
  including its numerical recovery for nearly constant kernel matrices.
- Corrected the local-wheel installation example and clarified that
  gamma='scale' resolves separately for each binary class pair.
- Added installation-metadata regressions for the ensemble wheel version and
  matching compiled/portable DWD dependency pins.

The ensemble's sampling, consensus selection, final refit, voting, stopping
defaults and parameter grids are unchanged. The cache synchronization repair
does not change numerical results. The DWD dependency update improves numerical
robustness while preserving the objective and unregularized intercept.
