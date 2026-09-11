# Changelog

## 0.3.2

- Added a shared `backend` parameter to `EnsembleKernelDWD`, forwarded to
  every base and final DWD fit, including each multiclass pair. Supported
  choices are `auto`, `spectral`, `cholesky`, and `lbfgs`; `auto` remains the
  default. Different backend choices use separate base-cache entries.
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
