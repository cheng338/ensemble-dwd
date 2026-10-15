# Changelog

This package originated in Chang Cheng's ensemble DWD research scripts
accompanying his dissertation.

## 0.3.9 — 2026-10-15

- Pair this release with DWD 1.3.14.
- End maintenance of the separate Python 3.10 companion. Python 3.11 or later
  remains required; historical release assets and validation records are retained.
- Allow optional adaptive residual ordering with restart acceleration when
  paired with DWD 1.3.14. Preserve `refinement_first` as
  the default, all numerical acceptance checks and the existing cache separation
  between policies. Keep adaptive-check counters in base-learner diagnostics.
- Preserve distinct represented lambda/gamma values in the legacy local
  refinement helper. Deduplicate exact coordinates only, retaining adjacent
  floating-point candidates and their original parent points. This can restore
  candidates formerly omitted by decimal rounding; the adaptive workflow's
  existing exact identities are unchanged.
- Report final selected-set class composition and held-out balanced accuracy
  and per-class recall from the existing predictions. Numerical summaries cover
  all base and final learners, distinguish current work from cached or recovered
  history, and avoid adding overlapping residual-check timers. Candidate
  selection continues to use ordinary accuracy. Final class reports use the
  checkpoint's owned training labels even if the caller changes its array.
- Separate final pipeline fitting, prediction and scoring intervals in the
  adaptive examples while preserving the meaning of existing elapsed-time
  fields.
- Expose the existing SVC kernel-cache allowance as `--cache-size` in the SVM
  tuning example, retaining the 200 MiB default. Display its concurrent cache
  allowance separately from other memory needs and from base-fit disk caching.
  The chosen value remains part of the estimator and checkpoint identity.

## 0.3.8 — 2026-10-09

- Pair this release with DWD 1.3.13. Keep Python 3.11 or later as the
  supported range. The Python 3.10 companion remains at ensemble-dwd 0.3.7
  with DWD 1.3.12 and does not include the changes below.

- Recover valid geometric boundary expansions that were refused because
  logarithms rounded together. Keep every previously successful coordinate,
  the factor-of-ten step cap and genuine representability limits. Apply the
  fallback to adaptive and optional ratio-refinement examples without merging
  distinct parameters. Changed workflow sources require a fresh checkpoint
  directory under the existing identity checks.
- Add optional `ensemble_dwd.profiling.fit_profile()` for binary ensemble-DWD
  fit stages, including base work, scoring, selection, cache reuse and the
  final fit. Distinguish inclusive and self times and completed or aborted
  calls. Profiles remain outside fitted models and cached base results; the
  estimator parameters, sampling and numerical methods stay unchanged. See
  the [profiling guide](docs/method.md#optional-fit-profiling).
- Preserve bounded ensemble failure context: stage, base position, completed
  fit and initial scoring calls, reused bases, and available class counts.
  Carry these facts through OvO wrapping and example failure receipts while
  preserving exception types, cleanup, sampling, selection and numerical work.
  Missing work counts remain unknown; cached work is separate from current work.
- Disclose newly introduced close but distinct search coordinates in adaptive
  example stage plans. Distinguish coordinate pairs from same-context candidate
  pairs, bound sampled details and whole-candidate comparisons, and suppress
  repeated notices when the same stage report is available. Keep all parameter
  values, candidate identities, fitted evaluations and search decisions intact.
- Report planned learner counts and remaining candidate allowance before each
  adaptive example stage. Save refused coefficient and ratio proposals with
  their reasons; large grid counts remain explicitly labeled lower bounds when
  full enumeration would be costly. Keep planned work separate from completed
  work and reuse, preserve exact candidate identities, and leave search rules
  and estimator mathematics unchanged. Supplementary report failures do not
  change search decisions. Existing workflow identity checks remain strict.
- Add optional completed-pair recovery to checkpointed multiclass DWD
  examples, disabled by default. Verified local models can complete an
  interrupted fold without refitting its finished pairs. Reuse preserves
  the existing fitting and prediction methods, checks the saved identity
  and fitted state, and reclaims pair files only after the completed fold
  receipt is verified. Reports separate current pair work from restored
  history. Storage limits and optional write failures preserve successful
  computations; unfinished binary fits and final refits still restart.
- Recover geometric refinement midpoints when logarithm/exponential rounding
  loses an available interior parameter value. Existing successful coordinates
  are unchanged, and adjacent floats still have no interior midpoint. This
  corrects premature refinement refusal in narrow grids; estimator mathematics
  and search acceptance rules are unchanged.
- Preserve existing exception notes, including the OvO class pair, in example
  failure records and failed checkpoint attempts. Bound the saved notes and
  mark truncation while retaining the original exception type, message and
  cause. Fitting and selection are unchanged.
- Correct the dependency-construction error message for optional numerical
  settings. It no longer claims the pinned DWD wheel lacks supported options;
  the original exception remains chained for diagnosis.
- Record correct-prediction and validation-row counts in checkpointed example
  folds. Aggregate accuracy with exact fractions, retaining equal seed and
  fold weights and the existing candidate-order tie rule. Use the same
  evidence for adaptive ranking, stable-score checks and numerical
  confirmation; keep arbitrary scalar scores and missing older evidence
  separate from exact count comparisons. Updated checkpoint identities require
  a fresh run directory.
- Extend compact learner diagnostics with the reported final objective,
  iteration attempts, solver/recovery summaries and distinct stopping flags.
  Example records also retain sampled/selected row set and order hashes and,
  when training labels are available, class counts. Missing older facts remain
  unknown. No predictions, solver steps or convergence claims are added.

## 0.3.7 — 2026-10-06

- Update the paired dependency to DWD 1.3.12 and refresh its validated
  native and portable wheel hashes. Eligible dense compensated score
  sums use DWD's compiled implementation on Windows CPython 3.12 x86-64;
  unsupported inputs and runtimes retain the Python calculation.
- Preserve the ensemble estimators, examples, defaults and mathematical
  methods. This release changes the dependency and release metadata only.
  The separate CPython 3.10 companion retains its existing compatibility
  overlay and uses the portable DWD companion.

## 0.3.6 — 2026-10-06

- Pair this release with DWD 1.3.11. Preserve Python 3.11 or later as the
  supported range, all estimator defaults and the mathematical methods.
- Update the legacy interruption test to check the final-runtime metadata
  added by the optional phase-thread control; runtime behavior is unchanged.

- Add optional `final_native_threads=1` to the tuning workflows and
  `--final-native-threads` to all three tuning commands. A wider positive
  integer budget applies only to the selected full-training pipeline fit;
  CV and held-out prediction keep one native thread, and CV `--jobs` still
  defaults to 1. Record nondefault phase runtime and restoration, distinguish
  the budget and helper source in checkpoint identity, and require explicit
  sequential nested estimators. Keep core estimators, search grids and
  defaults unchanged. Native rounding may change results; this is an optional
  resource control, not an automatic policy or universal speed improvement.
- Add opt-in `exact_recovery='extended'` to every DWD base learner and final
  refit, including through `OvOClassifier`. The extended exact-rank and integer-bit
  limits are 32 and 8192; the default remains 16 and 4096. Preserve the existing
  entry/operation caps, original kernel, recovery algebra and acceptance checks.
  The option covers additional small difficult kernels but is not a speedup
  or convergence guarantee. Retain fitted provenance and separate cache/CV
  identities. It requires DWD 1.3.11. See
  [limits and evidence](docs/method.md#optional-extended-exact-recovery).
- Add opt-in `affine_computation='joint'` to every DWD base learner and final
  refit, including through `OvOClassifier`. The standard default is unchanged.
  Joint scoring refines the complete `K @ alpha + b` expression when needed,
  addressing demonstrated intercept-cancellation rounding. It preserves the
  objective and ensemble method but may change rounded scores, selected rows
  and labels, and can cost more. Preserve the fitted policy through parameter
  changes and serialization, and distinguish it in base-cache and CV identity.
  The option requires DWD 1.3.11. See
  [scope and measured cost](docs/method.md#optional-joint-affine-scoring).
- Add opt-in `rbf_computation='direct'` and forward it to every DWD base
  learner and final refit, including through `OvOClassifier`. Keep
  `'standard'` as the default and identify the option in reusable base-stage
  and CV checkpoints. Direct computation requires DWD 1.3.11. It addresses
  demonstrated RBF numerical inconsistencies at additional computational cost.
  Sampling, averaging raw scores, ranking by absolute consensus score and
  final fitting are unchanged, although corrected scores can change selected
  rows and predictions. See the [scope and costs](docs/method.md#optional-direct-rbf-computation).

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
