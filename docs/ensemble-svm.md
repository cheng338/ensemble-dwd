# Ensemble kernel SVM comparator

`EnsembleKernelSVM` provides a binary RBF SVM comparison method alongside Chang
Cheng's ensemble kernel DWD. It uses scikit-learn's compiled LIBSVM `SVC`
implementation for every base learner and the final fit. No new numerical
dependency is required.

`OvOClassifier(EnsembleKernelSVM(...))` provides multiclass classification:
each class pair trains its own binary ensemble, and the pair's final SVC
casts one vote. The maximum-wins rule matches the ensemble DWD wrapper.
See [multiclass use](#multiclass-use) for an example and
[the ensemble DWD multiclass guide](multiclass.md) for the shared voting design.

## Method

1. Shuffle the training rows once and divide the first
   `M * max(2, floor(theta * n))` indices into `M` disjoint subsets of equal
   size. This is the same label-independent sampler as ensemble DWD. Every
   block must contain both classes; an ineligible draw is reported without
   resampling. See [sampling constraints](method.md#sampling-and-invalid-selections).
2. Fit one RBF SVC to each subset and calculate their arithmetic mean raw
   decision value g on every row supplied to this fit.
3. Encode the ordered binary labels as -1 and +1 and rank by
   `abs(y_signed * g - 1)`. Keep `floor(sv_selection_ratio * n)` observations,
   resolving exact ties by input row order.
4. Fit a fresh RBF SVC on the selected observations. Its predictions are the
   ensemble's predictions; the base models are used for selection only.

The signed score implements a ranking around Mao et al.'s condition
`y * g ≈ 1`. Mao's paper instead uses a squared-loss kernel coefficient fit
after selection. This package follows Mao's disjoint-subset sampling and
uses a hinge-loss SVC refit. It is an SVI-inspired comparison variant, not
an exact reproduction of Mao's algorithm. The paper does not
specify the retained-fraction grid used here. [Mao et al., *Fast Kernel SVM
Training via Support Vector Identification*, ICPR 2016, Eqs. 1–9](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2016/media/files/0649.pdf).

The dissertation describes the signed condition `y * g ≈ 1` in
Section 4.2.1 and uses `abs(y_signed * g - 1)` followed by a new kernel SVM in
Section 5.2.2. This comparator's selection rule and final SVC fit match that
practical SVM procedure. Mao's final squared-loss coefficient fit remains a
different refit, as described above.

Earlier dissertation wording used the unsigned score `abs(abs(g) - 1)`.
It agrees with the signed score on correctly classified observations and can
differ on misclassified observations. Historical results obtained with that
unsigned rule must retain that label. DWD continues to select by `abs(g)`;
its method is unchanged.

Neither rule is an exact detector of the full-data SVM's support vectors.
Accuracy and time advantages require measurement. Both ensemble methods
fit base classifiers, select rows, and fit a final classifier of the same family.

## Defaults and regularization

| Parameter | SVM default | Relationship to ensemble DWD |
|---|---:|---|
| M | 5 | Same |
| theta | 0.1 | Same |
| sv_selection_ratio | 0.3 | Same |
| random_state | 42 | Same sampler and repeatable subsets |
| gamma | `"scale"` | Resolve once from this fit's input, then share |
| prediction_batch_size | 1024 | Same |
| store_base_estimators | True | Same inspectable default |
| cache_dir | None | Optional reuse of deterministic base fits |
| C | 1.0 | Native SVC convention; independently tuned |
| tol | 1e-3 | LIBSVM stopping tolerance |
| max_iter | -1 | No iteration cap; standard SVC default |
| shrinking | True | LIBSVM optimization heuristic |
| cache_size | 200 MB | Per-active-SVC kernel cache |

SVC minimizes `0.5 * ||f||_H**2 + C * sum(hinge_loss)`, with a separate,
unregularized intercept. The same numerical C is passed to every base and
final fit. There is no automatic rescaling
by sample size. Equivalently, the mean-loss form has squared-norm coefficient
`1 / (2 * C * n_fit)`. Thus C is not interchangeable with DWD's `lambd`, and
sharing C across different-sized subsets is different from sharing a
mean-loss norm coefficient. [SVC mathematical formulation](https://scikit-learn.org/stable/modules/svm.html#svc).

`gamma="scale"` is calculated from the full input of the current binary fit,
which must be a training fold during CV. It is not recalculated on the sampled
or selected subsets. In OvO, it resolves separately in each binary pair;
specify a numerical gamma to share it across pairs. The first SVM release
supports dense RBF feature inputs; precomputed and other kernels are excluded
from this estimator's contract.

## External tuning and matched budgets

`mnist_svm_param_grid(prefix="")` returns fresh lists:

| Parameter | Candidates | Count |
|---|---|---:|
| C | `10**e`, e = -4, -2, 0, 2, 4 | 5 |
| gamma | `10**e`, e = -8, -6, -4, -2, 0 | 5 |
| theta | 0.1 (fixed) | 1 |
| sv_selection_ratio | 0.1, 0.15, 0.2, 0.25, 0.3 | 5 |

M remains fixed at the estimator default 5 unless explicitly changed. Both
presets have 125 coarse candidates and 625 ensemble fits with five-fold CV.
`broad=True` restores the historical 405-candidate grid: every integer C
exponent from -4 through 4 and gamma exponent from -8 through 0, with the
same fractions and outer endpoints. These are starting grids, not universal
optima or runtime guarantees. C and DWD lambda need separate tuning; equal
candidate counts do not imply equal regularization or wall-clock cost.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ensemble_dwd import EnsembleKernelSVM, mnist_svm_param_grid

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("ensemble", EnsembleKernelSVM(
        cache_dir=".ensemble-svm-cache", store_base_estimators=False,
    )),
])
search = GridSearchCV(
    pipeline, mnist_svm_param_grid("ensemble__"),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=2, pre_dispatch=2, error_score="raise",
)
# Supply only training data. Keep the final test set held out.
# search.fit(X_train, y_train)
```

This example stops on all fit errors. A broad grid may contain
ineligible single-class selections; the checkpointed example below can
exclude those specific candidates while allowing unexpected errors to stop
the run. Setting ordinary GridSearchCV's `error_score` to NaN alone would
also hide unrelated fit errors, so it is not the same policy.

The helper and estimator never run internal CV. The default CLI in
`examples/tune_mnist_svm.py` uses the shared external adaptive controller and
`CheckpointedCV`, with `regularization='C'` and prefix `ensemble__`. It expands
best/tied-best C/gamma boundaries, including new cross-combinations. Each
outward step is at most a factor of 10 or the smaller adjacent logarithmic
spacing. Repeated joint local refinement targets the top three distinct
C/gamma pairs within the explored bounds. All stages use the same five folds
and preserve earlier results. Sampling stays at 10% and M at five. Retention
stays on its five choices unless optional ratio refinement is enabled.

The initial grid remains 125 candidates, with three expansion rounds and
500 total unique candidates by default. Refinement requires valid bilateral
conditional neighbors within a factor of two and two completed stable local
rounds. Missing or invalid support cannot establish an interior winner.
Existing-coordinate repairs earn no stable-round credit. The full-strip
plateau rule can waive a distant tied outer edge after two complete
nonimproving outward levels with valid same-slice evidence; it cannot waive
the selected candidate's own bracket and is not a global-optimum certificate.
Improvement tolerance remains zero and boundary tie tolerance `1e-12`.

Use `--max-expansions`, `--max-candidates`, or `--broad-grid` explicitly.
`--no-refine` skips local rounds and factor-two tightening but still requires
valid conditional neighbors; `--refine` remains accepted. If a complete
proposed stage exceeds the remaining budget, none of that stage is evaluated.
An unresolved search records `needs_review` and preserves CV evidence without
final fitting or test scoring. These rules do not change SVC's objective
or its numerical controls.

`--refine-ratio` optionally follows the accepted C/gamma search with bounded
local retention-ratio refinement. The same complete-candidate, conditional
neighbor and budget checks apply; the SVM signed-margin selection rule and
final SVC fit remain unchanged. Defaults are three ratio rounds, two stable
rounds, an absolute ratio tolerance of `0.01`, and the initial ratio range
`[0.1, 0.3]`. Use `--ratio-bounds LOW HIGH`, `--max-ratio-rounds N`, and
`--ratio-tol X` to change those limits. These flags require `--refine-ratio`,
which is incompatible with `--no-refine`. See
[selection-ratio refinement](method.md#selection-ratio-refinement) for the
acceptance and resume rules, including acceptance restricted to the declared
range when a selected ratio is at an endpoint.

The default invocation previews costs without loading data. `--run` loads a
local archive while preserving a supplied test split. Final fitting follows
an accepted saved review, then held-out scoring. Each completed fold fit and
the final refit save compact diagnostics. Use `--resume --run` with the same
output directory, settings (including policy, budgets and jobs), data, folds
and immutable numerical runtime to reuse completed folds. Unfinished fits,
including an interrupted final fit, restart. If the completion receipt has
already been committed, a verified resume performs no fitting or scoring and
returns `model=None`. A crash before that receipt is committed can require
the final fit and scoring again. `--no-cache` disables base-fit reuse, not
fold checkpoints. See the
[shared workflow and records](method.md#validation-and-proposed-search-space).

`work-summary.json` separates this invocation's new CV evaluations,
equivalent-ratio reuse and checkpoint replay, plus final fitting and scoring.
It reports observed base-cache outcomes separately from missing metadata and
keeps worker timing sums distinct from elapsed time and stored receipt timings.
See [work and timing summaries](method.md#work-and-timing-summaries) for the
completed-replay and interruption rules.

`--reuse-equivalent-folds` optionally shares completed CV results between
retention ratios that select identical rows. It preserves every candidate
and the separate full-training refit; see the
[reuse conditions](method.md#equivalent-fold-reuse).

Use `--cv-seeds 42 43 44` to evaluate each candidate over the same prescribed
ensemble sampling seeds. Scores are averaged across seeds within each fold,
then across folds; all required scores must be valid. The single-seed default
and SVC settings remain unchanged. The seed count multiplies CV work, while
the candidate budget still counts parameter sets. Final fitting uses one
separate prescribed seed, set by `--refit-seed` or the configured default, 42.
See [multiple CV seeds](method.md#multiple-cv-seeds) for checkpoint, timing and
resume rules.

## Multiclass use

`OvOClassifier(EnsembleKernelSVM(...))` applies the same pairwise training and
maximum-wins voting to Chang Cheng's SVM implementation, which adapts the
support-vector-identification idea of Mao et al. (2016). Each pair selects
rows by `abs(y_signed * mean_score - 1)` and fits a final hinge-loss `SVC`.
The base subsets are disjoint within each pair. The final SVC distinguishes
the implementation from Mao's squared-loss kernel-coefficient refit; see
[the method and attribution above](#method).

Only each pair's final SVC votes, using SVC's native `predict` behavior.
The DWD-specific exactly-zero-margin convention should not be applied to
SVC. Equal total vote counts still select the smallest sorted class label.
For `M=5`, an uncached ten-class SVM fit likewise comprises 45 ensembles and
270 SVC fits. SVM uses native `C`, `tol` and `max_iter` conventions; it does
not use DWD's `lambd`, loss power or MM stopping policies.

One external CV candidate shares its parameter configuration across all
class pairs. Within each pair, the same numerical `C` is passed to all base
and final learners without sample-size rescaling. A numerical `gamma` gives
a common coefficient across pairs; `gamma="scale"` resolves separately for
each pair before sampling.

Use `mnist_svm_param_grid(prefix='ovo__estimator__')` for the nested estimator.
This constructs a training-only search; it does not launch it:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ensemble_dwd import EnsembleKernelSVM, OvOClassifier, mnist_svm_param_grid

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('ovo', OvOClassifier(EnsembleKernelSVM(
        M=5, random_state=42, store_base_estimators=False,
        cache_dir='.ensemble-svm-cache',
    ), n_jobs=1)),
])
search = GridSearchCV(
    pipeline, mnist_svm_param_grid(prefix='ovo__estimator__'),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy', n_jobs=1, pre_dispatch=1, error_score='raise',
)
# search.fit(X_train, y_train)  # Keep the final test set held out.
```

The default preset has 125 candidates; `broad=True` restores 405. Either can
be expensive across all class pairs. This example stops on any fit error.
As with DWD, an ineligible pair selection invalidates the complete multiclass
candidate.
Parameter tuning and any search-boundary review remain external to the
estimators. The checkpointed multiclass command-line example currently
demonstrates DWD; the code above shows the SVM composition.

## Parallel work, caching and diagnostics

Base fits are sequential inside one binary estimator. Parallelize external
GridSearchCV candidates/folds, or use `OvOClassifier(..., n_jobs=...)` for
pair-level parallelism in a standalone multiclass fit. Avoid running both
levels with unrestricted worker counts. A single LIBSVM SVC does not expose
an n_jobs setting; additional BLAS threads do not parallelize that solver.

The tuning command shares optional `--final-native-threads N` with the DWD
examples. It defaults to 1 and applies only to the selected full-training
pipeline fit; CV workers and held-out prediction retain one native thread.
`--jobs` separately controls concurrent CV work and defaults to 1. The native
setting limits each detected BLAS/OpenMP pool, not total CPU use or physical
core placement, and is not an SVC speedup claim. See
[native threads by phase](method.md#native-threads-by-phase) for nested-worker,
runtime identity and resume restrictions.

Changing only the selection ratio can reuse the same base models and margins
through the process-safe disk cache. The final SVC is always fitted anew.
Cache identity includes the training rows/labels, sampled rows, SVM settings
and runtime/source identity; it cannot share DWD results or results from a
different CV fold. A custom subclass or random_state=None bypasses caching.
The optional disk cache and SVC's in-memory `cache_size` are distinct.

Budget worker memory for feature copies, retained models, scoring buffers
and each active SVC's cache. `store_base_estimators=False` reduces retained
models while preserving selection and diagnostic arrays. Prediction batches
limit transient query storage. `prediction_batch_size` accepts `None` or a
positive integer and can be changed after fitting. Both `predict` and
`decision_function` revalidate it before allocating output or calling the
final SVC. Invalid values raise `ValueError` without invalidating the fitted
model; set a valid value to resume prediction.

`n_iter_`, `fit_status_` and `converged_` record each of the M base fits and
the final fit, with the final fit last. `support_counts_` and `n_support_`
record support-vector counts. Standard SVC convergence warnings remain
visible. A user-specified iteration cap may return an unconverged classifier;
it must not be reported as a converged result. Cached unconverged base fits
must retain their diagnostics and warnings.

An insufficient or single-class final selection raises `IneligibleSelectionError`,
as it does for DWD. An infeasible disjoint allocation or a one-class base block
raises its subclass `IneligibleSubsetError` before base fitting. No extra points are silently inserted and no different
classifier is substituted. Failed refits clear the fitted state. The final
SVC's own predict method is used, including its behavior at an exact zero
decision value.
