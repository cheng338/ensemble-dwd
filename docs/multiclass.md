# Multiclass ensemble kernel DWD

`OvOClassifier(EnsembleKernelDWD(...))` provides one-versus-one multiclass
classification using the binary ensemble DWD estimator for each class pair.
The package therefore supports both binary and multiclass classification.

## Method

Chapter 5's multiclass experiment in Chang Cheng's dissertation trains one
binary classifier for each unordered class pair. With ten classes there are
45 pairs. Each pair uses its own training rows and the four-stage binary
ensemble: sample `M` subsets, fit their kernel DWD models, select the rows
with smallest absolute mean raw score, and fit one final kernel DWD.
The final classifier of each pair casts one vote. The class with the most
votes is predicted. There is no vote among the base learners at test time.

`OvOClassifier(EnsembleKernelDWD(...))` implements this composition. It
preserves the binary estimator's objective, including its unregularized
intercept, sampling distribution, selection rule and stopping controls.
The wrapper does not change or reimplement the DWD solver. For `M=5`, a
successful uncached ten-class fit comprises 45 ensembles and 270 DWD fits.

For closest agreement with `multi.py`, each sorted pair `(a, b)` encodes
`a` as `+1` and `b` as `-1`. DWD predicts its negative encoded class at an
exactly zero margin, so an exact zero votes for `b`. Equal total win counts
select the smallest sorted original class label. The dissertation specifies
maximum wins; this explicit tie rule follows the script. sklearn's standard
`OneVsOneClassifier` uses confidence to break vote ties, so it is not an
identical replacement for this rule.

## Shared parameters and preprocessing

One external `GridSearchCV` candidate supplies shared binary parameters to
all class pairs. It is scored on the complete multiclass prediction. Individual
pairs do not run separate searches. Within each pair, all `M` base models
and the final refit share their DWD parameters. The parameters `theta` and
`sv_selection_ratio` denote the base sampling fraction and final retention
fraction respectively; both fractions are applied to that pair's row count.

Put a `StandardScaler` before the OVO estimator inside a sklearn `Pipeline`.
Each CV training fold then fits one scaler on that fold's multiclass rows.
All pairs within the fold share that feature transformation. Validation and
test rows do not participate in fitting the scaler or choosing subsets.

Use a numeric `gamma` for a common RBF coefficient across pairs. With the
convenience setting `gamma='scale'`, the binary estimator resolves it on
each pair's transformed rows, then shares it within that pair. The configured
binary random state is cloned for each pair without using global randomness.
As in the binary API, `lambda_dissertation = 2 * lambd` under the practical
mean-loss objective with the dissertation's one-half penalty convention.

The API's broad grid remains available through
`mnist_param_grid(prefix='ovo__estimator__')`. The compact grid in the example
is an execution and integration check, not a claim of optimal parameters.
The theoretical augmented-norm treatment in the dissertation remains distinct
from its practical unregularized-intercept implementation; this integration
does not supply a new justification for applying those bounds.

For an optional SVM comparison, the same wrapper accepts
`EnsembleKernelSVM`; use `mnist_svm_param_grid` with the
`ovo__estimator__` prefix. Its distinct method, attribution and usage example
are documented in [the SVM comparator guide](ensemble-svm.md#multiclass-use).

## Audit of the original script

The review used the author's `multi.py` and `formal2.tex` as reference inputs.
The original files were not edited. The following issues were addressed by
composition with the existing binary estimator and the new wrapper/example:

| Finding | Integrated behavior |
|---|---|
| Scaling before cross-validation leaks validation-fold statistics. | Fold-local scaling inside the pipeline. |
| `BaseEstimator` before `ClassifierMixin` can hide classifier tags from modern sklearn. | Classifier mixin first; classifier and stratified-CV checks. |
| Global random sampling and retry limits can train fewer than `M` learners. | Existing reproducible sampling conditional on both classes; exactly `M`. |
| Squeezing averaged scores breaks the sample dimension when `M=1`. | Existing binary estimator retains one score per row. |
| Failed selection can silently substitute the first base model or full-data model. | Pair-context failure; no substituted classifier or missing pair. |
| Float prediction matrices reject strings and can lose large integer labels. | Integer vote counts and original label dtype. |
| A failed refit can retain partially fitted pair state. | Failed fits leave the wrapper unfitted. |
| `error_score=0` masks unexpected failures. | Only expected ineligible selections may receive NaN; other errors propagate. |
| Automatic CV refit followed by manual refitting repeats work with new random subsets. | Explicit selection followed by one final refit. |
| Unrestricted CV workers can multiply memory use. | Bounded dispatch, configurable workers and one level of fit parallelism. |
| Whole-dataset random train/test splitting differs from official MNIST partitions. | Local archives retain their separate training and test arrays. |
| An `n_samples` by 45 float prediction matrix stores all pair votes. | Batched integer accumulation by class. |

The original script's additional DWSVM/nDWSVM implementations are not part of
this integration. Its lambda grid and full-data setting also differ from the
dissertation's reported multiclass experiment. Those historical scores should
not be treated as a controlled baseline for the new validation.

## Resource use and diagnostics

Pair fitting is sequential by default. Set `OvOClassifier.n_jobs` for a
standalone parallel fit, or leave it at 1 while parallelizing CV candidates.
Do not multiply both worker counts without an explicit resource budget.
The tuning example bounds dispatched jobs and limits native library work to
one thread per outer CV worker. Pair feature matrices are formed inside their
worker, rather than constructing all pair matrices before dispatch.

`predict` accumulates votes in batches. `vote_counts` returns all class vote
counts when requested. These counts are not probabilities or decision margins.
For lean retained models, set the binary estimator's
`store_base_estimators=False`. Training still uses dense kernel matrices;
these controls do not constitute an automatic out-of-memory guarantee.

After fitting, `pairs_` identifies each child in `estimators_`. To locate a
child's selected rows in the original multiclass training input, use
`model.pair_indices_[p][model.estimators_[p].selected_indices_]`.
Each child's `n_iter_` and `converged_` cover its `M` base fits followed by
its final fit. An objective-change stopping event does not certify optimality.
Ineligible selections invalidate the complete candidate and are reported with
the responsible pair. Changing the retention ratio is a new CV candidate,
not an automatic recovery action.
