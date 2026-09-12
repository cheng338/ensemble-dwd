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

For each sorted pair `(a, b)`, encode `a` as `+1` and `b` as `-1`. DWD predicts
its negative encoded class at an exactly zero margin, so an exact zero votes
for `b`. Equal total win counts select the smallest sorted original class label.
This makes the dissertation's maximum-wins rule explicit. scikit-learn's standard
`OneVsOneClassifier` uses confidence to break vote ties and therefore has
different tie behavior.

## Shared parameters and preprocessing

One external `GridSearchCV` candidate supplies shared binary parameters to
all class pairs. It is scored on the complete multiclass prediction. Individual
pairs do not run separate searches. Within each pair, all `M` base models
and the final refit share their DWD parameters. The parameters `theta` and
`sv_selection_ratio` denote the base sampling fraction and final retention
fraction respectively; both fractions are applied to that pair's row count.
For each pair, one permutation supplies `M` disjoint blocks of size
`k = max(2, floor(theta * N_pair))`, requiring `M * k <= N_pair`.
Every block must contain both pair labels; the sampler does not redraw an
ineligible allocation. Disjointness is enforced within each pair. The same
observation can participate in other class pairs and in the final selected set.

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

The default grid is `mnist_param_grid(prefix='ovo__estimator__')`, with
125 candidates. Passing `broad=True` restores the historical 405 candidates.
The four-candidate `--smoke-grid` remains an execution and integration check,
not a claim of optimal parameters.
The theoretical augmented-norm treatment in the dissertation remains distinct
from its practical unregularized-intercept implementation; this integration
does not supply a new justification for applying those bounds.

For an optional SVM comparison, the same wrapper accepts
`EnsembleKernelSVM`; use `mnist_svm_param_grid` with the
`ovo__estimator__` prefix. Its distinct method, attribution and usage example
are documented in [the SVM comparator guide](ensemble-svm.md#multiclass-use).

## Validation behavior

Preprocessing is fitted within each CV training fold. Each class pair trains
exactly `M` base learners, including when `M=1`. Selected sets must be large
enough and contain both classes; an ineligible pair fails the complete candidate.
The wrapper preserves original labels, clears fitted state after a failed fit,
and accumulates votes in batches. The supplied tuning example preserves separate
training and test arrays, limits dispatched work, and saves completed CV selection
before one explicit final refit. Compare historical dissertation results only
after matching data, parameter search, selection rules, and stopping settings.

## Adaptive command-line search

`examples/tune_mnist_multiclass.py` uses the same external adaptive controller
as the binary examples, with prefix `ovo__estimator__` and regularization
`lambd`. One complete candidate is shared across all 45 pairs. Every stage
uses the same stratified folds and evaluates only new complete candidates.
The default remains five folds, a stratified 6,000-row training subset and
all supplied test rows; the adaptive change does not alter those sizes.

Normal search starts at 125 candidates, expands best/tied-best lambda/gamma
boundaries with all cross-combinations, and refines the top three distinct
coefficient pairs within the explored bounds. Each outward step is at most one decade (a factor of 10), or the smaller
adjacent logarithmic spacing when the explored grid is finer. Sampling stays at 10%, M at
five, and retention on its five choices. Defaults are three expansion rounds,
500 unique candidates total, two nonimproving complete rounds for a plateau,
zero improvement tolerance, tie tolerance `1e-12`, and refinement enabled.
`--broad-grid` starts at 405, `--no-refine` disables refinement, and
`--max-expansions`/`--max-candidates` set explicit limits. `--refine` remains
accepted. A plateau is a finite search decision, not proof of global optimality.

For ten classes and five folds, the initial 125-candidate stage permits up to
168,750 DWD learner fits before caching; the 405-candidate stage permits up to
546,750. Adaptive stages are additional work within the total candidate budget.
The preview prints these separately. Pair fits stay sequential within each
outer CV worker. No data are loaded without `--run`.

`--smoke-grid` overrides the adaptive budget to four total candidates, zero
expansions and no refinement, even if other limit/refinement flags were supplied.
It cannot be combined with `--broad-grid`. All four smoke candidates lie on
their two-point coefficient axes, so their search will remain unresolved and
hold final fitting/test scoring. Use it to check CV execution and records;
it is not a completed accuracy search. Normal searches also hold when a
complete required stage cannot fit the budget. An accepted interior/plateau
review is saved before the final refit and separate test evaluation.

See the [shared workflow](method.md#validation-and-proposed-search-space)
for durable records, failure handling and limits on within-run score reuse.

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
Ineligible base subsets or final selections invalidate the complete candidate and are reported with
the responsible pair. Changing the retention ratio is a new CV candidate,
not an automatic recovery action.
