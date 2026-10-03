# ensemble-dwd 0.3.3: validation and justification of major changes

This historical record covers the package's evolution from the original binary experiment through its multiclass extension and the adaptive cross-validation workflow in version 0.3.3. Here, 'current' and 'latest' refer to that release and its validation. Ensemble DWD remains the primary method. The SVM implementation is a comparator with explicitly documented methodological differences.

Evidence is labeled as **current suite** (tests rerun on the latest source with the repaired DWD dependency), **audit** (focused independent comparisons or ablations), or **historical** (earlier experiments retained with their original scope). A historical result using overlapping draws is not presented as a result for the current disjoint sampling method.

The selected requirements \(M=5\), sampling fraction \(\theta=0.10\), and disjoint subsets are retained. Their methodological selection was not reopened by this audit; functional sampler regression tests remain part of the suite.

## Method preserved by the package

Each binary ensemble fits base learners on disjoint, uniformly selected subsets, averages their raw decision scores on the training observations, selects a fixed fraction of observations by the method's criterion, and fits a fresh final classifier on those selected rows. For ensemble DWD, ranking uses absolute consensus score. A missing class in a drawn or final selected set is reported explicitly; it is not repaired by changing the requested procedure.

Base and final DWD learners use the optimized **cheng338/dwd** implementation. The numerical kernel coefficient is resolved once within the current binary fit and shared across its base/final learners, along with lambda, \(q\), and stopping settings. Preprocessing belongs inside the CV pipeline so each fold learns it from training rows only. The mean-loss DWD lambda convention does not require an extra subset-size rescaling.

For multiclass data, independent binary ensembles are trained for class pairs and combined by votes. This package is therefore not limited to binary classification. See [method](method.md) and [multiclass behavior](multiclass.md).

## Evidence matrix

| Major change | Why it is retained | Validation and result | Limits or cost |
|---|---|---|---|
| sklearn estimator interface around the original procedure | Allows cloning, parameter search, pipelines, labels and fitted-state checks without rewriting the ensemble method. | Current [contract](../tests/test_estimator_contract.py) and [integration](../tests/test_integration.py) tests cover cloning, parameter validation, labels and fitting. | Packaging/API behavior is distinct from a proof of improved accuracy. |
| Optimized DWD dependency and shared numerical parameters | Prevent accidental use of the original dependency and avoid changing gamma from one subset to another when a symbolic default is used. | [Backend](../tests/test_backend.py), [integration](../tests/test_integration.py), and [numerical-boundary](../tests/test_numerical_boundaries.py) tests. Current integration suites passed with both native and portable repaired DWD builds. | Solver repairs can change finite-budget predictions relative to original SlicerSALT. The dependency's [validation report](https://github.com/cheng338/dwd/blob/v1.3.8/docs/validation-1.3.8.md) states its mathematical and numerical scope. |
| Raw-score consensus, absolute-score ranking and selected-row DWD refit | Implements the original ensemble DWD procedure without replacing it by voting among base classifiers. | Audit: four paired comparisons with the same optimized DWD and exactly matched subsets, including an extreme 3-vs-8 setting, gave exactly equal consensus, selected rows, final scores and predictions. | This isolates the wrapper. It does not claim equality between the original and repaired DWD solvers. Stable tie ordering is an explicit reproducibility policy. |
| Base-stage reuse across selection ratios | Selection ratio acts after base fitting, so unchanged base learners need not be retrained for every ratio. | Audit five-ratio controls matched repeated uncached fitting for DWD and SVM. Current [cache tests](../tests/test_cache.py) check reuse and final-stage isolation. | Reuse requires an enabled compatible cache. Without it, retraining identical bases is valid but repeats work. Every distinct final selected set still receives its own final fit. |
| Cache identity, mutation detection and callable bypass | Cached scores must correspond to the actual data, row order, subsets, parameters and implementation. Stateful callable kernels cannot safely use a simplistic identity. | Audit: in-place data/label mutation, reordering, lambda/gamma changes caused misses; restoration restored hits; stateful callables bypassed reuse. [Cache tests](../tests/test_cache.py). | Native and portable DWD builds can currently share a provenance identity. Use a consistent dependency build within a persistent cache, or a fresh cache when switching build type. The audit found this identity limitation, not a demonstrated wrong prediction. |
| Atomic cache publication, checksums and process locking | Multiple CV workers must not consume partial entries or compute under a broken lock. | Historical Windows test: 32 real workers produced one computation and 31 hits; owner-exit recovery and an initially empty locked file were exercised. Current [Windows lock tests](../tests/test_windows_cache_lock.py) retain those regressions. | Hashing, disk I/O and lock contention have costs. Disk reuse is not invariably faster on tiny workloads. |
| Lean cache records and bounded query batches | Avoid storing training-only solver state unnecessarily and avoid materializing an entire query-by-training kernel at once. | Historical cache validation measured 62.3% smaller serialization on its example. Current [cache](../tests/test_cache.py), [contract](../tests/test_estimator_contract.py), and [multiclass](../tests/test_multiclass.py) tests cover outputs and batching. | Batching reduces query memory, not the dense training kernel's quadratic memory requirement. Historical serialization size is not peak process RAM. |
| Failed-refit cleanup | An invalid new fit must not leave old predictions exposed as if the new fit succeeded. | Audit removal exposed stale predictions after an invalid gamma for both ensemble families. Current [contract](../tests/test_estimator_contract.py) and [SVM](../tests/test_svm.py) tests verify cleanup. | A failed fit remains a failure; there is no silent linear/full-model substitute. |
| Finite-mean overflow handling | Finite base scores can overflow in an intermediate sum even when their mean is representable. | Constructed finite operands \([F,F,-F,-F,1]\), with \(F\) the largest binary64 value, have mean 0.2; guarded averaging preserves this result. [Numerical-boundary tests](../tests/test_numerical_boundaries.py). | This is an arithmetic counterexample, not a claim that the observed MNIST scores reached that scale. |
| Explicit ineligible draws/selections | Redrawing, label conditioning or replacing the final classifier would alter the selected procedure. | [Disjoint-sampling](../tests/test_disjoint_sampling.py), [model-selection](../tests/test_model_selection.py), and [search-workflow](../tests/test_search_workflow.py) tests cover capacity, missing classes and error recording. | Some parameter/fold combinations are ineligible even for a valid dataset. They must not be scored as successful fits. |
| Integer multiclass votes and label preservation | Float storage can corrupt large integer class labels; exact vote counts should not depend on such conversion. | Audit: all 72 three/four-class pair-outcome patterns matched an integer oracle, including 34 ties; the original float representation misrepresented 38 of 64 large-integer-label patterns. [Multiclass tests](../tests/test_multiclass.py). | Tie behavior is explicit. A tie convention need not match every earlier script's incidental ordering. |
| Batched multiclass prediction | Accumulates pairwise votes without retaining all pair predictions for all queries. | Audit: all 45 MNIST class pairs, with 1,200 training and 1,000 query rows, gave zero prediction differences. Example intermediate storage decreased from 360,000 to 10,160 bytes. [Multiclass tests](../tests/test_multiclass.py). | These are array sizes, not measured peak RSS. Pointwise decision/predict behavior is assumed. |
| SVM comparator | Supplies a comparable base/select/refit architecture using optimized sklearn/LIBSVM routines. | [SVM tests](../tests/test_svm.py) cover signed-margin ranking, shared C/gamma, native prediction conventions, caching and multiclass integration. | The idea originates from Mao et al. (2016); Chang Cheng's implementation uses the signed criterion \(\lvert yg-1\rvert\) and a final hinge-loss SVC, differing from Mao's final squared-loss fit and the original experiment's unsigned ranking. This is an authorized methodological choice, not a numerical equivalence claim. See [comparator method](ensemble-svm.md). |
| External CV and fold-local preprocessing | Avoid test leakage and let established sklearn tooling perform candidate fitting/scoring. | Audit: six MNIST cases across three pairs and both ensemble families, using the same 18-candidate union, matched one ordinary GridSearchCV run bitwise in all five-fold scores, means, standard deviations and winner. There were 1,080 fold fits across the paired workflows. [Model-selection tests](../tests/test_model_selection.py). | Internal DWD CV is not used by this workflow. Equal-candidate correctness does not establish that adaptive search finds the best unexplored setting. |
| Smaller initial grids and adaptive boundary expansion | Reduce the initial workload while retaining explicit exploration of regularization/kernel boundaries. | Current [adaptive-search tests](../tests/test_adaptive_search.py), actual MNIST execution checks below, and [example tests](../tests/test_examples.py). | Predictive effectiveness and total runtime versus the broad preset remain a separate, deferred benchmark. |
| Search/refit/test transaction and failure records | An unresolved or incomplete search must not silently yield a final test result; failures must preserve useful records. | Tests cover tied boundary winners, all-invalid stages, invalid folds, budget holds, failed final refit, prediction failure and unexpected numerical errors. [Search-workflow](../tests/test_search_workflow.py) and [adaptive-search](../tests/test_adaptive_search.py) tests. | Expected ineligibility is recorded as such; unexpected numerical or memory failures propagate. No successful fallback model is fabricated. |
| Explicit parallelism controls | Outer CV workers or independent class pairs can run concurrently while limiting oversubscription and dense-kernel memory. | Current integration/workflow suites exercise process execution, cleanup and serial/parallel equality. Adaptive tests include a two-worker comparison with identical candidates and scores. | More workers are not always faster. Base learners within a binary fit are sequential; library-native threads and outer workers must be budgeted together. |

## Adaptive search defaults in this release

The estimator remains configurable. The experiment presets use the following initial candidates; each regularization and gamma axis has five choices.

| Parameter | Initial choices |
|---|---|
| DWD lambd | \(2^{[-30,-22,-14,-6,0]}\) |
| SVM C | \(10^{[-4,-2,0,2,4]}\) |
| RBF gamma | \(10^{[-8,-6,-4,-2,0]}\) |
| Base learners \(M\) | 5 |
| Sampling fraction \(\theta\) | 0.10 |
| Selection fraction \(\rho\) | 0.10, 0.15, 0.20, 0.25, 0.30 |

Each family starts with **125 candidates**, compared with **405** in the broad preset with nine regularization and nine gamma candidates. This is a **69.1% reduction in the initial count**, not a measured reduction in total runtime. The broad option remains available.

All stages use external GridSearchCV on the same five stratified training folds, with refit disabled until search acceptance. The official held-out test data are not CV input.

- Best and tied-best regularization/gamma endpoints trigger outward strips, including joint corners. Each outward step is at most a factor of ten.
- Sampling and selection fractions are not expanded.
- Previously evaluated complete candidates are reused within the search. This is not a claim of cross-run resume support.
- Local refinement adds logarithmic midpoints around the top three distinct regularization/gamma pairs, evaluated across all five selection ratios.
- Defaults allow three expansion rounds, 500 unique candidates, and two completed non-improving rounds for plateau acceptance; improvement tolerance is zero and tie tolerance is \(10^{-12}\).
- A budget that cannot accommodate the next complete required stage holds the search and saves its results. No final refit or test scoring occurs.
- A plateau can accept a boundary winner but is not proof of a global optimum. A candidate is eligible only if every fold has a finite valid score.
- Accepted selection is recorded before one explicit final training fit; test scoring follows only a successful final fit.

The estimator, cache, multiclass and SVM core modules were unchanged by the adaptive-search addition. Its changes concern model selection, examples, documentation and tests.

Later review of the 0.3.3 controller found that attempted global coordinates
could make a valid boundary candidate appear interior even when outer
candidates were invalid or neighbors at the same values of the other
parameters were missing. It also performed only one local-refinement stage,
leaving further improvements between tested points unchecked. Version 0.3.4
adds valid conditional-neighbor checks and repeated local refinement. The
eight execution checks below did not expose these cases; their recorded
outcomes remain unchanged. Their success does not establish that every
search accepted by 0.3.3 had adequate boundary or local coverage.

### Actual MNIST execution checks

Eight searches used 600 official-training examples per check: reduced execution grids for all three dissertation pairs and both ensemble families, plus two runs of the actual 125-candidate presets. Across these searches:

| Measure | Result |
|---|---:|
| Unique candidates evaluated, counted within runs | 1,030 |
| Five-fold fit attempts | 5,150 |
| Expected ineligible-selection records | 797 |
| Unexpected numerical/memory errors | 0 |
| Searches held at their exploration budget | 8 |
| Final test sets scored | 0 |

No candidate was duplicated within a run. Both actual-preset runs reached local refinement. All eight retained a best or tied boundary at the allowed exploration limit, so withholding final fitting/test scoring was the correct result. These checks validate execution, records and stopping safeguards; they do **not** establish the predictive quality or speed advantage of the smaller initial grids.

## Current dependency integration and cost

The complete ensemble suite passed **204 tests with native DWD** and **204 with portable DWD**, with no failures. The latest DWD source passed its own 467 native tests and 458 portable tests, with nine expected native-only skips. Validation used Windows/Conda CPython 3.12.14. These are local results rather than claims about remote CI jobs.

Six paired ensemble MNIST regressions compared the optimized DWD dependency before and after its new numerical checks: 2 vs 3, 4 vs 9 and 3 vs 8, at 2,400 and 6,000 stratified official-training rows per pair, seed 20260911. Pixels were float64 divided by 255; \(\gamma=0.01\), \(\mathrm{lambd}=2^{-18}\), \(q=1\), \(M=5\), \(\theta=0.10\), \(\rho=0.30\), disjoint subsets, seed 0 and default DWD stopping were fixed. The complete official pair-specific test sets were used only after fitting, without CV or tuning.

All six retained identical held-out labels, selected rows and iteration counts. Together with six full-DWD regressions, this was a 12-comparison dependency check. It does not promise unchanged accuracy on every future sample or parameter combination.

Controlled, sequential native timings on a Ryzen 9 9955HX used 16 native threads and medians of three repetitions for 2 vs 3:

| Ensemble training rows | Pre-check dependency | Repaired dependency | Ratio |
|---|---:|---:|---:|
| 2,400 | 0.169 s | 0.326 s | 1.93× |
| 6,000 | 0.743 s | 1.389 s | 1.87× |

These fit-only timings include the cost of the dependency's stronger checks. They are not comparisons against the original binary experiment or against SlicerSALT.

Earlier controlled cache studies found roughly 2.2× improvement when sharing five base learners across five selection ratios; see [cache validation](validation-0.1.1.md). That was an older sampling setup and an explicitly scoped cache experiment, not a current disjoint-method accuracy claim. Historical binary/multiclass accuracy reports retain their own data, sampling and tuning protocols.

## Scope and reproduction

Run **python -m unittest discover -s tests -v** from a checkout with test dependencies. Test native and portable DWD installations separately, confirming the imported build. The linked tests exercise estimator correctness, mathematical integration, failure behavior, voting, cache ownership and CV policy; they are complementary to classification experiments.

This release does not claim that all repairs improve accuracy, that every finite-budget prediction matches the original experiment, that no valid extreme DWD fit can fail, or that the adaptive preset is optimal. The current evidence supports the implemented methodology and its stated failure behavior. This validation did not establish the effectiveness of broad versus adaptive search, a numerical dispatch policy that is always optimal, or identification of every build artifact relevant to persistent-cache reuse.
