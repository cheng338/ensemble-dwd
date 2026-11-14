Updated accumulated comparison — November 14, 2026 (Indianapolis)

This update refreshes only S09 ordinary shared-fold CV and S10 wide shared-fold CV on local dwd_aocl and Robust mkl202602.
Both hosts compare the exact released DWD 1.3.19 baseline with the latest approved A-off/B-on integration, using the same environment per host. Here A-off means retaining the fold cache through final refitting; the previously accepted B prediction optimization is unchanged.
The timed estimand is released stock GridSearchCV.fit versus current explicit fit_shared_fold_search, not the default current CV route or a pure-A comparison.
Six fixed R,C,C,R blocks per case/host: 48 jobs per host, 96 total; 24 complete blocks; 960 timed estimator fits. All 96 original CV/scientific and lifecycle gates passed.
There were no omitted/replacement blocks, numerical retries, CPU-magnitude exclusions, package/environment edits, installations or publication.
Local uses 16 native threads and Robust 20; one outer worker. Original data/seeds/folds/timers, memory admission, scientific checks and acceptance remain unchanged.

Fresh results. Ratio is the median of six paired current/release block ratios; delta is the median paired current-minus-release time. These statistics need not equal ratios/differences of separate arm medians.
| Host | Case | Median C/R | Median paired delta ms | Slower blocks/6 | Ratio min–max |
|---|---|---:|---:|---:|---|
| local-aocl | S09-shared-fold-cv-ordinary | 1.070731 | +2.731925 | 4 | 0.915936–1.219493 |
| local-aocl | S10-shared-fold-cv-wide | 1.110135 | +3.161900 | 5 | 0.958855–1.154441 |
| robust-mkl | S09-shared-fold-cv-ordinary | 1.027883 | +4.610102 | 4 | 0.925057–1.166608 |
| robust-mkl | S10-shared-fold-cv-wide | 1.190011 | +14.858278 | 5 | 0.972286–1.355659 |

Recommendation unchanged: retain shared-fold CV as an explicit option, not a universal speedup/default. These small searches show additional whole-search cost; no new code change is justified by this refresh alone.
The earlier Robust ordinary-CV speedup (old median 0.816) did not recur in this refresh (1.028). Wide-case medians remain close to the earlier 1.098 local / 1.192 Robust. Those are separate runs/source endpoints: the difference does not isolate A removal, background activity, or a vendor-library effect.
Both independent scientific reviews cleared all 24 diagnostic entries. They record the expected stock/shared route difference and decision-score differences no larger than 4.45e-15, with no sign disagreements. Saved fold/train scores, ranks, selected parameters and existing independent-oracle checks passed. No tolerance changed.
Remote synthetic input archives were not redownloaded or decoded by the local reviewer; original on-host input/oracle/hash checks and downloaded raw records retain that evidence. No new numerical reproduction is claimed by the saved-data review.
All 96 jobs passed the unchanged recorded-owner protocol. One local unowned vanished-process discovery remains explicitly unproven; closure claims cover recorded identities, not every host process. CPU observations remain descriptive; no attribution/subtraction is inferred.
Fixed R,C,C,R order is not randomized. Six short blocks are descriptive, not proof of universal performance or statistical significance. Existing convergence/fixed-work scope is unchanged.

Dated accumulated view below: 4 rows newly measured on November 14; 180 rows carried unchanged from the November 13 accumulated report. The carried rows are not newly measured against the A-off endpoint. Evidence dates label the source reports; original per-job timestamps remain in those records. Carried ensemble-DWD rows retain the 0.3.15 release baseline.
The prior full-suite results also remain November 13 evidence; full suites were not rerun. 20 saved-report controls passed in this update. All 519 approved canonical file hashes still match.
The original report and raw results are preserved. No historical table or failure record was overwritten.

| Evidence report date | Host | Panel/workload | Workers | Phase/index | Median C/R | Released median ms | Current median ms | Median paired delta ms | Slower/6 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| 2026-11-13 | local-aocl | primary/T01-linear-fit | 1 | fresh_fit_first_sample/0 | 0.964036 | 11.895725 | 11.542900 | -0.423850 | 1 |
| 2026-11-13 | local-aocl | primary/T02-linear-hard | 1 | fresh_fit_first_sample/0 | 0.971360 | 11.783525 | 11.416575 | -0.328925 | 0 |
| 2026-11-13 | local-aocl | primary/T03-rbf-fit | 1 | fresh_fit_first_sample/0 | 0.929215 | 62.325850 | 57.221000 | -4.361825 | 0 |
| 2026-11-13 | local-aocl | primary/T04-linear-feature-lbfgs | 1 | fresh_fit_first_sample/0 | 1.038613 | 93.135100 | 96.564450 | +3.677425 | 6 |
| 2026-11-13 | local-aocl | primary/T05-linear-cv | 1 | fresh_fit_first_sample/0 | 1.005148 | 33.208200 | 33.321050 | +0.170000 | 4 |
| 2026-11-13 | local-aocl | primary/T06-rbf-query | 1 | first_explicit_query/0 | 0.985969 | 5.837100 | 5.740750 | -0.099300 | 3 |
| 2026-11-13 | local-aocl | primary/T06-rbf-query | 1 | warm_query/1 | 0.984295 | 5.475800 | 5.337450 | -0.122150 | 3 |
| 2026-11-13 | local-aocl | primary/T06-rbf-query | 1 | warm_query/2 | 0.985968 | 5.047950 | 5.134725 | -0.075950 | 2 |
| 2026-11-13 | local-aocl | primary/T06-rbf-query | 1 | warm_query/3 | 0.945250 | 5.257800 | 4.904325 | -0.289700 | 2 |
| 2026-11-13 | local-aocl | primary/T06-rbf-query | 1 | warm_query/4 | 1.044529 | 4.938525 | 5.069525 | +0.227975 | 4 |
| 2026-11-13 | local-aocl | primary/T06-rbf-query | 1 | warm_query/5 | 1.019236 | 5.171550 | 5.204225 | +0.109175 | 3 |
| 2026-11-13 | local-aocl | primary/T08-residual-inclusive-fit | 1 | fresh_fit_first_sample/0 | 0.973668 | 10.453050 | 10.238875 | -0.275100 | 1 |
| 2026-11-13 | local-aocl | primary/T09-conic-svm | 1 | fresh_fit_first_sample/0 | 1.012561 | 14.142850 | 14.357975 | +0.176150 | 5 |
| 2026-11-13 | local-aocl | primary/T10-linear-converged | 1 | fresh_fit_first_sample/0 | 0.954621 | 8.793175 | 8.233450 | -0.392250 | 1 |
| 2026-11-13 | local-aocl | primary/T11-rbf-converged | 1 | fresh_fit_first_sample/0 | 0.943618 | 55.536400 | 52.115400 | -3.134875 | 0 |
| 2026-11-13 | local-aocl | primary/T-checkpoint-create-and-replay | 1 | checkpoint-complete-replay/0 | 0.620907 | 1133.529975 | 701.590375 | -429.660725 | 0 |
| 2026-11-13 | local-aocl | primary/T-checkpoint-create-and-replay | 1 | checkpoint-initial/0 | 1.012762 | 4060.729975 | 4105.557000 | +51.760225 | 5 |
| 2026-11-13 | local-aocl | primary/T-checkpoint-create-and-replay | 1 | gridsearch-same-work-reference/0 | 0.937231 | 501.795750 | 467.924300 | -31.173575 | 0 |
| 2026-11-13 | local-aocl | primary/T-checkpoint-partial-restart | 1 | checkpoint-initial/0 | 1.001977 | 3216.878825 | 3221.130225 | +6.384000 | 4 |
| 2026-11-13 | local-aocl | primary/T-checkpoint-partial-restart | 1 | checkpoint-partial-resume/0 | 0.804066 | 1984.406075 | 1597.551350 | -390.458975 | 0 |
| 2026-11-13 | local-aocl | primary/T-checkpoint-partial-restart | 1 | gridsearch-same-work-reference/0 | 0.936342 | 498.359375 | 466.864625 | -31.686700 | 0 |
| 2026-11-13 | local-aocl | primary/T-fit-offset | 1 | fresh-estimator-fit/0 | 0.907212 | 79.608975 | 73.299550 | -7.428775 | 0 |
| 2026-11-13 | local-aocl | primary/T-fit-svm-control | 1 | fresh-estimator-fit/0 | 1.009434 | 10.052325 | 10.175750 | +0.095000 | 4 |
| 2026-11-13 | local-aocl | supplemental/S01-explicit-ordinary | 1 | fresh_fit/0 | 1.215825 | 10.320675 | 12.491750 | +2.197375 | 6 |
| 2026-11-13 | local-aocl | supplemental/S02-explicit-wide | 1 | fresh_fit/0 | 1.094022 | 12.300225 | 13.591675 | +1.156350 | 6 |
| 2026-11-13 | local-aocl | supplemental/S03-explicit-illconditioned | 1 | fresh_fit/0 | 1.226067 | 6.149200 | 7.591650 | +1.418175 | 6 |
| 2026-11-13 | local-aocl | supplemental/S04-explicit-integer | 1 | fresh_fit/0 | 1.152496 | 9.617975 | 11.073450 | +1.456325 | 6 |
| 2026-11-13 | local-aocl | supplemental/S05-zero-shift-full-rank | 1 | fresh_fit/0 | 1.001717 | 9.792900 | 9.912175 | +0.017075 | 3 |
| 2026-11-13 | local-aocl | supplemental/S06-wide-feature-lbfgs | 1 | fresh_fit/0 | 1.883928 | 52.500700 | 98.198925 | +46.149550 | 6 |
| 2026-11-13 | local-aocl | supplemental/S07-kernmd-ordinary-fit | 1 | fresh_fit/0 | 0.976760 | 15.671100 | 15.412300 | -0.367825 | 2 |
| 2026-11-13 | local-aocl | supplemental/S08-kernmd-wide-fit | 1 | fresh_fit/0 | 1.019789 | 14.518100 | 14.755125 | +0.287225 | 5 |
| 2026-11-14 | local-aocl | supplemental/S09-shared-fold-cv-ordinary | 1 | complete_cv/0 | 1.070731 | 38.323675 | 40.173625 | +2.731925 | 4 |
| 2026-11-14 | local-aocl | supplemental/S10-shared-fold-cv-wide | 1 | complete_cv/0 | 1.110135 | 28.697700 | 30.975525 | +3.161900 | 5 |
| 2026-11-13 | local-aocl | supplemental/S13-canonical-cosine-query | 1 | first_explicit_query/0 | 1.027758 | 1.444575 | 1.558075 | +0.044975 | 6 |
| 2026-11-13 | local-aocl | supplemental/S13-canonical-cosine-query | 1 | warm_query/1 | 1.026363 | 1.351875 | 1.379225 | +0.036875 | 4 |
| 2026-11-13 | local-aocl | supplemental/S13-canonical-cosine-query | 1 | warm_query/2 | 1.121783 | 1.297100 | 1.436400 | +0.157725 | 5 |
| 2026-11-13 | local-aocl | supplemental/S13-canonical-cosine-query | 1 | warm_query/3 | 0.943702 | 1.330725 | 1.232275 | -0.073775 | 2 |
| 2026-11-13 | local-aocl | supplemental/S13-canonical-cosine-query | 1 | warm_query/4 | 0.957519 | 1.266675 | 1.213200 | -0.055325 | 2 |
| 2026-11-13 | local-aocl | supplemental/S13-canonical-cosine-query | 1 | warm_query/5 | 1.069045 | 1.463250 | 1.545225 | +0.087350 | 4 |
| 2026-11-13 | local-aocl | supplemental/S14-canonical-laplacian-query | 1 | first_explicit_query/0 | 0.994301 | 4.572375 | 4.633650 | -0.028375 | 3 |
| 2026-11-13 | local-aocl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/1 | 0.822195 | 3.001000 | 2.556075 | -0.523275 | 1 |
| 2026-11-13 | local-aocl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/2 | 0.996895 | 2.576825 | 2.583450 | -0.007900 | 3 |
| 2026-11-13 | local-aocl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/3 | 1.002141 | 2.584000 | 2.563200 | +0.006100 | 3 |
| 2026-11-13 | local-aocl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/4 | 1.006729 | 2.478725 | 2.545500 | +0.015625 | 3 |
| 2026-11-13 | local-aocl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/5 | 0.990832 | 2.499300 | 2.551875 | -0.024650 | 3 |
| 2026-11-13 | local-aocl | supplemental/S15-scs-persistence-resolve | 1 | pickle_dump/0 | 1.014797 | 0.702275 | 0.746675 | +0.009975 | 3 |
| 2026-11-13 | local-aocl | supplemental/S15-scs-persistence-resolve | 1 | pickle_load/1 | 1.032788 | 0.657525 | 0.640950 | +0.022325 | 3 |
| 2026-11-13 | local-aocl | supplemental/S15-scs-persistence-resolve | 1 | restored_cold_resolve/3 | 1.007466 | 7.204825 | 7.232750 | +0.052950 | 3 |
| 2026-11-13 | local-aocl | supplemental/S15-scs-persistence-resolve | 1 | restored_warm_resolve/2 | 1.095984 | 1.305650 | 1.421150 | +0.123250 | 4 |
| 2026-11-13 | local-aocl | supplemental/S01-direct-bases-ordinary | 1 | complete-parallel-base-fit/0 | 0.984569 | 402.695500 | 397.249650 | -6.328425 | 1 |
| 2026-11-13 | local-aocl | supplemental/S02-direct-bases-strict | 1 | complete-parallel-base-fit/0 | 0.967629 | 405.026200 | 389.978375 | -13.175550 | 1 |
| 2026-11-13 | local-aocl | supplemental/S04-final-workflow-one | 1 | complete-cv-final-scoring-workflow/0 | 0.967843 | 137.317475 | 132.848275 | -4.417500 | 0 |
| 2026-11-13 | local-aocl | supplemental/S05-final-workflow-budget | 1 | complete-cv-final-scoring-workflow/0 | 0.095596 | 2178.803800 | 207.864775 | -1970.939025 | 0 |
| 2026-11-13 | local-aocl | portable-primary/T04-linear-feature-lbfgs | 1 | fresh_fit_first_sample/0 | 0.976803 | 100.930350 | 98.167400 | -2.358550 | 1 |
| 2026-11-13 | local-aocl | portable-supplemental/S06-wide-feature-lbfgs | 1 | fresh_fit/0 | 1.962379 | 55.743500 | 109.685100 | +54.028325 | 6 |
| 2026-11-13 | local-aocl | gap-primary/T-cache-cold-warm | 1 | cold-cache-fit/0 | 0.892864 | 398.902400 | 357.546125 | -43.114575 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-cache-cold-warm | 1 | warm-cache-fit/0 | 0.683680 | 104.192525 | 71.529525 | -33.761600 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-cache-cold-warm | 1 | warm-cache-fit/1 | 0.697027 | 93.496825 | 65.307600 | -28.383175 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-cache-cold-warm | 1 | warm-cache-fit/2 | 0.671003 | 96.066425 | 64.946425 | -31.569750 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-checkpoint-create-and-replay | 16 | checkpoint-complete-replay/0 | 0.608049 | 1168.621150 | 712.895000 | -458.028750 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-checkpoint-create-and-replay | 16 | checkpoint-initial/0 | 1.295939 | 8662.401650 | 11211.036950 | +2561.014850 | 6 |
| 2026-11-13 | local-aocl | gap-primary/T-checkpoint-create-and-replay | 16 | gridsearch-same-work-reference/0 | 0.998053 | 65.503200 | 65.603150 | -0.126550 | 3 |
| 2026-11-13 | local-aocl | gap-primary/T-checkpoint-partial-restart | 16 | checkpoint-initial/0 | 1.306072 | 6857.988550 | 8960.114000 | +2094.727725 | 6 |
| 2026-11-13 | local-aocl | gap-primary/T-checkpoint-partial-restart | 16 | checkpoint-partial-resume/0 | 0.672239 | 1340.225950 | 900.975350 | -438.677750 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-checkpoint-partial-restart | 16 | gridsearch-same-work-reference/0 | 1.000742 | 106.096250 | 106.172650 | +0.078025 | 3 |
| 2026-11-13 | local-aocl | gap-primary/T-fit-high-dimensional | 1 | fresh-estimator-fit/0 | 0.910379 | 96.585500 | 88.511075 | -8.716600 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-fit-ordinary | 1 | fresh-estimator-fit/0 | 0.963135 | 95.999200 | 91.749750 | -3.595750 | 1 |
| 2026-11-13 | local-aocl | gap-primary/T-gridsearch | 1 | complete-gridsearch-with-refit/0 | 0.879722 | 689.607425 | 628.589075 | -83.100650 | 1 |
| 2026-11-13 | local-aocl | gap-primary/T-gridsearch | 16 | complete-gridsearch-with-refit/0 | 0.995794 | 2788.822150 | 2784.969875 | -11.763300 | 2 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | first-decision_function/0 | 0.905011 | 4.135675 | 3.672125 | -0.378025 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | first-predict/0 | 0.903081 | 3.958425 | 3.530175 | -0.380275 | 1 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | query-setup-fit/0 | 0.972725 | 95.927575 | 91.992200 | -2.600425 | 2 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/0 | 0.844727 | 4.121775 | 3.515125 | -0.614775 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/1 | 0.871445 | 3.993650 | 3.485475 | -0.507075 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/2 | 0.880291 | 3.960550 | 3.499050 | -0.487300 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/3 | 0.885756 | 3.880550 | 3.445450 | -0.440550 | 1 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/4 | 0.936808 | 3.872150 | 3.649775 | -0.244475 | 2 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-predict/0 | 0.870649 | 3.902350 | 3.420125 | -0.507725 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-predict/1 | 0.863152 | 3.854475 | 3.436325 | -0.549625 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-predict/2 | 0.857108 | 4.029775 | 3.413750 | -0.570675 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-predict/3 | 0.857024 | 4.010150 | 3.437925 | -0.573625 | 0 |
| 2026-11-13 | local-aocl | gap-primary/T-predict-cold-warm | 1 | warm-predict/4 | 0.920081 | 3.902675 | 3.786350 | -0.313850 | 2 |
| 2026-11-13 | local-aocl | gap-supplemental/S11-sparse-kernel-cv-canonical | 1 | complete_cv/0 | 0.850469 | 74.527975 | 63.548275 | -10.928100 | 0 |
| 2026-11-13 | local-aocl | gap-supplemental/S03-cache-strict | 1 | strict-cold-cache-fit/0 | 0.891717 | 396.924925 | 354.649000 | -42.782225 | 0 |
| 2026-11-13 | local-aocl | gap-supplemental/S03-cache-strict | 1 | strict-warm-cache-fit/0 | 0.703103 | 101.125650 | 72.012775 | -29.992325 | 0 |
| 2026-11-13 | local-aocl | gap-supplemental/S03-cache-strict | 1 | strict-warm-cache-fit/1 | 0.681555 | 93.337525 | 63.614275 | -29.723250 | 0 |
| 2026-11-13 | local-aocl | gap-supplemental/S03-cache-strict | 1 | strict-warm-cache-fit/2 | 0.688916 | 94.654375 | 64.317100 | -29.043500 | 0 |
| 2026-11-13 | local-aocl | gap-larger/L01-wide-feature | 1 | first_fit/0 | 3.854404 | 66.220700 | 249.330375 | +184.143100 | 6 |
| 2026-11-13 | local-aocl | gap-larger/L01-wide-feature | 1 | repeat_fit/1 | 8.853293 | 18.203225 | 157.982725 | +140.043800 | 6 |
| 2026-11-13 | local-aocl | gap-larger/L02-final-workflow | 1 | complete-cv-final-scoring-workflow/0 | 0.591343 | 4917.983875 | 2898.756675 | -2013.770075 | 0 |
| 2026-11-13 | local-aocl | gap-larger/L03-checkpoint | 1 | checkpoint-complete-replay/0 | 0.607197 | 1137.264100 | 690.090925 | -444.307750 | 0 |
| 2026-11-13 | local-aocl | gap-larger/L03-checkpoint | 1 | checkpoint-initial/0 | 1.287560 | 7094.056350 | 9129.831100 | +2034.086950 | 6 |
| 2026-11-13 | robust-mkl | primary/T01-linear-fit | 1 | fresh_fit_first_sample/0 | 0.937216 | 33.632746 | 28.943140 | -3.373182 | 3 |
| 2026-11-13 | robust-mkl | primary/T02-linear-hard | 1 | fresh_fit_first_sample/0 | 1.183154 | 28.952451 | 34.267475 | +5.315024 | 4 |
| 2026-11-13 | robust-mkl | primary/T03-rbf-fit | 1 | fresh_fit_first_sample/0 | 0.892282 | 164.873507 | 149.737013 | -18.335600 | 0 |
| 2026-11-13 | robust-mkl | primary/T04-linear-feature-lbfgs | 1 | fresh_fit_first_sample/0 | 0.874686 | 275.695488 | 230.366973 | -33.427960 | 0 |
| 2026-11-13 | robust-mkl | primary/T05-linear-cv | 1 | fresh_fit_first_sample/0 | 0.918584 | 169.603620 | 159.552431 | -13.917411 | 2 |
| 2026-11-13 | robust-mkl | primary/T06-rbf-query | 1 | first_explicit_query/0 | 1.044543 | 20.430985 | 21.223753 | +0.926484 | 6 |
| 2026-11-13 | robust-mkl | primary/T06-rbf-query | 1 | warm_query/1 | 0.990255 | 16.714641 | 16.346114 | -0.187112 | 3 |
| 2026-11-13 | robust-mkl | primary/T06-rbf-query | 1 | warm_query/2 | 0.999436 | 16.459841 | 16.221008 | -0.021910 | 3 |
| 2026-11-13 | robust-mkl | primary/T06-rbf-query | 1 | warm_query/3 | 0.995311 | 16.528461 | 16.129008 | -0.087683 | 3 |
| 2026-11-13 | robust-mkl | primary/T06-rbf-query | 1 | warm_query/4 | 0.975686 | 16.629884 | 16.083753 | -0.418080 | 2 |
| 2026-11-13 | robust-mkl | primary/T06-rbf-query | 1 | warm_query/5 | 0.983653 | 16.317469 | 15.912199 | -0.276535 | 3 |
| 2026-11-13 | robust-mkl | primary/T08-residual-inclusive-fit | 1 | fresh_fit_first_sample/0 | 0.708219 | 26.706241 | 19.773787 | -7.898801 | 2 |
| 2026-11-13 | robust-mkl | primary/T09-conic-svm | 1 | fresh_fit_first_sample/0 | 1.229296 | 33.521876 | 40.600215 | +7.582439 | 4 |
| 2026-11-13 | robust-mkl | primary/T10-linear-converged | 1 | fresh_fit_first_sample/0 | 0.990773 | 40.857740 | 43.533977 | -0.422464 | 3 |
| 2026-11-13 | robust-mkl | primary/T11-rbf-converged | 1 | fresh_fit_first_sample/0 | 0.909446 | 141.087892 | 130.949087 | -13.201868 | 1 |
| 2026-11-13 | robust-mkl | primary/T-checkpoint-create-and-replay | 1 | checkpoint-complete-replay/0 | 0.526561 | 1958.949508 | 1023.651666 | -916.575896 | 0 |
| 2026-11-13 | robust-mkl | primary/T-checkpoint-create-and-replay | 1 | checkpoint-initial/0 | 0.876246 | 7172.201606 | 6298.286803 | -884.152265 | 0 |
| 2026-11-13 | robust-mkl | primary/T-checkpoint-create-and-replay | 1 | gridsearch-same-work-reference/0 | 0.909729 | 1548.558546 | 1405.793090 | -139.266391 | 0 |
| 2026-11-13 | robust-mkl | primary/T-checkpoint-partial-restart | 1 | checkpoint-initial/0 | 0.848080 | 5526.009406 | 4746.310896 | -835.759253 | 0 |
| 2026-11-13 | robust-mkl | primary/T-checkpoint-partial-restart | 1 | checkpoint-partial-resume/0 | 0.724425 | 3510.035238 | 2524.495392 | -964.590662 | 0 |
| 2026-11-13 | robust-mkl | primary/T-checkpoint-partial-restart | 1 | gridsearch-same-work-reference/0 | 0.906162 | 1563.260655 | 1409.497231 | -148.178177 | 0 |
| 2026-11-13 | robust-mkl | primary/T-fit-offset | 1 | fresh-estimator-fit/0 | 0.938089 | 204.586756 | 189.297067 | -12.530966 | 2 |
| 2026-11-13 | robust-mkl | primary/T-fit-svm-control | 1 | fresh-estimator-fit/0 | 1.300187 | 25.875667 | 33.082102 | +7.649708 | 4 |
| 2026-11-13 | robust-mkl | supplemental/S01-explicit-ordinary | 1 | fresh_fit/0 | 1.375975 | 27.275727 | 37.961130 | +10.020422 | 5 |
| 2026-11-13 | robust-mkl | supplemental/S02-explicit-wide | 1 | fresh_fit/0 | 0.940432 | 36.550331 | 34.679636 | -2.218376 | 3 |
| 2026-11-13 | robust-mkl | supplemental/S03-explicit-illconditioned | 1 | fresh_fit/0 | 1.228432 | 21.154902 | 25.763478 | +4.729061 | 4 |
| 2026-11-13 | robust-mkl | supplemental/S04-explicit-integer | 1 | fresh_fit/0 | 1.110244 | 25.276287 | 29.018352 | +2.473036 | 4 |
| 2026-11-13 | robust-mkl | supplemental/S05-zero-shift-full-rank | 1 | fresh_fit/0 | 1.186768 | 19.869103 | 22.668528 | +3.509372 | 4 |
| 2026-11-13 | robust-mkl | supplemental/S06-wide-feature-lbfgs | 1 | fresh_fit/0 | 1.786480 | 144.520729 | 273.290343 | +118.168724 | 6 |
| 2026-11-13 | robust-mkl | supplemental/S07-kernmd-ordinary-fit | 1 | fresh_fit/0 | 1.161105 | 27.920574 | 29.792487 | +4.155821 | 4 |
| 2026-11-13 | robust-mkl | supplemental/S08-kernmd-wide-fit | 1 | fresh_fit/0 | 0.931079 | 30.759223 | 27.301771 | -2.208233 | 3 |
| 2026-11-14 | robust-mkl | supplemental/S09-shared-fold-cv-ordinary | 1 | complete_cv/0 | 1.027883 | 139.620750 | 139.032599 | +4.610102 | 4 |
| 2026-11-14 | robust-mkl | supplemental/S10-shared-fold-cv-wide | 1 | complete_cv/0 | 1.190011 | 78.522802 | 94.417314 | +14.858278 | 5 |
| 2026-11-13 | robust-mkl | supplemental/S13-canonical-cosine-query | 1 | first_explicit_query/0 | 1.005110 | 5.878119 | 5.845782 | +0.046687 | 3 |
| 2026-11-13 | robust-mkl | supplemental/S13-canonical-cosine-query | 1 | warm_query/1 | 0.905248 | 4.889912 | 4.416484 | -0.477314 | 2 |
| 2026-11-13 | robust-mkl | supplemental/S13-canonical-cosine-query | 1 | warm_query/2 | 1.035232 | 4.353231 | 4.787005 | +0.190373 | 4 |
| 2026-11-13 | robust-mkl | supplemental/S13-canonical-cosine-query | 1 | warm_query/3 | 0.910925 | 5.333427 | 5.176667 | -0.468934 | 2 |
| 2026-11-13 | robust-mkl | supplemental/S13-canonical-cosine-query | 1 | warm_query/4 | 0.861324 | 5.145349 | 4.816378 | -0.693919 | 1 |
| 2026-11-13 | robust-mkl | supplemental/S13-canonical-cosine-query | 1 | warm_query/5 | 0.897164 | 5.095612 | 4.731014 | -0.517174 | 1 |
| 2026-11-13 | robust-mkl | supplemental/S14-canonical-laplacian-query | 1 | first_explicit_query/0 | 1.058147 | 8.423358 | 8.912774 | +0.436031 | 3 |
| 2026-11-13 | robust-mkl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/1 | 0.942927 | 7.460396 | 7.451311 | -0.435771 | 2 |
| 2026-11-13 | robust-mkl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/2 | 1.001609 | 7.571760 | 7.700547 | +0.003845 | 3 |
| 2026-11-13 | robust-mkl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/3 | 1.038360 | 7.438739 | 7.913931 | +0.279620 | 3 |
| 2026-11-13 | robust-mkl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/4 | 1.000044 | 7.207402 | 7.328024 | +0.000167 | 3 |
| 2026-11-13 | robust-mkl | supplemental/S14-canonical-laplacian-query | 1 | warm_query/5 | 0.997396 | 7.151855 | 7.155110 | -0.018684 | 3 |
| 2026-11-13 | robust-mkl | supplemental/S15-scs-persistence-resolve | 1 | pickle_dump/0 | 1.090326 | 1.899109 | 1.956854 | +0.158856 | 4 |
| 2026-11-13 | robust-mkl | supplemental/S15-scs-persistence-resolve | 1 | pickle_load/1 | 1.002051 | 1.866959 | 1.863494 | -0.004723 | 3 |
| 2026-11-13 | robust-mkl | supplemental/S15-scs-persistence-resolve | 1 | restored_cold_resolve/3 | 1.034931 | 25.754344 | 25.840345 | +0.853140 | 5 |
| 2026-11-13 | robust-mkl | supplemental/S15-scs-persistence-resolve | 1 | restored_warm_resolve/2 | 1.062212 | 4.743852 | 5.131320 | +0.300733 | 5 |
| 2026-11-13 | robust-mkl | supplemental/S01-direct-bases-ordinary | 1 | complete-parallel-base-fit/0 | 0.970610 | 2222.425208 | 2162.522771 | -65.597384 | 1 |
| 2026-11-13 | robust-mkl | supplemental/S02-direct-bases-strict | 1 | complete-parallel-base-fit/0 | 0.954674 | 2231.326970 | 2140.422167 | -101.516594 | 2 |
| 2026-11-13 | robust-mkl | supplemental/S04-final-workflow-one | 1 | complete-cv-final-scoring-workflow/0 | 0.959364 | 419.866457 | 404.378942 | -17.539269 | 1 |
| 2026-11-13 | robust-mkl | supplemental/S05-final-workflow-budget | 1 | complete-cv-final-scoring-workflow/0 | 0.164268 | 3804.412203 | 617.600019 | -3177.750348 | 0 |
| 2026-11-13 | robust-mkl | portable-primary/T04-linear-feature-lbfgs | 1 | fresh_fit_first_sample/0 | 0.851401 | 312.742587 | 264.405511 | -46.205611 | 1 |
| 2026-11-13 | robust-mkl | portable-supplemental/S06-wide-feature-lbfgs | 1 | fresh_fit/0 | 2.224860 | 149.062407 | 339.632638 | +188.001458 | 6 |
| 2026-11-13 | robust-mkl | gap-primary/T-cache-cold-warm | 1 | cold-cache-fit/0 | 0.894271 | 922.733176 | 838.323229 | -97.560399 | 1 |
| 2026-11-13 | robust-mkl | gap-primary/T-cache-cold-warm | 1 | warm-cache-fit/0 | 0.789553 | 156.406311 | 131.200996 | -33.870528 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-cache-cold-warm | 1 | warm-cache-fit/1 | 0.798207 | 160.455985 | 122.798517 | -32.305953 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-cache-cold-warm | 1 | warm-cache-fit/2 | 0.686162 | 140.721050 | 114.385136 | -44.196868 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-checkpoint-create-and-replay | 16 | checkpoint-complete-replay/0 | 0.504551 | 1969.892971 | 979.089270 | -981.657578 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-checkpoint-create-and-replay | 16 | checkpoint-initial/0 | 1.182622 | 9785.430524 | 11514.429583 | +1777.953861 | 6 |
| 2026-11-13 | robust-mkl | gap-primary/T-checkpoint-create-and-replay | 16 | gridsearch-same-work-reference/0 | 0.944222 | 283.704580 | 265.426192 | -16.471809 | 1 |
| 2026-11-13 | robust-mkl | gap-primary/T-checkpoint-partial-restart | 16 | checkpoint-initial/0 | 1.210181 | 9362.753402 | 11285.537354 | +1969.934642 | 6 |
| 2026-11-13 | robust-mkl | gap-primary/T-checkpoint-partial-restart | 16 | checkpoint-partial-resume/0 | 0.585855 | 2282.973681 | 1345.325417 | -949.994973 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-checkpoint-partial-restart | 16 | gridsearch-same-work-reference/0 | 0.950230 | 401.420877 | 383.901840 | -20.951857 | 2 |
| 2026-11-13 | robust-mkl | gap-primary/T-fit-high-dimensional | 1 | fresh-estimator-fit/0 | 0.962297 | 303.542684 | 295.692400 | -11.816789 | 2 |
| 2026-11-13 | robust-mkl | gap-primary/T-fit-ordinary | 1 | fresh-estimator-fit/0 | 0.895830 | 312.925815 | 266.570308 | -33.759227 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-gridsearch | 1 | complete-gridsearch-with-refit/0 | 0.909732 | 2264.610422 | 2033.367802 | -201.801106 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-gridsearch | 16 | complete-gridsearch-with-refit/0 | 0.979121 | 3455.381868 | 3368.924301 | -71.979257 | 1 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | first-decision_function/0 | 0.865179 | 13.010676 | 11.503728 | -1.813642 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | first-predict/0 | 0.961925 | 12.081363 | 11.055146 | -0.429532 | 3 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | query-setup-fit/0 | 0.792663 | 321.001502 | 252.988411 | -62.979164 | 0 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/0 | 0.867349 | 13.551854 | 10.944430 | -1.901066 | 2 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/1 | 0.923371 | 12.408840 | 11.233861 | -0.995269 | 1 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/2 | 0.905244 | 13.363575 | 10.790384 | -1.201240 | 2 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/3 | 0.953330 | 13.543360 | 11.575821 | -0.652927 | 2 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-decision_function/4 | 0.965339 | 12.814810 | 12.015329 | -0.525850 | 3 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-predict/0 | 0.995105 | 11.968058 | 10.696119 | -0.064728 | 3 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-predict/1 | 0.992456 | 11.992532 | 10.675689 | -0.095152 | 3 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-predict/2 | 0.940227 | 12.688981 | 12.355351 | -0.878244 | 2 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-predict/3 | 0.916669 | 12.603596 | 11.931668 | -1.163404 | 1 |
| 2026-11-13 | robust-mkl | gap-primary/T-predict-cold-warm | 1 | warm-predict/4 | 1.038853 | 11.037460 | 11.066163 | +0.406543 | 4 |
| 2026-11-13 | robust-mkl | gap-supplemental/S11-sparse-kernel-cv-canonical | 1 | complete_cv/0 | 0.694747 | 227.863021 | 162.574814 | -69.550436 | 0 |
| 2026-11-13 | robust-mkl | gap-supplemental/S03-cache-strict | 1 | strict-cold-cache-fit/0 | 0.935919 | 925.558567 | 869.819401 | -59.439833 | 0 |
| 2026-11-13 | robust-mkl | gap-supplemental/S03-cache-strict | 1 | strict-warm-cache-fit/0 | 0.702815 | 190.015967 | 132.901288 | -59.672923 | 0 |
| 2026-11-13 | robust-mkl | gap-supplemental/S03-cache-strict | 1 | strict-warm-cache-fit/1 | 0.707448 | 178.358561 | 123.154210 | -53.179966 | 0 |
| 2026-11-13 | robust-mkl | gap-supplemental/S03-cache-strict | 1 | strict-warm-cache-fit/2 | 0.687234 | 167.429012 | 113.148740 | -53.439325 | 0 |
| 2026-11-13 | robust-mkl | gap-larger/L01-wide-feature | 1 | first_fit/0 | 3.715477 | 166.178015 | 626.387476 | +459.522528 | 6 |
| 2026-11-13 | robust-mkl | gap-larger/L01-wide-feature | 1 | repeat_fit/1 | 5.686505 | 63.224290 | 376.845717 | +314.069909 | 6 |
| 2026-11-13 | robust-mkl | gap-larger/L02-final-workflow | 1 | complete-cv-final-scoring-workflow/0 | 0.547485 | 6932.813453 | 3779.261567 | -3125.903060 | 0 |
| 2026-11-13 | robust-mkl | gap-larger/L03-checkpoint | 1 | checkpoint-complete-replay/0 | 0.500869 | 1869.271062 | 934.779202 | -933.015247 | 0 |
| 2026-11-13 | robust-mkl | gap-larger/L03-checkpoint | 1 | checkpoint-initial/0 | 1.181362 | 9758.051323 | 11591.625526 | +1760.875002 | 6 |
