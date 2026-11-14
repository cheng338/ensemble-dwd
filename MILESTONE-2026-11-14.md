# Development milestone: November 14, 2026

This branch preserves the approved development source of **both DWD and
ensemble-DWD** after the accumulated correctness/performance investigation and
the November 14 shared-fold CV refresh. It is a source milestone, not a new
stable release or a claim that every third-party review item is finished.

The matching branches are:

- [DWD](https://github.com/cheng338/dwd/tree/milestone/validated-2026-11-14)
- [ensemble-DWD](https://github.com/cheng338/ensemble-dwd/tree/milestone/validated-2026-11-14)

## What is included

The source, tests, synthetic regression fixtures, and documentation are the
approved development snapshot. No runtime code, package defaults, environments,
or dependencies were changed while preparing this milestone. The source
inventory contains 351 DWD files and 168 ensemble-DWD files. The three milestone
documents in each repository are additional publication documentation.

Baseline releases are DWD **1.3.19** (`1bc3b41e695d5beb7f48ce35042542252a6164e5`)
and ensemble-DWD **0.3.15** (`bdd35668e59f2c81aa8705a39cdae47f97567c17`).
Existing version identifiers and release dependency metadata have not yet been
advanced; identify this milestone by its Git commit, not by `__version__`.

## Validation

- November 13 accumulated development suites: local AOCL **7,336 passed / 81
  skipped**; Robust MKL **7,383 passed / 34 skipped**. The focused functional
  panel passed **93/93 on each host**.
- The subsequent approved shared-fold lifetime change passed **153 focused
  checks per host**. Full suites were not rerun after this change.
- November 14 refresh: **96 original jobs / 24 complete blocks**, all scientific
  and lifecycle gates passed, with no retries, replacements, or CPU-magnitude
  exclusions; **20 saved-report controls passed**.
- All **519 approved source-file hashes** matched before milestone preparation.
  The manifest records the validated bytes; Git's existing line-ending rules
  may normalize text in a checkout.

[The accumulated performance comparison](docs/milestones/2026-11-14/comparison.md)
contains **4 refreshed rows** and **180 explicitly dated earlier rows**. The
earlier rows were not rerun against the final shared-fold lifetime change.
The compared package versions ran in the same environment on each host; this
does not isolate AOCL/MKL library-version effects.

Performance is workload dependent. Shared-fold CV remains explicit and can be
slower on small searches; ordinary GridSearchCV remains the general route.
The fixed six-block comparisons are descriptive, not universal speedup or
statistical-significance claims. Most fit timing lanes hold work budgets fixed,
with convergence assessed in separate lanes. Historical scientific and
attribution qualifications remain in the comparison.

## Outstanding review and release preparation

- Third-party ranked **No. 35 / IR-56 remains paused and unintegrated**: an
  interruption during initial checkpoint identity publication can strand an
  otherwise empty run directory. This bounded startup-recovery gap does not
  represent lost completed fold results. This milestone does not resolve it.
- This is a **paired source snapshot**, not a newly packaged install-ready
  release. The ensemble dependency and existing CI recipe still select the
  released DWD 1.3.19 wheel; they do not validate the paired development DWD API.
  Release version/dependency/CI alignment remains release-preparation work.
- Automatic CI is intentionally skipped for this archival milestone commit;
  the unchanged ensemble workflow targets the old wheel. The evidence above
  is the recorded local AOCL / Robust MKL validation, not a claim of new GitHub
  CI results. No wheels, package-index uploads, or stable releases are created.

## Evidence identity

The [source manifest](docs/milestones/2026-11-14/source-manifest.json) identifies
the approved source bytes and the saved validation receipts. Full raw
measurements remain in the development audit archive; the repository contains
the public comparison and receipt hashes rather than the machine-local archive.
