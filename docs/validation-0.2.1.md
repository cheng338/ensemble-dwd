# ensemble-dwd 0.2.1 validation

The release source passed **123 tests** under the `py12_df`
environment using conventional CPython 3.12 and the optimized DWD 1.3.5
compiled Windows wheel. Native numerical threads were limited to one for
the test run. The suite completed in 24.189 seconds with no failed or skipped
tests. These are functional and concurrency checks, not an accuracy benchmark.

## Dependency identity

The installation metadata pins the following cheng338/dwd 1.3.5 artifacts:

| Artifact | SHA-256 |
|---|---|
| `dwd-1.3.5-cp311-abi3-win_amd64.whl` | `f792d5fa251c57526990221c7d573274e50a99da60e5f508f51515f8f622435e` |
| `dwd-1.3.5-py3-none-any.whl` | `45ec234a373f6080d805089bb24fa869e6de57a1660c3988aeba6ae2c529cb13` |

Conventional Windows AMD64 CPython uses the compiled helper. Other supported
platforms use the portable wheel. The dependency update provides the tested
kernel numerical recovery while preserving the DWD objective and free
intercept. The README documents a separate portable-wheel installation for
free-threaded Windows Python; that interpreter configuration was not tested
locally.

## Cache synchronization repair

The original implementation initialized an empty lock file before acquiring
its lock. If another worker acquired byte zero first, the pending initialization
flush could fail with `PermissionError`. The repaired function opens or creates
the lock file and locks byte zero without writing file contents. Windows
supports locking a region beyond EOF; POSIX flock also supports empty files.
Existing one-byte lock files remain compatible.

Four real Windows multiprocessing regressions cover:

- 32 simultaneous workers on one new cache key: exactly one computation,
  31 cache hits, and identical returned results.
- An empty lock file held by another process: the contender waits without
  attempting the unsafe initialization write.
- Compatibility with existing nonempty lock files.
- Operating-system lock release after an owned worker exits unexpectedly.

The original 0.2.0 initialization failure was reproduced separately before the
repair. The earlier repair validation passed 121 tests: 117 existing tests
plus those four concurrency regressions. This release adds two metadata checks
preventing an obsolete README wheel filename or inconsistent/unhashed DWD
dependency links, bringing the source suite to 123.

`estimator.py`, `multiclass.py` and `model_selection.py` are byte-identical to
0.2.0. The cache module differs only in lock acquisition initialization.
Within the same DWD dependency and numerical settings, the repair preserves
sampled rows, consensus scores, selected observations and predictions. An
earlier MNIST cold/warm cache comparison verified these identities; this
release validation did not rerun MNIST or use official test images.

## Retained functionality

The suite covers binary parameter sharing, the unregularized-intercept DWD
configuration, exact-M conditional sampling without replacement, M=1 score
shape, deterministic subset seeds, absolute-mean consensus ranking and a
fresh selected-set final fit. Invalid selections propagate explicitly instead
of substituting another classifier.

Multiclass checks cover all 45 class pairs, pair-only training data, label
orientation, maximum-wins voting, exact ties, string and large integer labels,
serial/parallel parity, nested GridSearchCV tuning and fold-local scaling.
Cache tests verify reuse across retention ratios, distinct data/fold keys,
checksum handling and storage without base models. Search-workflow tests
preserve completed selection records if final fitting fails.

Normal upgrades produce fresh source-based cache keys. No old cache or
checkpoint is silently treated as belonging to the new release. The estimator
still performs no internal cross-validation; supplied examples use external
GridSearchCV.

Built-wheel checks and artifact hashes are recorded with the release
artifacts. Cross-platform GitHub Actions results are separate from the local
Windows validation above. Historical numerical comparisons remain in
[the 0.2.0 record](validation-0.2.0.md) and
[the 0.1.1 record](validation-0.1.1.md).

Chang Cheng's authorship, private research status and undecided software
license remain unchanged.
