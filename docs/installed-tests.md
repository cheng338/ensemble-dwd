# Testing installed ensemble-DWD and DWD wheels

The repository uses a `src` layout, so its root alone does not shadow the
installed `ensemble_dwd` package in the way DWD's flat-layout root can. Still,
running repository tests does not prove which exact artifacts were imported.
Editable installations, import configuration and a different installed DWD
dependency can change the tested code.

For artifact qualification, the paired DWD checkout's CI helper checks both imported package
origins, distribution versions and installed bytes against the chosen wheels.
It also verifies DWD's native or portable route contract. Tests run from a
detached bundle containing fixtures, examples and metadata but neither package's
importable source root.

## Development CI

The four jobs use Ubuntu and Windows with Python 3.11 and 3.12. Before enabling
CI for a new development pair, push the matching DWD revision and set the
repository Actions variable `DWD_DEVELOPMENT_REF` to its full 40-hex commit
SHA. Each job checks the requested SHA against the DWD checkout and the event
SHA against the Ensemble checkout. Its summary records both commits and the
SHA-256 of each verified wheel. The [release lock](../requirements-dwd.txt) records the hashes of the
DWD 2.0.2 wheels distributed for Ensemble DWD 0.4.1.

Three jobs check that CVXPY is absent from the isolated test environment.
Ubuntu 3.12 installs the paired DWD wheel with its `socp` extra and pins Clarabel
0.11.0 for its FAER backend. This job builds the Ensemble sdist, compares its
tests and fixtures with tracked checkout bytes, then builds the wheel and the
detached test bundle from the verified extraction. It checks the README body,
metadata separator and exact unconditional DWD requirement in both distributions.

Each job checks dependency consistency and runs the complete detached suite
once. On Ubuntu 3.12, the modules that use optional conic solvers run in a
separate Python process. This keeps their loaded native libraries out of the
cache-reuse and thread-budget tests. When adding a test module that loads a
conic solver, include it in the workflow's `conic_tests` list.
The conic JUnit report must show the real adaptive Clarabel
refit passed, including four FAER threads on the CV fits and one on the selected
refit. On each native Windows job the report must show the real ordered-gradient
reducer ran in both the Ensemble base and final learners.

## Prepare the paired artifacts

- Use the source and tests that produced the selected wheels. This checkout pairs
  Ensemble DWD `0.4.1` with DWD `2.0.2`.
  For the earlier Ensemble DWD 0.4.0 and DWD 2.0.1 wheels, use the
  [0.4.0 test procedure](https://github.com/cheng338/ensemble-dwd/blob/v0.4.0/docs/installed-tests.md)
  and the corresponding release sources.
- Put exactly one compatible DWD wheel in its dedicated directory and exactly
  one ensemble-DWD wheel in another. The commands below expect a native DWD
  wheel built for the target OS and interpreter. For a portable check, build
  the DWD wheel with `DWD_BUILD_ACCEL=0` and select `--backend portable`.
- Replace every quoted absolute-path placeholder. Choose a new environment and
  an absent test-bundle destination outside both source checkouts.
- This recipe's route checker targets standard GIL-enabled x86-64 CPython 3.11
  and 3.12 with binary64 arithmetic on Windows or Linux. It does not qualify
  other architectures or free-threaded interpreters. This bounded validation
  scope does not impose new restrictions on the packages' public support policy.

Create a fresh disposable virtual environment and activate it using your shell's
activation command. All subsequent `python` commands must select that environment:

```shell
python -m venv "/absolute/path/ensemble-wheel-env"
```

After activation, confirm the interpreter and prefix:

```shell
python -c "import sys; print(sys.executable); print(sys.prefix)"
```

Installations below affect only that disposable environment. Numerical and test
dependencies must be available from your configured package index or wheel
cache. The full ensemble test extra includes pandas for mandatory DataFrame
checks; pandas remains optional for ordinary package use. Do not use editable
installations for this artifact check.

## Install, verify and test

Build the development wheels from matching source revisions in the activated
environment. For each DWD build mode (native or portable), use a separate
clean checkout or fresh source-archive extraction and an empty wheel output
directory. This keeps compiled files from an earlier build out of the wheel.
The commands disable pip's wheel cache. See the
[DWD build instructions](https://github.com/cheng338/dwd/blob/v2.0.2/docs/compiled_residual.md#building-and-installing).

Set `DWD_BUILD_ACCEL=1` for a native build or `DWD_BUILD_ACCEL=0` for a portable build in the shell where you run these commands.

```shell
python -m pip wheel --no-cache-dir --no-deps --wheel-dir "/absolute/path/dwd-wheels" "/absolute/path/dwd-source"
python -m pip wheel --no-cache-dir --no-deps --wheel-dir "/absolute/path/ensemble-wheels" "/absolute/path/ensemble-source"
```

```shell
python -I -B "/absolute/path/ensemble-source/tests/ci_artifacts.py" install --wheel-dir "/absolute/path/dwd-wheels"
python -I -B "/absolute/path/ensemble-source/tests/ci_artifacts.py" install --wheel-dir "/absolute/path/ensemble-wheels" --extras test
python -m pip check
python -I -B "/absolute/path/ensemble-source/tests/ci_artifacts.py" bundle --source "/absolute/path/ensemble-source" --destination "/absolute/path/ensemble-installed-tests"
cd "/absolute/path/ensemble-installed-tests"
python -I -B "/absolute/path/dwd-source/tests/ci_artifacts.py" check --dwd-wheel-dir "/absolute/path/dwd-wheels" --ensemble-wheel-dir "/absolute/path/ensemble-wheels" --backend native --route-contract current --expected-platform windows --expected-python 3.12
python -I -B -m pytest tests -o "python_files=test*.py" -ra
```

Use `--expected-platform linux` on Linux and `--expected-python 3.11` for a 3.11
interpreter. The flags verify the intended runtime rather than emulate it.
The `cd` example works in PowerShell and POSIX shells; Windows Command Prompt
needs `cd /d` when changing drives.

Require installation, dependency consistency, artifact verification and pytest
to succeed. An artifact failure is not cured by a later passing test suite.
Keep both selected wheels and the verification output with the test results.
For portable DWD, use another fresh environment and bundle, a directory holding
only the portable DWD wheel, and `--backend portable`. The ensemble wheel is
pure Python; `--backend` describes its DWD dependency.

Isolation with `-I`, avoiding bytecode with `-B`, using `-P` or changing pytest's
import mode does not by itself bind an import to a particular wheel. The exact
artifact check and detached test bundle provide that evidence together.

Use `--route-contract current` for this package pair. The standalone
Ensemble helper retains `--route-contract released-1.3.19` for the historical
Ensemble DWD 0.3.15 and DWD 1.3.19 pair. The [CI workflow](../.github/workflows/tests.yml)
builds and checks the current pair.

### Targeted dependency checks before a release

When changing label handling or the numerical dependency bounds, use a fresh
Python 3.12 test environment with the paired wheels and detached tests described
above. NumPy 2.2.6 exposed a difference in how mixed byte and integer labels
are converted. The following cases check membership, overlapping classes and
the selected predictions across parallel fitting and refitting.

```shell
python -m pip install "numpy==2.2.6" "scipy==1.15.3" "pandas==2.2.2" "scikit-learn==1.6.1" "joblib==1.6.0" "threadpoolctl==3.7.0"
python -m pip check
python -I -B -m pytest tests/test_object_ovo_labels.py::test_mixed_iterator_report_membership tests/test_object_ovo_labels.py::test_mixed_report_rejects_overlapping_classes tests/test_parallel_base.py::ParallelBaseIntegrationTests::test_numerical_identity_clone_pickle_and_refit -q
```

For changes to scikit-learn support, run the following test with version 1.6.1
in the same environment. It checks all three public cross-validation functions
with metadata routing enabled and the default `params` argument, comparing
fold accuracies and out-of-fold predictions with manual fits.

```shell
python -I -B -m pytest tests/test_routed_cv.py -q
```
