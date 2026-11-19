# Testing installed ensemble-DWD and DWD wheels

The repository uses a `src` layout, so its root alone does not shadow the
installed `ensemble_dwd` package in the way DWD's flat-layout root can. Still,
running repository tests does not prove which exact artifacts were imported.
Editable installations, import configuration and a different installed DWD
dependency can change the tested code.

For artifact qualification, the existing CI helper checks both imported package
origins, distribution versions and installed bytes against the chosen wheels.
It also verifies DWD's native or portable route contract. Tests run from a
detached bundle containing fixtures, examples and metadata but neither package's
importable source root.

## Prepare the paired artifacts

- Use the source and tests for the same release as the selected wheels.
  Ensemble DWD `0.4.0` requires DWD `2.0.1`.
- Put exactly one compatible DWD wheel in its dedicated directory and exactly
  one ensemble-DWD wheel in another. The commands below expect a native DWD
  wheel. The staged native wheel is tagged `cp312-cp312-win_amd64` and fits
  64-bit Windows CPython 3.12. For a native check on Linux or CPython 3.11,
  build a wheel from the DWD 2.0.1 sdist for that OS and interpreter, with the
  native helpers enabled. Alternatively, select the portable wheel and the
  `--backend portable` check below. See the DWD [native build guide](https://github.com/cheng338/dwd/blob/v2.0.1/docs/compiled_residual.md).
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

```shell
python -I -B "/absolute/path/ensemble-source/tests/ci_artifacts.py" install --wheel-dir "/absolute/path/dwd-wheels"
python -I -B "/absolute/path/ensemble-source/tests/ci_artifacts.py" install --wheel-dir "/absolute/path/ensemble-wheels" --extras test
python -m pip check
python -I -B "/absolute/path/ensemble-source/tests/ci_artifacts.py" bundle --source "/absolute/path/ensemble-source" --destination "/absolute/path/ensemble-installed-tests"
cd "/absolute/path/ensemble-installed-tests"
python -I -B tests/ci_artifacts.py check --dwd-wheel-dir "/absolute/path/dwd-wheels" --ensemble-wheel-dir "/absolute/path/ensemble-wheels" --backend native --route-contract current --expected-platform windows --expected-python 3.12
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

Use `--route-contract current` for Ensemble DWD 0.4.0 with DWD 2.0.1.
For the historical Ensemble DWD 0.3.15 and DWD 1.3.19 pair, the checker also
accepts `--route-contract released-1.3.19`. The [CI workflow](../.github/workflows/tests.yml)
builds and checks paired artifacts.
