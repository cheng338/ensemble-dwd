"""Fresh-process identity regressions; edits are confined to temporary copies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import pytest

from process_test_support import popen_python

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'

DRIVER = r'''
import contextlib
import io
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
root, mode, record = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(root / 'examples'))
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
import adaptive_search as workflow
import cv_checkpoint as checkpoint
import tune_mnist
import dwd
import ensemble_dwd
from search_support import new_run_directory

class TiedSurface(ClassifierMixin, BaseEstimator):
    fits = []
    predictions = []
    def __init__(self, lambd=1., gamma=4.):
        self.lambd, self.gamma = lambd, gamma
    def fit(self, X, y):
        type(self).fits.append({'rows': len(y), 'lambd': self.lambd})
        self.classes_ = np.unique(y)
        return self
    def predict(self, X):
        type(self).predictions.append(len(X))
        return np.asarray(X)[:, 1].astype(int)

original_stage = workflow.next_search_stage
def stage(results, grid, config, state):
    if len(results['params']) == 2:
        if mode == 'interrupt':
            raise KeyboardInterrupt('controlled stop after saved coarse folds')
        return {'state': state, 'review': {'state': 'accepted', 'reasons': [],
                'stopping_reason': 'controlled test acceptance'}}
    return original_stage(results, grid, config, state)
workflow.next_search_stage = stage
rows = np.arange(12)
X = np.column_stack([rows, rows % 2])
y = rows % 2
test_rows = np.arange(100, 104)
test_X = np.column_stack([test_rows, test_rows % 2])
folds = [(np.setdiff1d(rows, np.arange(i, i + 4)), np.arange(i, i + 4))
         for i in range(0, 12, 4)]
run = root / 'run'
if mode != 'resume':
    run = new_run_directory(run)
result = {}
try:
    with contextlib.redirect_stdout(io.StringIO()), threadpool_limits(1):
        value = workflow.run_adaptive_search(
            Pipeline([('ensemble', TiedSurface())]),
            {'ensemble__lambd': [1., 2.], 'ensemble__gamma': [4.]},
            X, y, test_X, test_rows % 2, folds, run,
            config=workflow.AdaptiveSearchConfig(refine=False, max_candidates=2),
            resume=mode == 'resume')
        result.update(outcome='accepted', selection=value['selection'],
                      replay=bool(value.get('resumed_completed')))
except KeyboardInterrupt as error:
    result.update(outcome='interrupted', error=str(error))
except checkpoint.CheckpointError as error:
    result.update(outcome='identity_refused', error=str(error))
except Exception as error:
    result.update(outcome='other_error', error=repr(error))
result.update(fits=TiedSurface.fits, predictions=TiedSurface.predictions,
              checkpoint_source=str(Path(checkpoint.__file__).resolve()),
              selector_source=str(Path(tune_mnist.__file__).resolve()),
              workflow_source=str(Path(workflow.__file__).resolve()),
              dwd_source=str(Path(dwd.__file__).resolve()),
              ensemble_source=str(Path(ensemble_dwd.__file__).resolve()))
record.write_text(json.dumps(result, indent=2) + '\n')
'''


def subprocess_environment(examples):
    import dwd
    import ensemble_dwd
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               PYTHONOPTIMIZE=str(sys.flags.optimize))
    env.pop('PYTHONHOME', None)
    # Reuse the packages loaded by pytest, without depending on a runner's
    # PYTHONPATH or allowing inherited example directories to take priority.
    env['PYTHONPATH'] = os.pathsep.join([
        str(examples), str(Path(dwd.__file__).resolve().parent.parent),
        str(Path(ensemble_dwd.__file__).resolve().parent.parent)])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    return env


def copy_examples(destination):
    destination.mkdir()
    for source in EXAMPLES.glob('*.py'):
        shutil.copy2(source, destination / source.name)


def launch(root, mode, record):
    import dwd
    import ensemble_dwd
    env = subprocess_environment(root / 'examples')
    result = subprocess.run([sys.executable, '-B', str(root / 'probe.py'),
                             str(root), mode, str(record)], env=env,
                            capture_output=True, text=True, timeout=90,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, (result.stdout, result.stderr)
    value = json.loads(record.read_text())
    assert Path(value['checkpoint_source']) == (root / 'examples/cv_checkpoint.py').resolve()
    assert Path(value['selector_source']) == (root / 'examples/tune_mnist.py').resolve()
    assert Path(value['workflow_source']) == (root / 'examples/adaptive_search.py').resolve()
    assert Path(value['dwd_source']) == Path(dwd.__file__).resolve()
    assert Path(value['ensemble_source']) == Path(ensemble_dwd.__file__).resolve()
    return value


def saved_state(root):
    # Lock files can be touched to acquire ownership; saved records must not change.
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.pkl')}


@pytest.mark.parametrize('initial', ['interrupt', 'complete'])
@pytest.mark.parametrize('edit', ['none', 'tie', 'eligibility'])
def test_fresh_process_resume_source_identity(tmp_path, initial, edit):
    copy_examples(tmp_path / 'examples')
    (tmp_path / 'probe.py').write_text(DRIVER, encoding='utf-8')
    first = launch(tmp_path, initial, tmp_path / 'first.json')
    assert first['outcome'] == ('interrupted' if initial == 'interrupt' else 'accepted'), first
    assert len(first['fits']) == (6 if initial == 'interrupt' else 7), first
    if initial == 'complete':
        assert first['selection']['candidate_index'] == 0
    source = tmp_path / 'examples/tune_mnist.py'
    text = source.read_text(encoding='utf-8')
    if edit == 'tie':
        old = '    indices = np.flatnonzero(fully_valid_candidate_mask(cv_results))'
        assert text.count(old) == 1
        source.write_text(text.replace(old, old + '[::-1]'), encoding='utf-8')
    elif edit == 'eligibility':
        old = '    return valid\n'
        assert text.count(old) == 1
        source.write_text(text.replace(old, '    return np.zeros_like(valid)\n'), encoding='utf-8')
    before = saved_state(tmp_path / 'run')
    resumed = launch(tmp_path, 'resume', tmp_path / 'resumed.json')
    if edit != 'none':
        assert resumed['outcome'] == 'identity_refused', resumed
        assert resumed['error'] == 'Execution and coordinator common runtime differs: common_files_sha256', resumed
        assert resumed['fits'] == resumed['predictions'] == []
        assert saved_state(tmp_path / 'run') == before
    else:
        assert resumed['outcome'] == 'accepted', resumed
        assert len(resumed['fits']) == (1 if initial == 'interrupt' else 0), resumed
        assert all(row['rows'] == 12 for row in resumed['fits'])
        if initial == 'complete':
            assert resumed['replay'] and resumed['predictions'] == []
        assert resumed['selection']['candidate_index'] == 0


@pytest.mark.parametrize('order', ['cv_checkpoint, tune_mnist, adaptive_search',
                                 'tune_mnist, adaptive_search, cv_checkpoint'])
def test_fresh_import_order(order):
    env = subprocess_environment(EXAMPLES)
    code = 'import ' + order + '; print("SELECTION_IMPORT_OK")'
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True,
                            text=True, env=env, timeout=60,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, result.stderr
    assert 'SELECTION_IMPORT_OK' in result.stdout


def test_cli_preview_still_works():
    result = subprocess.run([sys.executable, '-B', str(EXAMPLES / 'tune_mnist.py')],
                            capture_output=True, text=True, timeout=60,
                            env=subprocess_environment(EXAMPLES),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, result.stderr
    assert result.stdout


def test_in_process_edit_uses_runtime_guard(tmp_path):
    copy_examples(tmp_path / 'examples')
    code = '''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import cv_checkpoint as checkpoint
from threadpoolctl import threadpool_limits
source = Path(sys.argv[1]) / 'tune_mnist.py'
with threadpool_limits(1):
    first = checkpoint.runtime_fingerprint()
    source.write_bytes(source.read_bytes() + b'\\n# controlled temporary source change\\n')
    try:
        second = checkpoint.runtime_fingerprint(check_inventory=False)
    except checkpoint.CheckpointError:
        print('SELECTION_RUNTIME_REFUSED')
    else:
        assert first == second
        print('SELECTION_RUNTIME_ACCEPTED')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path / 'examples')],
                            capture_output=True, text=True, timeout=90,
                            env=subprocess_environment(tmp_path / 'examples'),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, result.stderr
    assert 'SELECTION_RUNTIME_REFUSED' in result.stdout


DEFAULT_SOURCE_DRIVER = r'''
import ast
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import struct
import sys
from types import CodeType
import psutil

root, mode = Path(sys.argv[1]), sys.argv[2]
source = root / 'search_support.py'
original = source.read_bytes()
stat = source.stat()
old = b'def score_value(results, index, *, exact=None):'
new = b'def score_value(results, index, *, exact=True):'
if original.count(old) != 1 or len(old) != len(new):
    raise RuntimeError('Expected one same-size default declaration edit')
cache = Path(py_compile.compile(str(source), doraise=True,
    invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP))
cache_bytes = cache.read_bytes()
magic, flags, mtime_seconds, source_size = struct.unpack('<4sIII', cache_bytes[:16])
if mode == 'stale-default':
    source.write_bytes(original.replace(old, new))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
current = source.read_bytes()
node = next(node for node in ast.parse(current).body
            if isinstance(node, ast.FunctionDef) and node.name == 'score_value')
source_default = ast.literal_eval(node.args.kw_defaults[0])
sys.dont_write_bytecode = True
sys.path.insert(0, str(root))
import cv_checkpoint as checkpoint
import search_support as support
import tune_mnist
import dwd
import ensemble_dwd
import numpy as np
from threadpoolctl import threadpool_limits
loaded = support.score_value
compiled = compile(current, loaded.__code__.co_filename, 'exec',
                   dont_inherit=True, optimize=sys.flags.optimize)
body = next(code for code in compiled.co_consts
            if isinstance(code, CodeType) and code.co_name == 'score_value')
denominator = 2 ** 80
with threadpool_limits(1):
    evidence = support.accuracy_evidence([[{'correct': denominator // 2 + 1,
                                          'n_validation': denominator}]])
    results = {'params': [{}, {}], 'mean_test_score': np.asarray([.5, .5]),
               'split0_test_score': np.asarray([.5, .5]),
               'accuracy_evidence': [None, evidence]}
    loaded_rank = tune_mnist.ranked_valid_indices(results).tolist()
    explicit_values = [loaded(results, i, exact=source_default) for i in range(2)]
    explicit_rank = sorted(range(2), key=lambda i: explicit_values[i], reverse=True)
    cold = checkpoint._SELECTION_STATE is None
    try:
        checkpoint.check_selection_state()
    except checkpoint.CheckpointError:
        outcome = 'refused'
    else:
        outcome = 'accepted'
process = psutil.Process()
# This is a final live snapshot, not a claim to capture every transient child.
# Record provenance without guessing a descendant's purpose from its PID.
children = []
for child in process.children(recursive=True):
    row = {'pid': child.pid, 'created': None, 'ppid': None, 'name': None,
           'exe': None, 'cmdline': None, 'errors': []}
    for field, getter in (('created', child.create_time), ('ppid', child.ppid),
                          ('name', child.name), ('exe', child.exe), ('cmdline', child.cmdline)):
        try:
            row[field] = getter()
        except psutil.Error as error:
            row['errors'].append({'field': field, 'type': type(error).__name__,
                                  'message': str(error),
                                  'disappeared': isinstance(error, psutil.NoSuchProcess)})
    children.append(row)
print(json.dumps({
    'pid': process.pid, 'created': process.create_time(),
    'children': children,
    'guard': outcome, 'cold': cold,
    'source_default': source_default, 'loaded_default': loaded.__kwdefaults__,
    'body_matches': loaded.__code__ == body,
    'before_sha256': hashlib.sha256(original).hexdigest(),
    'after_sha256': hashlib.sha256(current).hexdigest(),
    'same_size': len(original) == len(current) == source_size,
    'same_mtime': stat.st_mtime_ns == source.stat().st_mtime_ns,
    'timestamp_cache': flags == 0 and mtime_seconds == (int(stat.st_mtime) & 0xffffffff),
    'magic_matches': magic == importlib.util.MAGIC_NUMBER,
    'cache_unchanged': cache.read_bytes() == cache_bytes,
    'cache_path': support.__cached__, 'generated_cache': str(cache),
    'loaded_rank': loaded_rank, 'explicit_source_rank': explicit_rank,
    'explicit_second_fraction': ([explicit_values[1].numerator, explicit_values[1].denominator]
                                 if isinstance(explicit_values[1], Fraction) else None),
    'origins': {module.__name__: str(Path(module.__file__).resolve())
                for module in (checkpoint, support, tune_mnist, dwd, ensemble_dwd)},
    'function_origin': str(Path(loaded.__code__.co_filename).resolve())}))
'''


def observed_descendant_state(row):
    """Check only the exact identity captured by the owned probe."""
    result = {'pid': row['pid'], 'created': row['created']}
    if row['created'] is None:
        return dict(result, state='unknown-identity')
    try:
        current = psutil.Process(row['pid'])
        if current.create_time() != row['created']:
            return dict(result, state='reused')
        return dict(result, state='present')
    except psutil.NoSuchProcess:
        return dict(result, state='absent')
    except psutil.Error as error:
        return dict(result, state='unavailable', error=repr(error))


@pytest.mark.parametrize('mode', ['unchanged', 'stale-default'])
def test_fresh_process_checks_source_default_against_timestamp_pyc(tmp_path, mode):
    import dwd
    import ensemble_dwd
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in EXAMPLES.glob('*.py')}
    copied = tmp_path / 'examples'
    copy_examples(copied)
    child = popen_python(
        ['-B', '-c', DEFAULT_SOURCE_DRIVER, str(copied), mode],
        env=subprocess_environment(copied), cwd=tmp_path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    observed = []
    try:
        identity = (child.pid, child.create_time())
        stdout, stderr = child.communicate(timeout=90)
        value = json.loads(stdout.strip().splitlines()[-1])
        (tmp_path / 'default-probe-result.json').write_text(
            json.dumps(value, indent=2) + '\n', encoding='utf-8')
        observed = value['children']
        assert child.returncode == 0, (stdout, stderr)
        assert (value['pid'], value['created']) == identity
        assert not child.is_running()
        deadline = time.monotonic() + 5
        while True:
            closures = [observed_descendant_state(row) for row in observed]
            if not any(row['state'] == 'present' for row in closures) or time.monotonic() >= deadline:
                break
            time.sleep(.02)
        (tmp_path / 'default-descendant-closures.json').write_text(
            json.dumps(closures, indent=2) + '\n', encoding='utf-8')
        assert all(row['state'] in ('absent', 'reused') for row in closures), (observed, closures)
        # Disappearance during metadata capture remains explicit. Access errors
        # or missing exact identities cannot silently qualify a test result.
        assert all(error['disappeared'] for row in observed for error in row['errors']), observed
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        cleanup, deadline = [], time.monotonic() + 5
        for row in observed:
            if observed_descendant_state(row)['state'] != 'present':
                continue
            action = {'pid': row['pid'], 'created': row['created']}
            try:
                survivor = psutil.Process(row['pid'])
                if survivor.create_time() == row['created']:
                    survivor.kill()
                    action['killed'] = True
                    survivor.wait(timeout=max(0., deadline - time.monotonic()))
            except psutil.NoSuchProcess:
                action['already_exited'] = True
            except psutil.Error as error:
                action['error'] = repr(error)
            cleanup.append(action)
        (tmp_path / 'default-descendant-cleanup.json').write_text(
            json.dumps(cleanup, indent=2) + '\n', encoding='utf-8')
        for stream in (child.stdout, child.stderr):
            if stream is not None:
                stream.close()
    assert cleanup == [], cleanup
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in EXAMPLES.glob('*.py')}
    for name in ('cv_checkpoint', 'search_support', 'tune_mnist'):
        assert Path(value['origins'][name]) == (copied / (name + '.py')).resolve()
    assert Path(value['origins']['dwd']) == Path(dwd.__file__).resolve()
    assert Path(value['origins']['ensemble_dwd']) == Path(ensemble_dwd.__file__).resolve()
    assert Path(value['function_origin']) == (copied / 'search_support.py').resolve()
    assert Path(value['cache_path']) == Path(value['generated_cache'])
    assert value['before_sha256'] == before['search_support.py']
    assert value['same_size'] and value['same_mtime'] and value['timestamp_cache']
    assert value['magic_matches'] and value['cache_unchanged']
    assert value['cold'] and value['body_matches']
    assert value['loaded_default'] == {'exact': None}
    assert value['loaded_rank'] == [0, 1]
    if mode == 'stale-default':
        assert value['source_default'] is True
        assert value['before_sha256'] != value['after_sha256']
        assert value['explicit_source_rank'] == [1, 0]
        assert value['explicit_second_fraction'] == [2 ** 79 + 1, 2 ** 80]
        assert value['guard'] == 'refused'
    else:
        assert value['source_default'] is None
        assert value['before_sha256'] == value['after_sha256']
        assert value['explicit_source_rank'] == [0, 1]
        assert value['explicit_second_fraction'] is None
        assert value['guard'] == 'accepted'
