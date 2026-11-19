"""Finite in-memory selection guards; edits stay test-owned.

No same-interpreter guard is a hostile-code security boundary. These tests cover
the explicit example-function, dependency and consumer-alias contracts only.
"""
import builtins
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import CodeType

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import adaptive_search
import cv_checkpoint as checkpoint
import numerical_confirmation
import ratio_refinement
import search_support
import tune_mnist


FUNCTIONS = (
    (tune_mnist, 'fully_valid_candidate_mask'),
    (tune_mnist, 'ranked_valid_indices'),
    (search_support, 'score_value'),
    (search_support, 'accuracy_evidence'),
    (search_support, 'accuracy_fraction'),
    (search_support, '_fraction_record'),
    (search_support, 'exact_accuracy_available'),
    (search_support, 'score_token'),
    (search_support, 'token_value'),
    (search_support, 'comparable_tokens'),
    (search_support, 'equal_score_tokens'),
    (search_support, 'score_nonimproving'),
    (search_support, 'apply_accuracy_evidence'),
    (search_support, 'score_ranks'),
)


@pytest.fixture
def uninitialized_guard(monkeypatch):
    monkeypatch.setattr(checkpoint, '_SELECTION_STATE', None)
    monkeypatch.setattr(checkpoint, '_SELECTION_COMPILE_EVIDENCE', ())
    monkeypatch.setattr(checkpoint, '_HASHES', {})
    monkeypatch.setattr(checkpoint, '_RESOLVED_PATHS', {})


@pytest.fixture
def initialized_guard(uninitialized_guard):
    checkpoint.check_selection_state()


def changed_code(function):
    def replacement(*args, **kwargs):
        return 'changed selection code'
    return replacement.__code__.replace(
        co_filename=function.__code__.co_filename,
        co_name=function.__code__.co_name,
        co_qualname=function.__code__.co_qualname)


@pytest.mark.parametrize('module,name', FUNCTIONS,
                         ids=[name for _, name in FUNCTIONS])
def test_same_function_code_mutation_is_refused(module, name, initialized_guard, monkeypatch):
    function = getattr(module, name)
    filename = function.__code__.co_filename
    original = function.__code__
    with monkeypatch.context() as scoped:
        scoped.setattr(function, '__code__', changed_code(function))
        assert getattr(module, name) is function
        assert function.__code__.co_filename == filename
        with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
            checkpoint.check_selection_state()
    assert function.__code__ is original
    checkpoint.check_selection_state()


@pytest.mark.parametrize('module,name', FUNCTIONS,
                         ids=[name for _, name in FUNCTIONS])
def test_positional_defaults_are_not_adopted(module, name, initialized_guard, monkeypatch):
    monkeypatch.setattr(getattr(module, name), '__defaults__', (None,))
    with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
        checkpoint.check_selection_state()


@pytest.mark.parametrize('name', ['fully_valid_candidate_mask', 'ranked_valid_indices'])
def test_selector_keyword_defaults_are_refused(name, initialized_guard, monkeypatch):
    monkeypatch.setattr(getattr(tune_mnist, name), '__kwdefaults__', {'unexpected': None})
    with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
        checkpoint.check_selection_state()


@pytest.mark.parametrize('name', ['score_value', 'score_ranks'])
def test_score_keyword_default_in_place_mutation_is_refused(name, initialized_guard, monkeypatch):
    function = getattr(search_support, name)
    keywords = function.__kwdefaults__
    monkeypatch.setitem(keywords, 'exact', False)
    assert function.__kwdefaults__ is keywords
    with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
        checkpoint.check_selection_state()


@pytest.mark.parametrize('module', [search_support, checkpoint], ids=['support', 'checkpoint'])
def test_accuracy_aggregation_constant_mutation_is_refused(module, initialized_guard, monkeypatch):
    monkeypatch.setattr(module, 'ACCURACY_AGGREGATION', 'changed-accuracy-policy')
    with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
        checkpoint.check_selection_state()


@pytest.mark.parametrize('module,name', [
    (tune_mnist, 'np'), (tune_mnist, 're'), (tune_mnist, 'score_value'),
    (tune_mnist, 'fully_valid_candidate_mask'), (tune_mnist, 'ranked_valid_indices'),
    (search_support, 'accuracy_evidence'), (search_support, 'Fraction'),
    (checkpoint, 'score_value'), (search_support, 'score_token'),
    (checkpoint, 'apply_accuracy_evidence'), (checkpoint, 'score_ranks'),
], ids=['tune-np', 'tune-re', 'tune-score', 'tune-mask', 'tune-rank',
        'support-evidence', 'support-Fraction', 'checkpoint-score', 'support-token',
        'checkpoint-evidence', 'checkpoint-ranks'])
def test_bound_global_alias_change_is_refused(module, name, initialized_guard, monkeypatch):
    monkeypatch.setattr(module, name, object())
    with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
        checkpoint.check_selection_state()


@pytest.mark.parametrize('owner,name', [
    (np, 'ma'), (np.ma, 'asarray'), (np, 'nan'), (np, 'isfinite'),
    (np, 'flatnonzero'), (np, 'asarray'), (tune_mnist.re, 'fullmatch'),
    (np, 'std'), (np, 'int32'),
], ids=['np-ma', 'ma-asarray', 'np-nan', 'np-isfinite', 'np-flatnonzero',
        'np-asarray', 're-fullmatch', 'np-std', 'np-int32'])
def test_direct_dependency_attribute_rebinding_is_refused(owner, name, initialized_guard, monkeypatch):
    monkeypatch.setattr(owner, name, object())
    with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
        checkpoint.check_selection_state()


@pytest.mark.parametrize('module,name', [
    (adaptive_search, 'fully_valid_candidate_mask'),
    (adaptive_search, 'ranked_valid_indices'),
    (adaptive_search, 'score_value'),
    (adaptive_search, 'score_token'),
    (adaptive_search, 'token_value'),
    (adaptive_search, 'comparable_tokens'),
    (adaptive_search, 'equal_score_tokens'),
    (adaptive_search, 'score_nonimproving'),
    (ratio_refinement, 'score_token'),
    (ratio_refinement, 'comparable_tokens'),
    (numerical_confirmation, 'score_value'),
    (numerical_confirmation, 'exact_accuracy_available'),
    (numerical_confirmation, 'score_ranks'),
], ids=['adaptive-mask', 'adaptive-rank', 'adaptive-score',
        'adaptive-token', 'adaptive-token-value', 'adaptive-comparable',
        'adaptive-equal', 'adaptive-nonimproving', 'ratio-token', 'ratio-comparable',
        'confirmation-score', 'confirmation-exact', 'confirmation-ranks'])
def test_consumer_alias_change_is_refused(module, name, initialized_guard, monkeypatch):
    monkeypatch.setattr(module, name, object())
    with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
        checkpoint.check_selection_state()


@pytest.mark.parametrize('module,name', [(tune_mnist, 'sorted'), (search_support, 'sum')])
def test_global_shadow_of_used_builtin_is_refused(module, name, initialized_guard, monkeypatch):
    monkeypatch.setattr(module, name, object(), raising=False)
    with pytest.raises(checkpoint.CheckpointError, match='changed in memory'):
        checkpoint.check_selection_state()


def test_initialization_compiles_without_executing_then_caches(uninitialized_guard, monkeypatch):
    compile_calls = []
    parse_calls = []
    real_compile = builtins.compile

    def observed_compile(source, filename, mode, *args, **kwargs):
        result = real_compile(source, filename, mode, *args, **kwargs)
        if isinstance(result, CodeType):
            compile_calls.append((type(source), filename, mode, args, kwargs))
        elif isinstance(result, ast.Module):
            parse_calls.append(filename)
        return result

    def forbidden_exec(*args, **kwargs):
        raise AssertionError('Selection initialization must not execute source')

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, 'compile', observed_compile)
        scoped.setattr(builtins, 'exec', forbidden_exec)
        checkpoint.check_selection_state()
        states = checkpoint._SELECTION_STATE
        evidence = checkpoint._SELECTION_COMPILE_EVIDENCE
        checkpoint.check_selection_state()
    assert checkpoint._SELECTION_STATE is states
    assert len(compile_calls) == 2
    assert len(parse_calls) == 2
    assert {Path(filename).name for filename in parse_calls} == {'tune_mnist.py', 'search_support.py'}
    assert {Path(row[1]).name for row in compile_calls} == {'tune_mnist.py', 'search_support.py'}
    for source_type, filename, mode, args, kwargs in compile_calls:
        assert source_type is bytes
        assert mode == 'exec'
        assert args == ()
        assert kwargs == {'dont_inherit': True, 'optimize': sys.flags.optimize}
    assert len(evidence) == len(FUNCTIONS) == 14
    assert {entry[3] for entry in evidence} == {name for _, name in FUNCTIONS}
    assert all(len(entry[1]) == 64 for entry in evidence)


def test_repeated_memory_check_does_not_read_source(initialized_guard, monkeypatch):
    def forbidden_read(*args, **kwargs):
        raise AssertionError('A warmed in-memory guard must not reread files')
    monkeypatch.setattr(Path, 'read_bytes', forbidden_read)
    checkpoint.check_selection_state()
    checkpoint.check_selection_state()


@pytest.mark.parametrize('case', ['ties', 'nonfinite', 'masked', 'exact-counts'])
def test_unmodified_selection_results_are_preserved(case, initialized_guard):
    if case == 'exact-counts':
        denominator = 2 ** 54
        evidence = [search_support.accuracy_evidence([[{'correct': numerator,
                     'n_validation': denominator}]])
                    for numerator in (denominator // 2, denominator // 2 + 1)]
        results = {'params': [{}, {}], 'mean_test_score': np.array([.5, .5]),
                   'split0_test_score': np.array([.5, .5]), 'accuracy_evidence': evidence}
        expected = [1, 0]
    else:
        means = np.array([.5, .5, np.nan, .7])
        results = {'params': [{}, {}, {}, {}], 'mean_test_score': means,
                   'split0_test_score': means.copy(), 'split1_test_score': means.copy()}
        expected = [3, 0, 1]
        if case == 'nonfinite':
            results['split1_test_score'][3] = np.inf
            expected = [0, 1]
        elif case == 'masked':
            results['mean_test_score'] = np.ma.array(means, mask=[False, False, False, True])
            expected = [0, 1]
    checkpoint.check_selection_state()
    actual = tune_mnist.ranked_valid_indices(results)
    assert actual.tolist() == expected
    assert tune_mnist.fully_valid_candidate_mask(results)[actual].all()
    checkpoint.check_selection_state()


PREINITIALIZATION_DRIVER = r'''
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import tune_mnist
import search_support
mode = sys.argv[2]
function = (tune_mnist.ranked_valid_indices if mode == 'selector-code'
            else search_support.token_value if mode == 'token-code'
            else search_support.score_value)
filename = function.__code__.co_filename
if mode.endswith('code'):
    def replacement(*args, **kwargs):
        return 'changed-before-guard'
    function.__code__ = replacement.__code__.replace(
        co_filename=filename, co_name=function.__code__.co_name,
        co_qualname=function.__code__.co_qualname)
elif mode == 'defaults':
    function.__defaults__ = (None,)
elif mode == 'kwdefaults':
    function.__kwdefaults__['exact'] = False
elif mode == 'support-constant':
    search_support.ACCURACY_AGGREGATION = 'changed-before-initialization'
import cv_checkpoint
if mode == 'checkpoint-constant':
    cv_checkpoint.ACCURACY_AGGREGATION = 'changed-before-initialization'
try:
    cv_checkpoint.check_selection_state()
except cv_checkpoint.CheckpointError as error:
    outcome = 'refused'
    message = str(error)
else:
    outcome = 'accepted'
    message = None
print(json.dumps({'outcome': outcome, 'message': message,
                  'filename_preserved': function.__code__.co_filename == filename}))
'''


@pytest.mark.parametrize('mode', ['selector-code', 'support-code', 'token-code', 'defaults', 'kwdefaults',
                                 'support-constant', 'checkpoint-constant'])
def test_preinitialization_mutation_is_refused_in_owned_copy(mode, tmp_path):
    import dwd
    import ensemble_dwd
    copied = tmp_path / 'examples'
    copied.mkdir()
    examples = Path(checkpoint.__file__).resolve().parent
    for source in examples.glob('*.py'):
        shutil.copy2(source, copied / source.name)
    environment = dict(os.environ)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONOPTIMIZE'] = str(sys.flags.optimize)
    environment.pop('PYTHONHOME', None)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    environment['PYTHONPATH'] = os.pathsep.join([
        str(Path(dwd.__file__).resolve().parent.parent),
        str(Path(ensemble_dwd.__file__).resolve().parent.parent),
        str(copied)])
    completed = subprocess.run(
        [sys.executable, '-B', '-c', PREINITIALIZATION_DRIVER, str(copied), mode],
        env=environment, cwd=tmp_path, text=True, capture_output=True, timeout=90,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result['outcome'] == 'refused', result
    assert result['filename_preserved']
