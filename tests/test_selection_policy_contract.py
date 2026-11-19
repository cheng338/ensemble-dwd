"""Release coverage for the finite selector dependency and declaration contract.

Only selected helper code is inspected, not all functions in the CLI modules.
Edited sources are copied and never executed; this is not an interpreter-wide
or concurrent-tampering guarantee.
"""
import ast
import builtins
import dis
import hashlib
from pathlib import Path
import shutil
import sys
from types import CodeType

import pytest

from test_memory_selection_guard import (
    FUNCTIONS, checkpoint, search_support, tune_mnist, uninitialized_guard)


def test_selected_helpers_have_complete_finite_dependency_bindings(uninitialized_guard):
    checkpoint.check_selection_state()
    states, bindings, attributes, missing_globals, _ = checkpoint._SELECTION_STATE
    names = {name for _, name in FUNCTIONS}
    assert len(states) == len(names) == 14
    assert {(module, name) for module, name, _, _ in states} == set(FUNCTIONS)
    evidence = checkpoint._SELECTION_COMPILE_EVIDENCE
    assert len(evidence) == 14 and {row[-1] for row in evidence} == names
    binding_map = {(id(mapping), key): value for mapping, key, value in bindings}
    missing = {(id(mapping), key) for mapping, key in missing_globals}
    attribute_map = {(id(owner), key): value for owner, key, value in attributes}
    compiled, declarations, digests, source_builtins = {}, {}, {}, {}
    imports = {'np', 're', 'Fraction'}
    constants = {'ACCURACY_AGGREGATION'}
    expected_attributes = {}
    for module, name, function, code in states:
        assert function is getattr(module, name) and function.__code__ is code
        assert function.__globals__ is module.__dict__
        path = Path(module.__file__).resolve()
        assert path.parent == Path(checkpoint.__file__).resolve().parent
        assert Path(code.co_filename).resolve() == path
        key = (module.__name__, code.co_filename)
        if key not in compiled:
            source = path.read_bytes()
            digests[module.__name__] = hashlib.sha256(source).hexdigest()
            parsed = ast.parse(source, filename=code.co_filename)
            declarations[module.__name__] = {node.name: node for node in parsed.body
                                             if isinstance(node, ast.FunctionDef)}
            module_code = compile(source, code.co_filename, 'exec', dont_inherit=True,
                                  optimize=sys.flags.optimize)
            compiled[key] = {item.co_name: item for item in module_code.co_consts
                             if isinstance(item, CodeType)}
        assert code == compiled[key][name]
        assert (module.__name__, digests[module.__name__], code.co_filename, name) in evidence
        reads, pending = set(), [compiled[key][name]]
        while pending:
            selected = pending.pop()
            reads.update(instruction.argval for instruction in dis.Bytecode(selected)
                         if instruction.opname in ('LOAD_GLOBAL', 'LOAD_NAME'))
            pending.extend(value for value in selected.co_consts if isinstance(value, CodeType))
        builtin_reads = reads - names - imports - constants
        source_builtins.setdefault(module, set()).update(builtin_reads)
        for global_name in reads:
            if global_name in builtin_reads:
                assert global_name in vars(builtins)
                assert global_name not in module.__dict__
                assert (id(module.__dict__), global_name) in missing
                assert binding_map[(id(function.__builtins__), global_name)] is getattr(builtins, global_name)
            else:
                assert binding_map[(id(module.__dict__), global_name)] is module.__dict__[global_name]
        for member in ast.walk(declarations[module.__name__][name]):
            if not isinstance(member, ast.Attribute):
                continue
            parts, base = [], member
            while isinstance(base, ast.Attribute):
                parts.append(base.attr)
                base = base.value
            if not isinstance(base, ast.Name) or base.id not in imports:
                continue
            parts.reverse()
            owner = module.__dict__[base.id]
            for part in parts[:-1]:
                owner = getattr(owner, part)
            expected_attributes[(id(owner), parts[-1])] = getattr(owner, parts[-1])
    for module in (tune_mnist, search_support):
        guarded = {name for mapping, name in missing_globals if mapping is module.__dict__}
        assert source_builtins[module] == guarded
    assert set(expected_attributes) == set(attribute_map)
    assert all(attribute_map[key] is value for key, value in expected_attributes.items())


DECLARATIONS = {
    'unchanged': None,
    'positional-default': ('def accuracy_fraction(counts):',
                           'def accuracy_fraction(counts=None):'),
    'wrong-keyword-name': ('def score_ranks(results, *, exact=None):',
                           'def score_ranks(results, *, different=None):'),
    'missing-keyword-argument': ('def score_value(results, index, *, exact=None):',
                                'def score_value(results, index):'),
    'missing-keyword-default': ('def score_value(results, index, *, exact=None):',
                               'def score_value(results, index, *, exact):'),
    'nonliteral-keyword-default': ('def score_value(results, index, *, exact=None):',
                                  'def score_value(results, index, *, exact=_must_not_execute()):'),
    'unexpected-ordinary-keyword': ('def accuracy_fraction(counts):',
                                    'def accuracy_fraction(counts, *, extra=None):'),
    'duplicate-selected-definition': ('def accuracy_fraction(counts):',
                                      'def accuracy_fraction(counts):\n    return counts\n\ndef accuracy_fraction(counts):'),
    'decorated-selected-definition': ('def accuracy_fraction(counts):',
                                      '@_must_not_execute\ndef accuracy_fraction(counts):'),
    'async-selected-definition': ('def accuracy_fraction(counts):',
                                  'async def accuracy_fraction(counts):'),
}


def relocated_code(code, filename):
    return code.replace(co_filename=str(filename), co_consts=tuple(
        relocated_code(value, filename) if isinstance(value, CodeType) else value
        for value in code.co_consts))


@pytest.fixture
def copied_policy(tmp_path, monkeypatch):
    modules = (checkpoint, tune_mnist, search_support)
    original_paths = {module: module.__file__ for module in modules}
    original_bytes = {module: Path(path).read_bytes() for module, path in original_paths.items()}
    assert {Path(path).resolve().parent for path in original_paths.values()} == {
        (Path(__file__).resolve().parents[1] / 'examples').resolve()}
    functions = {name: getattr(module, name) for module, name in FUNCTIONS}
    owners = {name: module for module, name in FUNCTIONS}
    original_codes = {name: function.__code__ for name, function in functions.items()}
    cache_names = ('_SELECTION_STATE', '_SELECTION_COMPILE_EVIDENCE', '_HASHES', '_RESOLVED_PATHS')
    original_caches = {name: getattr(checkpoint, name) for name in cache_names}
    copied = tmp_path / 'examples'
    copied.mkdir()
    for module, path in original_paths.items():
        shutil.copy2(path, copied / Path(path).name)
    with monkeypatch.context() as bounded:
        for module, path in original_paths.items():
            bounded.setattr(module, '__file__', str(copied / Path(path).name))
        for name, function in functions.items():
            bounded.setattr(function, '__code__', relocated_code(original_codes[name], owners[name].__file__))
        bounded.setattr(checkpoint, '_SELECTION_STATE', None)
        bounded.setattr(checkpoint, '_SELECTION_COMPILE_EVIDENCE', ())
        bounded.setattr(checkpoint, '_HASHES', {})
        bounded.setattr(checkpoint, '_RESOLVED_PATHS', {})
        yield copied
    assert all(module.__file__ == path for module, path in original_paths.items())
    assert all(functions[name].__code__ is code for name, code in original_codes.items())
    assert all(getattr(checkpoint, name) is value for name, value in original_caches.items())
    assert all(Path(path).read_bytes() == original_bytes[module]
               for module, path in original_paths.items())


@pytest.mark.parametrize('case_id', DECLARATIONS, ids=list(DECLARATIONS))
def test_selected_source_declaration_is_checked_before_bytecode(copied_policy, monkeypatch, case_id):
    path = copied_policy / 'search_support.py'
    before = path.read_bytes()
    edit = DECLARATIONS[case_id]
    if edit is not None:
        old, new = (value.encode('utf-8') for value in edit)
        if b'\r\n' in before:
            old, new = old.replace(b'\n', b'\r\n'), new.replace(b'\n', b'\r\n')
        assert before.count(old) == 1
        path.write_bytes(before.replace(old, new))
    real_compile, compiled_modules = builtins.compile, []

    def observe_compile(*args, **kwargs):
        result = real_compile(*args, **kwargs)
        if isinstance(result, CodeType) and Path(result.co_filename).parent == copied_policy:
            compiled_modules.append(Path(result.co_filename).name)
        return result

    def forbid_exec(*args, **kwargs):
        raise AssertionError('Edited declaration source must never execute')

    with monkeypatch.context() as bounded:
        bounded.setattr(builtins, 'compile', observe_compile)
        bounded.setattr(builtins, 'exec', forbid_exec)
        if edit is None:
            checkpoint.check_selection_state()
        else:
            with pytest.raises(checkpoint.CheckpointError, match='Selection policy changed in memory'):
                checkpoint.check_selection_state()
    if edit is None:
        assert set(compiled_modules) == {'tune_mnist.py', 'search_support.py'}
        assert len(checkpoint._SELECTION_COMPILE_EVIDENCE) == 14
        assert checkpoint._SELECTION_STATE is not None
    else:
        assert compiled_modules == []
        assert checkpoint._SELECTION_COMPILE_EVIDENCE == ()
        assert checkpoint._SELECTION_STATE is None
