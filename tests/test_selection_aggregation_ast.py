"""Focused controls for the deliberately stricter fixed-source declaration.

The historical predicate below reads unexecuted bytecode only. Source snippets
are never executed, and its observations do not replace the full fourteen-helper
source/code/global contract, exercised separately through copied_policy.
"""
import ast
import builtins
import dis
from pathlib import Path
import sys
from types import CodeType

import pytest

from test_selection_policy_contract import copied_policy
from test_memory_selection_guard import checkpoint


NAME = 'ACCURACY_AGGREGATION'
VALUE = 'exact-mean-seeds-within-fold-then-folds-v1'
LITERAL = NAME + ' = ' + repr(VALUE)
ERROR = 'Selection policy changed in memory; restart before resuming'


def historical_literal(source, *, optimize=None):
    """Exact previous extraction predicate, isolated from installed modules."""
    if optimize is None:
        optimize = sys.flags.optimize
    code = compile(source, '<aggregation-declaration>', 'exec', dont_inherit=True,
                   optimize=optimize)
    instructions = tuple(dis.get_instructions(code))
    stores = [index for index, instruction in enumerate(instructions)
              if instruction.opname == 'STORE_NAME' and instruction.argval == NAME]
    if len(stores) != 1 or stores[0] == 0:
        return None
    declaration = instructions[stores[0] - 1]
    if declaration.opname != 'LOAD_CONST' or type(declaration.argval) is not str:
        return None
    return declaration.argval


DECLARATIONS = [
    ('literal', LITERAL, True, True),
    ('parenthesized', NAME + ' = (' + repr(VALUE) + ')', True, True),
    ('adjacent-literals', NAME + " = 'exact-mean-' 'seeds-within-fold-then-folds-v1'", True, True),
    ('folded-expression', NAME + " = 'exact-mean-' + 'seeds-within-fold-then-folds-v1'", True, False),
    ('annotated', NAME + ': str = ' + repr(VALUE), True, False),
    ('conditional', 'if True:\n    ' + LITERAL, True, False),
    ('dead-duplicate', LITERAL + '\nif False:\n    ' + LITERAL, True, False),
    ('nested-local-reuse', LITERAL + '\ndef unrelated():\n    ' + LITERAL, True, False),
    ('direct-duplicate', LITERAL + '\n' + LITERAL, False, False),
    ('dynamic-expression', NAME + ' = must_not_execute()', False, False),
]


@pytest.mark.parametrize('case,source,old_accepts,new_accepts', DECLARATIONS,
                         ids=[row[0] for row in DECLARATIONS])
def test_historical_and_ast_acceptance_are_explicit(
        case, source, old_accepts, new_accepts, monkeypatch, record_property):
    def forbid_exec(*args, **kwargs):
        raise AssertionError('Declaration source must not execute')
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, 'exec', forbid_exec)
        old = historical_literal(source)
        assert (old == VALUE) is old_accepts
        tree = ast.parse(source, filename='<aggregation-declaration>')
        if new_accepts:
            assert checkpoint._selection_aggregation_literal(tree) == VALUE
        else:
            with pytest.raises(checkpoint.CheckpointError, match=ERROR):
                checkpoint._selection_aggregation_literal(tree)
    record_property('declaration_contract_case', case)
    record_property('historical_accepts', old_accepts)
    record_property('candidate_accepts', new_accepts)


def test_other_binding_shapes_and_nonliteral_values_refuse():
    # This is a finite syntactic contract, including conservative nested writes;
    # it does not analyze arbitrary namespace side effects or execute statements.
    suffixes = [
        NAME + " += 'suffix'", 'del ' + NAME,
        'import math as ' + NAME, 'from math import pi as ' + NAME,
        'def ' + NAME + '():\n    pass',
        'async def ' + NAME + '():\n    pass',
        'class ' + NAME + ':\n    pass',
        'try:\n    pass\nexcept Exception as ' + NAME + ':\n    pass',
        'match {}:\n    case {"x": ' + NAME + '}:\n        pass',
        'match []:\n    case [*' + NAME + ']:\n        pass',
        'match {}:\n    case {**' + NAME + '}:\n        pass',
        'def unrelated():\n    import math as ' + NAME,
        '(' + NAME + " := 'changed')",
        'for ' + NAME + ' in ():\n    pass',
    ]
    replacements = [
        NAME + ': str', NAME + ' = alias = ' + repr(VALUE),
        NAME + ', other = (' + repr(VALUE) + ', 0)',
        NAME + ' = b"bytes"', NAME + ' = 7', NAME + ' = f"literal"',
    ]
    for source in [*(LITERAL + '\n' + suffix for suffix in suffixes), *replacements]:
        with pytest.raises(checkpoint.CheckpointError, match=ERROR):
            checkpoint._selection_aggregation_literal(ast.parse(source))


def test_ast_policy_is_independent_of_compile_optimization():
    for optimization in (0, 1, 2):
        assert historical_literal(LITERAL, optimize=optimization) == VALUE
        assert checkpoint._selection_aggregation_literal(ast.parse(LITERAL)) == VALUE
        folded = DECLARATIONS[3][1]
        assert historical_literal(folded, optimize=optimization) == VALUE
        with pytest.raises(checkpoint.CheckpointError, match=ERROR):
            checkpoint._selection_aggregation_literal(ast.parse(folded))
    # The old source-level acceptance changed when __debug__ code was removed.
    duplicate = LITERAL + '\nif __debug__:\n    ' + LITERAL
    assert historical_literal(duplicate, optimize=0) is None
    assert historical_literal(duplicate, optimize=1) == VALUE
    assert historical_literal(duplicate, optimize=2) == VALUE
    with pytest.raises(checkpoint.CheckpointError, match=ERROR):
        checkpoint._selection_aggregation_literal(ast.parse(duplicate))


@pytest.mark.parametrize('case,replacement,accepted,compiled_count', [
    ('parenthesized', NAME + ' = (' + repr(VALUE) + ')', True, 2),
    ('adjacent', NAME + " = 'exact-mean-' 'seeds-within-fold-then-folds-v1'", True, 2),
    ('folded', NAME + " = 'exact-mean-' + 'seeds-within-fold-then-folds-v1'", False, 0),
    ('changed-literal', NAME + " = 'changed-but-literal'", False, 2),
])
def test_full_guard_keeps_code_and_loaded_value_checks(
        copied_policy, monkeypatch, case, replacement, accepted, compiled_count):
    path = copied_policy / 'search_support.py'
    original = path.read_bytes()
    old, new = LITERAL.encode('utf-8'), replacement.encode('utf-8')
    assert original.count(old) == 1 and b'\n' not in new
    path.write_bytes(original.replace(old, new))
    compiled = []
    real_compile = builtins.compile
    def observe_compile(*args, **kwargs):
        result = real_compile(*args, **kwargs)
        if isinstance(result, CodeType) and Path(result.co_filename).parent == copied_policy:
            compiled.append((Path(result.co_filename).name, kwargs))
        return result
    def forbid_exec(*args, **kwargs):
        raise AssertionError('Copied source must not execute')
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, 'compile', observe_compile)
        scoped.setattr(builtins, 'exec', forbid_exec)
        if accepted:
            checkpoint.check_selection_state()
            state = checkpoint._SELECTION_STATE
            checkpoint.check_selection_state()
            assert checkpoint._SELECTION_STATE is state
        else:
            with pytest.raises(checkpoint.CheckpointError, match=ERROR):
                checkpoint.check_selection_state()
    assert len(compiled) == compiled_count
    assert all(options == {'dont_inherit': True, 'optimize': sys.flags.optimize}
               for _, options in compiled)
    if accepted:
        assert {name for name, _ in compiled} == {'tune_mnist.py', 'search_support.py'}
        assert len(checkpoint._SELECTION_COMPILE_EVIDENCE) == 14
    else:
        assert checkpoint._SELECTION_STATE is None
        assert checkpoint._SELECTION_COMPILE_EVIDENCE == ()
