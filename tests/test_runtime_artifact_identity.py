"""Runtime alias and drift guards using owned temporary files.

Synthetic inventories exercise identity checks, not native-library execution.
All source/native/metadata mutations and links belong to the test temporary tree.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint

from runtime_test_support import owned_alias, remove_alias, physical_path


def reset_runtime(monkeypatch):
    values = {
        '_RUNTIME_FILES': None, '_RUNTIME_PACKAGE_FILES': None,
        '_RUNTIME_PACKAGES': None, '_RUNTIME_MODULES': None,
        '_RUNTIME_DIGEST': None, '_RUNTIME_NATIVE': None,
        '_RUNTIME_ACCELERATOR': None, '_RUNTIME_GUARDS': set(),
        '_RUNTIME_CONTROLLER': None, '_RUNTIME_MODULE_COUNT': None,
        '_RESOLVED_PATHS': {}, '_HASHES': {},
        '_SELECTION_STATE': None, '_SELECTION_COMPILE_EVIDENCE': (),
        '_RUNTIME_ARTIFACT_ALIASES': {}, '_RUNTIME_CANONICAL_ARTIFACTS': {},
        '_RUNTIME_FILE_COUNT': None, '_RUNTIME_COMMON_FILES': None,
        '_RUNTIME_COMMON_DIGEST': None, '_RUNTIME_COMMON_FILE_COUNT': None,
    }
    for name, value in values.items():
        monkeypatch.setattr(checkpoint, name, value)


@pytest.fixture(autouse=True)
def fresh_runtime(monkeypatch):
    reset_runtime(monkeypatch)
    assert checkpoint._RUNTIME_ARTIFACT_ALIASES == {}
    assert checkpoint._RUNTIME_CANONICAL_ARTIFACTS == {}
    assert checkpoint._RUNTIME_FILE_COUNT is None


@pytest.fixture
def tiny_runtime(tmp_path, monkeypatch):
    """Real file hashing/canonicalization with declared synthetic pool records."""
    import ensemble_dwd._cache_identity as cache_identity
    root = tmp_path / 'runtime'
    root.mkdir()
    source = root / '__init__.py'
    source.write_text('__version__ = "guard-fixture"\n', encoding='utf-8')
    binary = root / 'backend.dll'
    binary.write_bytes(b'IR14 nonexecutable native-identity fixture')
    second = root / 'unused_backend.dll'
    second.write_bytes(b'IR14 second nonexecutable identity fixture')
    metadata = tmp_path / 'guard_runtime.dist-info' / 'METADATA'
    metadata.parent.mkdir()
    metadata.write_text('Name: ir14-guard-runtime\nVersion: 1\n', encoding='utf-8')
    name = '_ir14_guard_runtime'
    module = SimpleNamespace(__file__=str(source), __path__=[str(root)], __version__='1')
    distribution = SimpleNamespace(version='1', files=['guard_runtime.dist-info/METADATA'],
                                   locate_file=lambda unused: metadata)
    pools = [dict(filepath=str(binary), user_api='blas', internal_api='synthetic',
                  num_threads=1, version='1', architecture='fixture')]
    real_import = checkpoint.importlib.import_module

    def import_module(requested, *args, **kwargs):
        return module if requested == name else real_import(requested, *args, **kwargs)

    monkeypatch.setattr(checkpoint, '_runtime_package_names', lambda: [(name, 'ir14-guard-runtime')])
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(checkpoint.importlib, 'import_module', import_module)
    monkeypatch.setattr(checkpoint.importlib.metadata, 'distribution', lambda unused: distribution)
    monkeypatch.setattr(cache_identity, '_package_identity', lambda: {'native': 'synthetic-fixture'})
    monkeypatch.setattr(checkpoint, 'ThreadpoolController',
                        lambda: SimpleNamespace(info=lambda: deepcopy(pools)))
    monkeypatch.setattr(checkpoint.np, 'dot', lambda *args, **kwargs: None)
    return SimpleNamespace(root=root, source=source, binary=binary, second=second,
                           metadata=metadata, module=module, pools=pools, name=name)


def assert_coordinator_refuses(frozen, temporary_parent):
    # Supply the new, valid session authority so the existing runtime-drift
    # assertion reaches the actual full-P comparison, not a missing-attribute fault.
    runtime = checkpoint._stable(frozen)
    identity = {'runtime': runtime}
    identity_hash = checkpoint._digest(identity)
    token = '1' * 32
    session_path = temporary_parent / 'synthetic-coordinator-session.json'
    session = dict(identity_sha256=identity_hash, owner_token=token,
                   runtime=runtime, execution_runtime_sha256=checkpoint._digest(runtime),
                   policy='execution-role-v1')
    checkpoint._write_receipt(session_path, session)
    subject = SimpleNamespace(identity=identity, identity_sha256=identity_hash,
                              pair_recovery=False, _pair_recovery_config=None,
                              runtime=runtime, execution_runtime=runtime, token=token,
                              _runtime_session=session, _runtime_session_path=session_path)
    with pytest.raises(checkpoint.CheckpointError, match='Coordinator runtime changed'):
        checkpoint.CheckpointedCV._check_runtime(subject)


def test_genuine_alias_is_one_identity_and_cached_read_keeps_live_stat(tmp_path, monkeypatch):
    target = tmp_path / 'target'
    target.mkdir()
    source = target / 'source.py'
    source.write_bytes(b'answer = 1\n')
    link = tmp_path / 'alias'
    receipt = owned_alias(link, physical_path(target), tmp_path)
    try:
        direct = checkpoint._runtime_artifact_identity(source)
        aliased = checkpoint._runtime_artifact_identity(link / source.name)
        assert direct == aliased
        assert direct[2] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert direct[1][:2] == (source.stat().st_dev, source.stat().st_ino)
        assert len(checkpoint._RUNTIME_ARTIFACT_ALIASES) == 2
        assert len(checkpoint._RUNTIME_CANONICAL_ARTIFACTS) == 1

        def forbidden(*args, **kwargs):
            raise AssertionError('Unchanged alias must not re-resolve or reread bytes')

        with monkeypatch.context() as repeated:
            repeated.setattr(Path, 'resolve', forbidden)
            repeated.setattr(Path, 'open', forbidden)
            assert checkpoint._runtime_artifact_identity(link / source.name) == direct
    finally:
        remove_alias(receipt, tmp_path)


def test_cached_alias_retarget_to_distinct_same_bytes_is_refused(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    old, new = first / 'source.py', second / 'source.py'
    old.write_bytes(b'answer = 1\n')
    shutil.copy2(old, new)
    original = old.read_bytes()
    assert not os.path.samefile(old, new)
    link = tmp_path / 'alias'
    receipt = owned_alias(link, physical_path(first), tmp_path)
    try:
        identity = checkpoint._runtime_artifact_identity(link / old.name)
        remove_alias(receipt, tmp_path)
        receipt = owned_alias(link, physical_path(second), tmp_path)
        assert new.read_bytes() == original
        with pytest.raises(checkpoint.CheckpointError, match='Runtime artifact changed in this process'):
            checkpoint._runtime_artifact_identity(link / old.name)
        assert old.read_bytes() == original
        assert checkpoint._runtime_artifact_identity(old)[2] == identity[2]
    finally:
        remove_alias(receipt, tmp_path)


def test_equal_bytes_in_distinct_files_do_not_establish_an_alias(tmp_path):
    first, second = tmp_path / 'first.py', tmp_path / 'second.py'
    first.write_bytes(b'answer = 1\n')
    shutil.copy2(first, second)
    a = checkpoint._runtime_artifact_identity(first)
    b = checkpoint._runtime_artifact_identity(second)
    assert a[2] == b[2]
    assert a[0] != b[0] and a[1][:2] != b[1][:2]
    assert len(checkpoint._RUNTIME_CANONICAL_ARTIFACTS) == 2


@pytest.mark.parametrize('artifact', ['source', 'binary', 'metadata'])
def test_actual_owned_artifact_mutation_is_not_normalized_away(tiny_runtime, artifact):
    checkpoint.runtime_fingerprint()
    target = getattr(tiny_runtime, artifact)
    target.write_bytes(target.read_bytes() + b'\nchanged')
    with pytest.raises(checkpoint.CheckpointError, match='Runtime artifact changed in this process'):
        checkpoint.runtime_fingerprint(check_inventory=False)


@pytest.mark.parametrize('field,value', [
    ('version', 'different-version'), ('architecture', 'different-architecture'),
    ('num_threads', 2),
])
def test_native_metadata_and_count_differences_still_refuse(tiny_runtime, field, value):
    before = checkpoint.runtime_fingerprint()
    tiny_runtime.pools[0][field] = value
    if field == 'num_threads':
        with pytest.raises(checkpoint.CheckpointError, match='one active native thread per worker'):
            checkpoint.runtime_fingerprint(check_inventory=False)
    else:
        after = checkpoint.runtime_fingerprint(check_inventory=False)
        assert checkpoint._stable(after) != checkpoint._stable(before)
        assert after['native_pools'][0][field] == value
        assert_coordinator_refuses(before, tiny_runtime.root.parent)


def test_extra_synthetic_pool_remains_a_strict_identity_difference(tiny_runtime):
    before = checkpoint.runtime_fingerprint()
    tiny_runtime.pools.append(dict(filepath=str(tiny_runtime.second), user_api='openmp',
                                   internal_api='synthetic-extra', num_threads=1, version='1'))
    after = checkpoint.runtime_fingerprint(check_inventory=False)
    assert len(after['native_pools']) == len(before['native_pools']) + 1
    assert_coordinator_refuses(before, tiny_runtime.root.parent)


def test_full_inventory_digest_and_count_deduplicate_only_proven_alias(
        tiny_runtime, tmp_path, monkeypatch):
    original = checkpoint.runtime_fingerprint()
    original_raw_count = len(checkpoint._RUNTIME_FILES)
    link = tmp_path / 'runtime-alias'
    receipt = owned_alias(link, physical_path(tiny_runtime.root), tmp_path)
    try:
        reset_runtime(monkeypatch)
        tiny_runtime.pools[0]['filepath'] = str(link / tiny_runtime.binary.name)
        aliased = checkpoint.runtime_fingerprint()
        assert len(checkpoint._RUNTIME_FILES) == original_raw_count + 1
        assert aliased['file_count'] == original['file_count']
        assert aliased['files_sha256'] == original['files_sha256']
        assert checkpoint._stable(aliased) == checkpoint._stable(original)
    finally:
        remove_alias(receipt, tmp_path)


def test_mismatch_diagnostics_are_bounded_fields_not_data_or_normalization():
    names = ('python', 'implementation', 'machine', 'packages', 'package_native',
             'files_sha256', 'file_count', 'native_pools')
    a = checkpoint._stable({name: 'first-private-value' for name in names})
    b = checkpoint._stable({name: 'second-private-value' for name in names})
    copies = deepcopy((a, b))
    text = checkpoint._runtime_mismatch_details(a, b)
    assert text == ' (differing fields: python, implementation, machine, packages; more)'
    assert 'private-value' not in text
    assert (a, b) == copies and a != b
    for malformed in (None, ['dict', [['python', 'x'], ['python', 'y']]], ['unknown']):
        assert checkpoint._runtime_mismatch_details(a, malformed) == ' (runtime record structure differs)'
