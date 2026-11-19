"""Selection source identity, cached reads and fail-closed runtime checks.

The small runtime fixture isolates source-guard behavior. A separate test uses
the actual runtime fingerprint. File edits are confined to owned temporary copies.
"""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from threadpoolctl import threadpool_limits

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint
import tune_mnist


@pytest.fixture
def fresh_runtime(monkeypatch):
    """Restore all checkpoint fingerprint caches after each test."""
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


@pytest.fixture
def tiny_runtime(tmp_path, monkeypatch, fresh_runtime):
    """Keep actual fingerprint/source-hash logic, replace package inventory."""
    import ensemble_dwd._cache_identity as cache_identity

    root = tmp_path / 'tiny_runtime'
    root.mkdir()
    source = root / '__init__.py'
    source.write_text('__version__ = "1"\n', encoding='utf-8')
    binary = root / 'test_backend.dll'
    binary.write_bytes(b'fixture identity bytes, never loaded as a library')
    metadata = tmp_path / 'tiny_runtime-1.dist-info' / 'METADATA'
    metadata.parent.mkdir()
    metadata.write_text('Name: tiny-runtime\nVersion: 1\n', encoding='utf-8')
    name = '_selection_identity_test_runtime'
    module = SimpleNamespace(__file__=str(source), __path__=[str(root)],
                             __version__='1')
    distribution = SimpleNamespace(version='1', files=['tiny_runtime-1.dist-info/METADATA'],
                                   locate_file=lambda unused: metadata)
    pool = {'filepath': str(binary), 'user_api': 'blas', 'internal_api': 'fixture',
            'num_threads': 1, 'version': '1'}
    controller = SimpleNamespace(info=lambda: [dict(pool)])
    real_import = checkpoint.importlib.import_module

    def import_module(requested, *args, **kwargs):
        if requested == name:
            return module
        return real_import(requested, *args, **kwargs)

    monkeypatch.setattr(checkpoint, '_runtime_package_names',
                        lambda: [(name, 'tiny-runtime')])
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(checkpoint.importlib, 'import_module', import_module)
    monkeypatch.setattr(checkpoint.importlib.metadata, 'distribution',
                        lambda unused: distribution)
    monkeypatch.setattr(cache_identity, '_package_identity',
                        lambda: {'test_native': 'fixed'})
    monkeypatch.setattr(checkpoint, 'ThreadpoolController', lambda: controller)
    monkeypatch.setattr(checkpoint.np, 'dot', lambda *args, **kwargs: None)
    return root


def selector_source():
    return Path(checkpoint.__file__).with_name('tune_mnist.py').resolve()


def test_selector_source_is_the_adjacent_defining_file():
    expected = selector_source()
    assert checkpoint._selection_source() == expected
    assert Path(tune_mnist.__file__).resolve() == expected
    for name in ('fully_valid_candidate_mask', 'ranked_valid_indices'):
        function = getattr(tune_mnist, name)
        assert Path(checkpoint.inspect.getsourcefile(function)).resolve() == expected


def test_shadow_module_origin_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mnist, '__file__', str(tmp_path / 'tune_mnist.py'))
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._selection_source()


@pytest.mark.parametrize('action', ['remove', 'replace'])
def test_lazy_import_module_rebinding_is_refused(action, monkeypatch):
    if action == 'remove':
        monkeypatch.delitem(sys.modules, 'tune_mnist')
    else:
        # Even equal metadata/functions must not make a second module object
        # equivalent to the module captured by checkpoint consumers.
        impostor = SimpleNamespace(
            __file__=tune_mnist.__file__,
            fully_valid_candidate_mask=tune_mnist.fully_valid_candidate_mask,
            ranked_valid_indices=tune_mnist.ranked_valid_indices)
        monkeypatch.setitem(sys.modules, 'tune_mnist', impostor)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._selection_source()


@pytest.mark.parametrize('name', ['fully_valid_candidate_mask', 'ranked_valid_indices'])
def test_replaced_selector_is_refused(name, monkeypatch):
    def replacement(*args, **kwargs):
        raise AssertionError('Source guard must not execute replacement selectors')

    monkeypatch.setattr(tune_mnist, name, replacement)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._selection_source()


@pytest.mark.parametrize('name', ['fully_valid_candidate_mask', 'ranked_valid_indices'])
def test_selector_defining_origin_is_checked(name, tmp_path, monkeypatch):
    function = getattr(tune_mnist, name)
    real_getsourcefile = checkpoint.inspect.getsourcefile

    def changed_source(value):
        if value is function:
            return str(tmp_path / 'unbound_selection.py')
        return real_getsourcefile(value)

    monkeypatch.setattr(checkpoint.inspect, 'getsourcefile', changed_source)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._selection_source()


@pytest.mark.parametrize('deep', [True, False])
def test_selector_is_hashed_on_deep_and_shallow_paths(deep, tiny_runtime, monkeypatch):
    expected = selector_source()
    checkpoint.runtime_fingerprint()
    calls = []
    real_hash = checkpoint._file_hash

    def tracked_hash(path):
        calls.append(Path(path).resolve())
        return real_hash(path)

    monkeypatch.setattr(checkpoint, '_file_hash', tracked_hash)
    result = checkpoint.runtime_fingerprint(check_inventory=deep)
    assert expected in calls
    assert str(expected) in checkpoint._RUNTIME_FILES
    assert checkpoint._HASHES[expected][1] == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert result['files_sha256'] == checkpoint._RUNTIME_DIGEST


def test_unchanged_selector_bytes_are_not_reread(tiny_runtime, monkeypatch):
    expected = selector_source()
    reads = []
    real_open = Path.open

    def tracked_open(path, *args, **kwargs):
        if path.resolve() == expected:
            reads.append((args, kwargs))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', tracked_open)
    assert checkpoint._SELECTION_STATE is None
    initial = checkpoint.runtime_fingerprint()
    # Hash once, then read the bytes once to compare compiled and loaded code.
    # This cold initialization must not recur on either warmed runtime path.
    cold_reads = 2
    assert len(reads) == cold_reads
    assert checkpoint._SELECTION_STATE is not None
    for deep in (False, True, False):
        assert checkpoint.runtime_fingerprint(check_inventory=deep) == initial
    assert len(reads) == cold_reads


@pytest.mark.parametrize('deep', [True, False])
def test_owned_selector_source_edit_is_refused(deep, tmp_path, tiny_runtime, monkeypatch):
    # Substitute only the path for this disk-change test. Origin/function checks
    # are tested independently above; only this owned temporary file is edited.
    source = tmp_path / 'owned_tune_mnist.py'
    source.write_bytes(selector_source().read_bytes())
    monkeypatch.setattr(checkpoint, '_selection_source', lambda: source)
    checkpoint.runtime_fingerprint()
    source.write_bytes(source.read_bytes() + b'\n# controlled source edit\n')
    with pytest.raises(checkpoint.CheckpointError, match='Runtime artifact changed'):
        checkpoint.runtime_fingerprint(check_inventory=deep)


@pytest.mark.parametrize('deep', [True, False])
def test_owned_selector_source_removal_is_refused(deep, tmp_path, tiny_runtime, monkeypatch):
    source = tmp_path / 'owned_tune_mnist.py'
    source.write_bytes(selector_source().read_bytes())
    monkeypatch.setattr(checkpoint, '_selection_source', lambda: source)
    checkpoint.runtime_fingerprint()
    source.unlink()
    # Existing fingerprint I/O failures deliberately propagate; no result may
    # be returned from a cached digest when its source disappeared.
    with pytest.raises((checkpoint.CheckpointError, FileNotFoundError)):
        checkpoint.runtime_fingerprint(check_inventory=deep)


def test_real_runtime_fingerprints_selector_source(fresh_runtime):
    # No fake package inventory, native identity, threadpool controller, source
    # path or fingerprint function is used in this integration test.
    expected = selector_source()
    with threadpool_limits(limits=1):
        initial = checkpoint.runtime_fingerprint()
        shallow = checkpoint.runtime_fingerprint(check_inventory=False)
    assert initial == shallow
    assert str(expected) in checkpoint._RUNTIME_FILES
    assert checkpoint._HASHES[expected][1] == hashlib.sha256(expected.read_bytes()).hexdigest()
