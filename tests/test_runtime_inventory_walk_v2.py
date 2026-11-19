"""Stable-tree inventory and live-drift parity with the original pathlib scan.

No benchmarks, package edits, or environment changes. File-system artifacts are
owned by pytest's temporary tree. Directory-link cases reuse the audited owned
junction/symlink helper; file symlinks skip only an unavailable OS privilege.
"""
from collections import Counter
import errno
import hashlib
import os
from pathlib import Path
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint
from runtime_test_support import owned_alias, physical_path, remove_alias
from test_runtime_artifact_identity import fresh_runtime, tiny_runtime


def literal_old_inventory(root, package='dwd'):
    """The original selection/hash body, without a redesigned test oracle."""
    files = {}
    for path in root.rglob('*'):
        if '__pycache__' in path.parts or not path.is_file():
            continue
        suffixes = ('.py', '.c', '.h', '.pyd', '.dll', '.so', '.dylib')
        if path.suffix.lower() in suffixes or '.so.' in path.name or (package == 'aocl_runtime' and path.suffix.lower() == '.json'):
            path = path.absolute()
            files[str(path)] = checkpoint._file_hash(path)
    return files


def candidate_inventory(root, package='dwd'):
    files = {}
    for path in checkpoint._runtime_package_paths(root):
        if '__pycache__' in path.parts or not path.is_file():
            continue
        suffixes = ('.py', '.c', '.h', '.pyd', '.dll', '.so', '.dylib')
        if path.suffix.lower() in suffixes or '.so.' in path.name or (package == 'aocl_runtime' and path.suffix.lower() == '.json'):
            path = path.absolute()
            files[str(path)] = checkpoint._file_hash(path)
    return files


SCANNERS = [literal_old_inventory, candidate_inventory]
MODERN_WALK = hasattr(type(Path()), 'walk')


def populate(root, records):
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in records.items():
        item = root / relative
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_bytes(content)


@pytest.mark.parametrize('package', ['dwd', 'aocl_runtime'])
def test_stable_tree_inventory_matches_literal_and_independent_expected(tmp_path, package):
    root = tmp_path / 'package'
    records = {
        '__init__.py': b'init', '.hidden.py': b'hidden', '.nested/source.PY': b'nested',
        'data.json': b'json', 'upper.JSON': b'json2', 'helper.c': b'c',
        'include/header.H': b'h', 'extension.PYD': b'pyd', 'native.DLL': b'dll',
        'libx.so.1': b'so1', 'upper-versioned.SO.1': b'upper-versioned-is-not-selected',
        'library.SO': b'so', 'library.dylib': b'dylib', 'notes.txt': b'notes',
        '__pycache__/ignored.py': b'ignored', 'sub/__pycache__/deep/hidden.so': b'ignored2',
        'mixed/__PyCache__/kept.py': b'case-sensitive-cache-name',
        'empty-name-like-dir.py/actual.py': b'directory-suffix',
        'left/same.py': b'equal', 'right/same.py': b'equal',
    }
    populate(root, records)
    selected = {'__init__.py', '.hidden.py', '.nested/source.PY', 'helper.c',
                'include/header.H', 'extension.PYD', 'native.DLL', 'libx.so.1',
                'library.SO', 'library.dylib', 'mixed/__PyCache__/kept.py',
                'empty-name-like-dir.py/actual.py', 'left/same.py', 'right/same.py'}
    if package == 'aocl_runtime':
        selected.update(('data.json', 'upper.JSON'))
    expected = {str((root / name).absolute()): hashlib.sha256(records[name]).hexdigest()
                for name in selected}
    old = literal_old_inventory(root, package)
    new = candidate_inventory(root, package)
    assert new == old == expected
    assert sorted(new) == sorted(old)
    assert checkpoint._digest(new) == checkpoint._digest(old)
    assert len(new) == len(expected)  # Equal content never collapses lexical entries.


@pytest.mark.parametrize('kind', ['missing', 'file', 'cache-root'])
def test_empty_missing_file_and_cache_roots_match_old(tmp_path, kind):
    root = tmp_path / ('__pycache__' if kind == 'cache-root' else kind)
    if kind == 'file':
        root.write_bytes(b'x')
    elif kind == 'cache-root':
        populate(root, {'nested/source.py': b'ignored'})
    assert candidate_inventory(root) == literal_old_inventory(root) == {}


def test_one_scan_per_directory_and_no_cache_descent(tmp_path, monkeypatch):
    root = tmp_path / 'package'
    populate(root, {'a.py': b'a', 'left/b.py': b'b', 'right/deep/c.py': b'c',
                    '__pycache__/deep/ignore.py': b'i', 'left/__pycache__/ignore.py': b'i'})
    real = os.scandir
    calls = []
    def counted(directory):
        calls.append(Path(directory))
        return real(directory)
    monkeypatch.setattr(os, 'scandir', counted)
    actual = candidate_inventory(root)
    assert len(actual) == 3
    if not MODERN_WALK:
        # On older interpreters the production helper delegates to rglob.
        # Compare actual delegated traversal rather than requiring the newer
        # single-scan/cache-pruning implementation on that route.
        candidate_calls = Counter(calls)
        calls.clear()
        assert literal_old_inventory(root) == actual
        assert Counter(calls) == candidate_calls
        return
    assert Counter(calls) == Counter({root: 1, root/'left': 1,
                                     root/'right': 1, root/'right'/'deep': 1})
    assert all('__pycache__' not in value.parts for value in calls)


@pytest.mark.parametrize('scanner', SCANNERS)
def test_live_file_checks_and_hashes_do_not_use_direntry_stat(tmp_path, monkeypatch, scanner):
    root = tmp_path / 'package'
    populate(root, {'source.py': b'first'})
    source = root/'source.py'
    scanner(root)  # Establish the live hash stamp.
    real = Path.stat
    calls = []
    def live_stat(item, *args, **kwargs):
        if item == source:
            calls.append(item)
        return real(item, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', live_stat)
    assert scanner(root)[str(source.absolute())] == hashlib.sha256(b'first').hexdigest()
    assert len(calls) >= 2  # is_file and the cached hash's live identity check.


@pytest.mark.parametrize('scanner', SCANNERS)
def test_change_between_is_file_and_hash_is_rejected(tmp_path, monkeypatch, scanner):
    root = tmp_path / 'package'
    populate(root, {'source.py': b'first'})
    source = root/'source.py'
    scanner(root)
    real = Path.is_file
    changed = False
    def mutate(item):
        nonlocal changed
        answer = real(item)
        if item == source and not changed:
            changed = True
            source.write_bytes(b'second-longer')
        return answer
    monkeypatch.setattr(Path, 'is_file', mutate)
    with pytest.raises(checkpoint.CheckpointError, match='Runtime artifact changed in this process'):
        scanner(root)
    assert changed


@pytest.mark.parametrize('scanner', SCANNERS)
def test_live_is_file_permission_error_is_not_suppressed(tmp_path, monkeypatch, scanner):
    root = tmp_path / 'package'
    populate(root, {'source.py': b'x'})
    source = root/'source.py'
    error = PermissionError(errno.EACCES, 'owned injected stat refusal', str(source))
    real = Path.stat
    def denied(item, *args, **kwargs):
        if item == source:
            raise error
        return real(item, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', denied)
    with pytest.raises(PermissionError) as caught:
        scanner(root)
    assert caught.value is error


@pytest.mark.parametrize('scanner', SCANNERS)
def test_unlistable_directory_open_preserves_pathlib_policy(tmp_path, monkeypatch, scanner):
    root = tmp_path / 'package'
    populate(root, {'ok.py': b'ok', 'blocked/source.py': b'blocked'})
    real = os.scandir
    def denied(directory):
        if Path(directory) == root/'blocked':
            raise PermissionError(errno.EACCES, 'owned injected scan refusal')
        return real(directory)
    monkeypatch.setattr(os, 'scandir', denied)
    assert scanner(root) == {str((root/'ok.py').absolute()): hashlib.sha256(b'ok').hexdigest()}


@pytest.mark.parametrize('scanner', SCANNERS)
def test_scan_iteration_failure_is_not_silently_hidden(tmp_path, monkeypatch, scanner):
    root = tmp_path / 'package'
    root.mkdir()
    error = OSError(errno.EIO, 'owned injected directory iteration failure')
    class FailedScan:
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            return False
        def __iter__(self):
            return self
        def __next__(self):
            raise error
    monkeypatch.setattr(os, 'scandir', lambda unused: FailedScan())
    if not MODERN_WALK:
        def outcome(scan):
            try:
                return ('inventory', scan(root))
            except OSError as caught:
                assert caught is error
                return ('error', type(caught), caught.errno)
        # Preserve this interpreter's actual delegated error policy.
        assert outcome(scanner) == outcome(literal_old_inventory)
        return
    # On the primary CPython 3.12 pathlib route its second Path.walk iteration
    # propagates this failure. Candidate does not need the redundant first scan.
    if scanner is literal_old_inventory and sys.version_info[:2] != (3, 12):
        pytest.skip('This exact original iteration-error cutpoint is CPython 3.12-specific')
    with pytest.raises(OSError) as caught:
        scanner(root)
    assert caught.value is error


def test_directory_alias_behavior_matches_original_on_this_platform(tmp_path):
    root, target = tmp_path/'package', tmp_path/'outside'
    populate(root, {'own.py': b'own'})
    populate(target, {'nested/other.py': b'other'})
    receipt = owned_alias(root/'alias', physical_path(target), tmp_path)
    try:
        before = literal_old_inventory(root)
        assert candidate_inventory(root) == before
        assert str((root/'own.py').absolute()) in before
        # POSIX directory symlinks are not traversed. Windows junction behavior
        # is delegated to the same is_dir(follow_symlinks=False) primitive.
        if os.name != 'nt':
            assert str((root/'alias'/'nested'/'other.py').absolute()) not in before
    finally:
        remove_alias(receipt, tmp_path)


def test_root_alias_is_still_traversed(tmp_path):
    target = tmp_path/'target'
    populate(target, {'source.py': b'source'})
    link = tmp_path/'root-alias'
    receipt = owned_alias(link, physical_path(target), tmp_path)
    try:
        assert candidate_inventory(link) == literal_old_inventory(link)
        assert list(candidate_inventory(link)) == [str((link/'source.py').absolute())]
    finally:
        remove_alias(receipt, tmp_path)


def test_file_and_broken_symlinks_match_literal(tmp_path):
    root = tmp_path/'package'
    populate(root, {'target.py': b'target'})
    try:
        (root/'linked.py').symlink_to(root/'target.py')
        (root/'broken.py').symlink_to(root/'missing.py')
    except OSError as error:
        if os.name == 'nt' and getattr(error, 'winerror', None) == 1314:
            pytest.skip('Windows file-symlink privilege unavailable; settings remain unchanged')
        raise
    old = literal_old_inventory(root)
    assert candidate_inventory(root) == old
    assert str((root/'linked.py').absolute()) in old
    assert str((root/'broken.py').absolute()) not in old
    assert len(old) == 2


def test_case_distinct_paths_are_not_folded_when_filesystem_supports_them(tmp_path):
    root = tmp_path/'package'
    populate(root, {'same.py': b'same'})
    other = root/'SAME.py'
    if other.exists():
        pytest.skip('The temporary filesystem folds these case-distinct names')
    other.write_bytes(b'SAME')
    assert len(candidate_inventory(root)) == 2
    assert candidate_inventory(root) == literal_old_inventory(root)


@pytest.mark.parametrize('original', [False, True], ids=['one-pass', 'literal-pathlib'])
@pytest.mark.parametrize('mutation', ['add', 'delete', 'rename', 'rewrite', 'add-directory'])
def test_persistent_changes_are_refused_at_next_deep_boundary(tiny_runtime, monkeypatch, original, mutation):
    if original:
        monkeypatch.setattr(checkpoint, '_runtime_package_paths', lambda root: root.rglob('*'))
    checkpoint.runtime_fingerprint()
    if mutation == 'add':
        (tiny_runtime.root/'new.py').write_bytes(b'new')
    elif mutation == 'delete':
        tiny_runtime.second.unlink()
    elif mutation == 'rename':
        tiny_runtime.second.rename(tiny_runtime.root/'renamed.dll')
    elif mutation == 'rewrite':
        tiny_runtime.second.write_bytes(b'new longer contents')
    else:
        populate(tiny_runtime.root/'new-dir', {'new.py': b'new'})
    with pytest.raises(checkpoint.CheckpointError, match='inventory changed|Runtime artifact changed'):
        checkpoint.runtime_fingerprint(check_inventory=True)


@pytest.mark.parametrize('original', [False, True], ids=['one-pass', 'literal-pathlib'])
def test_cache_changes_remain_excluded_at_deep_boundary(tiny_runtime, monkeypatch, original):
    if original:
        monkeypatch.setattr(checkpoint, '_runtime_package_paths', lambda root: root.rglob('*'))
    before = checkpoint.runtime_fingerprint()
    populate(tiny_runtime.root/'__pycache__', {'generated.py': b'excluded', 'nested/lib.so': b'excluded'})
    assert checkpoint.runtime_fingerprint(check_inventory=True) == before


def test_single_file_package_retains_direct_origin_route(tiny_runtime, monkeypatch):
    del tiny_runtime.module.__path__
    def forbidden(unused):
        raise AssertionError('Single-file module must not use directory traversal')
    monkeypatch.setattr(checkpoint, '_runtime_package_paths', forbidden)
    actual = checkpoint.runtime_fingerprint()
    assert str(tiny_runtime.source.absolute()) in checkpoint._RUNTIME_PACKAGE_FILES
    assert actual['packages'][tiny_runtime.name]['version'] == '1'


def test_hardlinked_lexical_entries_remain_distinct_until_alias_proof(tmp_path):
    root = tmp_path/'package'
    populate(root, {'first.py': b'same bytes'})
    os.link(root/'first.py', root/'second.py')
    current = candidate_inventory(root)
    assert current == literal_old_inventory(root)
    assert len(current) == 2
    first = checkpoint._runtime_artifact_identity(root/'first.py')
    second = checkpoint._runtime_artifact_identity(root/'second.py')
    assert first[1][:2] == second[1][:2] and first[2] == second[2]
    assert first[0] != second[0]
    # The common inode is evidence, but resolve() retains distinct hardlink
    # spellings: the existing alias policy is intentionally not broadened here.
    assert os.path.samefile(root/'first.py', root/'second.py')


def test_directory_classification_error_does_not_skip_live_file_check(tmp_path, monkeypatch):
    root = tmp_path/'package'
    populate(root, {'source.py': b'source', 'blocked/inner.py': b'inner'})
    real = os.scandir
    class Entry:
        def __init__(self, entry):
            self.name = entry.name
        def is_dir(self, *, follow_symlinks=True):
            raise OSError(errno.EIO, 'owned classification refusal')
    class Scan:
        def __init__(self, directory):
            self.inner = real(directory)
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            self.inner.close()
        def __iter__(self):
            return (Entry(value) for value in self.inner)
    monkeypatch.setattr(os, 'scandir', Scan)
    if not MODERN_WALK:
        def outcome(scan):
            try:
                return ('inventory', scan(root))
            except OSError as caught:
                return ('error', type(caught), caught.errno)
        assert outcome(candidate_inventory) == outcome(literal_old_inventory)
        return
    expected = {str((root/'source.py').absolute()): hashlib.sha256(b'source').hexdigest()}
    assert candidate_inventory(root) == expected
    assert literal_old_inventory(root) == expected
