"""Real production inventory composition and interpreter-specific traversal controls.

Only the external synthetic runtime is controlled by the retained tiny_runtime
fixture. The production fingerprint, inventory consumer, hashing and alias guards
remain real. No estimator fit or installed-file mutation is performed.
"""
from collections import Counter
import errno
import hashlib
import os
from pathlib import Path

import pytest

from test_runtime_artifact_identity import fresh_runtime, tiny_runtime, reset_runtime  # noqa: F401
import cv_checkpoint as checkpoint


FUSED = hasattr(checkpoint, '_inventory_regular_stat')
MODERN_WALK = hasattr(type(Path()), 'walk')


def populate(root, entries):
    for relative, content in entries.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def literal_members(root):
    result = {}
    for path in root.rglob('*'):
        if '__pycache__' in path.parts or not path.is_file():
            continue
        if path.suffix.lower() in ('.py', '.c', '.h', '.pyd', '.dll', '.so', '.dylib') or '.so.' in path.name:
            result[str(path.absolute())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_production_inventory_matches_independent_bytes_and_literal_tree(tiny_runtime):
    entries = {
        'extra.py': b'extra', '.hidden/nested.C': b'hidden',
        'nested/libx.so.1': b'versioned', 'nested/upper-versioned.SO.1': b'excluded',
        'nested/UPPER.PY': b'upper', 'notes.txt': b'notes', 'config.json': b'non-AOCL',
        '__pycache__/generated.py': b'excluded cache',
        'nested/__pycache__/lib.so': b'excluded cache binary',
    }
    populate(tiny_runtime.root, entries)
    expected_paths = [tiny_runtime.source, tiny_runtime.binary, tiny_runtime.second]
    expected_paths += [tiny_runtime.root / name for name in (
        'extra.py', '.hidden/nested.C', 'nested/libx.so.1', 'nested/UPPER.PY')]
    expected = {str(path.absolute()): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in expected_paths}
    initial = checkpoint.runtime_fingerprint(check_inventory=True)
    assert checkpoint._RUNTIME_PACKAGE_FILES == expected == literal_members(tiny_runtime.root)
    assert checkpoint.runtime_fingerprint(check_inventory=True) == initial
    assert checkpoint.runtime_fingerprint(check_inventory=False) == initial


def test_no_walk_delegates_before_any_custom_root_check():
    marker = object()
    calls = []
    class NoWalkPath:
        def rglob(self, pattern):
            calls.append(pattern)
            return iter((marker,))
        def __getattr__(self, name):
            raise AssertionError('Fallback must not inspect root ' + name)
    assert list(checkpoint._runtime_package_paths(NoWalkPath())) == [marker]
    assert calls == ['*']


def test_no_walk_keeps_delegated_iterator_failure_and_prior_yield():
    marker = object()
    failure = OSError(errno.EIO, 'owned delegated failure')
    class NoWalkPath:
        def rglob(self, pattern):
            assert pattern == '*'
            yield marker
            raise failure
        def __getattr__(self, name):
            raise AssertionError('Fallback must not inspect root ' + name)
    result = checkpoint._runtime_package_paths(NoWalkPath())
    assert next(result) is marker
    with pytest.raises(OSError) as raised:
        next(result)
    assert raised.value is failure


@pytest.mark.parametrize('stage', ['scan-open', 'scan-iteration', 'directory-classification'])
@pytest.mark.parametrize('code', [errno.EACCES, errno.EIO, errno.ENOENT])
def test_production_error_policy_matches_actual_interpreter_rglob(tiny_runtime, monkeypatch, stage, code):
    root = tiny_runtime.root
    blocked = root / 'blocked'
    populate(root, {'visible.py': b'visible', 'blocked/inside.py': b'inside'})
    failure = PermissionError(code, 'owned permission') if code == errno.EACCES else OSError(code, 'owned scan')
    real_scandir = os.scandir
    candidate = checkpoint._runtime_package_paths

    class Entry:
        def __init__(self, original):
            self.original, self.name = original, original.name
        def is_dir(self, *, follow_symlinks=True):
            if self.name == 'blocked':
                raise failure
            return self.original.is_dir(follow_symlinks=follow_symlinks)
        def __getattr__(self, name):
            return getattr(self.original, name)

    class Scan:
        def __init__(self, original, mode):
            self.original, self.mode = original, mode
        def __enter__(self):
            self.original.__enter__()
            return self
        def __exit__(self, *args):
            return self.original.__exit__(*args)
        def __iter__(self):
            if self.mode == 'scan-iteration':
                raise failure
            for entry in self.original:
                yield Entry(entry) if self.mode == 'directory-classification' else entry

    def observed_scan(directory):
        current = Path(directory)
        if current == blocked and stage == 'scan-open':
            raise failure
        original = real_scandir(directory)
        if current == blocked and stage == 'scan-iteration':
            return Scan(original, stage)
        if current == root and stage == 'directory-classification':
            return Scan(original, stage)
        return original

    def observe(use_original):
        with monkeypatch.context() as scoped:
            reset_runtime(scoped)
            scoped.setattr(os, 'scandir', observed_scan)
            scoped.setattr(checkpoint, '_runtime_package_paths',
                           (lambda path: path.rglob('*')) if use_original else candidate)
            try:
                checkpoint.runtime_fingerprint(check_inventory=True)
            except OSError as error:
                assert error is failure
                return ('error', type(error), error.errno)
            return ('inventory', dict(checkpoint._RUNTIME_PACKAGE_FILES))

    assert observe(False) == observe(True)


@pytest.mark.skipif(MODERN_WALK, reason='This exact legacy-root contract is exercised on actual Python without Path.walk')
def test_actual_legacy_cache_root_error_is_not_hidden(tiny_runtime, monkeypatch):
    cache_root = tiny_runtime.root / '__pycache__'
    populate(cache_root, {'__init__.py': b'owned cache-root module'})
    tiny_runtime.module.__file__ = str(cache_root / '__init__.py')
    failure = OSError(errno.EIO, 'owned old-runtime cache-root scan')
    real_scandir = os.scandir
    def observed_scan(directory):
        if Path(directory) == cache_root:
            raise failure
        return real_scandir(directory)
    monkeypatch.setattr(os, 'scandir', observed_scan)
    with pytest.raises(OSError) as raised:
        checkpoint.runtime_fingerprint(check_inventory=True)
    assert raised.value is failure


def test_production_repeated_scan_and_stat_counts(tiny_runtime, monkeypatch):
    root = tiny_runtime.root
    leaf = root / 'unloaded_leaf.py'
    populate(root, {'unloaded_leaf.py': b'unchanged', 'nested/other.py': b'other',
                    '__pycache__/ignored.py': b'ignored', 'nested/__pycache__/ignored.py': b'ignored'})
    initial = checkpoint.runtime_fingerprint(check_inventory=True)
    real_scandir, real_stat = os.scandir, Path.stat
    scans, stats = [], []
    def observed_scan(directory):
        current = Path(directory)
        if current == root or root in current.parents:
            scans.append(current)
        return real_scandir(directory)
    def observed_stat(item, *args, **kwargs):
        if item == leaf:
            stats.append(item)
        return real_stat(item, *args, **kwargs)
    monkeypatch.setattr(os, 'scandir', observed_scan)
    monkeypatch.setattr(Path, 'stat', observed_stat)
    assert checkpoint.runtime_fingerprint(check_inventory=True) == initial
    candidate_scans = Counter(scans)
    assert len(stats) == (1 if FUSED else 2)
    if MODERN_WALK:
        assert candidate_scans == Counter({root: 1, root / 'nested': 1})
    else:
        # Exercise the actual older implementation, not a simulated version flag.
        scans.clear()
        with monkeypatch.context() as scoped:
            scoped.setattr(checkpoint, '_runtime_package_paths', lambda path: path.rglob('*'))
            assert checkpoint.runtime_fingerprint(check_inventory=True) == initial
        assert candidate_scans == Counter(scans)
        assert root / '__pycache__' in candidate_scans


def test_uncached_production_read_keeps_post_read_mutation_check(tiny_runtime, monkeypatch):
    leaf = tiny_runtime.root / 'unloaded_leaf.py'
    leaf.write_bytes(b'original')
    real_stat = Path.stat
    count = 0
    def changed_after_pre_read_stat(item, *args, **kwargs):
        nonlocal count
        observed = real_stat(item, *args, **kwargs)
        if item == leaf:
            count += 1
            if count == (1 if FUSED else 2):
                leaf.write_bytes(b'changed to a longer source after retained pre-read stat')
        return observed
    monkeypatch.setattr(Path, 'stat', changed_after_pre_read_stat)
    with pytest.raises(checkpoint.CheckpointError, match='changed while fingerprinting'):
        checkpoint.runtime_fingerprint(check_inventory=True)
    assert leaf.absolute() not in checkpoint._HASHES


def test_cached_production_cutpoint_and_next_boundary_are_explicit(tiny_runtime, monkeypatch):
    leaf = tiny_runtime.root / 'unloaded_leaf.py'
    leaf.write_bytes(b'original')
    initial = checkpoint.runtime_fingerprint(check_inventory=True)
    original_digest = checkpoint._RUNTIME_PACKAGE_FILES[str(leaf.absolute())]
    real_stat = Path.stat
    changed = False
    def mutate_after_classification(item, *args, **kwargs):
        nonlocal changed
        observed = real_stat(item, *args, **kwargs)
        if item == leaf and not changed:
            changed = True
            leaf.write_bytes(b'changed to a longer source after classification')
        return observed
    monkeypatch.setattr(Path, 'stat', mutate_after_classification)
    if FUSED:
        # The unimported synthetic-package leaf is not a shallow loaded-source guard.
        # This exact transient cutpoint may return the cached digest once, as B does.
        assert checkpoint.runtime_fingerprint(check_inventory=True) == initial
        assert checkpoint._RUNTIME_PACKAGE_FILES[str(leaf.absolute())] == original_digest
    else:
        with pytest.raises(checkpoint.CheckpointError, match='changed in this process'):
            checkpoint.runtime_fingerprint(check_inventory=True)
    assert changed
    with pytest.raises(checkpoint.CheckpointError, match='changed in this process'):
        checkpoint.runtime_fingerprint(check_inventory=True)


@pytest.mark.parametrize('mutation', ['add', 'delete', 'rename', 'rewrite', 'add-directory'])
def test_persistent_production_inventory_change_refuses(tiny_runtime, mutation):
    checkpoint.runtime_fingerprint(check_inventory=True)
    if mutation == 'add':
        (tiny_runtime.root / 'new.py').write_bytes(b'new')
    elif mutation == 'delete':
        tiny_runtime.second.unlink()
    elif mutation == 'rename':
        tiny_runtime.second.rename(tiny_runtime.root / 'renamed.dll')
    elif mutation == 'rewrite':
        tiny_runtime.second.write_bytes(b'new longer source')
    else:
        populate(tiny_runtime.root, {'new-directory/new.py': b'new'})
    with pytest.raises(checkpoint.CheckpointError, match='inventory changed|Runtime artifact changed'):
        checkpoint.runtime_fingerprint(check_inventory=True)


def test_persistent_loaded_source_change_refuses_shallow_boundary(tiny_runtime):
    checkpoint.runtime_fingerprint(check_inventory=True)
    tiny_runtime.source.write_bytes(tiny_runtime.source.read_bytes() + b'\nchanged = True\n')
    with pytest.raises(checkpoint.CheckpointError, match='changed in this process'):
        checkpoint.runtime_fingerprint(check_inventory=False)


def test_production_cache_addition_remains_excluded(tiny_runtime):
    initial = checkpoint.runtime_fingerprint(check_inventory=True)
    populate(tiny_runtime.root, {'__pycache__/generated.py': b'ignored',
                                'nested/__pycache__/generated.dll': b'ignored'})
    assert checkpoint.runtime_fingerprint(check_inventory=True) == initial


def test_production_hardlink_entries_keep_lexical_identity(tiny_runtime):
    first, second = tiny_runtime.root / 'first.py', tiny_runtime.root / 'second.py'
    first.write_bytes(b'owned hardlink source')
    os.link(first, second)
    checkpoint.runtime_fingerprint(check_inventory=True)
    members = checkpoint._RUNTIME_PACKAGE_FILES
    assert str(first.absolute()) in members and str(second.absolute()) in members
    assert members[str(first.absolute())] == members[str(second.absolute())]
    a = checkpoint._runtime_artifact_identity(first)
    b = checkpoint._runtime_artifact_identity(second)
    assert a[0] != b[0] and a[1][:2] == b[1][:2]
