"""Checkpoint cache exclusion through the actual runtime fingerprint.

Owned temporary artifacts and a synthetic native-pool record isolate these
tests while production inventory, hashing, canonicalization and drift guards
remain active.
"""
from pathlib import Path

import pytest

from test_runtime_artifact_identity import fresh_runtime, tiny_runtime  # noqa: F401
import cv_checkpoint as checkpoint


@pytest.mark.parametrize('denied_kind', ['cache_directory', 'cache_source'])
def test_cache_stat_denial_is_skipped_with_equal_identity(
        tiny_runtime, monkeypatch, denied_kind):
    cache = tiny_runtime.root / '__pycache__'
    cache.mkdir()
    cached_source = cache / 'otherwise_selected.py'
    cached_source.write_text('cached = 1\n', encoding='utf-8')
    denied = cache if denied_kind == 'cache_directory' else cached_source
    assert denied in list(tiny_runtime.root.rglob('*'))
    before = checkpoint.runtime_fingerprint()
    inventory = dict(checkpoint._RUNTIME_PACKAGE_FILES)
    original_stat = Path.stat
    denied_calls = []

    def guarded_stat(path, *args, **kwargs):
        if path == denied:
            denied_calls.append(path)
            raise PermissionError('owned cache stat denial')
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', guarded_stat)
    after = checkpoint.runtime_fingerprint(check_inventory=True)
    assert denied_calls == []
    assert checkpoint._stable(after) == checkpoint._stable(before)
    assert checkpoint._RUNTIME_PACKAGE_FILES == inventory
    assert all('__pycache__' not in Path(name).parts for name in inventory)


@pytest.mark.parametrize('denied_kind', ['selected_source', 'ordinary_directory'])
def test_noncache_stat_denial_still_propagates(
        tiny_runtime, monkeypatch, denied_kind):
    ordinary = tiny_runtime.root / 'ordinary_directory'
    ordinary.mkdir()
    denied = tiny_runtime.source if denied_kind == 'selected_source' else ordinary
    assert denied in list(tiny_runtime.root.rglob('*'))
    checkpoint.runtime_fingerprint()
    original_stat = Path.stat
    failure = PermissionError('owned non-cache stat denial')

    def guarded_stat(path, *args, **kwargs):
        if path == denied:
            raise failure
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', guarded_stat)
    with pytest.raises(PermissionError) as raised:
        checkpoint.runtime_fingerprint(check_inventory=True)
    assert raised.value is failure


def test_cache_file_addition_and_removal_leave_inventory_unchanged(tiny_runtime):
    before = checkpoint.runtime_fingerprint()
    inventory = dict(checkpoint._RUNTIME_PACKAGE_FILES)
    cache = tiny_runtime.root / '__pycache__'
    cache.mkdir()
    cached_source = cache / 'otherwise_selected.py'
    cached_source.write_text('cached = 1\n', encoding='utf-8')
    after_addition = checkpoint.runtime_fingerprint(check_inventory=True)
    assert checkpoint._stable(after_addition) == checkpoint._stable(before)
    assert checkpoint._RUNTIME_PACKAGE_FILES == inventory
    assert str(cached_source.absolute()) not in checkpoint._RUNTIME_PACKAGE_FILES
    cached_source.unlink()
    cache.rmdir()
    after_removal = checkpoint.runtime_fingerprint(check_inventory=True)
    assert checkpoint._stable(after_removal) == checkpoint._stable(before)
    assert checkpoint._RUNTIME_PACKAGE_FILES == inventory


@pytest.mark.parametrize('check_inventory', [False, True], ids=['shallow', 'full'])
def test_selected_source_mutation_still_refuses(tiny_runtime, check_inventory):
    checkpoint.runtime_fingerprint()
    assert str(tiny_runtime.source.absolute()) in checkpoint._RUNTIME_PACKAGE_FILES
    tiny_runtime.source.write_bytes(tiny_runtime.source.read_bytes() + b'\nchanged = True\n')
    with pytest.raises(checkpoint.CheckpointError,
                       match='Runtime artifact changed in this process'):
        checkpoint.runtime_fingerprint(check_inventory=check_inventory)
