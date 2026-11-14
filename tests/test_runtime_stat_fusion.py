"""Fresh-stat fusion invariants, including its deliberately changed cutpoint."""
import errno
import hashlib
import os
from pathlib import Path

import pytest

from test_runtime_artifact_identity import fresh_runtime, tiny_runtime  # noqa: F401
import cv_checkpoint as checkpoint


def fused(path):
    info = checkpoint._inventory_regular_stat(path)
    return None if info is None else checkpoint._file_hash_observed(path.absolute(), info)


def test_new_file_two_stats_cached_file_one_live_stat(tmp_path, monkeypatch):
    path = tmp_path/'owned.py'
    path.write_bytes(b'original source\n')
    real = Path.stat
    calls = []

    def observed(item, *args, **kwargs):
        if item == path:
            calls.append(item)
        return real(item, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', observed)
    expected = hashlib.sha256(b'original source\n').hexdigest()
    assert fused(path) == expected
    assert len(calls) == 2  # one classification/pre-read plus one post-read
    calls.clear()
    assert fused(path) == expected
    assert len(calls) == 1
    calls.clear()
    assert checkpoint._file_hash(path) == expected
    assert len(calls) == 1  # ordinary noninventory entry remains fresh


@pytest.mark.parametrize('code', [errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP])
def test_original_missing_path_errno_policy(tmp_path, monkeypatch, code):
    failure = OSError(code, 'owned synthetic classification error')
    monkeypatch.setattr(Path, 'stat', lambda *a, **k: (_ for _ in ()).throw(failure))
    assert checkpoint._inventory_regular_stat(tmp_path/'owned.py') is None


@pytest.mark.parametrize('code', [21, 123, 1921])
def test_original_windows_classification_error_policy(tmp_path, monkeypatch, code):
    failure = OSError('owned Windows classification error')
    failure.winerror = code
    monkeypatch.setattr(Path, 'stat', lambda *a, **k: (_ for _ in ()).throw(failure))
    assert checkpoint._inventory_regular_stat(tmp_path/'owned.py') is None


@pytest.mark.parametrize('failure', [PermissionError(errno.EACCES, 'owned denial'),
                                   OSError(errno.EIO, 'owned I/O failure'), RuntimeError('owned unexpected')])
def test_nonmissing_stat_error_propagates_exact_object(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(Path, 'stat', lambda *a, **k: (_ for _ in ()).throw(failure))
    with pytest.raises(type(failure)) as raised:
        checkpoint._inventory_regular_stat(tmp_path/'owned.txt')
    assert raised.value is failure


def test_nonencodable_classification_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'stat', lambda *a, **k: (_ for _ in ()).throw(ValueError('owned encoding')))
    assert checkpoint._inventory_regular_stat(tmp_path/'owned.py') is None


def test_nonselected_suffix_still_observes_real_stat_error(tiny_runtime, monkeypatch):
    denied = tiny_runtime.root/'not-selected.txt'
    denied.write_text('not a fingerprint member', encoding='utf-8')
    real = Path.stat
    failure = PermissionError(errno.EACCES, 'owned nonselected file denial')

    def guarded(path, *args, **kwargs):
        if path == denied:
            raise failure
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', guarded)
    with pytest.raises(PermissionError) as raised:
        checkpoint.runtime_fingerprint(check_inventory=True)
    assert raised.value is failure


def test_directory_and_fifo_are_never_opened(tmp_path, monkeypatch):
    directory = tmp_path/'directory.py'
    directory.mkdir()
    items = [directory]
    if hasattr(os, 'mkfifo'):
        fifo = tmp_path/'owned-pipe.py'
        os.mkfifo(fifo)
        items.append(fifo)
    real = Path.open

    def refuse(path, *args, **kwargs):
        if path in items:
            raise AssertionError('nonregular object must not be opened')
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', refuse)
    for path in items:
        assert fused(path) is None


def test_symlink_file_digest_and_directory_rejection(tmp_path):
    source = tmp_path/'source.py'
    source.write_bytes(b'owned target')
    directory = tmp_path/'directory'
    directory.mkdir()
    file_alias, directory_alias = tmp_path/'alias.py', tmp_path/'directory-alias.py'
    try:
        file_alias.symlink_to(source)
        directory_alias.symlink_to(directory, target_is_directory=True)
    except OSError as error:
        unavailable = (getattr(error, 'winerror', None) == 1314 or error.errno in
                       {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP})
        if unavailable:
            pytest.skip('Owned symbolic-link capability unavailable: '+str(error))
        raise
    assert fused(file_alias) == hashlib.sha256(source.read_bytes()).hexdigest()
    assert fused(directory_alias) is None


def test_uncached_mutation_after_classification_still_refuses(tmp_path, monkeypatch):
    path = tmp_path/'owned.py'
    path.write_bytes(b'original')
    real = Path.stat
    touched = False

    def mutate_after_observation(item, *args, **kwargs):
        nonlocal touched
        observed = real(item, *args, **kwargs)
        if item == path and not touched:
            touched = True
            path.write_bytes(b'changed to a longer source')
        return observed

    monkeypatch.setattr(Path, 'stat', mutate_after_observation)
    with pytest.raises(checkpoint.CheckpointError, match='changed while fingerprinting'):
        fused(path)
    assert path.absolute() not in checkpoint._HASHES


@pytest.mark.parametrize('route', ['original-two-observations', 'fused-one-observation'])
def test_cached_concurrent_mutation_cutpoint_is_explicit(tmp_path, monkeypatch, route):
    path = tmp_path/'owned.py'
    path.write_bytes(b'original')
    original_digest = checkpoint._file_hash(path)
    real = Path.stat
    touched = False

    def mutate_after_observation(item, *args, **kwargs):
        nonlocal touched
        observed = real(item, *args, **kwargs)
        if item == path and not touched:
            touched = True
            path.write_bytes(b'changed to a longer source')
        return observed

    monkeypatch.setattr(Path, 'stat', mutate_after_observation)
    if route == 'original-two-observations':
        # The old inventory's classification succeeds, then its extra live
        # _file_hash observation detects this specifically placed mutation.
        assert path.is_file()
        with pytest.raises(checkpoint.CheckpointError, match='changed in this process'):
            checkpoint._file_hash(path)
    else:
        # Under violated immutable-installation assumptions, B can return the
        # old digest at this exact cutpoint. The NEXT observation must refuse.
        assert fused(path) == original_digest
        with pytest.raises(checkpoint.CheckpointError, match='changed in this process'):
            fused(path)
    with pytest.raises(checkpoint.CheckpointError, match='changed in this process'):
        checkpoint._file_hash(path)


@pytest.mark.parametrize('deep', [False, True])
def test_persistent_selected_change_refuses_at_next_full_or_shallow_boundary(tiny_runtime, deep):
    checkpoint.runtime_fingerprint()
    tiny_runtime.source.write_bytes(tiny_runtime.source.read_bytes()+b'\npersistent = True\n')
    with pytest.raises(checkpoint.CheckpointError, match='changed in this process'):
        checkpoint.runtime_fingerprint(check_inventory=deep)


def test_stable_full_and_shallow_inventory_agree(tiny_runtime):
    first = checkpoint.runtime_fingerprint()
    members = dict(checkpoint._RUNTIME_PACKAGE_FILES)
    for deep in (True, False, True):
        assert checkpoint.runtime_fingerprint(check_inventory=deep) == first
        assert checkpoint._RUNTIME_PACKAGE_FILES == members
