"""Explicit synthetic role records for unit fixtures, never live-pool evidence."""
from copy import deepcopy
from pathlib import Path
import os
import stat
import subprocess


def synthetic_runtime(*, source='fixed-unit-fixture', pools=None, packages=None):
    if pools is None:
        pools = [dict(filepath=str(Path(__file__).resolve().with_name('__synthetic_blas__')),
                      user_api='blas', num_threads=1,
                      sha256='b' * 64, version='test-build')]
    else:
        pools = [dict(row, sha256=row.get('sha256', 'b' * 64)) for row in pools]
    return dict(python=source, implementation='CPython-unit-fixture', machine='unit-fixture',
                packages={} if packages is None else deepcopy(packages), package_native={},
                files_sha256='a' * 64, file_count=1 + len({row['filepath'] for row in pools}),
                native_pools=pools,
                runtime_policy='execution-role-v1', common_files_sha256='c' * 64,
                common_file_count=1)


def _is_junction(link):
    """Recognize the Windows mount-point tag on Python 3.11 and later."""
    entry = link.lstat()
    return (stat.S_ISDIR(entry.st_mode)
            and entry.st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT)


def beneath(path, root):
    return path == root or root in path.parents


def physical_path(path):
    """Normalize evidence spelling only; admission receives untouched objects."""
    value = str(Path(path).resolve(strict=True))
    if os.name == 'nt':
        prefix = '\\\\?\\'
        if value[:8].upper() == prefix + 'UNC\\':
            value = '\\\\' + value[8:]
        elif value.startswith(prefix):
            value = value[len(prefix):]
    return Path(value)


def owned_alias(link, target, owned_root):
    """Create only a link in the verified owned temporary tree."""
    if not beneath(link.absolute(), owned_root.absolute()):
        raise AssertionError('Alias must stay inside the owned case directory')
    if link.exists() or link.is_symlink():
        raise AssertionError('Alias destination already exists')
    if os.name == 'nt':
        completed = subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J', str(link), str(target)],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode:
            raise RuntimeError('Owned junction creation failed: ' + completed.stderr)
        assert _is_junction(link)
    else:
        link.symlink_to(target, target_is_directory=True)
        assert link.is_symlink()
    assert physical_path(link) == target
    assert os.path.samefile(link, target)
    return dict(link=str(link), target=str(target), same_file=True)


def remove_alias(receipt, owned_root):
    link, target = Path(receipt['link']), Path(receipt['target'])
    if not beneath(link.absolute(), owned_root.absolute()) or physical_path(link) != target:
        raise AssertionError('Owned link binding changed; cleanup refused')
    if os.name == 'nt':
        assert _is_junction(link)
        os.rmdir(link)  # Link only: never recurse through a junction.
    else:
        assert link.is_symlink()
        link.unlink()


