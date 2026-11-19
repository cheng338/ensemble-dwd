"""Checkpoint metadata membership, hashing, guards, and fallback behavior.

The literal Distribution.files loop and explicit expectations protect selected
metadata behavior. Unrelated RECORD filesystem errors have an explicit narrower
policy. All file mutations use owned temporary trees; no estimator fit,
prediction, or score runs. Simulated version checks prove routing only.
"""
import csv
import errno
import importlib.metadata as metadata
import io
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile

import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as cp

FAST = sys.implementation.name == 'cpython' and sys.version_info[:2] >= (3, 12)


def original(distribution):
    try:
        installed = metadata.distribution(distribution)
    except metadata.PackageNotFoundError:
        return None
    for member in installed.files or ():
        if str(member).replace('\\', '/').endswith('.dist-info/METADATA'):
            path = Path(installed.locate_file(member)).absolute()
            cp._file_hash(path)
            cp._RUNTIME_GUARDS.add(path)
    return installed.version


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    for name, value in (('_HASHES', {}), ('_RESOLVED_PATHS', {}),
                        ('_RUNTIME_GUARDS', set())):
        monkeypatch.setattr(cp, name, value)


def csv_text(rows):
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\r\n').writerows(rows)
    return stream.getvalue()


def installed(tmp_path, rows=None, text=None):
    site = tmp_path / 'site'
    directory = site / 'demo.dist-info'
    directory.mkdir(parents=True)
    (directory / 'METADATA').write_text('Name: demo\nVersion: 1.2.3\n', encoding='utf-8')
    if rows is not None or text is not None:
        (directory / 'RECORD').write_text(csv_text(rows) if text is None else text,
                                        encoding='utf-8', newline='')
    return metadata.PathDistribution(directory), site


def reset():
    cp._HASHES.clear()
    cp._RESOLVED_PATHS.clear()
    cp._RUNTIME_GUARDS.clear()


def capture(function, name='demo'):
    reset()
    try:
        result = ('value', function(name))
    except Exception as error:
        result = ('error', type(error).__module__, type(error).__qualname__)
    guards = sorted(str(path) for path in cp._RUNTIME_GUARDS)
    hashes = sorted((str(path), value[1]) for path, value in cp._HASHES.items())
    return result, guards, hashes


MAIN = 'demo.dist-info/METADATA'
CASES = [
    ('normal', [MAIN, 'unrelated.py', ''], [MAIN], [MAIN]),
    ('quoted-comma', [MAIN, 'other,one.dist-info/METADATA'], [MAIN, 'other,one.dist-info/METADATA'], [MAIN, 'other,one.dist-info/METADATA']),
    ('backslash', ['demo.dist-info\\METADATA'], ['demo.dist-info\\METADATA'], ['demo.dist-info\\METADATA']),
    ('dot-components', ['./demo.dist-info/../demo.dist-info/METADATA'], [MAIN], ['./demo.dist-info/../demo.dist-info/METADATA']),
    ('duplicate', [MAIN, MAIN], [MAIN], [MAIN]),
    ('second-metadata', [MAIN, 'second.dist-info/METADATA'], [MAIN, 'second.dist-info/METADATA'], [MAIN, 'second.dist-info/METADATA']),
    ('case-sensitive-suffix', ['other.DIST-INFO/METADATA'], ['other.DIST-INFO/METADATA'], []),
    ('case-sensitive-name', ['other.dist-info/metadata'], ['other.dist-info/metadata'], []),
    ('missing-member', [MAIN, 'absent.dist-info/METADATA'], [MAIN], [MAIN]),
    ('unlisted-metadata', ['unrelated.py'], ['unrelated.py'], []),
    ('relative-alias', ['../outside.dist-info/METADATA'], ['../outside.dist-info/METADATA'], ['../outside.dist-info/METADATA']),
    ('permissive-unclosed-quote', [MAIN], [MAIN], [MAIN]),
]


@pytest.mark.parametrize('case,names,existing,expected', CASES, ids=[r[0] for r in CASES])
def test_record_membership_and_guard_equivalence(tmp_path, monkeypatch, case, names, existing, expected):
    dist, site = installed(tmp_path, [[name, 'unchecked-hash-string', '0'] for name in names],
                           text=('"' + MAIN) if case == 'permissive-unclosed-quote' else None)
    for name in existing:
        # Create the physical backing file; RECORD keeps its original spelling.
        target = Path(os.path.normpath(site / name))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target != site / MAIN:
            target.write_text('Name: extra\nVersion: 9\n', encoding='utf-8')
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    old = capture(original)
    new = capture(cp._metadata_identity)
    assert new == old
    if site.drive.startswith('\\\\?\\') and case in ('dot-components', 'relative-alias'):
        # Windows verbatim paths do not interpret '..' as the parent directory.
        with pytest.raises(OSError) as invalid:
            (site / expected[0]).stat()
        assert invalid.value.winerror == 123
        outcome = (('error', 'builtins', 'OSError') if sys.version_info[:2] < (3, 12)
                   else ('value', '1.2.3'))
        assert new == (outcome, [], [])
    elif case == 'missing-member' and sys.version_info[:2] < (3, 12):
        # Real 3.11 Distribution.files retains absent records; the old hash fails.
        assert new[0][0] == 'error'
    else:
        assert new[0] == ('value', '1.2.3')
        assert new[1] == sorted(set(str(Path(site / name).absolute()) for name in expected))


MALFORMED = [
    ('zero-fields', '\n'),
    ('four-fields', 'other,hash,1,extra\n'),
    ('invalid-integer', 'other,hash,nope\n'),
    ('float-size', 'other,hash,1.5\n'),
    ('late-arity', MAIN + ',,1\nother,h,1,extra\n'),
    ('late-size', MAIN + ',,1\nother,h,NaN\n'),
    ('quoted-size', '"other,comma",hash,"not-an-int"\n'),
]


@pytest.mark.parametrize('case,text', MALFORMED, ids=[r[0] for r in MALFORMED])
def test_malformed_row_exception_equivalence(tmp_path, monkeypatch, case, text):
    dist, site = installed(tmp_path, text=text)
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    old = capture(original)
    new = capture(cp._metadata_identity)
    assert new == old
    assert new[0][0] == 'error'
    assert new[1:] == ([], [])


@pytest.mark.parametrize('record', [None, ''], ids=['absent', 'empty'])
def test_missing_or_empty_record_uses_literal_fallback(tmp_path, monkeypatch, record):
    dist, site = installed(tmp_path, text=record)
    (site / 'demo.dist-info' / 'SOURCES.txt').write_text(MAIN + '\n', encoding='utf-8')
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    new = capture(cp._metadata_identity)
    assert new == capture(original)
    assert new[0] == ('value', '1.2.3')
    assert new[1] == [str(site / MAIN)]
    assert cp._metadata_identity('demo') == '1.2.3'


def test_custom_distribution_uses_literal_files(tmp_path, monkeypatch):
    dist, site = installed(tmp_path, [[MAIN, '', '1']])
    class Custom(metadata.PathDistribution):
        def read_text(self, name):
            return super().read_text(name)
    custom = Custom(site / 'demo.dist-info')
    calls = []
    base = Custom.files
    monkeypatch.setattr(Custom, 'files', property(lambda self: (calls.append('files'), base.fget(self))[1]))
    monkeypatch.setattr(metadata, 'distribution', lambda name: custom)
    assert capture(cp._metadata_identity) == capture(original)
    assert calls == ['files', 'files']


@pytest.mark.parametrize('implementation,version', [('cpython', (3, 11, 15)), ('pypy', (3, 12, 0)), ('cpython', (3, 10, 0))])
def test_nonprimary_python_uses_literal_files(tmp_path, monkeypatch, implementation, version):
    dist, site = installed(tmp_path, [[MAIN, '', '1']])
    old = dist.files
    with monkeypatch.context() as patch:
        patch.setattr(cp.sys, 'implementation', SimpleNamespace(name=implementation))
        patch.setattr(cp.sys, 'version_info', version)
        patch.setattr(metadata.PathDistribution, 'files', property(lambda self: old))
        patch.setattr(metadata.PathDistribution, 'locate_file', lambda self, member: (_ for _ in ()).throw(AssertionError('fast root probe')))
        assert list(cp._metadata_record_members(dist)) == old


def test_metadata_permission_error_retained(tmp_path, monkeypatch):
    dist, site = installed(tmp_path, [[MAIN, '', '1']])
    target = site / MAIN
    old_exists = Path.exists
    def exists(path):
        if path == target:
            raise PermissionError(errno.EACCES, 'metadata refusal', str(path))
        return old_exists(path)
    monkeypatch.setattr(Path, 'exists', exists)
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    assert capture(cp._metadata_identity) == capture(original)
    assert capture(original)[0] == (('error', 'builtins', 'PermissionError')
                                    if sys.version_info[:2] >= (3, 12) else ('value', '1.2.3'))


def test_unrelated_permission_error_is_explicit_policy(tmp_path, monkeypatch):
    dist, site = installed(tmp_path, [['unrelated.py', '', '1'], [MAIN, '', '1']])
    old_exists = Path.exists
    def exists(path):
        if path == site / 'unrelated.py':
            raise OSError(errno.EIO, 'unrelated record refusal', str(path))
        return old_exists(path)
    monkeypatch.setattr(Path, 'exists', exists)
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    old = capture(original)
    new = capture(cp._metadata_identity)
    if FAST:
        assert old[0][-1] == 'OSError'
        assert new[0] == ('value', '1.2.3')
        assert new[1] == [str(site / MAIN)]
    else:
        assert new == old


def test_all_rows_validate_before_hashing(tmp_path, monkeypatch):
    dist, site = installed(tmp_path, text=MAIN + ',,1\nlate,h,broken\n')
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    calls = []
    monkeypatch.setattr(cp, '_file_hash', lambda path: calls.append(path))
    with pytest.raises(ValueError):
        cp._metadata_identity('demo')
    assert calls == []


def test_metadata_drift_rejected_next_observation(tmp_path, monkeypatch):
    dist, site = installed(tmp_path, [[MAIN, '', '1']])
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    assert cp._metadata_identity('demo') == '1.2.3'
    (site / MAIN).write_text('Name: demo\nVersion: 123.456.789\n', encoding='utf-8')
    with pytest.raises(cp.CheckpointError, match='Runtime artifact changed'):
        cp._metadata_identity('demo')


def test_record_member_locate_order(tmp_path, monkeypatch):
    dist, site = installed(tmp_path, [[MAIN + '/.', '', '1'], ['other.py', '', '2'],
                                     ['demo.dist-info//METADATA', '', '1']])
    (site / 'other.py').write_text('x', encoding='utf-8')
    actual = metadata.PathDistribution.locate_file
    calls = []
    def locate(self, member):
        calls.append(str(member))
        return actual(self, member)
    monkeypatch.setattr(metadata.PathDistribution, 'locate_file', locate)
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    old = capture(original)
    old_calls = list(calls)
    calls.clear()
    new = capture(cp._metadata_identity)
    assert new == old
    if FAST:
        assert calls[0] == ''  # Deliberate public native-Path eligibility probe.
        assert calls[1:] == [x for x in old_calls if x != 'other.py']
    else:
        assert calls == old_calls


def test_zip_root_uses_public_locator(tmp_path, monkeypatch):
    archive = tmp_path / 'package.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('demo.dist-info/METADATA', 'Name: demo\nVersion: 1.2.3\n')
        stream.writestr('demo.dist-info/RECORD', MAIN + ',,1\n')
    with zipfile.ZipFile(archive) as stream:
        dist = metadata.PathDistribution(zipfile.Path(stream, 'demo.dist-info/'))
        expected = dist.files
        assert list(cp._metadata_record_members(dist)) == expected
        monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
        # Preserve the literal caller's unsupported pathlib conversion, if any.
        assert capture(cp._metadata_identity) == capture(original)
    native, site = installed(tmp_path / 'native', [[MAIN, '', '1']])
    genuine = metadata.PathDistribution.locate_file
    for root_error in (False, True):
        def locate(self, member):
            if member == '':
                if root_error:
                    raise OSError(errno.EIO, 'eligibility probe unavailable')
                return object()
            return genuine(self, member)
        with monkeypatch.context() as patch:
            patch.setattr(metadata.PathDistribution, 'locate_file', locate)
            patch.setattr(metadata, 'distribution', lambda name: native)
            assert capture(cp._metadata_identity) == capture(original)


def test_current_distributions_equivalent():
    for _, distribution in cp._runtime_package_names():
        old = capture(original, distribution)
        new = capture(cp._metadata_identity, distribution)
        assert new == old, distribution
        assert new[0][0] == 'value', distribution
