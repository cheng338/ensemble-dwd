"""Reject malformed release metadata and skipped required tests on either artifact path."""
import xml.etree.ElementTree as ET
import zipfile

import pytest

from ci_pair import REQUIRED_CASES, require_pass, verify_pair_metadata


README = b'# Ensemble DWD\n\nExample README.\n'
HEADERS = (b'Metadata-Version: 2.4\nName: ensemble-dwd\nVersion: 0.4.1.dev0\n'
           b'Description-Content-Type: text/markdown\nRequires-Dist: dwd==2.0.2.dev0')
ENSEMBLE = HEADERS + b'\n\n' + README


def wheel(directory, name, payload):
    directory.mkdir()
    with zipfile.ZipFile(directory / f'{name}.whl', 'w') as archive:
        archive.writestr(f'{name}.dist-info/METADATA', payload)
    return directory


def verify(tmp_path, pkg_info, rebuilt):
    source = tmp_path / 'verified-sdist'
    source.mkdir()
    (source / 'README.md').write_bytes(README)
    (source / 'PKG-INFO').write_bytes(pkg_info)
    ensemble = wheel(tmp_path / 'ensemble', 'ensemble_dwd', rebuilt)
    dwd = wheel(tmp_path / 'dwd', 'dwd', b'Name: dwd\nVersion: 2.0.2.dev0\n\n')
    return verify_pair_metadata(source, ensemble, dwd)


def test_metadata_accepts_flit_line_endings_and_trailing_newline(tmp_path):
    assert verify(tmp_path, ENSEMBLE.replace(b'\n', b'\r\n'), ENSEMBLE + b'\n') == {
        'ensemble_version': '0.4.1.dev0', 'dwd_requirement': 'dwd==2.0.2.dev0'}


@pytest.mark.parametrize('artifact', ['sdist', 'rebuilt'])
@pytest.mark.parametrize('invalid', [
    HEADERS + b'\n' + README,  # Parser can recover a body despite the missing blank line.
    ENSEMBLE.replace(b'Example README.', b'Changed README.'),
    ENSEMBLE.replace(b'dwd==2.0.2.dev0', b'dwd>=2.0.2.dev0'),
    ENSEMBLE.replace(b'dwd==2.0.2.dev0', b'dwd==2.0.1'),
    ENSEMBLE.replace(b'dwd==2.0.2.dev0', b'dwd==2.0.2.dev0; python_version < "3.13"'),
    ENSEMBLE.replace(b'dwd==2.0.2.dev0', b'dwd==2.0.2.dev0\nRequires-Dist: DWD==2.0.2.dev0'),
], ids=['separator', 'readme', 'range', 'wrong-version', 'conditional', 'duplicate'])
def test_metadata_requires_strict_readme_boundary_and_exact_pin(tmp_path, artifact, invalid):
    with pytest.raises(RuntimeError, match='boundary|unconditional'):
        verify(tmp_path, invalid if artifact == 'sdist' else ENSEMBLE,
               invalid if artifact == 'rebuilt' else ENSEMBLE)


@pytest.mark.parametrize('route', ['conic', 'native'])
def test_required_result_uses_the_full_suite_report(tmp_path, route):
    path = tmp_path / 'junit.xml'
    classname, name = REQUIRED_CASES[route]
    suite = ET.Element('testsuite')
    ET.SubElement(suite, 'testcase', classname='another.module', name=name)
    ET.ElementTree(suite).write(path)
    with pytest.raises(RuntimeError, match='found 0'):
        require_pass(path, route)
    case = ET.SubElement(suite, 'testcase', classname=classname, name=name)
    ET.ElementTree(suite).write(path)
    assert require_pass(path, route) == {'passed': f'{classname}.{name}'}
    for status in ('skipped', 'failure', 'error'):
        child = ET.SubElement(case, status)
        ET.ElementTree(suite).write(path)
        with pytest.raises(RuntimeError, match='did not pass'):
            require_pass(path, route)
        case.remove(child)
    ET.SubElement(suite, 'testcase', classname=classname, name=name)
    ET.ElementTree(suite).write(path)
    with pytest.raises(RuntimeError, match='found 2'):
        require_pass(path, route)
