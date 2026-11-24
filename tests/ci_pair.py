"""Check the Ensemble source distribution, paired metadata and required CI results."""
import argparse
from email.parser import BytesHeaderParser
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import xml.etree.ElementTree as ET
import zipfile


def extract(source, sdist_dir, destination, dwd_source):
    # Use the same extraction preflight as the paired DWD CI. Only Ensemble's
    # expected resources differ. Git supplies their names independently of Flit.
    source = Path(source).resolve()
    roots = ['tests', 'examples', 'docs', '.github', 'src', '.gitattributes',
             'README.md', 'pyproject.toml', 'LICENSE', 'CITATION.cff', 'CHANGELOG.md',
             'VALIDATION.md', 'requirements-dwd.txt']
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z', '--', *roots])
    required = {}
    for entry in names.split(b'\0'):
        if entry:
            name = os.fsdecode(entry)
            path = source / name
            if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
                raise RuntimeError(f'Expected a regular checkout file: {name}')
            required[name] = path.read_bytes()
    if not any(name.startswith('tests/') for name in required):
        raise RuntimeError('Tracked Ensemble test inventory is empty.')
    script = Path(dwd_source).resolve() / 'tests' / 'ci_sdist.py'
    spec = importlib.util.spec_from_file_location('_paired_dwd_ci_sdist', script)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier.verify_and_extract(source, sdist_dir, destination, required=required)


def wheel_metadata(directory):
    wheels = list(Path(directory).glob('*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(f'Expected one wheel in {directory}, found {wheels}')
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise RuntimeError(f'Expected one METADATA in {wheels[0]}')
        return archive.read(names[0])


def check_metadata(raw, readme, dwd_version, label):
    from packaging.requirements import Requirement

    # A tolerant email parser can recover an apparent body even when the
    # required blank line is missing. Inspect the raw boundary first.
    headers, boundary, body = raw.replace(b'\r\n', b'\n').partition(b'\n\n')
    readme = readme.replace(b'\r\n', b'\n')
    if not boundary or body not in (readme, readme + b'\n'):
        raise RuntimeError(f'{label}: malformed header/README boundary or changed README')
    metadata = BytesHeaderParser().parsebytes(headers + boundary)
    if (metadata.defects or metadata.get_all('Name') != ['ensemble-dwd']
            or metadata.get_all('Description-Content-Type') != ['text/markdown']
            or len(metadata.get_all('Version', [])) != 1):
        raise RuntimeError(f'{label}: malformed Ensemble distribution headers')
    requirements = [Requirement(value) for value in metadata.get_all('Requires-Dist', [])]
    pins = [r for r in requirements if r.name.lower() == 'dwd']
    if (len(pins) != 1 or str(pins[0].specifier) != f'=={dwd_version}'
            or pins[0].marker or pins[0].extras or pins[0].url):
        raise RuntimeError(f'{label}: expected exactly one unconditional dwd=={dwd_version}')
    return metadata['Version']


def verify_pair_metadata(source, wheel_dir, dwd_wheel_dir):
    source = Path(source)
    dwd = BytesHeaderParser().parsebytes(wheel_metadata(dwd_wheel_dir))
    if dwd.get_all('Name') != ['dwd'] or len(dwd.get_all('Version', [])) != 1:
        raise RuntimeError('Expected the metadata of the paired DWD wheel.')
    readme = (source / 'README.md').read_bytes()
    version = check_metadata((source / 'PKG-INFO').read_bytes(), readme, dwd['Version'], 'PKG-INFO')
    rebuilt = check_metadata(wheel_metadata(wheel_dir), readme, dwd['Version'], 'wheel METADATA')
    if rebuilt != version:
        raise RuntimeError('Ensemble sdist and rebuilt wheel versions differ.')
    return {'ensemble_version': version, 'dwd_requirement': f'dwd=={dwd["Version"]}'}


REQUIRED_CASES = {
    'conic': ('tests.test_adaptive_conic_refit',
              'test_conic_selector_retains_cv_and_applies_final_timing_policy[clarabel-adaptive]'),
    'native': ('tests.test_ordered_gradient_dependency',
               'test_actual_native_reduction_reaches_both_base_and_final_fit'),
}


def require_pass(junit, route):
    classname, name = REQUIRED_CASES[route]
    cases = [case for case in ET.parse(junit).iter('testcase')
             if (case.get('classname'), case.get('name')) == (classname, name)]
    if len(cases) != 1:
        raise RuntimeError(f'Expected exactly one result for {classname}.{name}, found {len(cases)}')
    for child in cases[0]:
        if child.tag not in {'properties', 'system-out', 'system-err'}:
            raise RuntimeError(f'Required {classname}.{name} did not pass: {child.tag} {child.attrib}')
    return {'passed': f'{classname}.{name}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    archive = commands.add_parser('extract')
    archive.add_argument('--source', default='.')
    archive.add_argument('--sdist-dir', required=True)
    archive.add_argument('--destination', required=True)
    archive.add_argument('--dwd-source', required=True)
    archive.add_argument('--github-output', action='store_true')
    metadata = commands.add_parser('metadata')
    metadata.add_argument('--source', required=True)
    metadata.add_argument('--wheel-dir', required=True)
    metadata.add_argument('--dwd-wheel-dir', required=True)
    result = commands.add_parser('result')
    result.add_argument('--junit', required=True)
    result.add_argument('--require', choices=REQUIRED_CASES, required=True)
    args = parser.parse_args()
    if args.command == 'extract':
        record = extract(args.source, args.sdist_dir, args.destination, args.dwd_source)
        if args.github_output:
            with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
                output.write('source=' + record['source'] + '\n')
    elif args.command == 'metadata':
        record = verify_pair_metadata(args.source, args.wheel_dir, args.dwd_wheel_dir)
    else:
        record = require_pass(args.junit, args.require)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
