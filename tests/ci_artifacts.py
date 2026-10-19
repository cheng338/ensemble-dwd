"""Verify wheel installations and prepare tests without importable source code."""
import argparse
from email.parser import BytesParser
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


def single_wheel(directory):
    wheels = sorted(Path(directory).resolve().glob('*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(f'Expected exactly one wheel in {directory}: {wheels}')
    return wheels[0]


def verify_artifact(wheel, module_name, *, allowed_roots=None):
    """Bind imports and installed payload bytes to the supplied wheel."""
    roots = [Path(root).resolve() for root in (allowed_roots or [sys.prefix])]
    wheel = Path(wheel).resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(metadata_names) != 1:
            raise RuntimeError('Wheel must contain exactly one distribution metadata file.')
        expected = BytesParser().parsebytes(archive.read(metadata_names[0]))
        distribution = importlib.metadata.distribution(expected['Name'])
        module = importlib.import_module(module_name)
        origin = Path(module.__file__).resolve()
        installed_origin = Path(distribution.locate_file(module_name + '/__init__.py')).resolve()
        if origin != installed_origin or not any(origin.is_relative_to(root) for root in roots):
            raise RuntimeError(f'Import is outside the selected installation: {origin}')
        if module.__version__ != expected['Version'] or distribution.version != expected['Version']:
            raise RuntimeError('Imported, installed and wheel versions differ.')
        checked = 0
        for member in names:
            path = PurePosixPath(member)
            if path.is_absolute() or '..' in path.parts:
                raise RuntimeError('Unsafe wheel member path: ' + member)
            if member.endswith('/') or member.endswith('.dist-info/RECORD'):
                continue  # Installers update RECORD to include generated files.
            installed = Path(distribution.locate_file(member)).resolve()
            if not any(installed.is_relative_to(root) for root in roots):
                raise RuntimeError('Installed member is outside the selected prefix: ' + member)
            if not installed.is_file() or installed.read_bytes() != archive.read(member):
                raise RuntimeError('Installed member differs from the exact wheel: ' + member)
            checked += 1
    return {'distribution': expected['Name'], 'version': expected['Version'],
            'origin': str(origin), 'wheel': str(wheel),
            'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
            'verified_members': checked,
            'declares_accelerator': any(name.startswith('dwd/_residual_accel.')
                                       and name.endswith(('.pyd', '.so')) for name in names)}


def verify_backend(dwd_record, expected):
    import numpy as np
    from dwd import _compiled_residual

    declared = 'native' if dwd_record['declares_accelerator'] else 'portable'
    if expected != 'wheel' and declared != expected:
        raise RuntimeError(f'Expected {expected} wheel, found {declared}.')
    accelerator = _compiled_residual._ACCEL
    values = _compiled_residual.compiled_values(np.eye(8), np.ones(8), np.zeros(8), .5, .25)
    if declared == 'native':
        if accelerator is None:
            raise RuntimeError('Native wheel silently fell back to Python.')
        origin = Path(accelerator.__file__).resolve()
        if origin.parent != Path(dwd_record['origin']).parent:
            raise RuntimeError('Accelerator was imported from another installation.')
        if values is None or not all(np.all(np.isfinite(value)) for value in values):
            raise RuntimeError('Native residual helper did not execute successfully.')
    elif accelerator is not None or values is not None:
        raise RuntimeError('Portable wheel unexpectedly uses a compiled accelerator.')
    return {'backend': declared, 'compiled_helper_executed': values is not None}


def prepare_bundle(source, destination):
    """Keep tests, examples and their metadata/fixtures, but no package roots."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError('Test bundle must be outside the source checkout.')
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('tests', 'examples', 'doc', 'docs', '.github'):
        if (source / name).is_dir():
            shutil.copytree(source / name, destination / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in source.iterdir():
        if path.is_file() and (path.suffix in {'.md', '.toml', '.txt', '.cff'}
                               or path.name in {'setup.py', 'MANIFEST.in', 'LICENSE'}):
            shutil.copy2(path, destination / path.name)
    for name in ('dwd', 'ensemble_dwd', 'src'):
        if (destination / name).exists():
            raise RuntimeError('Test bundle contains an importable package source root.')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    install = commands.add_parser('install')
    install.add_argument('--wheel-dir', required=True)
    install.add_argument('--extras', default='')
    bundle = commands.add_parser('bundle')
    bundle.add_argument('--source', default='.')
    bundle.add_argument('--destination', required=True)
    check = commands.add_parser('check')
    check.add_argument('--dwd-wheel-dir', required=True)
    check.add_argument('--ensemble-wheel-dir')
    check.add_argument('--backend', choices=['native', 'portable', 'wheel'], required=True)
    args = parser.parse_args()
    if args.command == 'install':
        target = str(single_wheel(args.wheel_dir))
        if args.extras:
            target += '[' + args.extras + ']'
        subprocess.run([sys.executable, '-m', 'pip', 'install', target], check=True)
    elif args.command == 'bundle':
        print(prepare_bundle(args.source, args.destination))
    else:
        record = verify_artifact(single_wheel(args.dwd_wheel_dir), 'dwd')
        result = {'python': sys.executable, 'dwd': record,
                  **verify_backend(record, args.backend)}
        if args.ensemble_wheel_dir:
            result['ensemble'] = verify_artifact(single_wheel(args.ensemble_wheel_dir), 'ensemble_dwd')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
