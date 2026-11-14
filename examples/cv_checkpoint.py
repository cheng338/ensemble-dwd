"""Durable fixed-fold CV for the example workflows.

Completed folds are strict, checksummed JSON. Optional completed-pair recovery
loads only trusted local estimator pickles after identity and integrity checks.
An OS lock excludes a live coordinator. Per-fold worker locks and a generation
token prevent surviving workers from racing a replacement coordinator. A resume
refuses while an old worker is still fitting; it never terminates processes.

Runtime fingerprints describe disk artifacts and observed loaded native pools.
They require an installation that is immutable during the process lifetime;
they are not an attestation of machine code already mapped before import.
A checkpoint pins the complete worker execution runtime. The current coordinator
is pinned per session; only additional coordinator pools may differ across
sessions, with common source/library identity and worker pools unchanged.
"""
from datetime import datetime, timezone
from copy import deepcopy
from contextlib import ExitStack
from collections import Counter
import ast
import csv
import builtins
import errno
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
import threading
from time import perf_counter
from types import CodeType, FunctionType, SimpleNamespace
import uuid
import warnings

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import ThreadpoolController, threadpool_limits

from search_support import (CandidateEvaluator, array_fingerprint, data_fingerprint,
                            eligible_accuracy, exception_details, json_safe,
                            write_failure_json, write_json_atomic)
from search_support import (ACCURACY_AGGREGATION, accuracy_fraction, accuracy_evidence,
                            apply_accuracy_evidence, score_ranks, score_value)
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier
from work_reporting import summarize_records
import mnist_preprocessing
from mnist_preprocessing import UnitPixelScaler
import tune_mnist
import search_support as _selection_support
from tune_mnist import fully_valid_candidate_mask, ranked_valid_indices

SCHEMA = 2
_HASHES = {}
_HASH_LOCK = threading.RLock()
_RUNTIME_FILES = None
_RUNTIME_PACKAGE_FILES = None
_RUNTIME_PACKAGES = None
_RUNTIME_MODULES = None
_RUNTIME_DIGEST = None
_RUNTIME_NATIVE = None
_RUNTIME_ACCELERATOR = None
_RUNTIME_GUARDS = set()
_RUNTIME_CONTROLLER = None
_RUNTIME_MODULE_COUNT = None
_RESOLVED_PATHS = {}
_RUNTIME_ARTIFACT_ALIASES = {}
_RUNTIME_CANONICAL_ARTIFACTS = {}
_RUNTIME_FILE_COUNT = None
_RUNTIME_COMMON_FILES = None
_RUNTIME_COMMON_DIGEST = None
_RUNTIME_COMMON_FILE_COUNT = None
_RUNTIME_ROLE_POLICY = 'execution-role-v1'
_SELECTION_STATE = None
_SELECTION_COMPILE_EVIDENCE = ()
_EQUIVALENT_POLICY = 'retained-count-v2'
_SEED_AGGREGATION_POLICY = 'mean-seeds-within-fold-v1'
_REUSE_METHODS = {
    cls: {name: value for name, value in vars(cls).items() if callable(value)}
    for cls in (Pipeline, StandardScaler, UnitPixelScaler, OvOClassifier, EnsembleKernelDWD, EnsembleKernelSVM)
}


class CheckpointError(RuntimeError):
    """A saved identity or receipt cannot safely be reused."""


class CheckpointBusyError(CheckpointError):
    """Another coordinator or surviving fold worker still owns this run."""


def validate_seed(value, *, name='seed'):
    """Return a Python integer accepted by the examples' random-state policy."""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= value < 2 ** 32):
        raise ValueError(f'{name} must be an integer between 0 and 2**32 - 1')
    return int(value)


def validate_cv_seeds(value):
    """Normalize an explicit, ordered, nonempty list of unique sampling seeds."""
    if value is None:
        return None
    if type(value) is np.ndarray:
        if value.ndim != 1 or value.dtype.kind not in 'iu':
            raise ValueError('cv_seeds must be a one-dimensional integer array')
    elif type(value) not in (list, tuple):
        raise ValueError('cv_seeds must be an ordered list, tuple or integer array')
    seeds = tuple(validate_seed(seed, name='cv_seeds entry') for seed in value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('cv_seeds must be nonempty and contain no duplicate seeds')
    return seeds


def _seed_ancestors(seed_parameter):
    if (type(seed_parameter) is not str or not seed_parameter
            or any(not part for part in seed_parameter.split('__'))
            or seed_parameter.split('__')[-1] != 'random_state'):
        raise ValueError('seed_parameter must name an existing random_state parameter')
    parts = seed_parameter.split('__')
    return ['__'.join(parts[:length]) for length in range(1, len(parts))]


def validate_seed_parameters(parameters, seed_parameter):
    """Reject tuning the sampling seed or replacing its containing estimator.

    This accepts a candidate or a grid dictionary; only its parameter names
    are inspected. CheckpointedCV also checks the actual estimator path after
    setting each candidate, before any fold is dispatched.
    """
    ancestors = _seed_ancestors(seed_parameter)
    replaced_steps = {'__'.join(path + ['steps']) for path in
                      [seed_parameter.split('__')[:length] for length in range(len(ancestors))]}
    conflicts = {seed_parameter, *ancestors, *replaced_steps}.intersection(parameters)
    if conflicts:
        raise ValueError('Multi-seed CV cannot tune or replace the sampling-seed path: '
                         + ', '.join(sorted(conflicts)))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _io_path(path):
    """Use extended Windows paths for all internal receipt and lock I/O."""
    absolute = os.path.abspath(os.fspath(path))
    if os.name == 'nt' and not absolute.startswith('\\\\?\\'):
        absolute = ('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                    else '\\\\?\\' + absolute)
    return Path(absolute)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError('Duplicate JSON key in checkpoint')
        result[key] = value
    return result


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(CheckpointError('Nonfinite JSON token')))
    except (ValueError, OSError) as error:
        raise CheckpointError('Unreadable checkpoint: ' + str(path)) from error


def _write_receipt(path, payload):
    write_json_atomic(path, {'payload': payload, 'sha256': _digest(payload)}, replace=True)


def _read_receipt(path, identity_sha256):
    record = _read_json(path)
    if not isinstance(record, dict) or set(record) != {'payload', 'sha256'}:
        raise CheckpointError('Malformed checkpoint envelope: ' + str(path))
    payload = record['payload']
    if (not isinstance(payload, dict) or record['sha256'] != _digest(payload)
            or payload.get('identity_sha256') != identity_sha256):
        raise CheckpointError('Checkpoint checksum or identity mismatch: ' + str(path))
    return payload


class _Lease:
    """Non-inherited advisory lock; lockfiles remain as harmless stable names."""
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = None
        try:
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            os.set_inheritable(descriptor, False)
            self.stream = os.fdopen(descriptor, 'r+b', buffering=0)
            descriptor = None  # The stream owns the descriptor from here.
            if os.fstat(self.stream.fileno()).st_size == 0:
                self.stream.write(b'\0')
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as error:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            elif descriptor is not None:
                os.close(descriptor)
            if isinstance(error, OSError):
                raise CheckpointBusyError('Live or inaccessible checkpoint owner: ' + str(self.path)) from error
            raise
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            try:
                self.stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


def _file_hash(path):
    # Keep the lexical path: its live stat must detect alias retargeting.
    raw = os.fspath(path)
    if raw not in _RESOLVED_PATHS:
        _RESOLVED_PATHS[raw] = Path(path).absolute()
    path = _RESOLVED_PATHS[raw]
    info = path.stat()
    return _file_hash_observed(path, info)


def _inventory_regular_stat(path):
    """Fresh lexical classification with Path.is_file's primary-target errors.

    This is not DirEntry's cached metadata. Keep the observation before suffix
    filtering, including error propagation for files outside the inventory.
    """
    try:
        info = path.stat()
    except OSError as error:
        if (error.errno in (errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP)
                or getattr(error, 'winerror', None) in (21, 123, 1921)):
            return None
        raise
    except ValueError:
        return None
    return info if stat.S_ISREG(info.st_mode) else None


def _file_hash_observed(path, info):
    """Reuse this inventory visit's live stat; later observations stay fresh.

    An installation must remain immutable. Removing the adjacent second stat
    does not preserve every possible concurrent-mutation detection cutpoint.
    """
    stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    with _HASH_LOCK:
        if path in _HASHES:
            previous, digest = _HASHES[path]
            if previous != stamp:
                raise CheckpointError('Runtime artifact changed in this process: ' + str(path))
            return digest
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise CheckpointError('Runtime artifact changed while fingerprinting: ' + str(path))
        result = digest.hexdigest()
        _HASHES[path] = (stamp, result)
        return result


def _runtime_artifact_identity(path):
    """Emit a verified physical spelling while retaining the live lexical guard.

    Only first registration resolves the path. Every use still hashes/checks the
    original lexical name, so a cached resolved target cannot hide retargeting.
    The identity is for an immutable installation, not mapped-code attestation.
    """
    raw = Path(path).absolute()
    with _HASH_LOCK:
        digest = _file_hash(raw)
        stamp = _HASHES[raw][0]
        previous = _RUNTIME_ARTIFACT_ALIASES.get(str(raw))
        if previous is not None:
            if previous[1:] != (stamp, digest):
                raise CheckpointError('Runtime artifact alias changed: ' + str(raw))
            return previous
        # Windows public and extended spellings share one extended I/O form.
        # resolve(strict=True) supplies the physical spelling; no case folding
        # or unverified prefix/suffix matching establishes an alias.
        resolved = _io_path(raw.resolve(strict=True))
        physical = resolved.stat()
        physical_stamp = (physical.st_dev, physical.st_ino, physical.st_size,
                          physical.st_mtime_ns, physical.st_ctime_ns)
        if stamp[1] == 0 or physical_stamp != stamp:
            raise CheckpointError('Runtime artifact physical identity is unavailable: ' + str(raw))
        # Recheck both names around the existing hash guard before caching.
        verified_digest = _file_hash(raw)
        after_raw, after_resolved = raw.stat(), resolved.stat()
        stamps = [(item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
                   item.st_ctime_ns) for item in (after_raw, after_resolved)]
        if verified_digest != digest or any(item != stamp for item in stamps):
            raise CheckpointError('Runtime artifact alias changed while identifying: ' + str(raw))
        canonical = str(resolved)
        known = _RUNTIME_CANONICAL_ARTIFACTS.get(canonical)
        if known is not None and known != (stamp, digest):
            raise CheckpointError('Conflicting physical runtime artifact: ' + canonical)
        result = (canonical, stamp, digest)
        _RUNTIME_CANONICAL_ARTIFACTS[canonical] = (stamp, digest)
        _RUNTIME_ARTIFACT_ALIASES[str(raw)] = result
        return result


def _runtime_mismatch_details(expected, observed):
    """Bounded field names only; this never normalizes the equality predicate."""
    def fields(value):
        if (type(value) is not list or len(value) != 2 or value[0] != 'dict'
                or type(value[1]) is not list or len(value[1]) > 16):
            return None
        result = {}
        for pair in value[1]:
            if (type(pair) is not list or len(pair) != 2
                    or type(pair[0]) is not str or pair[0] in result):
                return None
            result[pair[0]] = pair[1]
        return result
    left, right = fields(expected), fields(observed)
    if left is None or right is None:
        return ' (runtime record structure differs)'
    names = ('python', 'implementation', 'machine', 'packages', 'package_native',
             'files_sha256', 'file_count', 'native_pools')
    changed = [name for name in names
               if (name in left) != (name in right) or left.get(name) != right.get(name)]
    if not changed:
        return ' (runtime field order or structure differs)'
    shown = changed[:4]
    return ' (differing fields: ' + ', '.join(shown) + ('; more' if len(changed) > 4 else '') + ')'


def _class_identity(value):
    cls = type(value)
    source = inspect.getsourcefile(cls)
    if not source:
        raise CheckpointError('Estimator class source is unavailable: ' + cls.__qualname__)
    return {'class': cls.__module__ + '.' + cls.__qualname__, 'source_sha256': _file_hash(source)}


def _stable(value):
    """Encode values and types without changing their numerical semantics.

    Every node is tagged, including ordinary containers, so user parameters
    cannot impersonate an array, estimator or RNG encoding. Unsupported
    subclasses are rejected rather than identified only by their base value.
    """
    cls = type(value)
    if value is None:
        return ['none']
    if cls in (str, bool, int):
        return [cls.__name__, value]
    if cls is float:
        if not math.isfinite(value):
            raise CheckpointError('Nonfinite parameter identity')
        return ['float', value.hex()]
    if isinstance(value, np.generic) and cls is value.dtype.type:
        if isinstance(value, np.floating):
            if not np.isfinite(value):
                raise CheckpointError('Nonfinite parameter identity')
            # A round-trippable value avoids longdouble's platform padding bytes
            # and the precision loss of conversion to a Python float.
            scalar = np.format_float_scientific(value, unique=True, trim='k')
        elif isinstance(value, np.integer):
            scalar = str(int(value))
        elif isinstance(value, np.bool_):
            scalar = bool(value)
        elif isinstance(value, np.str_):
            scalar = str(value)
        else:
            raise CheckpointError('Unsupported NumPy scalar parameter identity: ' + cls.__qualname__)
        return ['numpy_scalar', cls.__module__ + '.' + cls.__qualname__, value.dtype.str, scalar]
    if cls is type(Path()):
        return ['path', cls.__module__ + '.' + cls.__qualname__, str(value)]
    if cls is np.ndarray:
        if (value.dtype.hasobject or value.dtype.fields is not None
                or value.dtype.subdtype is not None or value.dtype.metadata is not None):
            raise CheckpointError('Object, structured and metadata dtypes lack a supported parameter identity')
        return ['array', value.dtype.type.__module__ + '.' + value.dtype.type.__qualname__,
                array_fingerprint(value), list(value.strides),
                [bool(value.flags[name]) for name in ('C_CONTIGUOUS', 'F_CONTIGUOUS', 'WRITEABLE', 'ALIGNED')]]
    if cls is np.random.RandomState:
        return ['RandomState', _stable(value.get_state())]
    if cls is np.random.Generator:
        bit_cls = type(value.bit_generator)
        if bit_cls is not getattr(np.random, bit_cls.__name__, None):
            raise CheckpointError('Unsupported random bit-generator identity')
        return ['Generator', bit_cls.__module__ + '.' + bit_cls.__qualname__,
                _stable(value.bit_generator.state)]
    if isinstance(value, BaseEstimator):
        parameters = value.get_params(deep=False)
        if parameters.get('random_state', 0) is None:
            raise CheckpointError('Checkpointed estimators require an explicit random_state; None cannot safely resume stochastic work')
        return ['estimator', _class_identity(value), _stable(parameters)]
    if cls is dict:
        if any(type(key) is not str for key in value):
            raise CheckpointError('Parameter dictionaries require string keys')
        return ['dict', [[key, _stable(item)] for key, item in value.items()]]
    if cls in (tuple, list):
        return [cls.__name__, [_stable(item) for item in value]]
    raise CheckpointError('Unsupported reusable parameter identity: ' + type(value).__qualname__)


def _runtime_package_names(modules=None):
    names = [('dwd', 'dwd'), ('ensemble_dwd', 'ensemble-dwd'),
             ('numpy', 'numpy'), ('scipy', 'scipy'), ('sklearn', 'scikit-learn'),
             ('joblib', 'joblib'), ('threadpoolctl', 'threadpoolctl')]
    modules = sys.modules if modules is None else modules
    # Fingerprint an already active bridge, without activating an absent backend.
    if modules.get('aocl_runtime') is not None:
        names.append(('aocl_runtime', 'aocl-runtime'))
    return names


def _metadata_record_members(installed):
    """Select ordinary RECORD metadata with metadata-only existence checks.

    This path uses the CPython 3.12+ missing-member policy. Older interpreters,
    nonstandard providers and non-filesystem roots retain Distribution.files.
    Validation/performance evidence is limited to the tested interpreters.
    """
    if (sys.implementation.name != 'cpython' or sys.version_info[:2] < (3, 12)
            or type(installed) is not importlib.metadata.PathDistribution):
        return installed.files or ()
    try:
        root = installed.locate_file('')
    except Exception:
        return installed.files or ()
    if type(root) is not type(Path()):
        return installed.files or ()
    record = installed.read_text('RECORD')
    if not record:
        return installed.files or ()

    def member(name, hash=None, size_str=None):
        path = importlib.metadata.PackagePath(name)
        if size_str:
            int(size_str)
        return path

    selected = []
    for row in csv.reader(record.splitlines()):
        path = member(*row)
        # B deliberately omits filesystem failure checks for unrelated records.
        if (str(path).replace('\\', '/').endswith('.dist-info/METADATA')
                and installed.locate_file(path).exists()):
            selected.append(path)
    return selected


def _metadata_identity(distribution):
    try:
        installed = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    for member in _metadata_record_members(installed):
        if str(member).replace('\\', '/').endswith('.dist-info/METADATA'):
            path = Path(installed.locate_file(member)).absolute()
            _file_hash(path)
            _RUNTIME_GUARDS.add(path)
    return installed.version


def _loaded_source_guards():
    """Only executed module sources need a per-fit stat; inventory checks are deeper."""
    prefixes = tuple(name for name, _ in _runtime_package_names())
    for name, module in tuple(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            continue
        path = getattr(module, '__file__', None)
        if path is None:
            continue
        if path.endswith('.pyc'):
            path = importlib.util.source_from_cache(path)
        path = Path(path).absolute()
        if path.suffix.lower() in ('.py', '.pyd', '.dll', '.so', '.dylib') or '.so.' in path.name:
            # A newly imported file must be one of the initially identified artifacts.
            if str(path) not in _RUNTIME_PACKAGE_FILES:
                raise CheckpointError('A numerical module loaded outside the frozen inventory: ' + str(path))
            _RUNTIME_GUARDS.add(path)


def _preprocessing_source():
    """Keep the example transformer tied to the adjacent source file."""
    expected = Path(__file__).with_name('mnist_preprocessing.py').resolve()
    if (Path(mnist_preprocessing.__file__).resolve() != expected
            or mnist_preprocessing.UnitPixelScaler is not UnitPixelScaler
            or Path(inspect.getsourcefile(UnitPixelScaler)).resolve() != expected):
        raise CheckpointError('MNIST preprocessing loaded outside the frozen example source')
    return expected


def _selection_source():
    """Bind the eligibility and ranking rules to the adjacent example source."""
    expected = Path(__file__).with_name('tune_mnist.py').resolve()
    if (sys.modules.get('tune_mnist') is not tune_mnist
            or Path(tune_mnist.__file__).resolve() != expected
            or tune_mnist.fully_valid_candidate_mask is not fully_valid_candidate_mask
            or tune_mnist.ranked_valid_indices is not ranked_valid_indices
            or Path(inspect.getsourcefile(fully_valid_candidate_mask)).resolve() != expected
            or Path(inspect.getsourcefile(ranked_valid_indices)).resolve() != expected):
        raise CheckpointError('Selection policy loaded outside the frozen example source')
    check_selection_state()
    return expected


def _selection_failure():
    raise CheckpointError('Selection policy changed in memory; restart before resuming')


def _selection_defaults(function, name):
    if function.__defaults__ is not None or function.__closure__ is not None:
        _selection_failure()
    keywords = function.__kwdefaults__
    if name in ('score_value', 'score_ranks'):
        if type(keywords) is not dict or set(keywords) != {'exact'} or keywords['exact'] is not None:
            _selection_failure()
    elif keywords is not None:
        _selection_failure()


def _selection_aggregation_literal(tree):
    """Read the fixed source's sole direct string declaration, without execution.

    This conservative syntax contract also declines nested-local rebinding. It
    is not a general Python binding analyzer or a hostile-code security boundary.
    """
    name = 'ACCURACY_AGGREGATION'
    declarations = [node for node in tree.body
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == name]
    if len(declarations) != 1:
        _selection_failure()
    declaration = declarations[0]
    if not isinstance(declaration.value, ast.Constant) or type(declaration.value.value) is not str:
        _selection_failure()
    bindings = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id == name
                and isinstance(node.ctx, (ast.Store, ast.Del))):
            bindings.append(node)
        elif (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                               ast.ExceptHandler, ast.MatchAs, ast.MatchStar))
              and node.name == name):
            bindings.append(node)
        elif isinstance(node, ast.MatchMapping) and node.rest == name:
            bindings.append(node)
        elif isinstance(node, ast.alias) and (node.asname or node.name.split('.')[0]) == name:
            bindings.append(node)
    if len(bindings) != 1 or bindings[0] is not declaration.targets[0]:
        _selection_failure()
    return declaration.value.value

def _initialize_selection_state():
    """Compare fourteen loaded example functions with compiled, unexecuted source.

    This is a bounded consistency check, not an interpreter security boundary.
    Library internals, hostile changes to this guard and concurrent mutation
    between a check and use remain outside its guarantee.
    """
    global _SELECTION_COMPILE_EVIDENCE
    directory = Path(__file__).resolve().parent
    modules = ((tune_mnist, 'tune_mnist'), (_selection_support, 'search_support'))
    sources = {}
    for module, name in modules:
        path = directory / (name + '.py')
        if (sys.modules.get(name) is not module
                or Path(module.__file__).resolve() != path):
            _selection_failure()
        digest = _file_hash(path)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != digest or _file_hash(path) != digest:
            _selection_failure()
        sources[name] = (source, digest)
    functions = (
        (tune_mnist, 'fully_valid_candidate_mask', fully_valid_candidate_mask),
        (tune_mnist, 'ranked_valid_indices', ranked_valid_indices),
        (_selection_support, 'score_value', score_value),
        (_selection_support, 'accuracy_evidence', accuracy_evidence),
        (_selection_support, 'accuracy_fraction', accuracy_fraction),
        (_selection_support, '_fraction_record', _selection_support._fraction_record),
        (_selection_support, 'exact_accuracy_available', _selection_support.exact_accuracy_available),
        (_selection_support, 'score_token', _selection_support.score_token),
        (_selection_support, 'token_value', _selection_support.token_value),
        (_selection_support, 'comparable_tokens', _selection_support.comparable_tokens),
        (_selection_support, 'equal_score_tokens', _selection_support.equal_score_tokens),
        (_selection_support, 'score_nonimproving', _selection_support.score_nonimproving),
        (_selection_support, 'apply_accuracy_evidence', apply_accuracy_evidence),
        (_selection_support, 'score_ranks', score_ranks),
    )
    # Function code objects do not include evaluated defaults. Check the fixed
    # release's source declarations too, before trusting loaded None defaults.
    constants = {}
    for module, module_name in modules:
        names = {name for owner, name, _ in functions if owner is module}
        definitions = {name: [] for name in names}
        tree = ast.parse(sources[module_name][0], filename=str(directory / (module_name + '.py')))
        if module_name == 'search_support':
            constants['ACCURACY_AGGREGATION'] = _selection_aggregation_literal(tree)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in definitions:
                definitions[node.name].append(node)
        for name, matches in definitions.items():
            if (len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef)
                    or matches[0].decorator_list):
                _selection_failure()
            arguments = matches[0].args
            if arguments.defaults:
                _selection_failure()
            if name in ('score_value', 'score_ranks'):
                if (len(arguments.kwonlyargs) != 1 or arguments.kwonlyargs[0].arg != 'exact'
                        or len(arguments.kw_defaults) != 1
                        or not isinstance(arguments.kw_defaults[0], ast.Constant)
                        or arguments.kw_defaults[0].value is not None):
                    _selection_failure()
            elif arguments.kwonlyargs or arguments.kw_defaults:
                _selection_failure()
    compiled, states, evidence = {}, [], []
    for module, name, function in functions:
        if (type(function) is not FunctionType or getattr(module, name) is not function
                or function.__globals__ is not module.__dict__):
            _selection_failure()
        _selection_defaults(function, name)
        module_name = 'tune_mnist' if module is tune_mnist else 'search_support'
        source, digest = sources[module_name]
        filename = function.__code__.co_filename
        if Path(filename).resolve() != directory / (module_name + '.py'):
            _selection_failure()
        key = (module_name, filename)
        if key not in compiled:
            # compile constructs code objects only; it does not execute imports
            # or module statements. Use the loaded filename/optimization mode
            # so normal interpreter-specific code metadata compares correctly.
            module_code = compile(source, filename, 'exec', dont_inherit=True,
                                  optimize=sys.flags.optimize)
            compiled[key] = {item.co_name: item for item in module_code.co_consts
                             if type(item) is CodeType}
        expected = compiled[key].get(name)
        if expected is None or function.__code__ != expected:
            _selection_failure()
        states.append((module, name, function, function.__code__))
        evidence.append((module_name, digest, filename, name))
    by_name = {name: function for _, name, function, _ in states}
    declared_aggregation = constants['ACCURACY_AGGREGATION']
    for loaded in (_selection_support.ACCURACY_AGGREGATION, ACCURACY_AGGREGATION):
        if type(loaded) is not str or loaded != declared_aggregation:
            _selection_failure()
    bindings = [
        (sys.modules, 'tune_mnist', tune_mnist),
        (sys.modules, 'search_support', _selection_support),
        (sys.modules, 'numpy', np), (sys.modules, 're', re),
        (tune_mnist.__dict__, 'np', np), (tune_mnist.__dict__, 're', re),
        (tune_mnist.__dict__, 'score_value', by_name['score_value']),
        (_selection_support.__dict__, 'np', np),
        (_selection_support.__dict__, 'Fraction', Fraction),
        (_selection_support.__dict__, 'ACCURACY_AGGREGATION', _selection_support.ACCURACY_AGGREGATION),
        (globals(), 'ACCURACY_AGGREGATION', ACCURACY_AGGREGATION),
    ]
    for name in ('fully_valid_candidate_mask', 'ranked_valid_indices'):
        bindings.append((tune_mnist.__dict__, name, by_name[name]))
        bindings.append((globals(), name, by_name[name]))
    for name in ('score_value', 'accuracy_evidence', 'accuracy_fraction',
                 '_fraction_record', 'exact_accuracy_available', 'score_token',
                 'token_value', 'comparable_tokens', 'equal_score_tokens',
                 'score_nonimproving', 'apply_accuracy_evidence', 'score_ranks'):
        bindings.append((_selection_support.__dict__, name, by_name[name]))
    for name in ('score_value', 'accuracy_evidence', 'accuracy_fraction',
                 'apply_accuracy_evidence', 'score_ranks'):
        bindings.append((globals(), name, by_name[name]))
    # A global shadow of a builtin also changes these functions' semantics.
    # The fixed names below are the builtins used by the fourteen source functions.
    builtin_names = {
        'tune_mnist': ('sorted', 'int', 'list', 'range', 'len', 'float', 'ValueError'),
        'search_support': ('isinstance', 'dict', 'set', 'ValueError', 'type', 'int',
                           'len', 'any', 'sum', 'enumerate', 'float', 'range'),
    }
    missing_globals = []
    for module, name, function, _ in states:
        module_name = 'tune_mnist' if module is tune_mnist else 'search_support'
        for builtin_name in builtin_names[module_name]:
            if builtin_name in module.__dict__:
                _selection_failure()
            missing_globals.append((module.__dict__, builtin_name))
            bindings.append((function.__builtins__, builtin_name, getattr(builtins, builtin_name)))
    attributes = (
        (np, 'ma', np.ma), (np.ma, 'asarray', np.ma.asarray),
        (np, 'nan', np.nan), (np, 'isfinite', np.isfinite),
        (np, 'flatnonzero', np.flatnonzero), (np, 'asarray', np.asarray),
        (np, 'std', np.std), (np, 'int32', np.int32),
        (re, 'fullmatch', re.fullmatch),
    )
    unique_bindings, binding_keys = [], set()
    for mapping, name, expected in bindings:
        if mapping.get(name) is not expected:
            _selection_failure()
        key = (id(mapping), name)
        if key not in binding_keys:
            binding_keys.add(key)
            unique_bindings.append((mapping, name, expected))
    unique_missing = {(id(mapping), name): (mapping, name)
                      for mapping, name in missing_globals}
    consumers = (
        ('adaptive_search', (('fully_valid_candidate_mask', by_name['fully_valid_candidate_mask']),
                             ('ranked_valid_indices', by_name['ranked_valid_indices']),
                             ('score_value', by_name['score_value']),
                             ('score_token', by_name['score_token']),
                             ('token_value', by_name['token_value']),
                             ('comparable_tokens', by_name['comparable_tokens']),
                             ('equal_score_tokens', by_name['equal_score_tokens']),
                             ('score_nonimproving', by_name['score_nonimproving']))),
        ('ratio_refinement', (('score_token', by_name['score_token']),
                              ('comparable_tokens', by_name['comparable_tokens']))),
        ('numerical_confirmation', (('score_value', by_name['score_value']),
                                    ('score_ranks', by_name['score_ranks']),
                                    ('exact_accuracy_available', by_name['exact_accuracy_available']))),
    )
    _SELECTION_COMPILE_EVIDENCE = tuple(evidence)
    return tuple(states), tuple(unique_bindings), attributes, tuple(unique_missing.values()), consumers


def check_selection_state():
    """Check the finite selection dependency set without rescanning the runtime."""
    global _SELECTION_STATE
    if _SELECTION_STATE is None:
        with _HASH_LOCK:
            if _SELECTION_STATE is None:
                _SELECTION_STATE = _initialize_selection_state()
    states, bindings, attributes, missing_globals, consumers = _SELECTION_STATE
    for module, name, function, code in states:
        if (getattr(module, name, None) is not function or function.__code__ is not code
                or function.__globals__ is not module.__dict__):
            _selection_failure()
        _selection_defaults(function, name)
    for mapping, name, expected in bindings:
        if mapping.get(name) is not expected:
            _selection_failure()
    for owner, name, expected in attributes:
        if getattr(owner, name, None) is not expected:
            _selection_failure()
    if any(name in mapping for mapping, name in missing_globals):
        _selection_failure()
    # Consumers may load after cv_checkpoint. Do not import them here: adaptive
    # and optional workflows import this module themselves. Ratio/confirmation
    # lazy selector imports are bound by the tune_mnist module/function checks.
    for module_name, aliases in consumers:
        module = sys.modules.get(module_name)
        if module is not None:
            for name, expected in aliases:
                if getattr(module, name, None) is not expected:
                    _selection_failure()


def _native_pool_fingerprint(files, *, native_threads):
    """Observe native pools; callers own complete runtime boundary checks.

    ``files`` receives the same lexical artifact guards used by the full
    fingerprint. This helper does not claim to observe package/source identity.
    """
    global _RUNTIME_CONTROLLER, _RUNTIME_MODULE_COUNT
    _RUNTIME_CONTROLLER = ThreadpoolController()
    _RUNTIME_MODULE_COUNT = len(sys.modules)
    pools = []
    for pool in _RUNTIME_CONTROLLER.info():
        record = {key: value for key, value in pool.items() if key != 'filepath'}
        path = Path(pool['filepath']).absolute()
        files[str(path)] = _file_hash(path)
        canonical, _, verified = _runtime_artifact_identity(path)
        if verified != files[str(path)]:
            raise CheckpointError('Runtime pool artifact identity changed')
        record.update(filepath=canonical, sha256=verified)
        if pool.get('num_threads') != native_threads:
            if native_threads == 1:
                raise CheckpointError('Checkpointed CV requires one active native thread per worker')
            raise CheckpointError('Final-fit runtime does not match its requested native-thread budget')
        pools.append(record)
    if not any(pool.get('user_api') == 'blas' for pool in pools):
        raise CheckpointError('Loaded BLAS identity is unavailable')
    return pools


def _runtime_package_paths(root):
    """Enumerate a stable package tree, retaining live caller file checks.

    Directory entries select traversal only; the caller uses fresh lexical
    regular-file classification and hash checks. Do not descend into directory symlinks
    or excluded bytecode caches. A changed installation is rejected by the
    existing cross-observation inventories and live artifact guards; no traversal
    is an atomic filesystem snapshot.
    """
    if not hasattr(type(root), 'walk'):
        # Preserve the older pathlib traversal and its narrower error policy.
        yield from root.rglob('*')
        return
    if not root.is_dir() or '__pycache__' in root.parts:
        return
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            scan = os.scandir(directory)
        except OSError:
            # Match pathlib's default handling of an unlistable directory.
            continue
        with scan:
            entries = list(scan)
        children = []
        for entry in entries:
            if entry.name == '__pycache__':
                continue
            path = directory / entry.name
            yield path
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_directory = False
            if is_directory:
                children.append(path)
        pending.extend(reversed(children))


def runtime_fingerprint(*, check_inventory=True, native_threads=1):
    """Hash once, check complete inventory per stage and core/native guards per fit.

    Workers avoid traversing large numerical package trees on every fast fit.
    Their first call still identifies the full runtime. The coordinator checks
    the full inventory before and after each search, while every worker checks
    the DWD/ensemble cache identity, loaded DLLs and checkpoint source each fit.
    """
    if isinstance(native_threads, (bool, np.bool_)) or not isinstance(native_threads, (int, np.integer)) or native_threads < 1:
        raise ValueError('native_threads must be a positive integer')
    global _RUNTIME_FILES, _RUNTIME_PACKAGE_FILES, _RUNTIME_PACKAGES, _RUNTIME_MODULES
    global _RUNTIME_DIGEST, _RUNTIME_NATIVE, _RUNTIME_ACCELERATOR, _RUNTIME_FILE_COUNT
    global _RUNTIME_CONTROLLER, _RUNTIME_MODULE_COUNT
    global _RUNTIME_COMMON_FILES, _RUNTIME_COMMON_DIGEST, _RUNTIME_COMMON_FILE_COUNT
    from ensemble_dwd._cache_identity import _package_identity
    from dwd import _compiled_residual, _native_residual
    import scipy.linalg  # Load numerical dependencies before observing thread pools.
    deep = check_inventory or _RUNTIME_PACKAGE_FILES is None
    files, packages = ({} if deep else dict(_RUNTIME_PACKAGE_FILES)), {}
    module_guards = {}
    if _RUNTIME_NATIVE is None:
        np.dot(np.ones((2, 2)), np.ones((2, 2)))
        _RUNTIME_NATIVE = _package_identity()
        _RUNTIME_ACCELERATOR = (_compiled_residual._ACCEL, bool(_native_residual._native_supported()))
    elif (_compiled_residual._ACCEL is not _RUNTIME_ACCELERATOR[0]
          or bool(_native_residual._native_supported()) != _RUNTIME_ACCELERATOR[1]):
        raise CheckpointError('Loaded accelerator availability or identity changed')
    for name, distribution in _runtime_package_names():
        module = importlib.import_module(name)
        origin = Path(module.__file__).absolute()
        root = origin.parent
        if _RUNTIME_PACKAGES is None:
            installed = _metadata_identity(distribution)
        else:
            if name not in _RUNTIME_PACKAGES:
                raise CheckpointError('An optional numerical backend activated after runtime initialization')
            installed = _RUNTIME_PACKAGES[name]['distribution_version']
        packages[name] = {'version': str(getattr(module, '__version__', installed)),
                          'distribution_version': installed, 'origin': str(origin)}
        module_guards[name] = (module, str(origin), packages[name]['version'])
        if deep:
            candidates = [origin] if not hasattr(module, '__path__') else _runtime_package_paths(root)
            for path in candidates:
                if '__pycache__' in path.parts:
                    continue
                info = _inventory_regular_stat(path)
                if info is None:
                    continue
                suffixes = ('.py', '.c', '.h', '.pyd', '.dll', '.so', '.dylib')
                if path.suffix.lower() in suffixes or '.so.' in path.name or (name == 'aocl_runtime' and path.suffix.lower() == '.json'):
                    path = path.absolute()
                    files[str(path)] = _file_hash_observed(path, info)
                    if name in ('dwd', 'ensemble_dwd', 'aocl_runtime'):
                        _RUNTIME_GUARDS.add(path)
    if _RUNTIME_PACKAGES is None:
        _RUNTIME_PACKAGES, _RUNTIME_MODULES = packages, module_guards
    elif packages != _RUNTIME_PACKAGES or module_guards != _RUNTIME_MODULES:
        raise CheckpointError('Loaded numerical package identity changed')
    if deep:
        if _RUNTIME_PACKAGE_FILES is not None and files != _RUNTIME_PACKAGE_FILES:
            raise CheckpointError('Numerical package inventory changed; restart before resuming')
        _RUNTIME_PACKAGE_FILES = dict(files)
    _loaded_source_guards()
    for path in _RUNTIME_GUARDS:
        _file_hash(path)
    # Native loads (including ctypes) need not change sys.modules. Discover
    # pools at every observation; shallow calls still reuse package inventories.
    common_files = dict(files)
    pools = _native_pool_fingerprint(files, native_threads=native_threads)
    for path in (Path(sys.executable), Path(__file__), Path(__file__).with_name('phase_threads.py'),
                 _preprocessing_source(), _selection_source(),
                 Path(inspect.getsourcefile(CandidateEvaluator))):
        name = str(path.absolute())
        files[name] = common_files[name] = _file_hash(path)
    for path in _RUNTIME_GUARDS:
        if path.name == 'METADATA':
            files[str(path)] = common_files[str(path)] = _file_hash(path)
    inventory = sorted(files)
    common_inventory = sorted(common_files)
    if _RUNTIME_FILES is None:
        canonical_files = {}
        canonical_stamps = {}
        canonical_common = {}
        for name, digest in files.items():
            canonical, stamp, verified = _runtime_artifact_identity(name)
            if verified != digest:
                raise CheckpointError('Runtime artifact inventory identity changed')
            if canonical in canonical_files and (canonical_files[canonical] != digest
                                                  or canonical_stamps[canonical] != stamp):
                raise CheckpointError('Conflicting canonical runtime inventory: ' + canonical)
            canonical_files[canonical], canonical_stamps[canonical] = digest, stamp
            if name in common_files:
                canonical_common[canonical] = digest
        _RUNTIME_FILES = inventory
        _RUNTIME_DIGEST = _digest(canonical_files)
        _RUNTIME_FILE_COUNT = len(canonical_files)
        _RUNTIME_COMMON_FILES = common_inventory
        _RUNTIME_COMMON_DIGEST = _digest(canonical_common)
        _RUNTIME_COMMON_FILE_COUNT = len(canonical_common)
    elif inventory != _RUNTIME_FILES or common_inventory != _RUNTIME_COMMON_FILES:
        raise CheckpointError('Loaded runtime inventory changed in this process; restart before resuming')
    emitted_packages = {name: {**record, 'origin': _runtime_artifact_identity(record['origin'])[0]}
                        for name, record in packages.items()}
    return {'python': sys.version, 'implementation': platform.python_implementation(),
            'machine': platform.machine(), 'packages': emitted_packages, 'package_native': _RUNTIME_NATIVE,
            'files_sha256': _RUNTIME_DIGEST, 'file_count': _RUNTIME_FILE_COUNT,
            'native_pools': sorted(pools, key=lambda row: row['filepath']),
            'runtime_policy': _RUNTIME_ROLE_POLICY,
            'common_files_sha256': _RUNTIME_COMMON_DIGEST,
            'common_file_count': _RUNTIME_COMMON_FILE_COUNT}


def _runtime_fields(value):
    """Read only the closed runtime encoding; never decode estimator identities."""
    if (type(value) is not list or len(value) != 2 or value[0] != 'dict'
            or type(value[1]) is not list):
        raise CheckpointError('Malformed execution-role runtime identity')
    fields = {}
    for row in value[1]:
        if (type(row) is not list or len(row) != 2 or type(row[0]) is not str
                or row[0] in fields):
            raise CheckpointError('Malformed or duplicate runtime identity field')
        fields[row[0]] = row[1]
    return fields


def _role_runtime(value):
    fields = _runtime_fields(value)
    required = {'python', 'implementation', 'machine', 'packages', 'package_native',
                'files_sha256', 'file_count', 'native_pools', 'runtime_policy',
                'common_files_sha256', 'common_file_count'}
    if set(fields) != required or fields['runtime_policy'] != _stable(_RUNTIME_ROLE_POLICY):
        raise CheckpointError('Unsupported execution-role runtime policy')
    for name in ('files_sha256', 'common_files_sha256'):
        encoded = fields[name]
        if (type(encoded) is not list or len(encoded) != 2 or encoded[0] != 'str'
                or type(encoded[1]) is not str
                or re.fullmatch(r'[0-9a-f]{64}', encoded[1]) is None):
            raise CheckpointError('Malformed runtime artifact digest')
    for name in ('file_count', 'common_file_count'):
        encoded = fields[name]
        if (type(encoded) is not list or len(encoded) != 2 or encoded[0] != 'int'
                or type(encoded[1]) is not int or encoded[1] < 1):
            raise CheckpointError('Malformed runtime artifact count')
    if fields['common_file_count'][1] > fields['file_count'][1]:
        raise CheckpointError('Common runtime inventory exceeds full inventory')
    encoded_pools = fields['native_pools']
    if (type(encoded_pools) is not list or len(encoded_pools) != 2
            or encoded_pools[0] != 'list' or type(encoded_pools[1]) is not list):
        raise CheckpointError('Malformed runtime native-pool list')
    blas = False
    for row in encoded_pools[1]:
        pool = _runtime_fields(row)
        if not {'filepath', 'sha256', 'num_threads', 'user_api'}.issubset(pool):
            raise CheckpointError('Incomplete runtime native-pool identity')
        for name in ('filepath', 'sha256'):
            encoded = pool[name]
            if (type(encoded) is not list or len(encoded) != 2 or encoded[0] != 'str'
                    or type(encoded[1]) is not str or not encoded[1]):
                raise CheckpointError('Malformed native-pool artifact identity')
        if (not Path(pool['filepath'][1]).is_absolute()
                or re.fullmatch(r'[0-9a-f]{64}', pool['sha256'][1]) is None):
            raise CheckpointError('Malformed native-pool artifact path or digest')
        if pool['user_api'] not in (_stable('blas'), _stable('openmp')):
            raise CheckpointError('Unsupported native-pool API identity')
        count = pool['num_threads']
        if (type(count) is not list or len(count) != 2 or count[0] != 'int'
                or type(count[1]) is not int or count[1] != 1):
            raise CheckpointError('Execution-role runtime requires native threads equal to one')
        blas = blas or pool['user_api'] == _stable('blas')
    if not blas:
        raise CheckpointError('Loaded BLAS identity is unavailable')
    return fields


def _require_execution_compatible(execution, coordinator):
    """Admit only P-only pools; actual workers still require exact full W."""
    worker, parent = _role_runtime(execution), _role_runtime(coordinator)
    common = ('python', 'implementation', 'machine', 'packages', 'package_native',
              'runtime_policy', 'common_files_sha256', 'common_file_count')
    changed = [name for name in common
               if _canonical(worker[name]) != _canonical(parent[name])]
    if changed:
        raise CheckpointError('Execution and coordinator common runtime differs: '
                              + ', '.join(changed[:4]))
    worker_pools = Counter(_canonical(row) for row in worker['native_pools'][1])
    parent_pools = Counter(_canonical(row) for row in parent['native_pools'][1])
    if worker_pools - parent_pools:
        raise CheckpointError('Execution native pools are not contained in coordinator runtime')
    worker_paths = {_runtime_fields(row)['filepath'][1] for row in worker['native_pools'][1]}
    parent_paths = {_runtime_fields(row)['filepath'][1] for row in parent['native_pools'][1]}
    worker_count, parent_count = worker['file_count'][1], parent['file_count'][1]
    if not worker_count <= parent_count <= worker_count + len(parent_paths - worker_paths):
        raise CheckpointError('Execution and coordinator artifact counts are inconsistent')
    # Exact common identity plus pool inclusion permits only additional files.
    # These are necessary aggregate joins, not reconstruction of a hidden map.
    if (worker_pools == parent_pools or worker_count == parent_count) and (
            worker['file_count'] != parent['file_count']
            or worker['files_sha256'] != parent['files_sha256']):
        raise CheckpointError('Execution and coordinator full artifact identities are inconsistent')


def _execution_runtime_probe(estimator):
    # The real task payload has already deserialized/imported its estimator.
    # No fitting, prediction, scoring or random-state draws occur here.
    _stable(estimator)
    with threadpool_limits(limits=1):
        return _stable(runtime_fingerprint())


def _execution_import_probe():
    # Resolving this module-qualified task imports the checkpoint module.
    # No estimator payload, runtime fingerprint or identity is created here.
    return None


def _probe_execution_runtime(estimator, *, jobs):
    if type(jobs) is not int or jobs <= 1:
        raise CheckpointError('A parallel execution probe requires jobs greater than one')
    with parallel_config(backend='loky', inner_max_num_threads=1), threadpool_limits(limits=1):
        result = Parallel(n_jobs=jobs, pre_dispatch=jobs, batch_size=1)(
            [delayed(_execution_runtime_probe)(estimator)]
            + [delayed(_execution_import_probe)() for _ in range(jobs - 1)])
    if len(result) != jobs or any(row is not None for row in result[1:]):
        raise CheckpointError('Invalid execution runtime probe result')
    return result[0]


def _saved_identity(path, *, schema=SCHEMA):
    """Authenticate a saved identity before extracting its execution runtime."""
    record = _read_json(path)
    if type(record) is not dict or set(record) != {'payload', 'sha256'}:
        raise CheckpointError('Malformed saved identity envelope')
    payload = record['payload']
    if (type(payload) is not dict or set(payload) != {'identity_sha256', 'identity'}
            or record['sha256'] != _digest(payload)):
        raise CheckpointError('Saved identity checksum or structure mismatch')
    identity = payload['identity']
    if (type(identity) is not dict or identity.get('schema') != schema
            or payload['identity_sha256'] != _digest(identity)):
        raise CheckpointError('Saved identity digest or schema mismatch')
    return identity, payload['identity_sha256']


def _row_diagnostics(rows, y=None):
    """Compact row identities, relative to this fitted model's training data."""
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu':
        return {'available': False}
    rows = rows.astype(np.int64, copy=False)
    result = {'size': len(rows), 'order_sha256': array_fingerprint(rows)['sha256'],
              'set_sha256': array_fingerprint(np.unique(rows))['sha256']}
    if y is not None:
        labels = np.asarray(y)
        if labels.ndim == 1 and np.all((rows >= 0) & (rows < len(labels))):
            classes, counts = np.unique(labels[rows], return_counts=True)
            result['class_counts'] = [{'label': json_safe(label), 'count': int(count)}
                                      for label, count in zip(classes, counts)]
    return result


def diagnostics(estimator, y=None):
    """Copy reported scalar/small-array facts without inventing convergence claims."""
    from ensemble_dwd import OvOClassifier
    from ensemble_dwd.estimator import _base_diagnostics

    if isinstance(estimator, CandidateEvaluator):
        estimator = getattr(estimator, 'estimator_', estimator.estimator)
    if hasattr(estimator, 'steps') and estimator.steps:
        estimator = estimator.steps[-1][1]
    result = {}
    names = ('n_iter_', 'total_iterations_', 'iterations_per_attempt_', 'converged_',
             'criterion_reached_', 'objective_tolerance_met_', 'optimality_met_', 'fit_status_',
             'stopping_reason_', 'termination_reason_', 'backend_', 'backend_used_',
             'gradient_inf_norm_', 'rkhs_gradient_norm_', 'dual_gap_',
             'equality_residual_', 'dual_equality_residual_', 'stationarity_residual_',
             'returned_iteration_', 'support_counts_', 'n_support_', 'gamma_',
             'cache_hit_', 'cache_status_', 'base_diagnostics_', 'final_objective_',
             'stationarity_checked_', 'C_conversion_finite_', 'prediction_precision_')
    for name in names:
        if not hasattr(estimator, name):
            continue
        value = json_safe(getattr(estimator, name))
        if len(_canonical(value)) <= 32768:
            result[name] = value
    solver = _base_diagnostics(estimator).get('solver_diagnostics')
    if solver:
        result['solver_diagnostics'] = solver
    final = getattr(estimator, 'final_estimator_', None)
    if hasattr(estimator, 'subset_indices_'):
        result['subset_sizes'] = [len(rows) for rows in estimator.subset_indices_]
        result['subset_rows'] = [_row_diagnostics(rows, y) for rows in estimator.subset_indices_]
    if hasattr(estimator, 'selected_indices_'):
        result['selected_size'] = len(estimator.selected_indices_)
        result['selected_rows'] = _row_diagnostics(estimator.selected_indices_, y)
    if final is not None and final is not estimator:
        result['final_estimator'] = diagnostics(final)
    if isinstance(estimator, OvOClassifier) and hasattr(estimator, 'estimators_'):
        # Only OvO's fitted children represent class pairs. Binary ensembles
        # already summarize their base fits in base_diagnostics_.
        result['pairwise_estimators'] = []
        for pair, child in zip(estimator.pairs_, estimator.estimators_, strict=True):
            pair_y = None
            if y is not None:
                labels = np.asarray(y)
                included = (labels == pair[0]) | (labels == pair[1])
                pair_y = np.where(labels[included] == pair[0], 1, -1)
            result['pairwise_estimators'].append(
                {'pair': json_safe(pair), 'diagnostics': diagnostics(child, pair_y)})
    return result


def _folds(folds, n_rows):
    result = []
    for train, validation in folds:
        pair = []
        for values in (train, validation):
            values = np.asarray(values)
            if (values.ndim != 1 or values.dtype.kind not in 'iu' or len(values) == 0
                    or np.any(values < 0) or np.any(values >= n_rows)
                    or len(np.unique(values)) != len(values)):
                raise ValueError('CV indices must be nonempty, unique integer rows inside the dataset')
            pair.append(values.astype(np.int64, copy=True))
        if np.intersect1d(*pair).size:
            raise ValueError('CV training and validation rows overlap')
        result.append(tuple(pair))
    if len(result) < 2:
        raise ValueError('At least two fixed folds are required')
    return result


def _unchanged_methods(estimator):
    cls = type(estimator)
    return (cls in _REUSE_METHODS
            and {name: value for name, value in vars(cls).items() if callable(value)} == _REUSE_METHODS[cls])


def _reuse_parameter_types(value):
    """Limit semantic reuse to the parameter types covered by its proof.

    NumPy scalars can multiply differently from Python floats. Array and custom
    parameter behavior is also outside this proof. Typed exact checkpoints do
    not by themselves establish equivalence between different ratio values.
    """
    if value is None or type(value) in (str, bool, int, float, type(Path())):
        return True
    if type(value) in _REUSE_METHODS:
        # Decline altered built-in methods before querying their parameters.
        if not _unchanged_methods(value):
            return False
        return _reuse_parameter_types(value.get_params(deep=False))
    if type(value) is dict:
        return all(type(key) is str and _reuse_parameter_types(item)
                   for key, item in value.items())
    if type(value) in (tuple, list):
        return all(_reuse_parameter_types(item) for item in value)
    return False


def _retention_identity(candidate_identity):
    """Return a normalized identity and the original built-in scalar ratio."""
    normalized = deepcopy(candidate_identity)
    node = normalized

    def parameters(item):
        if (type(item) is not list or len(item) != 3 or item[0] != 'estimator'
                or type(item[1]) is not dict or type(item[2]) is not list
                or len(item[2]) != 2 or item[2][0] != 'dict'
                or type(item[2][1]) is not list):
            raise CheckpointError('Unsupported estimator in equivalent-fold identity')
        result = {}
        for pair in item[2][1]:
            if (type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str
                    or pair[0] in result):
                raise CheckpointError('Malformed parameters in equivalent-fold identity')
            result[pair[0]] = pair[1]
        return result

    params = parameters(node)
    if node[1]['class'] == Pipeline.__module__ + '.' + Pipeline.__qualname__:
        steps = params['steps']
        if steps[0] not in ('list', 'tuple') or steps[1][-1][0] not in ('list', 'tuple'):
            raise CheckpointError('Unsupported pipeline steps in equivalent-fold identity')
        node = steps[1][-1][1][1]
        params = parameters(node)
    if node[1]['class'] == OvOClassifier.__module__ + '.' + OvOClassifier.__qualname__:
        node = params['estimator']
        params = parameters(node)
    if node[1]['class'] not in {cls.__module__ + '.' + cls.__qualname__
                               for cls in (EnsembleKernelDWD, EnsembleKernelSVM)}:
        raise CheckpointError('Unsupported estimator in equivalent-fold identity')
    encoded = params['sv_selection_ratio']
    if type(encoded) is not list or len(encoded) != 2:
        raise CheckpointError('Invalid retention ratio in equivalent-fold identity')
    if encoded[0] == 'float' and type(encoded[1]) is str:
        ratio = float.fromhex(encoded[1])
        if ratio.hex() != encoded[1]:
            raise CheckpointError('Noncanonical retention ratio in equivalent-fold identity')
    elif encoded[0] == 'int' and type(encoded[1]) is int:
        ratio = encoded[1]
    else:
        ratio = None
    if type(ratio) not in (float, int) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise CheckpointError('Invalid retention ratio in equivalent-fold identity')
    encoded[:] = _stable(0.0)
    return normalized, ratio


def _equivalence_key(proof, identity, fold):
    normalized, ratio = _retention_identity(proof['candidate_identity'])
    populations = proof['populations']
    counts = [int(item['n_train'] * ratio) for item in populations]
    if proof['policy'] != _EQUIVALENT_POLICY or counts != proof['retained_counts']:
        raise CheckpointError('Invalid equivalent-fold retained-count proof')
    return _digest({'identity': identity, 'fold': fold, 'policy': _EQUIVALENT_POLICY,
                    'estimator': normalized, 'populations': populations, 'retained_counts': counts})


def _fold_equivalence(candidate, identity, fold, y_train):
    """Conservative semantic identity; unsupported candidates still fit normally.

    Only supported built-in parameter types are eligible. Row-count
    normalization does not merge concrete parameter identities or extend
    this reuse proof to NumPy scalar parameters.
    """
    if not _reuse_parameter_types(candidate):
        return None
    model = candidate
    if type(model) is Pipeline:
        if (not _unchanged_methods(model) or len(model.steps) != 2
                or type(model.steps[0][1]) not in (StandardScaler, UnitPixelScaler)
                or not _unchanged_methods(model.steps[0][1])
                or getattr(model, 'transform_input', None) is not None):
            return None
        try:
            model._validate_params()
            model._validate_steps()
            scaler = model.steps[0][1]
            if type(scaler) is UnitPixelScaler:
                scaler._validate_options()
            else:
                scaler._validate_params()
        except (TypeError, ValueError):
            return None
        model = model.steps[-1][1]
    is_ovo = type(model) is OvOClassifier
    if is_ovo:
        if not _unchanged_methods(model):
            return None
        try:
            model._validate_options()
        except (TypeError, ValueError):
            return None
        model = model.estimator
    if type(model) not in (EnsembleKernelDWD, EnsembleKernelSVM) or not _unchanged_methods(model):
        return None
    # Apply the estimator's own validation before granting any equivalence.
    try:
        model._validate_options()
    except (TypeError, ValueError):
        # Preserve the ordinary fit's validation and failure-recording path.
        return None
    if (type(model.sv_selection_ratio) not in (float, int)
            or type(model.random_state) is not int or callable(model.kernel)):
        return None
    def contains_callable(value):
        if callable(value):
            return True
        if isinstance(value, dict):
            return any(contains_callable(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(contains_callable(item) for item in value)
        return False
    if contains_callable(getattr(model, 'kernel_kws', None)):
        return None
    labels, sizes = np.unique(y_train, return_counts=True)
    if len(labels) < 2 or (not is_ovo and len(labels) != 2):
        return None
    if is_ovo:
        populations = [{'pair': json_safe([labels[first], labels[second]]),
                        'n_train': int(sizes[first] + sizes[second])}
                       for first in range(len(labels)) for second in range(first + 1, len(labels))]
    else:
        populations = [{'pair': json_safe(labels), 'n_train': len(y_train)}]
    proof = {'policy': _EQUIVALENT_POLICY, 'candidate_identity': _stable(candidate),
             'populations': populations,
             'retained_counts': [int(item['n_train'] * model.sv_selection_ratio) for item in populations]}
    proof['key'] = _equivalence_key(proof, identity, fold)
    return proof


def _validate_equivalence(record, identity, task):
    proof = record.get('equivalent_fold')
    if proof is None:
        return None
    try:
        if (_equivalence_key(proof, identity, record['fold']) != proof['key']
                or _digest({'identity': identity, 'estimator': proof['candidate_identity'],
                            'fold': record['fold']}) != task):
            raise CheckpointError('Equivalent-fold proof does not match its task')
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as error:
        raise CheckpointError('Malformed equivalent-fold proof') from error
    return proof


def _validate_accuracy_counts(record, validation_size=None):
    counts = record.get('accuracy_counts')
    if counts is not None:
        try:
            if (record['state'] != 'complete' or float(accuracy_fraction(counts)) != record['score']
                    or (validation_size is not None and counts['n_validation'] != validation_size)):
                raise ValueError('Count/score or validation-size mismatch')
        except (TypeError, ValueError) as error:
            raise CheckpointError('Invalid fold accuracy counts') from error


def _valid_fold(path, identity, task, *, parameters=None, fold=None, _allow_alias=True,
                validation_size=None, validation_sizes=None):
    record = _read_receipt(path, identity)
    if (record.get('schema') != SCHEMA or record.get('task_sha256') != task
            or record.get('state') not in ('complete', 'ineligible')):
        raise CheckpointError('Fold receipt does not identify a completed matching fit')
    if (parameters is not None
            and _canonical(record.get('parameters')) != _canonical(_stable(parameters))):
        raise CheckpointError('Fold receipt parameter mismatch')
    if fold is not None and (isinstance(record.get('fold'), bool) or record.get('fold') != fold):
        raise CheckpointError('Fold receipt index mismatch')
    for field in ('fit_time', 'score_time'):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise CheckpointError('Invalid fold timing')
    if record['state'] == 'complete':
        score = record.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise CheckpointError('Invalid accuracy score in completed fold')
        if record.get('failure') is not None:
            raise CheckpointError('Successful fold cannot carry a fit failure')
    elif (record.get('score') is not None or record.get('failure', {}).get('category')
          not in ('ineligible_selection', 'ineligible_subset')):
        raise CheckpointError('Only a recorded expected ineligibility may omit a score')
    if validation_sizes is not None:
        index = record.get('fold')
        if type(index) is not int or not 0 <= index < len(validation_sizes):
            raise CheckpointError('Fold receipt index is outside the fixed folds')
        validation_size = validation_sizes[index]
    _validate_accuracy_counts(record, validation_size)
    if 'pair_recovery' in record:
        from pair_recovery import validate_summary
        validate_summary(record)
    proof = _validate_equivalence(record, identity, task)
    if 'equivalent_reuse' in record:
        if not _allow_alias or proof is None or record['state'] != 'complete':
            raise CheckpointError('Equivalent fold must point directly to a completed fitted source')
        _validate_alias(record, path, identity, validation_size=validation_size)
    return record


def _source_fold(folder, identity, source_task, key, *, validation_size=None):
    if not isinstance(source_task, str) or not re.fullmatch(r'[a-f0-9]{64}', source_task):
        raise CheckpointError('Invalid equivalent-fold source task')
    source = _valid_fold(Path(folder) / 'folds' / (source_task + '.json'), identity,
                         source_task, _allow_alias=False, validation_size=validation_size)
    if (source['state'] != 'complete' or source.get('equivalent_fold', {}).get('key') != key):
        raise CheckpointError('Equivalent-fold source is not a matching completed fit')
    return source


def _validate_alias(record, path, identity, *, validation_size=None):
    reuse = record['equivalent_reuse']
    try:
        source = _source_fold(Path(path).parent.parent, identity, reuse['source_task_sha256'],
                              record['equivalent_fold']['key'], validation_size=validation_size)
        elapsed = reuse['reuse_seconds']
        if (reuse['policy'] != _EQUIVALENT_POLICY
                or reuse['source_receipt_sha256'] != _digest(source)
                or reuse['source_parameters'] != source['parameters']
                or reuse['source_fit_time'] != source['fit_time']
                or reuse['source_score_time'] != source['score_time']
                or reuse['diagnostics_source_task_sha256'] != source['task_sha256']
                or record['score'] != source['score']
                or record.get('accuracy_counts') != source.get('accuracy_counts')
                or record.get('diagnostics') != source.get('diagnostics')
                or record['fit_time'] != 0.0 or record['score_time'] != 0.0
                or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed < 0):
            raise CheckpointError('Equivalent-fold alias disagrees with its source')
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError('Malformed equivalent-fold alias') from error


def _fit_fold(folder, identity, runtime, token, task, estimator, parameters, X, y,
              train, validation, fold, error_policy, failure_dir, *, equivalent_fold=None,
              pair_recovery=None):
    folder = Path(folder)
    with _Lease(folder / 'locks' / (task + '.lock')):
        owner = _read_json(folder / 'owner.json')
        if owner.get('token') != token or owner.get('state') != 'active':
            raise CheckpointBusyError('The dispatching coordinator no longer owns this run')
        context = None
        if pair_recovery is not None:
            from pair_recovery import FoldContext, cleanup_context
            context = FoldContext(folder, identity, runtime, token, task, parameters, fold,
                                  train, validation, X, y, pair_recovery)
            context.assert_owner()
        result_path = folder / 'folds' / (task + '.json')
        if result_path.exists():
            saved = _valid_fold(result_path, identity, task, parameters=parameters, fold=fold,
                                validation_size=len(validation))
            if context is not None:
                result = cleanup_context(context, already_validated=saved)
                saved = deepcopy(saved)
                saved['pair_recovery']['cleanup_status'] = result['status']
            return saved, True
        attempt = {'schema': SCHEMA, 'identity_sha256': identity, 'task_sha256': task,
                   'attempt': uuid.uuid4().hex, 'owner_token': token, 'pid': os.getpid(),
                   'fold': fold, 'parameters': _stable(parameters), 'state': 'running',
                   'started_at': _now()}
        if equivalent_fold is not None:
            attempt['equivalent_fold'] = equivalent_fold
        path = folder / 'attempts' / (attempt['attempt'] + '.json')
        _write_receipt(path, attempt)
        if context is not None:
            context.attempt = attempt
        fold_receipt_committed = False
        try:
            with threadpool_limits(limits=1):
                observed_runtime = _stable(runtime_fingerprint(check_inventory=False))
                if observed_runtime != runtime:
                    raise CheckpointError('Worker runtime differs from the frozen CV identity'
                                          + _runtime_mismatch_details(runtime, observed_runtime))
                model = clone(estimator).set_params(**parameters)
                if context is None:
                    evaluator = CandidateEvaluator(model, error_policy=error_policy, failure_dir=failure_dir)
                else:
                    from pair_recovery import FoldEvaluator
                    evaluator = FoldEvaluator(model, context, error_policy=error_policy, failure_dir=failure_dir)
                started = perf_counter()
                evaluator.fit(X[train], y[train])
                fit_time = perf_counter() - started
                started = perf_counter()
                score = eligible_accuracy(evaluator, X[validation], y[validation])
                score_time = perf_counter() - started
                if evaluator.eligible_:
                    if not math.isfinite(score):
                        raise FloatingPointError('An eligible fold returned a nonfinite score')
                    state, failure = 'complete', None
                else:
                    state, failure = 'ineligible', evaluator.failure_
                    if error_policy != 'nan' or failure['category'] not in ('ineligible_selection', 'ineligible_subset'):
                        raise CheckpointError('Unexpected ineligible result')
                # Check core package/native guards and newly loaded pools before committing.
                observed_runtime = _stable(runtime_fingerprint(check_inventory=False))
                if observed_runtime != runtime:
                    raise CheckpointError('Worker runtime changed during the fit'
                                          + _runtime_mismatch_details(runtime, observed_runtime))
                record = {**attempt, 'state': state, 'completed_at': _now(),
                          'score': float(score) if state == 'complete' else None,
                          'fit_time': fit_time, 'score_time': score_time,
                          'accuracy_counts': deepcopy(getattr(evaluator, 'accuracy_counts_', None)),
                          'failure': failure, 'diagnostics': diagnostics(evaluator, y[train])}
                _validate_accuracy_counts(record, len(validation))
                if context is not None:
                    from pair_recovery import annotate_record, validate_summary
                    annotate_record(record, context)
                    validate_summary(record)
                    context.assert_owner()
                _write_receipt(result_path, record)
                fold_receipt_committed = True
                if context is None:
                    _write_receipt(path, record)
                else:
                    # Attempt/evidence records are auxiliary once the scientific
                    # fold is committed. Optional reporting cannot discard it.
                    try:
                        _write_receipt(path, record)
                    except Exception:
                        pass
                    result = cleanup_context(context, already_validated=record)
                    record = deepcopy(record)
                    record['pair_recovery']['cleanup_status'] = result['status']
                    context.write_evidence()
                return record, False
        except BaseException as error:
            failed = {**attempt, 'state': 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                      'finished_at': _now(), 'failure': exception_details(error)}
            if context is not None:
                failed['fold_receipt_committed'] = fold_receipt_committed
                failed['pair_recovery'] = context.summary()
            write_failure_json(path, {'payload': failed, 'sha256': _digest(failed)}, replace=True)
            if context is not None:
                # Never replace the original numerical/interruption exception.
                try:
                    context.write_evidence()
                except Exception:
                    pass
            raise


def _reuse_fold(folder, identity, runtime, token, task, parameters, fold, proof, source_task,
                *, validation_size=None):
    started = perf_counter()
    folder = Path(folder)
    with _Lease(folder / 'locks' / (task + '.lock')), threadpool_limits(limits=1):
        owner = _read_json(folder / 'owner.json')
        if owner.get('token') != token or owner.get('state') != 'active':
            raise CheckpointBusyError('The dispatching coordinator no longer owns this run')
        observed_runtime = _stable(runtime_fingerprint(check_inventory=False))
        if observed_runtime != runtime:
            raise CheckpointError('Worker runtime differs from the frozen CV identity'
                                  + _runtime_mismatch_details(runtime, observed_runtime))
        path = folder / 'folds' / (task + '.json')
        if path.exists():
            return _valid_fold(path, identity, task, parameters=parameters, fold=fold,
                               validation_size=validation_size), True
        source = _source_fold(folder, identity, source_task, proof['key'], validation_size=validation_size)
        record = {'schema': SCHEMA, 'identity_sha256': identity, 'task_sha256': task,
                  'attempt': uuid.uuid4().hex, 'owner_token': token, 'pid': os.getpid(),
                  'fold': fold, 'parameters': _stable(parameters), 'state': 'complete',
                  'started_at': _now(), 'completed_at': _now(), 'failure': None,
                  'score': source['score'], 'fit_time': 0.0, 'score_time': 0.0,
                  'accuracy_counts': deepcopy(source.get('accuracy_counts')),
                  'diagnostics': deepcopy(source.get('diagnostics', {})), 'equivalent_fold': proof,
                  'equivalent_reuse': {
                      'policy': _EQUIVALENT_POLICY, 'source_task_sha256': source_task,
                      # This is the checksummed payload digest in the source envelope.
                      'source_receipt_sha256': _digest(source), 'source_parameters': source['parameters'],
                      'source_fit_time': source['fit_time'], 'source_score_time': source['score_time'],
                      'diagnostics_source_task_sha256': source_task, 'reuse_seconds': perf_counter() - started}}
        _validate_equivalence(record, identity, task)
        _validate_alias(record, path, identity, validation_size=validation_size)
        observed_runtime = _stable(runtime_fingerprint(check_inventory=False))
        if observed_runtime != runtime:
            raise CheckpointError('Worker runtime changed before equivalent-fold publication'
                                  + _runtime_mismatch_details(runtime, observed_runtime))
        # Include all verification before publication; the atomic write itself
        # remains part of the enclosing stage's wall time.
        record['equivalent_reuse']['reuse_seconds'] = perf_counter() - started
        _write_receipt(path, record)
        return record, False


def _fit_fold_group(folder, identity, runtime, token, estimator, X, y, error_policy,
                    failure_dir, members, source_task=None, *, pair_recovery=None):
    """Dispatch one representative, then commit aliases; never reuse failures."""
    completed = []
    if pair_recovery is not None and source_task is None:
        # Scheduling hint only: fit_pipeline still validates every parent/pair
        # identity under its task lease before restoring any model. Keep the
        # caller's candidate indices, so tie order and candidate labels survive.
        members = sorted(members, key=lambda member: not
                         (folder / 'pair-stores' / member[2]).exists())
    for ci, fi, task, parameters, train, validation, proof in members:
        if source_task is not None and proof is not None:
            result = _reuse_fold(folder, identity, runtime, token, task, parameters, fi, proof, source_task,
                                 validation_size=len(validation))
        else:
            result = _fit_fold(folder, identity, runtime, token, task, estimator, parameters,
                               X, y, train, validation, fi, error_policy, failure_dir, equivalent_fold=proof, pair_recovery=pair_recovery)
            if proof is not None and result[0]['state'] == 'complete':
                source_task = task
        if pair_recovery is not None and 'equivalent_reuse' in result[0]:
            from pair_recovery import cleanup_saved_fold
            result = (cleanup_saved_fold(folder, identity, runtime, token, task, parameters, fi,
                       train, validation, X, y, pair_recovery), result[1])
        completed.append((ci, fi, result))
    return completed


class _EntryImportPreparation:
    """Overlap fixed import-only tasks; never qualify or serialize an estimator."""

    def __init__(self, manifest, *, jobs, resume):
        self.manifest, self.jobs, self.resume = manifest, jobs, resume
        self.enabled = False
        self.failure = None
        self.stack = self.pool = self.pending = None
        self.pool_enter_attempted = False

    def start(self):
        if self.jobs <= 1 or self.resume:
            return
        try:
            if self.manifest.exists():
                return
        except OSError:
            # This is only a routing hint. The original authoritative lookup
            # still follows the first parent fingerprint and observes errors.
            return
        except Exception as error:
            self.enabled = True
            self.failure = (error, error.__traceback__)
            return
        self.enabled = True
        try:
            self.stack = ExitStack()
            self.stack.enter_context(parallel_config(backend='loky', inner_max_num_threads=1))
            self.stack.enter_context(threadpool_limits(limits=1))
            self.pool = Parallel(n_jobs=self.jobs, pre_dispatch=self.jobs,
                                 batch_size=1, return_as='generator')
            # A failed __enter__ can still have partially initialized a backend.
            self.pool_enter_attempted = True
            self.pool.__enter__()
            self.pending = self.pool(
                [delayed(_execution_import_probe)() for _ in range(self.jobs - 1)])
        except Exception as error:
            # Retain the failure until the original parent/manifest checks have
            # had their precedence. No preparation or real task is retried.
            self.failure = (error, error.__traceback__)
            self.close(primary=error)
        except BaseException as error:
            self.close(primary=error)
            raise

    def probe(self, estimator):
        if not self.enabled:
            return _probe_execution_runtime(estimator, jobs=self.jobs)
        if self.failure is not None:
            error, traceback = self.failure
            raise error.with_traceback(traceback)
        imported = list(self.pending)
        self.pending = None
        if len(imported) != self.jobs - 1 or any(value is not None for value in imported):
            raise CheckpointError('Invalid entry import preparation result')
        # The caller has completed its first parent fingerprint and original
        # manifest validation. This is the only estimator-bearing submission.
        self.pending = self.pool([delayed(_execution_runtime_probe)(estimator)])
        result = list(self.pending)
        self.pending = None
        if len(result) != 1:
            raise CheckpointError('Invalid execution runtime probe result')
        return result[0]

    def close(self, *, primary=None):
        if primary is not None and self.failure is not None and self.failure[0] is not primary:
            try:
                secondary = self.failure[0]
                primary.add_note('Entry preparation also failed: ' + repr(secondary))
                for note in getattr(secondary, '__notes__', ()):
                    primary.add_note(note)
            except BaseException:
                pass
        pending, pool, stack = self.pending, self.pool, self.stack
        attempted = self.pool_enter_attempted
        self.pending = self.pool = self.stack = None
        self.pool_enter_attempted = False
        errors = []
        if pending is not None:
            try:
                pending.close()
            except BaseException as error:
                errors.append(('pending import/authority generator', error))
        if pool is not None and attempted:
            try:
                pool.__exit__(None, None, None)
            except BaseException as error:
                errors.append(('managed preparation pool', error))
        if stack is not None:
            try:
                stack.close()
            except BaseException as error:
                errors.append(('preparation configuration/limits', error))
        if errors:
            target = primary if primary is not None else errors[0][1]
            for where, error in errors:
                try:
                    target.add_note('Entry preparation cleanup failed at ' + where + ': ' + repr(error))
                except BaseException:
                    # Diagnostics must not replace the primary error.
                    pass
            if primary is None:
                raise target



class CheckpointedCV:
    """Context-managed, fixed-fold, accuracy-only whole-pipeline CV.

    ``provenance`` must contain stable run settings, not timestamps or the
    subsequently returned ``identity``. ``resume=True`` permits only exact
    provenance/execution-runtime/data/fold matches. Common source/library
    identity remains exact; coordinator-only native pools are session provenance.
    New parallel runs establish one worker runtime before fitting. Resume uses
    that saved runtime and every actual worker checks it before and after work.
    Explicit random seeds are required whenever an estimator exposes a
    random_state parameter.

    Schema 2 preserves parameter types, including NumPy scalar types and
    list/tuple distinctions. Older checkpoint directories cannot be resumed;
    use a new run directory. Custom scalar/container subclasses and object,
    structured or metadata-bearing array parameters are unsupported here.

    ``reuse_equivalent_folds='auto'`` (the default) or True
    reuses successful folds for exact built-in
    ensembles with identical retained counts and all other inputs unchanged.
    Supported wrappers are the package's OvOClassifier and a two-step Pipeline
    containing exact StandardScaler or UnitPixelScaler followed by a supported classifier. Python
    float/int ratios and fixed integer seeds are required; unsupported cases
    retain ordinary fitting and validation. Explicit False disables reuse.

    ``new_folds`` counts newly committed candidate/fold receipts, including
    aliases. When enabled, ``executed_folds`` counts actual new fits and
    ``equivalent_reused_folds`` counts aliases in this result, including saved
    aliases. Aliases have zero fit/score time, separate reuse time, and explicit
    source fit/score timings and diagnostics provenance. Final refitting is
    outside this class and is unaffected.

    Explicit ``cv_seeds`` evaluates each candidate on the same ordered sampling
    seeds and folds. ``seed_parameter`` names the existing integer random_state
    parameter to vary. Seed scores are averaged within each fold, then across
    folds; any ineligible seed makes the candidate ineligible. Candidate grids
    cannot tune that seed or replace its containing estimator. The default
    ``cv_seeds=None`` keeps the ordinary single-seed behavior.

    In multi-seed results, ``seed_results_`` and its summary copy use explicitly
    labeled candidate/seed/fold axes and expose actual receipt task identities,
    timings and diagnostics. Fold counts in ``new_folds``, ``reused_folds`` and
    ``executed_folds`` count fold-seed evaluations; ``n_splits_`` still counts
    folds. Mean fit/score timings sum seed work within each fold before averaging
    across folds. ``wall_seconds`` measures the search's elapsed time, including
    aggregation, rather than summing parallel worker times.

    ``pair_recovery=True`` optionally restores completed binary pairs inside an
    unfinished multiclass fold. This requires a plain StandardScaler or UnitPixelScaler/OvO
    pipeline, one inner pair worker, exact KernGDWD or EnsembleKernelDWD, fixed
    integer seeds, named RBF and a positive numeric gamma for every candidate.
    It cannot combine multi-seed CV or ensemble caches. Equivalent-fold reuse
    preserves the representative's pair recovery and records aliases separately.
    ``pair_recovery_max_bytes`` limits committed pair payloads/manifests per
    fold; zero keeps fitting while saving no pairs. Trusted local checkpoints
    only: integrity checks do not make untrusted pickle loading safe. Pair work
    and restored historical fit times are reported separately. Pair recovery
    defaults to disabled and its size limit defaults to unlimited.
    """
    diagnostics = staticmethod(diagnostics)

    def __init__(self, run_dir, estimator, X, y, folds, *, jobs=1, error_policy='nan',
                 resume=False, provenance=None, reuse_equivalent_folds='auto',
                 cv_seeds=None, seed_parameter=None, pair_recovery=False,
                 pair_recovery_max_bytes=None):
        if isinstance(jobs, bool) or not isinstance(jobs, (int, np.integer)) or jobs < 1:
            raise ValueError('jobs must be a positive integer')
        if error_policy not in ('nan', 'raise'):
            raise ValueError('error_policy must be nan or raise')
        requested_reuse_mode = reuse_equivalent_folds
        if type(reuse_equivalent_folds) is str and reuse_equivalent_folds == 'auto':
            reuse_equivalent_folds = True
        self.requested_reuse_mode = requested_reuse_mode
        if type(reuse_equivalent_folds) is not bool:
            raise ValueError('reuse_equivalent_folds must be boolean or auto')
        if type(pair_recovery) is not bool:
            raise ValueError('pair_recovery must be boolean')
        if pair_recovery_max_bytes is not None and (
                type(pair_recovery_max_bytes) is not int or pair_recovery_max_bytes < 0):
            raise ValueError('pair_recovery_max_bytes must be a nonnegative integer or None')
        if not pair_recovery and pair_recovery_max_bytes is not None:
            raise ValueError('pair_recovery_max_bytes requires pair_recovery=True')
        self.pair_recovery, self.pair_recovery_max_bytes = pair_recovery, pair_recovery_max_bytes
        self._pair_recovery_config = None
        if pair_recovery:
            from pair_recovery import normalize_config, validate_supported
            if cv_seeds is not None:
                raise ValueError('Pair recovery cannot combine multi-seed CV')
            validate_supported(estimator, require_numeric_gamma=False)
            self._pair_recovery_config = normalize_config(True, pair_recovery_max_bytes)
        self.folder = _io_path(Path(run_dir) / 'cv-checkpoints')
        # Caller-owned arrays and RNG objects may change after entering a context.
        # Own immutable data and a cloned estimator so saved identities stay true.
        self.estimator, self.X, self.y = clone(estimator), np.array(X, copy=True), np.array(y, copy=True)
        self.cv_seeds = validate_cv_seeds(cv_seeds)
        self.seed_parameter = seed_parameter
        if self.cv_seeds is None:
            if seed_parameter is not None:
                raise ValueError('seed_parameter requires explicit cv_seeds')
        else:
            _seed_ancestors(seed_parameter)
            parameters = self.estimator.get_params(deep=True)
            if seed_parameter not in parameters:
                raise ValueError('seed_parameter must name an existing random_state parameter')
            validate_seed(parameters[seed_parameter], name='estimator sampling seed')
        if self.X.ndim != 2 or self.y.ndim != 1 or len(self.X) != len(self.y) or self.X.dtype.hasobject or self.y.dtype.hasobject:
            raise ValueError('Checkpointed examples require dense numeric X and one-dimensional labels')
        if self.pair_recovery and (self.X.dtype.kind not in 'biuf'
                or self.y.dtype.kind not in 'biuf' or not np.isfinite(self.X).all()
                or not np.isfinite(self.y).all()):
            raise ValueError('Pair recovery requires dense finite numeric data and labels')
        self.X.setflags(write=False)
        self.y.setflags(write=False)
        self.folds = _folds(folds, len(self.y))
        self.jobs, self.error_policy = int(jobs), error_policy
        self.resume, self.provenance = bool(resume), {} if provenance is None else provenance
        self.reuse_equivalent_folds = reuse_equivalent_folds
        self.lease, self.token, self.identity = None, None, None

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        preparation = None
        self.lease = _Lease(self.folder / 'session.lock')
        try:
            self.lease.__enter__()
            for name in ('folds', 'attempts', 'locks', 'artifacts', 'sessions'):
                (self.folder / name).mkdir(exist_ok=True)
            preparation = _EntryImportPreparation(self.folder / 'identity.json', jobs=self.jobs, resume=self.resume)
            preparation.start()
            with threadpool_limits(limits=1):
                self.runtime = _stable(runtime_fingerprint())
            manifest = self.folder / 'identity.json'
            saved_identity = None
            if manifest.exists():
                if not self.resume:
                    raise CheckpointError('Checkpoint already exists; explicitly request resume')
                saved_identity, _ = _saved_identity(manifest)
                self.execution_runtime = saved_identity.get('runtime')
            else:
                if self.resume:
                    raise CheckpointError('No checkpoint identity exists for this run')
                self.execution_runtime = (self.runtime if self.jobs == 1 else
                    preparation.probe(self.estimator))
            _require_execution_compatible(self.execution_runtime, self.runtime)
            if self.jobs == 1 and _canonical(self.execution_runtime) != _canonical(self.runtime):
                raise CheckpointError('Serial execution runtime differs from the frozen CV identity')
            # Probe serialization/imports must not silently redefine this session.
            if saved_identity is None and self.jobs > 1:
                with threadpool_limits(limits=1):
                    if _canonical(_stable(runtime_fingerprint())) != _canonical(self.runtime):
                        raise CheckpointError('Coordinator runtime changed while establishing execution identity')
            preparation.close()
            preparation = None
            self.identity = {'schema': SCHEMA, 'data': data_fingerprint(self.X, self.y),
                             'folds': [{'train': array_fingerprint(t), 'validation': array_fingerprint(v)} for t, v in self.folds],
                             'estimator': _stable(self.estimator), 'error_policy': self.error_policy,
                             'scoring': 'accuracy', 'accuracy_aggregation': ACCURACY_AGGREGATION,
                             'native_threads': 1, 'runtime': self.execution_runtime,
                             'provenance': _stable(self.provenance)}
            if self.pair_recovery:
                self.identity['pair_recovery'] = deepcopy(self._pair_recovery_config)
            if self.reuse_equivalent_folds:
                self.identity['equivalent_fold_policy'] = _EQUIVALENT_POLICY
                self.identity['equivalent_fold_mode'] = self.requested_reuse_mode
            if self.cv_seeds is not None:
                self.identity['seed_policy'] = self._seed_policy()
            self.identity_sha256 = _digest(self.identity)
            if saved_identity is not None:
                saved = _read_receipt(manifest, self.identity_sha256)
                if _canonical(saved.get('identity')) != _canonical(self.identity):
                    raise CheckpointError('Run identity changed; use a new run directory')
            else:
                _write_receipt(manifest, {'identity_sha256': self.identity_sha256, 'identity': self.identity})
            # Invalidate unstarted work from a dead parent before checking live workers.
            self.token = uuid.uuid4().hex
            self._runtime_session = {
                'identity_sha256': self.identity_sha256, 'owner_token': self.token,
                'policy': _RUNTIME_ROLE_POLICY, 'runtime': self.runtime,
                'execution_runtime_sha256': _digest(self.execution_runtime),
                'pid': os.getpid(), 'started_at': _now()}
            self._runtime_session_path = self.folder / 'sessions' / (self.token + '.json')
            if self._runtime_session_path.exists():
                raise CheckpointError('Coordinator session identifier already exists')
            _write_receipt(self._runtime_session_path, self._runtime_session)
            write_json_atomic(self.folder / 'owner.json', {'token': self.token, 'state': 'active',
                              'pid': os.getpid(), 'started_at': _now()}, replace=True)
            for path in (self.folder / 'locks').glob('*.lock'):
                with _Lease(path):
                    pass
            return self
        except BaseException as error:
            if preparation is not None:
                preparation.close(primary=error)
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *args):
        try:
            if self.token is not None:
                write_failure_json(self.folder / 'owner.json', {'token': self.token, 'state': 'inactive',
                                   'pid': os.getpid(), 'finished_at': _now()}, replace=True)
        finally:
            if self.lease is not None:
                self.lease.__exit__(*args)
                self.lease = None

    def _active(self):
        if self.lease is None or self.identity is None or self.token is None:
            raise CheckpointError('Use CheckpointedCV inside its context manager')

    def write_artifact(self, name, payload):
        self._active()
        self._check_runtime()
        path = self._artifact_path(name)
        _write_receipt(path, {'identity_sha256': self.identity_sha256, 'artifact': name,
                             'value': json_safe(payload)})
        return path

    def read_artifact(self, name):
        self._active()
        self._check_runtime()
        path = self._artifact_path(name)
        if not path.exists():
            return None
        payload = _read_receipt(path, self.identity_sha256)
        if payload.get('artifact') != name:
            raise CheckpointError('Artifact name mismatch')
        return payload['value']

    def _artifact_path(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('Artifact names must be simple filenames')
        return self.folder / 'artifacts' / (name + '.json')

    def _check_runtime(self):
        if (_canonical(self.execution_runtime) != _canonical(self.identity.get('runtime'))
                or _digest(self.execution_runtime) != self._runtime_session['execution_runtime_sha256']
                or self.token != self._runtime_session['owner_token']
                or _canonical(self.runtime) != _canonical(self._runtime_session['runtime'])
                or _canonical(_read_receipt(self._runtime_session_path, self.identity_sha256))
                   != _canonical(self._runtime_session)):
            raise CheckpointError('Coordinator session or execution runtime identity changed')
        frozen = self.identity.get('pair_recovery')
        if (self.pair_recovery is not (frozen is not None)
                or (frozen is not None and self.pair_recovery_max_bytes != frozen['max_bytes'])
                or self._pair_recovery_config != frozen):
            raise CheckpointError('Pair-recovery configuration changed during this run')
        if frozen is not None:
            from pair_recovery import check_source_hashes
            check_source_hashes(frozen)
        with threadpool_limits(limits=1):
            observed_runtime = _stable(runtime_fingerprint())
            if observed_runtime != self.runtime:
                raise CheckpointError('Coordinator runtime changed during this run'
                                      + _runtime_mismatch_details(self.runtime, observed_runtime))

    def check_runtime(self):
        """Verify full frozen provenance, including immediately after final refit."""
        self._active()
        self._check_runtime()

    def search(self, candidate_grid, *, stage='cv'):
        call_started = perf_counter()
        if self.cv_seeds is not None or (self.identity is not None and 'seed_policy' in self.identity):
            self._active()
            if self.identity.get('seed_policy') != self._seed_policy():
                raise CheckpointError('Sampling-seed policy changed during this run')
        if self.cv_seeds is None:
            search = self._search_single_seed(candidate_grid, stage=stage)
            if search.work_summary_ is not None:
                search.work_summary_['search_call_wall_seconds'] = perf_counter() - call_started
                search.work_summary_['search_call_count'] = 1
            return search
        started = perf_counter()
        candidates = list(ParameterGrid(candidate_grid))
        if not candidates:
            raise ValueError('At least one candidate is required')
        expanded = []
        ancestors = _seed_ancestors(self.seed_parameter)
        for parameters in candidates:
            validate_seed_parameters(parameters, self.seed_parameter)
            candidate = clone(self.estimator)
            original = candidate.get_params(deep=True)
            candidate.set_params(**parameters)
            changed = candidate.get_params(deep=True)
            if (any(changed.get(path) is not original[path] for path in ancestors)
                    or self.seed_parameter not in changed
                    or changed[self.seed_parameter] != original[self.seed_parameter]):
                raise ValueError('Multi-seed CV cannot replace the sampling-seed path')
            for seed in self.cv_seeds:
                expanded.append({name: [value] for name, value in
                                 {**parameters, self.seed_parameter: seed}.items()})
        raw = self._search_single_seed(expanded, stage=stage, _include_records=True)
        search = self._aggregate_seeds(candidates, raw, started)
        if search.work_summary_ is not None:
            search.work_summary_['search_call_wall_seconds'] = perf_counter() - call_started
            search.work_summary_['search_call_count'] = 1
        return search

    def _seed_policy(self):
        if self.cv_seeds is None:
            return None
        return {'seeds': list(self.cv_seeds), 'seed_parameter': self.seed_parameter,
                'aggregation_policy': _SEED_AGGREGATION_POLICY}

    def _aggregate_seeds(self, candidates, raw, started):
        n_candidates, n_seeds, n_folds = len(candidates), len(self.cv_seeds), len(self.folds)
        records = raw._fold_records_

        def rows(function):
            return [[[function(*records[ci * n_seeds + si, fi]) for fi in range(n_folds)]
                     for si in range(n_seeds)] for ci in range(n_candidates)]

        details = {'axes': ['candidate', 'seed', 'fold'], 'candidate_params': candidates,
                   **self._seed_policy(),
                   'scores': np.asarray(rows(lambda row, reused:
                       np.nan if row['state'] == 'ineligible' else row['score'])),
                   'fit_times': np.asarray(rows(lambda row, reused: row['fit_time'])),
                   'score_times': np.asarray(rows(lambda row, reused: row['score_time'])),
                   'states': rows(lambda row, reused: row['state']),
                   'diagnostics': rows(lambda row, reused: row.get('diagnostics', {})),
                   'accuracy_counts': rows(lambda row, reused: row.get('accuracy_counts')),
                   'receipt_task_sha256': rows(lambda row, reused: row['task_sha256']),
                   'receipt_parameters': rows(lambda row, reused: row['parameters']),
                   'reused': rows(lambda row, reused: bool(reused)),
                   'equivalent_reuse': rows(lambda row, reused: row.get('equivalent_reuse'))}
        results = {'params': [{'estimator__' + key: value for key, value in params.items()}
                              for params in candidates]}
        # Every requested seed must succeed in every fold. No nanmean or
        # selection of a favorable seed is used to make a candidate eligible.
        scores = np.mean(details['scores'], axis=1)
        for fi in range(n_folds):
            results[f'split{fi}_test_score'] = scores[:, fi]
        results['mean_test_score'], results['std_test_score'] = np.mean(scores, axis=1), np.std(scores, axis=1)
        apply_accuracy_evidence(results, [accuracy_evidence(counts) for counts in details['accuracy_counts']])
        # Sum fit work across seeds within a fold. These receipt timings are
        # not the elapsed time of parallel dispatch; wall_seconds reports that.
        for field in ('fit_time', 'score_time'):
            times = np.sum(details[field + 's'], axis=1)
            results['mean_' + field], results['std_' + field] = np.mean(times, axis=1), np.std(times, axis=1)
        means = results['mean_test_score']
        valid = np.isfinite(means)
        results['rank_test_score'] = score_ranks(results)
        for name in sorted(set().union(*(params.keys() for params in candidates))):
            results['param_estimator__' + name] = np.ma.array(
                [params.get(name) for params in candidates], mask=[name not in params for params in candidates], dtype=object)
        counts = []
        for si, seed in enumerate(self.cv_seeds):
            selected = [records[ci * n_seeds + si, fi] for ci in range(n_candidates) for fi in range(n_folds)]
            counts.append({'seed': seed, 'evaluation_count': len(selected),
                           'reused_folds': sum(bool(reused) for _, reused in selected),
                           'new_folds': sum(not reused for _, reused in selected),
                           'executed_folds': sum(not reused and 'equivalent_reuse' not in row for row, reused in selected),
                           'equivalent_reused_folds': sum('equivalent_reuse' in row for row, _ in selected)})
        summary = dict(raw.checkpoint_summary_)
        summary.update(candidate_count=n_candidates, seed_count=n_seeds,
                       evaluation_count=n_candidates * n_seeds * n_folds,
                       seed_policy=self._seed_policy(), seed_counts=counts,
                       axes=details['axes'], diagnostics=details['diagnostics'],
                       seed_results=details, wall_seconds=perf_counter() - started,
                       timing_policy='sum-seed-times-within-fold-v1',
                       executed_folds=sum(item['executed_folds'] for item in counts))
        if self.reuse_equivalent_folds:
            summary['equivalent_reuse'] = details['equivalent_reuse']
        search = SimpleNamespace(cv_results_=results, n_splits_=n_folds,
                                 checkpoint_summary_=summary, seed_results_=details,
                                 work_summary_=raw.work_summary_)
        if np.any(valid):
            index = int(max(np.flatnonzero(valid), key=lambda i: score_value(results, i)))
            search.best_index_, search.best_params_, search.best_score_ = index, results['params'][index], float(means[index])
        return search

    def _search_single_seed(self, candidate_grid, *, stage='cv', _include_records=False):
        self._active()
        self._check_runtime()
        candidates = list(ParameterGrid(candidate_grid))
        if not candidates:
            raise ValueError('At least one candidate is required')
        sources = {}
        if self.reuse_equivalent_folds:
            # Receipts are the index: a representative committed before a parent
            # crash is immediately discoverable, with no separate index to repair.
            for path in sorted((self.folder / 'folds').glob('*.json')):
                saved = _valid_fold(path, self.identity_sha256, path.stem,
                                    validation_sizes=[len(v) for _, v in self.folds])
                if (saved['state'] == 'complete' and saved.get('equivalent_fold') is not None
                        and 'equivalent_reuse' not in saved):
                    sources.setdefault(saved['equivalent_fold']['key'], saved['task_sha256'])
        work, records, seen = [], {}, set()
        for candidate_index, parameters in enumerate(candidates):
            # Validate changed estimator parameters, including any newly unseeded setting.
            candidate = clone(self.estimator).set_params(**parameters)
            candidate_identity = _stable(candidate)
            if self.pair_recovery:
                from pair_recovery import validate_supported
                validate_supported(candidate)
            for fold, (train, validation) in enumerate(self.folds):
                proof = (_fold_equivalence(candidate, self.identity_sha256, fold, self.y[train])
                         if self.reuse_equivalent_folds else None)
                task = _digest({'identity': self.identity_sha256, 'estimator': candidate_identity, 'fold': fold})
                if task in seen:
                    raise ValueError('Duplicate candidate/fold in one checkpointed search')
                seen.add(task)
                result_path = self.folder / 'folds' / (task + '.json')
                if result_path.exists():
                    saved = _valid_fold(result_path, self.identity_sha256, task, parameters=parameters, fold=fold,
                                        validation_size=len(validation))
                    if self.reuse_equivalent_folds and saved.get('equivalent_fold') != proof:
                        raise CheckpointError('Saved equivalent-fold eligibility or identity changed')
                    if self.pair_recovery:
                        from pair_recovery import cleanup_saved_fold
                        saved = cleanup_saved_fold(self.folder, self.identity_sha256, self.execution_runtime,
                            self.token, task, parameters, fold, train, validation, self.X, self.y,
                            self._pair_recovery_config)
                    records[candidate_index, fold] = (saved, True)
                else:
                    work.append((candidate_index, fold, task, parameters, train, validation, proof))
        started = perf_counter()
        # Workers commit each result before returning; bounded dispatch never queues an unbounded grid.
        with parallel_config(backend='loky', inner_max_num_threads=1), threadpool_limits(limits=1):
            if self.reuse_equivalent_folds:
                groups = {}
                for member in work:
                    proof = member[-1]
                    key = proof['key'] if proof is not None else member[2]
                    groups.setdefault(key, []).append(member)
                arguments = (self.folder, self.identity_sha256, self.execution_runtime, self.token,
                             self.estimator, self.X, self.y, self.error_policy,
                             str(self.folder.parent / 'failures'))
                if self.jobs == 1:
                    for key, members in groups.items():
                        for ci, fi, result in _fit_fold_group(*arguments, members, sources.get(key), pair_recovery=self._pair_recovery_config):
                            records[ci, fi] = result
                else:
                    completed = Parallel(n_jobs=self.jobs, pre_dispatch=self.jobs, batch_size=1)(
                        delayed(_fit_fold_group)(*arguments, members, sources.get(key), pair_recovery=self._pair_recovery_config)
                        for key, members in groups.items())
                    for group in completed:
                        for ci, fi, result in group:
                            records[ci, fi] = result
            elif self.jobs == 1:
                for ci, fi, task, params, train, validation, _ in work:
                    records[ci, fi] = _fit_fold(self.folder, self.identity_sha256, self.execution_runtime, self.token,
                        task, self.estimator, params, self.X, self.y, train, validation, fi,
                        self.error_policy, str(self.folder.parent / 'failures'),
                        pair_recovery=self._pair_recovery_config)
            else:
                completed = Parallel(n_jobs=self.jobs, pre_dispatch=self.jobs, batch_size=1)(
                    delayed(_fit_fold)(self.folder, self.identity_sha256, self.execution_runtime, self.token,
                        task, self.estimator, params, self.X, self.y, train, validation, fi,
                        self.error_policy, str(self.folder.parent / 'failures'),
                        pair_recovery=self._pair_recovery_config)
                    for ci, fi, task, params, train, validation, _ in work)
                for (ci, fi, *_), result in zip(work, completed):
                    records[ci, fi] = result
        self._check_runtime()
        n_candidates, n_folds = len(candidates), len(self.folds)
        results = {'params': [{'estimator__' + key: value for key, value in params.items()} for params in candidates]}
        scores = np.empty((n_candidates, n_folds))
        for ci in range(n_candidates):
            for fi in range(n_folds):
                row = records[ci, fi][0]
                scores[ci, fi] = np.nan if row['state'] == 'ineligible' else row['score']
        for fi in range(n_folds):
            results[f'split{fi}_test_score'] = scores[:, fi]
        results['mean_test_score'], results['std_test_score'] = np.mean(scores, axis=1), np.std(scores, axis=1)
        apply_accuracy_evidence(results, [accuracy_evidence([
            [records[ci, fi][0].get('accuracy_counts') for fi in range(n_folds)]])
            for ci in range(n_candidates)])
        for field in ('fit_time', 'score_time'):
            values = np.asarray([[records[ci, fi][0][field] for fi in range(n_folds)] for ci in range(n_candidates)])
            results['mean_' + field], results['std_' + field] = np.mean(values, axis=1), np.std(values, axis=1)
        valid = np.isfinite(results['mean_test_score'])
        means = results['mean_test_score']
        results['rank_test_score'] = score_ranks(results)
        for name in sorted(set().union(*(params.keys() for params in candidates))):
            results['param_estimator__' + name] = np.ma.array([params.get(name) for params in candidates],
                                                            mask=[name not in params for params in candidates], dtype=object)
        summary = {'stage': str(stage), 'candidate_count': n_candidates, 'fold_count': n_folds,
                   'reused_folds': sum(reused for _, reused in records.values()),
                   'new_folds': sum(not reused for _, reused in records.values()),
                   'wall_seconds': perf_counter() - started,
                   'diagnostics': [[records[ci, fi][0].get('diagnostics', {}) for fi in range(n_folds)] for ci in range(n_candidates)]}
        if self.reuse_equivalent_folds:
            summary.update(
                equivalent_fold_policy=_EQUIVALENT_POLICY,
                equivalent_reused_folds=sum('equivalent_reuse' in row for row, _ in records.values()),
                executed_folds=sum(not reused and 'equivalent_reuse' not in row for row, reused in records.values()),
                equivalent_reuse=[[records[ci, fi][0].get('equivalent_reuse') for fi in range(n_folds)]
                                  for ci in range(n_candidates)])
        if self.pair_recovery:
            summary['pair_recovery'] = [[records[ci, fi][0].get('pair_recovery') for fi in range(n_folds)]
                                        for ci in range(n_candidates)]
        try:
            work_summary = summarize_records(records)
        except Exception as error:
            # Auxiliary reporting must not discard successful numerical work.
            work_summary = None
            summary['work_summary_error'] = type(error).__name__
            try:
                warnings.warn('Work summary unavailable: ' + type(error).__name__, RuntimeWarning)
            except Exception:
                pass
        summary['work_summary'] = work_summary
        search = SimpleNamespace(cv_results_=results, n_splits_=n_folds,
                                 checkpoint_summary_=summary, work_summary_=work_summary)
        if _include_records:
            search._fold_records_ = records
        if np.any(valid):
            index = int(max(np.flatnonzero(valid), key=lambda i: score_value(results, i)))
            search.best_index_, search.best_params_, search.best_score_ = index, results['params'][index], float(means[index])
        return search
