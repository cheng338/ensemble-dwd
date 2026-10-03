"""Durable fixed-fold CV for the example workflows.

Completed folds are strict, checksummed JSON; no estimator pickle is loaded.
An OS lock excludes a live coordinator. Per-fold worker locks and a generation
token prevent surviving workers from racing a replacement coordinator. A resume
refuses while an old worker is still fitting; it never terminates processes.

Runtime fingerprints describe disk artifacts and observed loaded native pools.
They require an installation that is immutable during the process lifetime;
they are not an attestation of machine code already mapped before import.
"""
from datetime import datetime, timezone
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
import sys
import threading
from time import perf_counter
from types import SimpleNamespace
import uuid

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid
from threadpoolctl import ThreadpoolController, threadpool_limits

from search_support import (CandidateEvaluator, array_fingerprint, data_fingerprint,
                            eligible_accuracy, exception_details, json_safe,
                            write_failure_json, write_json_atomic)

SCHEMA = 1
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


class CheckpointError(RuntimeError):
    """A saved identity or receipt cannot safely be reused."""


class CheckpointBusyError(CheckpointError):
    """Another coordinator or surviving fold worker still owns this run."""


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
    # resolve() walks path components on Windows; do that only once per name.
    raw = os.fspath(path)
    if raw not in _RESOLVED_PATHS:
        _RESOLVED_PATHS[raw] = Path(path).absolute()
    path = _RESOLVED_PATHS[raw]
    info = path.stat()
    stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
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
        if stamp != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise CheckpointError('Runtime artifact changed while fingerprinting: ' + str(path))
        result = digest.hexdigest()
        _HASHES[path] = (stamp, result)
        return result


def _class_identity(value):
    cls = type(value)
    source = inspect.getsourcefile(cls)
    if not source:
        raise CheckpointError('Estimator class source is unavailable: ' + cls.__qualname__)
    return {'class': cls.__module__ + '.' + cls.__qualname__, 'source_sha256': _file_hash(source)}


def _stable(value):
    """No opaque repr/type-only fallbacks are safe for reusable fit identity."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise CheckpointError('Nonfinite parameter identity')
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise CheckpointError('Object arrays do not have a supported parameter identity')
        return {'array': array_fingerprint(value)}
    if isinstance(value, np.random.RandomState):
        return {'RandomState': _stable(value.get_state())}
    if isinstance(value, np.random.Generator):
        return {'Generator': _stable(value.bit_generator.state)}
    if isinstance(value, BaseEstimator):
        parameters = value.get_params(deep=False)
        if parameters.get('random_state', 0) is None:
            raise CheckpointError('Checkpointed estimators require an explicit random_state; None cannot safely resume stochastic work')
        return {**_class_identity(value), 'parameters': _stable(parameters)}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CheckpointError('Parameter dictionaries require string keys')
        return {key: _stable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
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


def _metadata_identity(distribution):
    try:
        installed = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    for member in installed.files or ():
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


def runtime_fingerprint(*, check_inventory=True):
    """Hash once, check complete inventory per stage and core/native guards per fit.

    Workers avoid traversing large numerical package trees on every fast fit.
    Their first call still identifies the full runtime. The coordinator checks
    the full inventory before and after each search, while every worker checks
    the DWD/ensemble cache identity, loaded DLLs and checkpoint source each fit.
    """
    global _RUNTIME_FILES, _RUNTIME_PACKAGE_FILES, _RUNTIME_PACKAGES, _RUNTIME_MODULES
    global _RUNTIME_DIGEST, _RUNTIME_NATIVE, _RUNTIME_ACCELERATOR
    global _RUNTIME_CONTROLLER, _RUNTIME_MODULE_COUNT
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
            candidates = [origin] if not hasattr(module, '__path__') else root.rglob('*')
            for path in candidates:
                if not path.is_file() or '__pycache__' in path.parts:
                    continue
                suffixes = ('.py', '.c', '.h', '.pyd', '.dll', '.so', '.dylib')
                if path.suffix.lower() in suffixes or '.so.' in path.name or (name == 'aocl_runtime' and path.suffix.lower() == '.json'):
                    path = path.absolute()
                    files[str(path)] = _file_hash(path)
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
    # Enumerate newly loaded libraries only at a stage boundary or after a
    # Python extension/module import; the controller queries live thread counts.
    if deep or _RUNTIME_CONTROLLER is None or _RUNTIME_MODULE_COUNT != len(sys.modules):
        _RUNTIME_CONTROLLER = ThreadpoolController()
        _RUNTIME_MODULE_COUNT = len(sys.modules)
    pools = []
    for pool in _RUNTIME_CONTROLLER.info():
        record = {key: value for key, value in pool.items() if key != 'filepath'}
        path = Path(pool['filepath']).absolute()
        files[str(path)] = _file_hash(path)
        record.update(filepath=str(path), sha256=files[str(path)])
        if pool.get('num_threads') != 1:
            raise CheckpointError('Checkpointed CV requires one active native thread per worker')
        pools.append(record)
    if not any(pool.get('user_api') == 'blas' for pool in pools):
        raise CheckpointError('Loaded BLAS identity is unavailable')
    for path in (Path(sys.executable), Path(__file__), Path(inspect.getsourcefile(CandidateEvaluator))):
        files[str(path.absolute())] = _file_hash(path)
    for path in _RUNTIME_GUARDS:
        if path.name == 'METADATA':
            files[str(path)] = _file_hash(path)
    inventory = sorted(files)
    if _RUNTIME_FILES is None:
        _RUNTIME_FILES = inventory
        _RUNTIME_DIGEST = _digest(files)
    elif inventory != _RUNTIME_FILES:
        raise CheckpointError('Loaded runtime inventory changed in this process; restart before resuming')
    return {'python': sys.version, 'implementation': platform.python_implementation(),
            'machine': platform.machine(), 'packages': packages, 'package_native': _RUNTIME_NATIVE,
            'files_sha256': _RUNTIME_DIGEST, 'file_count': len(files),
            'native_pools': sorted(pools, key=lambda row: row['filepath'])}


def diagnostics(estimator):
    """Copy reported scalar/small-array facts without inventing convergence claims."""
    if isinstance(estimator, CandidateEvaluator):
        estimator = getattr(estimator, 'estimator_', estimator.estimator)
    if hasattr(estimator, 'steps') and estimator.steps:
        estimator = estimator.steps[-1][1]
    result = {}
    names = ('n_iter_', 'total_iterations_', 'iterations_per_attempt_', 'converged_',
             'criterion_reached_', 'objective_tolerance_met_', 'fit_status_',
             'stopping_reason_', 'termination_reason_', 'backend_', 'backend_used_',
             'gradient_inf_norm_', 'rkhs_gradient_norm_', 'dual_gap_',
             'equality_residual_', 'dual_equality_residual_', 'stationarity_residual_',
             'returned_iteration_', 'support_counts_', 'n_support_', 'gamma_',
             'cache_hit_', 'base_diagnostics_')
    for name in names:
        if not hasattr(estimator, name):
            continue
        value = json_safe(getattr(estimator, name))
        if len(_canonical(value)) <= 32768:
            result[name] = value
    final = getattr(estimator, 'final_estimator_', None)
    if hasattr(estimator, 'subset_indices_'):
        result['subset_sizes'] = [len(rows) for rows in estimator.subset_indices_]
    if hasattr(estimator, 'selected_indices_'):
        result['selected_size'] = len(estimator.selected_indices_)
    if final is not None and final is not estimator:
        result['final_estimator'] = diagnostics(final)
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


def _valid_fold(path, identity, task, *, parameters=None, fold=None):
    record = _read_receipt(path, identity)
    if (record.get('schema') != SCHEMA or record.get('task_sha256') != task
            or record.get('state') not in ('complete', 'ineligible')):
        raise CheckpointError('Fold receipt does not identify a completed matching fit')
    if parameters is not None and record.get('parameters') != _stable(parameters):
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
    return record


def _fit_fold(folder, identity, runtime, token, task, estimator, parameters, X, y,
              train, validation, fold, error_policy, failure_dir):
    folder = Path(folder)
    with _Lease(folder / 'locks' / (task + '.lock')):
        owner = _read_json(folder / 'owner.json')
        if owner.get('token') != token or owner.get('state') != 'active':
            raise CheckpointBusyError('The dispatching coordinator no longer owns this run')
        result_path = folder / 'folds' / (task + '.json')
        if result_path.exists():
            return _valid_fold(result_path, identity, task, parameters=parameters, fold=fold), True
        attempt = {'schema': SCHEMA, 'identity_sha256': identity, 'task_sha256': task,
                   'attempt': uuid.uuid4().hex, 'owner_token': token, 'pid': os.getpid(),
                   'fold': fold, 'parameters': _stable(parameters), 'state': 'running',
                   'started_at': _now()}
        path = folder / 'attempts' / (attempt['attempt'] + '.json')
        _write_receipt(path, attempt)
        try:
            with threadpool_limits(limits=1):
                if runtime_fingerprint(check_inventory=False) != runtime:
                    raise CheckpointError('Worker runtime differs from the frozen CV identity')
                model = clone(estimator).set_params(**parameters)
                evaluator = CandidateEvaluator(model, error_policy=error_policy, failure_dir=failure_dir)
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
                if runtime_fingerprint(check_inventory=False) != runtime:
                    raise CheckpointError('Worker runtime changed during the fit')
                record = {**attempt, 'state': state, 'completed_at': _now(),
                          'score': float(score) if state == 'complete' else None,
                          'fit_time': fit_time, 'score_time': score_time,
                          'failure': failure, 'diagnostics': diagnostics(evaluator)}
                _write_receipt(result_path, record)
                _write_receipt(path, record)
                return record, False
        except BaseException as error:
            failed = {**attempt, 'state': 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                      'finished_at': _now(), 'failure': exception_details(error)}
            write_failure_json(path, {'payload': failed, 'sha256': _digest(failed)}, replace=True)
            raise


class CheckpointedCV:
    """Context-managed, fixed-fold, accuracy-only whole-pipeline CV.

    ``provenance`` must contain stable run settings, not timestamps or the
    subsequently returned ``identity``. ``resume=True`` permits only exact
    provenance/runtime/data/fold matches. Explicit random seeds are required
    whenever an estimator exposes a random_state parameter.
    """
    diagnostics = staticmethod(diagnostics)

    def __init__(self, run_dir, estimator, X, y, folds, *, jobs=1, error_policy='nan',
                 resume=False, provenance=None):
        if isinstance(jobs, bool) or not isinstance(jobs, (int, np.integer)) or jobs < 1:
            raise ValueError('jobs must be a positive integer')
        if error_policy not in ('nan', 'raise'):
            raise ValueError('error_policy must be nan or raise')
        self.folder = _io_path(Path(run_dir) / 'cv-checkpoints')
        # Caller-owned arrays and RNG objects may change after entering a context.
        # Own immutable data and a cloned estimator so saved identities stay true.
        self.estimator, self.X, self.y = clone(estimator), np.array(X, copy=True), np.array(y, copy=True)
        if self.X.ndim != 2 or self.y.ndim != 1 or len(self.X) != len(self.y) or self.X.dtype.hasobject or self.y.dtype.hasobject:
            raise ValueError('Checkpointed examples require dense numeric X and one-dimensional labels')
        self.X.setflags(write=False)
        self.y.setflags(write=False)
        self.folds = _folds(folds, len(self.y))
        self.jobs, self.error_policy = int(jobs), error_policy
        self.resume, self.provenance = bool(resume), {} if provenance is None else provenance
        self.lease, self.token, self.identity = None, None, None

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lease = _Lease(self.folder / 'session.lock')
        try:
            self.lease.__enter__()
            for name in ('folds', 'attempts', 'locks', 'artifacts'):
                (self.folder / name).mkdir(exist_ok=True)
            with threadpool_limits(limits=1):
                self.runtime = runtime_fingerprint()
            self.identity = {'schema': SCHEMA, 'data': data_fingerprint(self.X, self.y),
                             'folds': [{'train': array_fingerprint(t), 'validation': array_fingerprint(v)} for t, v in self.folds],
                             'estimator': _stable(self.estimator), 'error_policy': self.error_policy,
                             'scoring': 'accuracy', 'native_threads': 1, 'runtime': self.runtime,
                             'provenance': _stable(self.provenance)}
            self.identity_sha256 = _digest(self.identity)
            manifest = self.folder / 'identity.json'
            if manifest.exists():
                if not self.resume:
                    raise CheckpointError('Checkpoint already exists; explicitly request resume')
                saved = _read_receipt(manifest, self.identity_sha256)
                if saved.get('identity') != self.identity:
                    raise CheckpointError('Run identity changed; use a new run directory')
            else:
                if self.resume:
                    raise CheckpointError('No checkpoint identity exists for this run')
                _write_receipt(manifest, {'identity_sha256': self.identity_sha256, 'identity': self.identity})
            # Invalidate unstarted work from a dead parent before checking live workers.
            self.token = uuid.uuid4().hex
            write_json_atomic(self.folder / 'owner.json', {'token': self.token, 'state': 'active',
                              'pid': os.getpid(), 'started_at': _now()}, replace=True)
            for path in (self.folder / 'locks').glob('*.lock'):
                with _Lease(path):
                    pass
            return self
        except BaseException:
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
        with threadpool_limits(limits=1):
            if runtime_fingerprint() != self.runtime:
                raise CheckpointError('Coordinator runtime changed during this run')

    def check_runtime(self):
        """Verify full frozen provenance, including immediately after final refit."""
        self._active()
        self._check_runtime()

    def search(self, candidate_grid, *, stage='cv'):
        self._active()
        self._check_runtime()
        candidates = list(ParameterGrid(candidate_grid))
        if not candidates:
            raise ValueError('At least one candidate is required')
        work, records, seen = [], {}, set()
        for candidate_index, parameters in enumerate(candidates):
            # Validate changed estimator parameters, including any newly unseeded setting.
            candidate_identity = _stable(clone(self.estimator).set_params(**parameters))
            for fold, (train, validation) in enumerate(self.folds):
                task = _digest({'identity': self.identity_sha256, 'estimator': candidate_identity, 'fold': fold})
                if task in seen:
                    raise ValueError('Duplicate candidate/fold in one checkpointed search')
                seen.add(task)
                result_path = self.folder / 'folds' / (task + '.json')
                if result_path.exists():
                    records[candidate_index, fold] = (_valid_fold(result_path, self.identity_sha256, task,
                                                                parameters=parameters, fold=fold), True)
                else:
                    work.append((candidate_index, fold, task, parameters, train, validation))
        started = perf_counter()
        # Workers commit each result before returning; bounded dispatch never queues an unbounded grid.
        with parallel_config(backend='loky', inner_max_num_threads=1), threadpool_limits(limits=1):
            if self.jobs == 1:
                for ci, fi, task, params, train, validation in work:
                    records[ci, fi] = _fit_fold(self.folder, self.identity_sha256, self.runtime, self.token,
                        task, self.estimator, params, self.X, self.y, train, validation, fi,
                        self.error_policy, str(self.folder.parent / 'failures'))
            else:
                completed = Parallel(n_jobs=self.jobs, pre_dispatch=self.jobs, batch_size=1)(
                    delayed(_fit_fold)(self.folder, self.identity_sha256, self.runtime, self.token,
                        task, self.estimator, params, self.X, self.y, train, validation, fi,
                        self.error_policy, str(self.folder.parent / 'failures'))
                    for ci, fi, task, params, train, validation in work)
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
        for field in ('fit_time', 'score_time'):
            values = np.asarray([[records[ci, fi][0][field] for fi in range(n_folds)] for ci in range(n_candidates)])
            results['mean_' + field], results['std_' + field] = np.mean(values, axis=1), np.std(values, axis=1)
        valid = np.isfinite(results['mean_test_score'])
        means = results['mean_test_score']
        results['rank_test_score'] = np.asarray([1 + int(np.sum(means[valid] > value)) if math.isfinite(value)
                                                else (int(np.sum(valid)) + 1 if np.any(valid) else 1) for value in means], dtype=np.int32)
        for name in sorted(set().union(*(params.keys() for params in candidates))):
            results['param_estimator__' + name] = np.ma.array([params.get(name) for params in candidates],
                                                            mask=[name not in params for params in candidates], dtype=object)
        summary = {'stage': str(stage), 'candidate_count': n_candidates, 'fold_count': n_folds,
                   'reused_folds': sum(reused for _, reused in records.values()),
                   'new_folds': sum(not reused for _, reused in records.values()),
                   'wall_seconds': perf_counter() - started,
                   'diagnostics': [[records[ci, fi][0].get('diagnostics', {}) for fi in range(n_folds)] for ci in range(n_candidates)]}
        search = SimpleNamespace(cv_results_=results, n_splits_=n_folds, checkpoint_summary_=summary)
        if np.any(valid):
            index = int(np.flatnonzero(valid)[np.argmax(means[valid])])
            search.best_index_, search.best_params_, search.best_score_ = index, results['params'][index], float(means[index])
        return search
