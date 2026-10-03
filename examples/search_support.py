"""Failure handling and durable records for external parameter searches."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

from ensemble_dwd import IneligibleSelectionError, IneligibleSubsetError


def json_safe(value):
    """Convert numerical results to strict JSON; nonfinite values become null."""
    if isinstance(value, np.ma.MaskedArray):
        data = value.astype(object).data.copy()
        data[np.ma.getmaskarray(value)] = None
        return json_safe(data.tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return {'type': type(value).__module__ + '.' + type(value).__qualname__}


def write_json_atomic(path, payload, *, replace=False):
    """Write within an existing directory, replacing only when requested."""
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.record-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(json_safe(payload), stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _warn_logging_failure(error):
    # A warnings-as-errors policy must not hide the original numerical failure.
    try:
        warnings.warn('Could not write the failure record: ' + str(error), RuntimeWarning)
    except Exception:
        pass


def write_failure_json(path, payload, *, replace=False):
    """Best-effort failure-only logging; successful checkpoints stay strict."""
    try:
        return write_json_atomic(path, payload, replace=replace)
    except Exception as logging_error:
        _warn_logging_failure(logging_error)
        return None


def new_run_directory(path=None, *, resume=False):
    """Create a fresh directory, or locate an explicitly requested saved run.

    Numerical checkpoint identity and exclusive ownership are checked by the
    search runner before it reuses or writes any evaluation records.
    """
    if resume:
        if path is None:
            raise ValueError('Resuming requires an explicit run directory.')
        path = Path(path).resolve()
        if not path.is_dir() or not (path / 'failures').is_dir():
            raise ValueError('The resume directory must contain an existing search run.')
        return path
    if path is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path = Path('results') / ('mnist-' + stamp + '-' + uuid.uuid4().hex[:8])
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    (path / 'failures').mkdir()
    return path


def array_fingerprint(values):
    array = np.asarray(values)
    digest = hashlib.sha256()
    identity = {'shape': list(array.shape), 'dtype': array.dtype.str}
    digest.update(json.dumps(identity, sort_keys=True).encode('utf-8'))
    if array.dtype.hasobject:
        # Object pointers are not stable data identities. Serialize values for
        # hashing only; the record contains the resulting digest, not the data.
        digest.update(json.dumps(json_safe(array.tolist()), sort_keys=True,
                                 ensure_ascii=False, allow_nan=False).encode('utf-8'))
    else:
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast('B'))
    return {**identity, 'sha256': digest.hexdigest()}


def data_fingerprint(X, y):
    return {'X': array_fingerprint(X), 'y': array_fingerprint(y)}


def _parameter_value(value):
    if isinstance(value, np.ndarray):
        return {'array': array_fingerprint(value)}
    if isinstance(value, dict):
        return {str(key): _parameter_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_parameter_value(item) for item in value]
    return json_safe(value)


def parameter_snapshot(estimator):
    return {key: _parameter_value(value) for key, value in estimator.get_params(deep=True).items()}


def exception_details(error):
    if isinstance(error, IneligibleSubsetError):
        category = 'ineligible_subset'
    elif isinstance(error, IneligibleSelectionError):
        category = 'ineligible_selection'
    elif isinstance(error, MemoryError):
        category = 'memory_error'
    elif isinstance(error, FloatingPointError):
        category = 'numerical_error'
    else:
        category = 'unexpected_error'
    return {'category': category,
            'exception_type': type(error).__module__ + '.' + type(error).__qualname__,
            'message': str(error)}


def record_failure(directory, error, *, phase, estimator, training_data):
    """Atomically append one uniquely named record without exposing input rows."""
    details = exception_details(error)
    if directory is None:
        return details
    path = Path(directory) / ('failure-' + uuid.uuid4().hex + '.json')
    try:
        payload = {**details, 'phase': phase, 'parameters': parameter_snapshot(estimator),
                   'training_data': training_data, 'pid': os.getpid(),
                   'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        write_json_atomic(path, payload)
    except Exception as logging_error:
        # Preserve the original fit/scoring exception even if the disk is full.
        _warn_logging_failure(logging_error)
    return details


class CandidateEvaluator(ClassifierMixin, BaseEstimator):
    """Cloneable search adapter for explicitly ineligible training-row selections.

    Only IneligibleSelectionError (including IneligibleSubsetError) may be
    represented as an ineligible candidate.
    All other fit or prediction exceptions are recorded and propagated.
    """
    def __init__(self, estimator, error_policy='nan', failure_dir=None):
        self.estimator = estimator
        self.error_policy = error_policy
        self.failure_dir = failure_dir

    def fit(self, X, y):
        for name in list(vars(self)):
            if name.endswith('_'):
                delattr(self, name)
        self.eligible_ = False
        self.failure_ = None
        if self.error_policy not in ('nan', 'raise'):
            raise ValueError("error_policy must be 'nan' or 'raise'.")
        self.training_data_ = None
        candidate = self.estimator
        try:
            self.training_data_ = data_fingerprint(X, y)
            candidate = clone(self.estimator)
            candidate.fit(X, y)
            classes = np.asarray(candidate.classes_)
        except Exception as error:
            self.failure_ = record_failure(
                self.failure_dir, error, phase='cv_fit', estimator=candidate,
                training_data=self.training_data_)
            if isinstance(error, IneligibleSelectionError) and self.error_policy == 'nan':
                return self
            raise
        self.estimator_ = candidate
        self.eligible_ = True
        self.classes_ = classes
        if hasattr(candidate, 'n_features_in_'):
            self.n_features_in_ = candidate.n_features_in_
        return self

    def predict(self, X):
        if not getattr(self, 'eligible_', False) or not hasattr(self, 'estimator_'):
            if getattr(self, 'failure_', None):
                if self.failure_['category'] == 'ineligible_subset':
                    raise IneligibleSubsetError(self.failure_['message'])
                if self.failure_['category'] == 'ineligible_selection':
                    raise IneligibleSelectionError(self.failure_['message'])
            raise NotFittedError('The search candidate did not finish fitting successfully.')
        try:
            return self.estimator_.predict(X)
        except Exception as error:
            record_failure(self.failure_dir, error, phase='cv_predict',
                           estimator=self.estimator_, training_data=self.training_data_)
            raise


def eligible_accuracy(estimator, X, y):
    """NaN is reserved for an explicitly recorded ineligible subset or selection."""
    if not estimator.eligible_:
        if estimator.failure_ and estimator.failure_['category'] in (
                'ineligible_subset', 'ineligible_selection'):
            return np.nan
        raise RuntimeError('The candidate has no eligible fitted model.')
    predictions = estimator.predict(X)
    try:
        return float(accuracy_score(y, predictions))
    except Exception as error:
        record_failure(estimator.failure_dir, error, phase='cv_scoring',
                       estimator=estimator.estimator_, training_data=estimator.training_data_)
        raise


def prefix_grid(grid):
    def one(mapping):
        return {'estimator__' + name: values for name, values in mapping.items()}
    return one(grid) if isinstance(grid, dict) else [one(mapping) for mapping in grid]


def bare_parameters(parameters):
    prefix = 'estimator__'
    if any(not name.startswith(prefix) for name in parameters):
        raise ValueError('Search parameters do not identify the constituent estimator.')
    return {name[len(prefix):]: value for name, value in parameters.items()}
