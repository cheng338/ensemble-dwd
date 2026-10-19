"""Failure handling and durable records for external parameter searches."""
from datetime import datetime, timezone
from fractions import Fraction
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
from ensemble_dwd._failure_context import failure_context


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


def _io_path(path):
    """Use extended Windows paths for I/O without changing public path spelling."""
    path = Path(path)
    if os.name != 'nt':
        return path
    absolute = os.path.abspath(os.fspath(path))
    if not absolute.startswith('\\\\?\\'):
        absolute = ('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                    else '\\\\?\\' + absolute)
    return Path(absolute)


def _resolved_public_path(path):
    """Resolve through a long-path-safe name, retaining ordinary caller spelling."""
    requested = Path(path)
    resolved = _io_path(requested).resolve()
    if os.name == 'nt' and not str(requested).startswith('\\\\?\\'):
        value = str(resolved)
        if value.startswith('\\\\?\\UNC\\'):
            return Path('\\\\' + value[8:])
        if value.startswith('\\\\?\\'):
            return Path(value[4:])
    return resolved


def write_json_atomic(path, payload, *, replace=False):
    """Write within an existing directory, replacing only when requested."""
    path = Path(path)
    io_path = _io_path(path)
    if io_path.exists() and not replace:
        raise FileExistsError(path)
    # Serialization errors should not leave even an unpublished temporary file.
    serialized = json.dumps(json_safe(payload), indent=2, allow_nan=False, sort_keys=True) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=io_path.parent,
                                         prefix='.record-', suffix='.tmp', delete=False) as stream:
            temporary = _io_path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if io_path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(temporary, io_path)
        temporary = None
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException:
                pass
        raise
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
        path = _resolved_public_path(path)
        io_path = _io_path(path)
        if not io_path.is_dir() or not (io_path / 'failures').is_dir():
            raise ValueError('The resume directory must contain an existing search run.')
        return path
    if path is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path = Path('results') / ('mnist-' + stamp + '-' + uuid.uuid4().hex[:8])
    path = _resolved_public_path(path)
    io_path = _io_path(path)
    io_path.mkdir(parents=True, exist_ok=False)
    (io_path / 'failures').mkdir()
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
    details = {'category': category,
               'exception_type': type(error).__module__ + '.' + type(error).__qualname__,
               'message': str(error)}
    try:
        context = failure_context(error)
        if context is not None:
            details['ensemble_failure_context'] = context
    except Exception:
        # Supplementary failure evidence must not replace the original error.
        pass
    try:
        notes = getattr(error, '__notes__', None)
    except Exception:
        return details
    # Bound existing diagnostic notes without invoking user-defined containers.
    # Python and the older-interpreter compatibility helper use plain lists.
    if type(notes) not in (list, tuple):
        return details
    kept = []
    truncated = len(notes) > 8
    for note in notes[:8]:
        if not isinstance(note, str):
            continue
        # Avoid custom __str__, length or slicing methods on a str subclass.
        plain = str.__str__(note)
        truncated = truncated or len(plain) > 1024
        kept.append(plain[:1024])
    if kept:
        details['exception_notes'] = kept
    if truncated:
        details['exception_notes_truncated'] = True
    return details


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

    def _fit_candidate(self, candidate, X, y):
        """Fit one cloned candidate; specialized examples may dispatch its work."""
        return candidate.fit(X, y)

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
            self._fit_candidate(candidate, X, y)
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
    estimator.accuracy_counts_ = None
    if not estimator.eligible_:
        if estimator.failure_ and estimator.failure_['category'] in (
                'ineligible_subset', 'ineligible_selection'):
            return np.nan
        raise RuntimeError('The candidate has no eligible fitted model.')
    predictions = estimator.predict(X)
    try:
        score = float(accuracy_score(y, predictions))
        actual, predicted = np.asarray(y), np.asarray(predictions)
        if actual.ndim == predicted.ndim == 1 and actual.shape == predicted.shape and len(actual):
            counts = {'correct': int(np.count_nonzero(actual == predicted)),
                      'n_validation': len(actual)}
            # Keep custom or unsupported scorer behavior unchanged. Counts are
            # evidence only for the unweighted one-label-per-row accuracy above.
            if float(accuracy_fraction(counts)) == score:
                estimator.accuracy_counts_ = counts
        return score
    except Exception as error:
        record_failure(estimator.failure_dir, error, phase='cv_scoring',
                       estimator=estimator.estimator_, training_data=estimator.training_data_)
        raise


ACCURACY_AGGREGATION = 'exact-mean-seeds-within-fold-then-folds-v1'


def accuracy_fraction(counts):
    """Validate recorded counts; never infer them from a rounded score."""
    if not isinstance(counts, dict) or set(counts) != {'correct', 'n_validation'}:
        raise ValueError('Accuracy evidence must contain correct and n_validation counts.')
    correct, size = counts['correct'], counts['n_validation']
    if (type(correct) is not int or type(size) is not int
            or size <= 0 or not 0 <= correct <= size):
        raise ValueError('Invalid accuracy counts.')
    return Fraction(correct, size)


def _fraction_record(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def accuracy_evidence(seed_folds):
    """Exact equal-seed-within-fold, then equal-fold mean; missing stays unknown."""
    if not seed_folds or not seed_folds[0]:
        return None
    n_folds = len(seed_folds[0])
    if any(len(folds) != n_folds for folds in seed_folds):
        raise ValueError('Accuracy evidence has inconsistent fold counts.')
    if any(counts is None for folds in seed_folds for counts in folds):
        return None
    fractions = [[accuracy_fraction(counts) for counts in folds] for folds in seed_folds]
    means = [sum((folds[fi] for folds in fractions), Fraction()) / len(fractions)
             for fi in range(n_folds)]
    return {'policy': ACCURACY_AGGREGATION, 'counts': seed_folds,
            'axes': ['seed', 'fold'], 'fold_means': [_fraction_record(v) for v in means],
            'mean': _fraction_record(sum(means, Fraction()) / n_folds)}


def exact_accuracy_available(*collections):
    """A comparison uses counts only if every eligible candidate has them."""
    for results in collections:
        evidence = results.get('accuracy_evidence')
        for index, score in enumerate(results['mean_test_score']):
            if np.isfinite(float(score)) and (evidence is None or evidence[index] is None):
                return False
    return True


def score_value(results, index, *, exact=None):
    """Compare a whole candidate set exactly or use its reported scalars."""
    score = float(results['mean_test_score'][index])
    evidence = results.get('accuracy_evidence')
    evidence = None if evidence is None else evidence[index]
    if evidence is None:
        return score
    if evidence != accuracy_evidence(evidence.get('counts')):
        raise ValueError('Accuracy aggregation evidence does not match its counts.')
    mean = evidence['mean']
    value = Fraction(mean['numerator'], mean['denominator'])
    if float(value) != score:
        raise ValueError('Reported accuracy does not match its count evidence.')
    if exact is None:
        exact = exact_accuracy_available(results)
    return value if exact else score


def score_token(results, index):
    value = score_value(results, index)
    return _fraction_record(value) if isinstance(value, Fraction) else {'score': value}


def token_value(token, fallback):
    """Read new exact policy evidence while retaining scalar-only old results."""
    if token is None or 'score' in token:
        return fallback if token is None else token['score']
    return Fraction(token['numerator'], token['denominator'])


def comparable_tokens(first, second, first_fallback, second_fallback):
    """Historical scalar-only stages cannot supply an exact comparison."""
    left, right = token_value(first, first_fallback), token_value(second, second_fallback)
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left, right
    return (None if left is None else float(left), None if right is None else float(right))


def equal_score_tokens(first, second, first_fallback, second_fallback):
    left, right = comparable_tokens(first, second, first_fallback, second_fallback)
    return left == right


def score_nonimproving(after, before, tolerance):
    """Retain scalar addition rounding; count comparisons use exact differences."""
    if isinstance(after, Fraction) and isinstance(before, Fraction):
        return after - before <= tolerance
    return after <= before + tolerance


def apply_accuracy_evidence(results, evidence):
    """Keep public scores as floats, rounding exact fold/overall means once."""
    results['accuracy_evidence'] = evidence
    for index, item in enumerate(evidence):
        if item is None:
            continue
        folds = [float(Fraction(v['numerator'], v['denominator'])) for v in item['fold_means']]
        for fi, value in enumerate(folds):
            results[f'split{fi}_test_score'][index] = value
        mean = item['mean']
        results['mean_test_score'][index] = float(Fraction(mean['numerator'], mean['denominator']))
        results['std_test_score'][index] = float(np.std(folds))


def score_ranks(results, *, exact=None):
    values = [score_value(results, index, exact=exact) for index in range(len(results['params']))]
    valid = [value for value in values if np.isfinite(float(value))]
    return np.asarray([1 + sum(other > value for other in valid) if np.isfinite(float(value))
                       else len(valid) + 1 if valid else 1 for value in values], dtype=np.int32)


def prefix_grid(grid):
    def one(mapping):
        return {'estimator__' + name: values for name, values in mapping.items()}
    return one(grid) if isinstance(grid, dict) else [one(mapping) for mapping in grid]


def bare_parameters(parameters):
    prefix = 'estimator__'
    if any(not name.startswith(prefix) for name in parameters):
        raise ValueError('Search parameters do not identify the constituent estimator.')
    return {name[len(prefix):]: value for name, value in parameters.items()}
