"""Optional completed-pair recovery for checkpointed multiclass examples.

Only trusted local pickle checkpoints are supported. The CV coordinator's
existing task lease and generation token own all access and reclamation.
"""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier
from ensemble_dwd.multiclass import _fit_pair
from search_support import CandidateEvaluator, data_fingerprint, json_safe
from _pair_store import PairStore, StoreMismatch, canonical, digest, _read_envelope
import _pair_store

POLICY = 'completed-pair-v1'


def checkpoint():
    # Lazy import avoids a coordinator/helper import cycle.
    import cv_checkpoint
    return cv_checkpoint


def array_identity(value):
    value = np.asarray(value)
    return {'shape': list(value.shape), 'dtype': value.dtype.str,
            'sha256': digest(np.ascontiguousarray(value).tobytes())}


VOLATILE_TIMERS = frozenset(('seconds', 'elapsed_seconds', 'setup_seconds', 'optimization_seconds',
    'total_seconds', 'factorization_seconds', 'initial_state_check_seconds', 'cholesky_setup_seconds',
    'spectral_setup_seconds', 'eigenvalue_check_seconds', 'adaptive_residual_probe_seconds',
    'native_refinement_seconds', 'compensated_residual_seconds', 'aposteriori_rkhs_seconds',
    'native_residual_seconds', 'reconstruction_seconds', 'preparation_seconds', 'readout_seconds'))


def snapshot(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise StoreMismatch('Object array is outside the comparison contract')
        return {'array': array_identity(value)}
    if isinstance(value, np.generic):
        return {'numpy_scalar': str(value.dtype), 'value': str(value)}
    if isinstance(value, BaseEstimator):
        return {'class': type(value).__module__ + '.' + type(value).__qualname__,
                'state': {name: snapshot(item) for name, item in vars(value).items()}}
    if isinstance(value, dict):
        return {name: snapshot(item) for name, item in value.items() if name not in VOLATILE_TIMERS}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if isinstance(value, float):
        return {'float_hex': value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise StoreMismatch('Unsupported fitted-state comparison value: ' + type(value).__qualname__)


def validate_model(model, estimator, X, y):
    if type(model) is not type(estimator) or checkpoint()._stable(model.get_params()) != checkpoint()._stable(estimator.get_params()):
        raise StoreMismatch('Restored model type or parameters mismatch')
    check_is_fitted(model)
    if not np.array_equal(model.classes_, np.unique(y)) or model.n_features_in_ != X.shape[1]:
        raise StoreMismatch('Restored model classes or feature count mismatch')
    if type(model) is KernGDWD:
        if not np.array_equal(model._Xfit, X) or model.dual_coef_.shape != (1, len(X)) or model.intercept_.shape != (1,):
            raise StoreMismatch('Restored DWD training state mismatch')
        if not np.isfinite(model.dual_coef_).all() or not np.isfinite(model.intercept_).all():
            raise StoreMismatch('Nonfinite restored DWD coefficients')
    elif type(model) is EnsembleKernelDWD:
        expected_gamma, expected_kws = estimator._kernel_parameters(X)
        expected_child = estimator._make_estimator(expected_kws)
        if model.gamma_ != expected_gamma or model.kernel_kws_ != expected_kws:
            raise StoreMismatch('Restored ensemble kernel controls mismatch')
        if len(model.subset_indices_) != model.M or model.consensus_scores_.shape != (len(X),):
            raise StoreMismatch('Restored ensemble dimensions mismatch')
        for rows in [*model.subset_indices_, model.selected_indices_]:
            if rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(X)):
                raise StoreMismatch('Restored ensemble row indices invalid')
        if len(np.unique(np.concatenate(model.subset_indices_))) != sum(map(len, model.subset_indices_)):
            raise StoreMismatch('Restored ensemble base subsets overlap')
        final = model.final_estimator_
        validate_model(final, expected_child, X[model.selected_indices_], y[model.selected_indices_])
        if model.store_base_estimators:
            if model.estimators_ is None or len(model.estimators_) != model.M:
                raise StoreMismatch('Restored retained bases missing')
            for child, rows in zip(model.estimators_, model.subset_indices_, strict=True):
                validate_model(child, expected_child, X[rows], y[rows])
        elif model.estimators_ is not None:
            raise StoreMismatch('Unexpected retained models in lean state')
    else:
        raise StoreMismatch('Unsupported estimator type')



def normalize_config(enabled, max_bytes):
    if type(enabled) is not bool:
        raise ValueError('pair_recovery must be boolean')
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError('pair_recovery_max_bytes must be a nonnegative integer or None')
    if not enabled:
        if max_bytes is not None:
            raise ValueError('pair_recovery_max_bytes requires pair_recovery=True')
        return None
    cp = checkpoint()
    expected_store = Path(__file__).with_name('_pair_store.py').resolve()
    if Path(_pair_store.__file__).resolve() != expected_store:
        raise cp.CheckpointError('Pair storage loaded outside the example source directory')
    paths = [Path(__file__).absolute(), expected_store]
    return {'policy': POLICY, 'max_bytes': max_bytes,
            'source_hashes': {str(path): cp._file_hash(path) for path in paths}}


def check_source_hashes(config):
    cp = checkpoint()
    if Path(_pair_store.__file__).resolve() != Path(__file__).with_name('_pair_store.py').resolve():
        raise cp.CheckpointError('Pair storage source origin changed')
    for path, expected in config['source_hashes'].items():
        try:
            actual = cp._file_hash(Path(path))
        except OSError as error:
            raise cp.CheckpointError('Pair-recovery source is unavailable') from error
        if actual != expected:
            raise cp.CheckpointError('Pair-recovery source changed; restart before resuming')


def validate_supported(model, *, require_numeric_gamma=True):
    if type(model) is not Pipeline or len(model.steps) != 2 or model.memory is not None or model.verbose:
        raise ValueError('Pair recovery requires a plain two-step Pipeline')
    if getattr(model, 'transform_input', None) is not None:
        raise ValueError('Pair recovery does not support metadata routing')
    scaler, ovo = (item[1] for item in model.steps)
    if (type(scaler) is not StandardScaler or type(ovo) is not OvOClassifier
            or type(ovo.n_jobs) is not int or ovo.n_jobs != 1):
        raise ValueError('Pair recovery requires StandardScaler and serial exact OvOClassifier')
    binary = ovo.estimator
    if type(binary) not in (KernGDWD, EnsembleKernelDWD):
        raise ValueError('Pair recovery supports exact KernGDWD or EnsembleKernelDWD only')
    if type(binary.random_state) is not int or not 0 <= binary.random_state < 2**32:
        raise ValueError('Pair recovery requires an integer random_state')
    if binary.kernel != 'rbf':
        raise ValueError('Pair recovery supports named RBF only')
    if type(binary) is KernGDWD:
        gamma = (binary.kernel_kws or {}).get('gamma')
    else:
        gamma = binary.gamma
        if binary.cache_dir is not None:
            raise ValueError('Pair recovery cannot combine a separate ensemble cache')
    if require_numeric_gamma and (type(gamma) not in (int, float)
            or not math.isfinite(gamma) or gamma <= 0):
        raise ValueError('Pair recovery requires positive numeric gamma for each candidate')


class FoldContext:
    """Explicit per-fold state; the caller holds its authoritative task lease."""
    def __init__(self, folder, identity, runtime, token, task, parameters, fold,
                 train, validation, X, y, config, attempt=None):
        self.folder, self.identity, self.runtime = Path(folder), identity, runtime
        self.token, self.task, self.parameters, self.fold = token, task, parameters, fold
        self.train, self.validation, self.X, self.y = train, validation, X, y
        self.config, self.attempt = config, attempt
        self.events, self.reporting_errors = [], []
        self.parent = None
        self.cleanup_status = 'pending'

    def assert_owner(self):
        cp = checkpoint()
        owner = cp._read_json(self.folder / 'owner.json')
        if owner.get('token') != self.token or owner.get('state') != 'active':
            raise cp.CheckpointBusyError('The dispatching coordinator no longer owns this run')
        check_source_hashes(self.config)

    def set_parent(self, model, transformed, scaler):
        cp = checkpoint()
        self.parent = {'schema': 1, 'policy': POLICY,
                       'checkpoint_identity_sha256': self.identity,
                       'task_sha256': self.task, 'parameters': cp._stable(self.parameters),
                       'fold': self.fold, 'runtime': self.runtime, 'config': self.config,
                       'candidate': cp._stable(model),
                       'raw_training_data': data_fingerprint(self.X[self.train], self.y[self.train]),
                       'training_rows': array_identity(self.train),
                       'validation_rows': array_identity(self.validation),
                       'transformed_data': data_fingerprint(transformed, self.y[self.train]),
                       'scaler': snapshot(scaler)}

    @property
    def store_root(self):
        return self.folder / 'pair-stores' / self.task

    def summary(self):
        return {'schema': 1, 'policy': POLICY,
                'fitted_pairs': sum(e['work_origin'] == 'fitted' for e in self.events),
                'recovered_pairs': sum(e['work_origin'] == 'recovered' for e in self.events),
                'fitted_unsaved_pairs': sum(e['status'] == 'fitted_unsaved' for e in self.events),
                'current_pair_fit_seconds': sum(e.get('fit_seconds', 0.0) for e in self.events),
                'pair_load_seconds': sum(e.get('load_seconds', 0.0) for e in self.events),
                'pair_save_seconds': sum(e.get('save_seconds', 0.0) for e in self.events),
                'historical_reused_pair_fit_seconds': sum(e.get('historical_fit_seconds', 0.0) for e in self.events),
                'pairs': deepcopy(self.events), 'cleanup_status': self.cleanup_status,
                'timing_policy': 'current-attempt-wall-time; restored diagnostics retain historical fit observations'}

    def write_evidence(self):
        if self.attempt is None:
            return
        cp = checkpoint()
        payload = {'schema': 1, 'identity_sha256': self.identity, 'task_sha256': self.task,
                   'attempt': self.attempt['attempt'], 'fold': self.fold,
                   'pair_recovery': self.summary()}
        destination = self.folder / 'pair-evidence' / (self.attempt['attempt'] + '.json')
        self.assert_owner()
        try:
            cp._write_receipt(destination, payload)
        except Exception as error:
            self.reporting_errors.append({'phase': 'pair_evidence', 'type': type(error).__name__})
            return False
        return True

    def _pair_finished(self, event):
        """Record a fully validated fitted or restored model before next pair."""
        self.events.append(event)
        self.write_evidence()

    def dispatch(self, estimator, X, y, indices, pairs, session):
        models = []
        for rows, pair in zip(indices, pairs, strict=True):
            self.assert_owner()
            pair_y = np.where(y[rows] == pair[0], 1, -1)
            key = {'labels': snapshot(pair), 'rows': array_identity(rows),
                   'data': data_fingerprint(X[rows], pair_y),
                   'estimator': checkpoint()._stable(estimator)}

            def compute():
                self.assert_owner()
                model = _fit_pair(estimator, X, y, rows, pair)
                self.assert_owner()
                return model

            def validate(model):
                self.assert_owner()
                validate_model(model, estimator, X[rows], pair_y)

            model, event = session.get_or_compute(key, compute, validate)
            event.update(pair=json_safe(pair), pair_index=len(models),
                         work_origin='recovered' if event['status'] == 'reused' else 'fitted')
            models.append(model)
            self._pair_finished(event)
        return models


def fit_pipeline(model, X, y, context):
    validate_supported(model)
    if (np.asarray(X).ndim != 2 or np.asarray(X).dtype.kind not in 'biuf'
            or not np.isfinite(X).all() or np.asarray(y).dtype.kind not in 'biuf'
            or not np.isfinite(y).all()):
        raise ValueError('Pair recovery requires dense finite numeric data and labels')
    context.assert_owner()
    model._validate_steps()
    scaler, ovo = (item[1] for item in model.steps)
    transformed = scaler.fit_transform(X, y)
    context.set_parent(model, transformed, scaler)
    store = PairStore(context.store_root, context.parent,
                      max_bytes=context.config['max_bytes'], snapshot=snapshot,
                      assert_owner=context.assert_owner)
    with store.session() as session:
        context.assert_owner()
        ovo._fit(transformed, y, pair_dispatch=lambda *args: context.dispatch(*args, session=session))
    return model


class FoldEvaluator(CandidateEvaluator):
    def __init__(self, estimator, context, error_policy='nan', failure_dir=None):
        super().__init__(estimator, error_policy=error_policy, failure_dir=failure_dir)
        self.context = context

    def _fit_candidate(self, candidate, X, y):
        return fit_pipeline(candidate, X, y, self.context)


def annotate_record(record, context):
    """Preserve child history while identifying work done by this attempt."""
    record['pair_recovery'] = context.summary()
    diagnostics = record['diagnostics']
    diagnostics['pair_diagnostics_scope'] = 'model_fit_history'
    if 'pairwise_estimators' in diagnostics:
        for pair, event in zip(diagnostics['pairwise_estimators'], context.events, strict=True):
            pair['work_origin'] = event['work_origin']


def validate_summary(record):
    """Reject malformed work attribution in a checksummed fold receipt."""
    cp = checkpoint()
    summary = record.get('pair_recovery')
    if (type(summary) is not dict or summary.get('schema') != 1
            or summary.get('policy') != POLICY or type(summary.get('pairs')) is not list
            or summary.get('cleanup_status') != 'pending'):
        raise cp.CheckpointError('Invalid pair-recovery fold summary')
    events = summary['pairs']
    for index, event in enumerate(events):
        if (type(event) is not dict or event.get('pair_index') != index
                or type(event.get('pair_index')) is not int
                or type(event.get('pair')) is not list or len(event['pair']) != 2
                or event.get('status') not in ('reused', 'fitted_saved', 'fitted_unsaved')
                or event.get('work_origin') != ('recovered' if event['status'] == 'reused' else 'fitted')
                or event.get('cache_committed') is not (event['status'] != 'fitted_unsaved')):
            raise cp.CheckpointError('Invalid pair-recovery event')
        for field in ('fit_seconds', 'load_seconds', 'save_seconds', 'historical_fit_seconds'):
            value = event.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise cp.CheckpointError('Invalid pair-recovery event timing')
        if ((event['work_origin'] == 'recovered' and (event['fit_seconds'] != 0 or event['save_seconds'] != 0))
                or (event['work_origin'] == 'fitted' and (event['historical_fit_seconds'] != 0 or event['load_seconds'] != 0))):
            raise cp.CheckpointError('Pair-recovery timing mixes historical and current work')
    counts = {'fitted_pairs': sum(e['work_origin'] == 'fitted' for e in events),
              'recovered_pairs': sum(e['work_origin'] == 'recovered' for e in events),
              'fitted_unsaved_pairs': sum(e['status'] == 'fitted_unsaved' for e in events)}
    for name, value in counts.items():
        if type(summary.get(name)) is not int or summary[name] != value:
            raise cp.CheckpointError('Pair-recovery count disagrees with events')
    times = {'current_pair_fit_seconds': 'fit_seconds', 'pair_load_seconds': 'load_seconds',
             'pair_save_seconds': 'save_seconds', 'historical_reused_pair_fit_seconds': 'historical_fit_seconds'}
    for name, source in times.items():
        value = summary.get(name)
        if type(value) not in (int, float) or value != sum(e[source] for e in events):
            raise cp.CheckpointError('Pair-recovery timing disagrees with events')
    diagnostics = record.get('diagnostics', {})
    pairs = diagnostics.get('pairwise_estimators')
    if record['state'] == 'complete' and (type(pairs) is not list or len(pairs) != len(events)):
        raise cp.CheckpointError('Completed pair diagnostics disagree with recovery events')
    if pairs is not None:
        if len(pairs) != len(events) or any(
                pair.get('pair') != event['pair'] or pair.get('work_origin') != event['work_origin']
                for pair, event in zip(pairs, events)):
            raise cp.CheckpointError('Pair diagnostics have incorrect work origins')


def _cleanup_store(context, record):
    cp = checkpoint()
    if not context.store_root.exists():
        return {'status': 'no_pair_store'}
    parent_record, _ = _read_envelope(context.store_root / 'identity.json')
    if not isinstance(parent_record, dict):
        raise StoreMismatch('Malformed pair-store parent')
    parent = parent_record.get('identity')
    if (not isinstance(parent, dict) or parent.get('checkpoint_identity_sha256') != context.identity
            or parent.get('task_sha256') != context.task or parent.get('fold') != context.fold
            or parent.get('parameters') != cp._stable(context.parameters)
            or parent.get('config') != context.config or parent.get('runtime') != context.runtime
            or cp._digest({'identity': context.identity, 'estimator': parent.get('candidate'),
                           'fold': context.fold}) != context.task
            or parent.get('raw_training_data') != data_fingerprint(context.X[context.train], context.y[context.train])
            or parent.get('training_rows') != array_identity(context.train)
            or parent.get('validation_rows') != array_identity(context.validation)):
        raise StoreMismatch('Pair-store parent does not match authoritative fold')
    def assert_cleanup_owner():
        context.assert_owner()
        authoritative = cp._valid_fold(context.folder / 'folds' / (context.task + '.json'),
            context.identity, context.task, parameters=context.parameters,
            fold=context.fold, validation_size=len(context.validation))
        if authoritative != record:
            raise cp.CheckpointError('Fold receipt changed before cleanup')

    store = PairStore(context.store_root, parent, max_bytes=context.config['max_bytes'],
                      snapshot=snapshot, assert_owner=assert_cleanup_owner)
    with store.session() as session:
        context.assert_owner()
        authoritative = cp._valid_fold(context.folder / 'folds' / (context.task + '.json'),
            context.identity, context.task, parameters=context.parameters,
            fold=context.fold, validation_size=len(context.validation))
        if authoritative != record:
            raise cp.CheckpointError('Fold receipt changed before cleanup')
        return session.complete(cp._digest(authoritative))


def cleanup_context(context, *, already_validated=None):
    """Validate under the task lease, then reclaim only optional pair files."""
    cp = checkpoint()
    context.assert_owner()
    path = context.folder / 'folds' / (context.task + '.json')
    record = cp._valid_fold(path, context.identity, context.task,
                           parameters=context.parameters, fold=context.fold,
                           validation_size=len(context.validation))
    validate_summary(record)
    if already_validated is not None and record != already_validated:
        raise cp.CheckpointError('Fold receipt changed before cleanup')
    try:
        result = _cleanup_store(context, record)
    except (OSError, StoreMismatch) as error:
        result = {'status': 'cleanup_deferred', 'error_type': type(error).__name__,
                  'remaining_pair_bytes': None}
    context.cleanup_status = result['status']
    destination = context.folder / 'pair-cleanup' / (context.task + '.json')
    context.assert_owner()
    try:
        cp._write_receipt(destination, {'identity_sha256': context.identity,
            'task_sha256': context.task, 'fold': context.fold,
            'authoritative_receipt_sha256': cp._digest(record), **result})
    except Exception as error:
        context.reporting_errors.append({'phase': 'pair_cleanup', 'type': type(error).__name__})
    return result


def cleanup_saved_fold(folder, identity, runtime, token, task, parameters, fold,
                       train, validation, X, y, config):
    cp = checkpoint()
    with cp._Lease(Path(folder) / 'locks' / (task + '.lock')):
        context = FoldContext(folder, identity, runtime, token, task, parameters, fold,
                              train, validation, X, y, config)
        context.assert_owner()
        record = cp._valid_fold(Path(folder) / 'folds' / (task + '.json'), identity, task,
                                parameters=parameters, fold=fold, validation_size=len(validation))
        result = cleanup_context(context, already_validated=record)
        returned = deepcopy(record)
        returned['pair_recovery']['cleanup_status'] = result['status']
        return returned
