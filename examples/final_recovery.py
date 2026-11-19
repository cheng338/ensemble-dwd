"""Explicit final-only attempts from a prospective accepted-search handoff.

This is a separate execution identity, not a relaxed CV resume or a historical
checkpoint importer. The source session lease protects every finalization
operation. A durable scoring-start record is shared by normal and recovery
paths; an interrupted uncommitted score is deliberately not retried.
"""
from pathlib import Path
from time import perf_counter
import uuid
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

import cv_checkpoint as checkpoint_module
import evaluation_reporting
import phase_threads
import work_reporting
from search_support import (_resolved_public_path, data_fingerprint, exception_message,
                            json_safe, write_json_atomic)


_HANDOFF_SCHEMA = 'accepted-final-handoff-role-v2'
_ATTEMPT_SCHEMA = 'final-only-attempt-role-v2'


def _error(message):
    raise checkpoint_module.CheckpointError(message)


def _same(left, right):
    return checkpoint_module._canonical(left) == checkpoint_module._canonical(right)


def _tagged_item(value, key):
    """Read one key of an existing typed identity; never decode estimators."""
    if (not isinstance(value, list) or len(value) != 2
            or value[0] != 'dict' or not isinstance(value[1], list)):
        _error('Malformed parent provenance identity')
    pairs = value[1]
    if any(not isinstance(row, list) or len(row) != 2 for row in pairs):
        _error('Malformed parent provenance identity')
    matches = [row[1] for row in pairs if row[0] == key]
    if len(matches) != 1:
        _error('Missing or duplicate parent provenance field: ' + key)
    return matches[0]


def _sources():
    return {
        'workflow_sha256': checkpoint_module._file_hash(
            Path(__file__).with_name('adaptive_search.py')),
        'final_thread_policy_sha256': checkpoint_module._file_hash(phase_threads.__file__),
        'final_recovery_sha256': checkpoint_module._file_hash(__file__),
        'checkpoint_sha256': checkpoint_module._file_hash(checkpoint_module.__file__),
        'reporting_sha256': {
            'evaluation_reporting.py': checkpoint_module._file_hash(evaluation_reporting.__file__),
            'work_reporting.py': checkpoint_module._file_hash(work_reporting.__file__),
        },
    }


def _check_sources(parent, sources):
    if not _same(sources, _sources()):
        _error('Final recovery source identity changed')
    for key in ('workflow_sha256', 'final_thread_policy_sha256', 'final_recovery_sha256',
                'reporting_sha256'):
        if not _same(_tagged_item(parent['provenance'], key),
                     checkpoint_module._stable(sources[key])):
            _error('Parent workflow does not authorize this recovery source: ' + key)


def _parent(folder):
    path = folder / 'identity.json'
    record = checkpoint_module._read_json(path)
    try:
        identity = record['payload']['identity']
    except (KeyError, TypeError):
        _error('Malformed source checkpoint identity')
    if not isinstance(identity, dict) or identity.get('schema') != checkpoint_module.SCHEMA:
        _error('Unsupported source checkpoint identity')
    digest = checkpoint_module._digest(identity)
    payload = checkpoint_module._read_receipt(path, digest)
    if set(payload) != {'identity_sha256', 'identity'}:
        _error('Malformed source checkpoint identity envelope')
    return identity, digest


def _completed_parent(folder, parent_digest):
    path = folder / 'artifacts' / 'completed.json'
    if path.exists():
        result = checkpoint_module._read_receipt(path, parent_digest)
        if result.get('artifact') != 'completed':
            _error('Malformed original completion artifact')
        _error('The original workflow already completed; replay its saved result')


def _immutable(path, payload):
    """The caller holds the source lease; never replace an existing decision."""
    if path.exists():
        existing = checkpoint_module._read_receipt(path, payload['identity_sha256'])
        if not _same(existing, payload):
            _error('Immutable finalization record differs: ' + str(path))
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_module._write_receipt(path, payload)
    return payload


def _runtime(parent, *, frozen_runtime=None):
    runtime = checkpoint_module.runtime_fingerprint()
    encoded = checkpoint_module._stable(runtime)
    checkpoint_module._require_execution_compatible(parent['runtime'], encoded)
    if frozen_runtime is not None and not _same(encoded, frozen_runtime):
        _error('Final recovery runtime changed within this execution attempt')
    return runtime


def _held_out_matches_identity(value, identity):
    """Join the fixed fingerprint fields without retyping sorted JSON maps."""
    if type(value) is not dict or set(value) != {'X', 'y'}:
        return False
    for name in ('X', 'y'):
        descriptor = value[name]
        if (type(descriptor) is not dict
                or set(descriptor) != {'shape', 'dtype', 'sha256'}):
            return False
        tagged = _tagged_item(identity, name)
        for field in ('shape', 'dtype', 'sha256'):
            if not _same(checkpoint_module._stable(descriptor[field]),
                         _tagged_item(tagged, field)):
                return False
    return True


def _handoff(folder, parent, parent_digest):
    path = folder / 'finalization' / 'handoff.json'
    value = checkpoint_module._read_receipt(path, parent_digest)
    if (value.get('schema') != _HANDOFF_SCHEMA
            or value.get('artifact') != 'accepted_final_selection'
            or value.get('review_state') != 'accepted'
            or not _same(value.get('parent_provenance'), parent['provenance'])
            or not _same(value.get('training'), parent['data'])):
        _error('Invalid accepted final-selection handoff')
    _check_sources(parent, value.get('sources'))
    parent_data = _tagged_item(parent['provenance'], 'data')
    held_out_identity = value.get('held_out_identity')
    if (not _same(_tagged_item(parent_data, 'held_out'), held_out_identity)
            or not _held_out_matches_identity(value.get('held_out'), held_out_identity)):
        _error('Handoff held-out identity differs from its parent')
    return value, checkpoint_module._digest(value)


def _barrier(folder):
    return folder / 'finalization' / 'scoring-started.json'


def _check_no_score(folder):
    if _barrier(folder).exists():
        _error('Held-out scoring has already been reserved; use a committed result, '
               'not an automatic scoring retry')


def assert_normal_final_allowed(checkpoint):
    """Call after original completed replay, before ordinary CV/refit work."""
    checkpoint._active()
    _check_no_score(checkpoint.folder)
    attempts = checkpoint.folder / 'finalization' / 'attempts'
    if attempts.exists() and any(attempts.iterdir()):
        _error('An explicit final-only attempt exists; ordinary resume cannot ignore it')


def publish_final_handoff(checkpoint, model, X_test, y_test, selection, review):
    """Commit the accepted branch's exact configured model, before any fit.

    The accepted workflow branch is the decision producer. This function does
    not reconstruct historical selection from supplementary reports or rescan
    all completed folds. The prospective handoff becomes the decision artifact.
    """
    checkpoint._active()
    assert_normal_final_allowed(checkpoint)
    parent, digest = _parent(checkpoint.folder)
    if digest != checkpoint.identity_sha256 or not _same(parent, checkpoint.identity):
        _error('Active checkpoint identity changed')
    _completed_parent(checkpoint.folder, digest)
    sources = _sources()
    _check_sources(parent, sources)
    checkpoint.check_runtime()
    checkpoint_module._require_execution_compatible(parent['runtime'], checkpoint.runtime)
    checkpoint_module.check_selection_state()
    if (not isinstance(review, dict) or review.get('state') != 'accepted'
            or review.get('final_refit_performed', False)
            or review.get('held_out_scored', False)):
        _error('A prospective accepted, unscored selection is required')
    if not isinstance(selection, dict) or not isinstance(selection.get('parameters'), dict):
        _error('A selected parameter mapping is required')
    if not _same(checkpoint_module._stable(checkpoint.provenance), parent['provenance']):
        _error('Active workflow provenance changed')
    if not _same(checkpoint_module._stable(checkpoint.estimator), parent['estimator']):
        _error('Active checkpoint estimator changed')
    if not _same(data_fingerprint(checkpoint.X, checkpoint.y), parent['data']):
        _error('Active training data changed')
    held_out = data_fingerprint(X_test, y_test)
    if not _same(held_out, checkpoint.provenance['data']['held_out']):
        _error('Held-out data differs from the original workflow')
    held_out_identity = checkpoint_module._stable(held_out)
    parent_data = _tagged_item(parent['provenance'], 'data')
    if not _same(held_out_identity, _tagged_item(parent_data, 'held_out')):
        _error('Held-out data differs from the original workflow')
    parameters = dict(selection['parameters'])
    if checkpoint.cv_seeds is not None:
        sampling = checkpoint.provenance['settings']['sampling']
        if (selection.get('refit_seed') != sampling['refit_seed']
                or selection.get('seed_parameter') != checkpoint.seed_parameter
                or selection.get('cv_seeds') != list(checkpoint.cv_seeds)):
            _error('Final sampling seed differs from the accepted selection')
        parameters[checkpoint.seed_parameter] = sampling['refit_seed']
    expected = clone(checkpoint.estimator).set_params(**parameters)
    model_identity = checkpoint_module._stable(model)
    if not _same(model_identity, checkpoint_module._stable(expected)):
        _error('Configured final model differs from the accepted parameters')
    if ('numerical_confirmation' in checkpoint.provenance
            and review.get('numerical_confirmation', {}).get('state') != 'stable'):
        _error('Numerical confirmation has not authorized final fitting')
    value = {
        'schema': _HANDOFF_SCHEMA, 'identity_sha256': digest,
        'artifact': 'accepted_final_selection', 'review_state': 'accepted',
        'parent_provenance': parent['provenance'], 'sources': sources,
        'training': parent['data'], 'held_out': held_out,
        'held_out_identity': held_out_identity,
        'final_estimator': model_identity,
        'selection_identity': checkpoint_module._stable(selection),
        'review_identity': checkpoint_module._stable(review),
        'selection': json_safe(selection), 'review': json_safe(review),
        'original_final_native_threads': checkpoint.provenance['settings']['final_native_threads'],
    }
    path = checkpoint.folder / 'finalization' / 'handoff.json'
    value = _immutable(path, value)
    return {'path': str(path), 'parent_identity_sha256': digest,
            'handoff_sha256': checkpoint_module._digest(value)}


def _reserve_score(folder, parent_digest, handoff_digest, actor):
    _check_no_score(folder)
    value = {'identity_sha256': parent_digest, 'artifact': 'held_out_scoring_started',
             'handoff_sha256': handoff_digest, 'actor': actor}
    _immutable(_barrier(folder), value)
    return value


def reserve_normal_scoring(checkpoint):
    """Reserve the one automatic held-out evaluation immediately before predict."""
    checkpoint._active()
    assert_normal_final_allowed(checkpoint)
    parent, digest = _parent(checkpoint.folder)
    if digest != checkpoint.identity_sha256:
        _error('Active source identity changed before scoring')
    _completed_parent(checkpoint.folder, digest)
    handoff, handoff_digest = _handoff(checkpoint.folder, parent, digest)
    checkpoint.check_runtime()
    checkpoint_module._require_execution_compatible(parent['runtime'], checkpoint.runtime)
    if not _same(data_fingerprint(checkpoint.X, checkpoint.y), handoff['training']):
        _error('Training data changed before scoring')
    return _reserve_score(checkpoint.folder, digest, handoff_digest,
                          {'kind': 'normal_workflow', 'identity_sha256': digest,
                           'session_owner_token': checkpoint.token,
                           'execution_runtime': checkpoint.runtime})


def _own_data(X_train, y_train, X_test, y_test):
    arrays = tuple(np.array(value, copy=True) for value in (X_train, y_train, X_test, y_test))
    for X, y in (arrays[:2], arrays[2:]):
        if (X.ndim != 2 or y.ndim != 1 or len(X) != len(y)
                or X.dtype.hasobject or y.dtype.hasobject):
            _error('Final recovery requires dense numeric X and one-dimensional labels')
    for value in arrays:
        value.setflags(write=False)
    return arrays


def _check_data(arrays, handoff):
    if (not _same(data_fingerprint(*arrays[:2]), handoff['training'])
            or not _same(data_fingerprint(*arrays[2:]), handoff['held_out'])):
        _error('Final recovery data differs from the accepted handoff')


def _check_current(folder, parent, handoff, arrays, *, frozen_runtime=None):
    current_parent, parent_digest = _parent(folder)
    if not _same(current_parent, parent):
        _error('Source checkpoint identity changed during final recovery')
    current_handoff, _ = _handoff(folder, current_parent, parent_digest)
    if not _same(current_handoff, handoff):
        _error('Accepted handoff changed during final recovery')
    _check_sources(parent, handoff['sources'])
    _check_data(arrays, handoff)
    return _runtime(parent, frozen_runtime=frozen_runtime)


def _failure(attempt_dir, identity, invocation, phase, error):
    """Failure reporting must never replace a fit/predict/interrupt exception."""
    try:
        value = {'identity_sha256': identity, 'artifact': 'failed_final_invocation',
                 'invocation': invocation, 'phase': phase, 'state': 'failed',
                 'exception_type': type(error).__name__, 'message': str(error)}
        _immutable(attempt_dir / 'invocations' / (invocation + '.json'), value)
        write_json_atomic(attempt_dir / 'status.json', value, replace=True)
    except BaseException:
        pass


def _optional_report(factory):
    try:
        return factory()
    except Exception as error:
        message = exception_message(error)
        try:
            warnings.warn('Work/report metadata unavailable: ' + message, RuntimeWarning)
        except Exception:
            pass
        return {'reporting_error': message, 'coverage': 'unavailable'}


def recover_final_fit(final_pipeline, X_train, y_train, X_test, y_test,
                      source_run, attempt_dir, *, final_native_threads, resume=False):
    """Run only an explicit budget-specific final fit and held-out evaluation.

    Supply the already selected, unfitted pipeline. Its typed identity must
    exactly match the handoff; parameters are never reconstructed from JSON.
    Changed data, common source/libraries or model settings are not migrations.
    A new named attempt may have different coordinator-only pools; its full
    runtime is then fixed for every execution of that attempt. Completed replay
    retains the historical runtime and returns metadata with model=None.
    A scoring-start record
    without a committed result blocks automatic retries, including new paths.
    """
    if type(resume) is not bool:
        raise ValueError('resume must be boolean')
    source_run, attempt_dir = _resolved_public_path(source_run), _resolved_public_path(attempt_dir)
    source_io = checkpoint_module._io_path(source_run)
    output = checkpoint_module._io_path(attempt_dir)
    if (source_io == output or source_io in output.parents
            or output in source_io.parents):
        raise ValueError('Source and final-attempt directories must not overlap')
    folder = source_io / 'cv-checkpoints'
    if not folder.is_dir() or not (folder / 'identity.json').is_file():
        _error('The source is not an existing checkpointed adaptive run')
    model = clone(final_pipeline)
    requested = phase_threads.validate_final_native_threads(final_native_threads, model)
    model_identity = checkpoint_module._stable(model)
    arrays = _own_data(X_train, y_train, X_test, y_test)
    with checkpoint_module._Lease(folder / 'session.lock'), threadpool_limits(limits=1):
        parent, parent_digest = _parent(folder)
        _completed_parent(folder, parent_digest)
        handoff, handoff_digest = _handoff(folder, parent, parent_digest)
        final_runtime = checkpoint_module._stable(_check_current(folder, parent, handoff, arrays))
        if not _same(model_identity, handoff['final_estimator']):
            _error('Supplied configured pipeline differs from the accepted final model')
        identity = {
            'schema': _ATTEMPT_SCHEMA, 'parent_identity_sha256': parent_digest,
            'handoff_sha256': handoff_digest, 'attempt_dir': str(attempt_dir),
            'final_native_threads': requested, 'final_estimator': model_identity,
            'training': handoff['training'], 'held_out': handoff['held_out'],
            'runtime': final_runtime, 'sources': handoff['sources'],
            'cv_execution_runtime_sha256': checkpoint_module._digest(parent['runtime']),
        }
        replay_complete = resume and (output / 'completed.json').exists()
        if replay_complete:
            # No new execution: authenticate the historical role pin, and keep
            # it in the identity while validating all current non-runtime inputs.
            saved_identity, _ = checkpoint_module._saved_identity(
                output / 'identity.json', schema=_ATTEMPT_SCHEMA)
            checkpoint_module._require_execution_compatible(
                parent['runtime'], saved_identity.get('runtime'))
            identity['runtime'] = saved_identity['runtime']
        digest = checkpoint_module._digest(identity)
        registration = {'identity_sha256': parent_digest, 'artifact': 'final_attempt_registration',
                        'attempt_identity_sha256': digest, 'attempt_dir': str(attempt_dir),
                        'handoff_sha256': handoff_digest}
        registration_path = folder / 'finalization' / 'attempts' / (digest + '.json')
        actor = {'kind': 'final_only_attempt', 'identity_sha256': digest,
                 'attempt_dir': str(attempt_dir)}
        if resume:
            saved = checkpoint_module._read_receipt(output / 'identity.json', digest)
            if not _same(saved.get('identity'), identity):
                _error('Final-attempt identity changed')
            saved_registration = checkpoint_module._read_receipt(registration_path, parent_digest)
            if not _same(saved_registration, registration):
                _error('Final-attempt registration changed')
            if (output / 'completed.json').exists():
                completed = checkpoint_module._read_receipt(output / 'completed.json', digest)
                claim = checkpoint_module._read_receipt(_barrier(folder), parent_digest)
                completed_invocation = completed.get('invocation')
                if (type(completed_invocation) is not str or len(completed_invocation) != 32
                        or any(c not in '0123456789abcdef' for c in completed_invocation)):
                    _error('Invalid completed final invocation identifier')
                refit = checkpoint_module._read_receipt(
                    output / 'refits' / (completed_invocation + '.json'), digest)
                if (completed.get('artifact') != 'completed_final_attempt'
                        or claim.get('artifact') != 'held_out_scoring_started'
                        or not _same(claim.get('actor'), actor)
                        or claim.get('handoff_sha256') != handoff_digest
                        or completed.get('handoff_sha256') != handoff_digest
                        or completed.get('parent_identity_sha256') != parent_digest
                        or completed.get('scoring_claim_sha256') != checkpoint_module._digest(claim)
                        or refit.get('artifact') != 'completed_final_refit'
                        or refit.get('invocation') != completed_invocation
                        or not _same(refit.get('execution_runtime'), identity['runtime'])
                        or refit.get('cv_execution_runtime_sha256') != identity['cv_execution_runtime_sha256']
                        or not _same(completed.get('refit'), refit)
                        or completed.get('refit_receipt_sha256') != checkpoint_module._digest(refit)):
                    _error('Completed final attempt does not match the shared scoring claim')
                return {'model': None, 'resumed_completed': True, 'completed': completed,
                        'attempt_dir': attempt_dir, 'source_run': source_run}
        elif output.exists():
            raise FileExistsError('Use a new final-attempt directory or explicit resume')
        _check_no_score(folder)
        # A corrected request may use a different named attempt after the
        # preceding coordinator releases this same source lease. Prior attempt
        # registrations and failure receipts are never deleted or re-keyed.
        phase_threads.preflight_final_native_threads(requested, model,
                                                     frozen_runtime=final_runtime)
        if not resume:
            output.mkdir(parents=True, exist_ok=False)
            _immutable(output / 'identity.json', {'identity_sha256': digest, 'identity': identity})
            _immutable(registration_path, registration)
        invocation = uuid.uuid4().hex
        phase = 'final_fit'
        try:
            _check_current(folder, parent, handoff, arrays, frozen_runtime=final_runtime)
            if not _same(checkpoint_module._stable(model), model_identity):
                _error('Configured pipeline changed before fitting')
            runtime_receipt = {}
            started = perf_counter()
            with phase_threads.final_fit_threads(requested, model, runtime_receipt,
                                                frozen_runtime=final_runtime):
                pipeline_started = perf_counter()
                model.fit(*arrays[:2])
                pipeline_fit_seconds = perf_counter() - pipeline_started
            fit_seconds = perf_counter() - started
            phase = 'final_diagnostics'
            _check_current(folder, parent, handoff, arrays, frozen_runtime=final_runtime)
            # This diagnostic extraction is strict in the original final path.
            # Secondary summaries remain optional and use the same helpers.
            refit = {
                'identity_sha256': digest, 'artifact': 'completed_final_refit',
                'invocation': invocation, 'state': 'completed',
                'selection': handoff['selection'], 'wall_seconds': fit_seconds,
                'pipeline_fit_seconds': pipeline_fit_seconds,
                'fit_management_seconds': fit_seconds - pipeline_fit_seconds,
                'diagnostics': checkpoint_module.diagnostics(model, arrays[1]),
                'native_runtime': runtime_receipt, 'execution_runtime': final_runtime,
                'cv_execution_runtime_sha256': checkpoint_module._digest(parent['runtime']),
            }
            refit['base_cache'] = _optional_report(
                lambda: work_reporting.summarize_cache_diagnostics(refit['diagnostics']))
            refit['numerical_checks'] = _optional_report(
                lambda: evaluation_reporting.numerical_check_summary(refit['diagnostics']))
            refit['selection_composition'] = _optional_report(
                lambda: evaluation_reporting.selection_composition(refit['diagnostics'], arrays[1]))
            _immutable(output / 'refits' / (invocation + '.json'), refit)
            phase = 'held_out_scoring'
            _check_current(folder, parent, handoff, arrays, frozen_runtime=final_runtime)
            claim = _reserve_score(folder, parent_digest, handoff_digest, actor)
            started = perf_counter()
            prediction = model.predict(arrays[2])
            prediction_finished = perf_counter()
            accuracy = float(accuracy_score(arrays[3], prediction))
            scoring_finished = perf_counter()
            scoring_seconds = scoring_finished - started
            classification = _optional_report(
                lambda: evaluation_reporting.classification_report(arrays[3], prediction))
            phase = 'completion_commit'
            _check_current(folder, parent, handoff, arrays, frozen_runtime=final_runtime)
            value = {
                'identity_sha256': digest, 'artifact': 'completed_final_attempt',
                'parent_identity_sha256': parent_digest, 'handoff_sha256': handoff_digest,
                'invocation': invocation, 'final_native_threads': requested,
                'selection': handoff['selection'], 'held_out_accuracy': accuracy,
                'fit_seconds': fit_seconds, 'scoring_seconds': scoring_seconds,
                'pipeline_fit_seconds': pipeline_fit_seconds,
                'fit_management_seconds': fit_seconds - pipeline_fit_seconds,
                'held_out_prediction_seconds': prediction_finished - started,
                'held_out_accuracy_scoring_seconds': scoring_finished - prediction_finished,
                'held_out_classification': classification,
                'refit': refit,
                'refit_receipt_sha256': checkpoint_module._digest(refit),
                'native_runtime': runtime_receipt,
                'scoring_claim_sha256': checkpoint_module._digest(claim),
                'cv_fits_performed': 0, 'full_pipeline_fit_calls': 1,
                'held_out_predict_calls': 1,
            }
            _immutable(output / 'completed.json', value)
            # A supplementary status failure cannot undo the committed result.
            try:
                write_json_atomic(output / 'status.json', {'state': 'completed',
                                  'identity_sha256': digest}, replace=True)
            except Exception:
                pass
            return {'model': model, 'resumed_completed': False, 'completed': value,
                    'attempt_dir': attempt_dir, 'source_run': source_run}
        except BaseException as error:
            _failure(output, digest, invocation, phase, error)
            raise
