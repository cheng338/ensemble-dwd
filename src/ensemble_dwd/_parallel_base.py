"""Ordered base fitting with one native thread per worker.

No numerical solver settings are changed. Worker timing sums are kept separate
from parent wall-clock stages. Only known main-thread, named-kernel base work
uses this path; caller subclasses/callables retain their sequential behavior.
The caller must own native-thread configuration during the fit: being on the
main Python thread does not exclude unrelated numerical threads in the process.
"""
from concurrent.futures import ThreadPoolExecutor, CancelledError, wait
from contextlib import nullcontext
from threading import Barrier, Lock, local, current_thread, main_thread
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from ._failure_context import attach_failure_context, uses_implementation
from .profiling import _active_profile, _stage, fit_profile, _record_parallel_profiles

_LOCAL=local()

def eligibility(model, kernel_kws, implementation):
    if model.base_jobs==1:return False,'sequential requested'
    if current_thread() is not main_thread():return False,'outer Python worker: preserve caller runtime'
    if not uses_implementation(model,'_make_estimator',implementation):return False,'custom learner factory'
    if callable(model.kernel):return False,'callable kernel'
    from .estimator import _contains_callable
    if _contains_callable(kernel_kws):return False,'callable kernel keyword'
    return True,'qualified native-one workers'

def _runtime():
    return threadpool_info()

def _native_one(runtime):
    return bool(runtime) and all(x.get('num_threads')==1 for x in runtime)

def parallel_base_stage(model,X,y,classes,subsets,kernel_kws):
    from .estimator import _binary_scores,_base_diagnostics,_overflow_safe_consensus
    workers=min(model.base_jobs,len(subsets))
    collect=_active_profile() is not None
    before=_runtime()
    lock=Lock();barrier=Barrier(workers)
    def initialize():
        with lock:_LOCAL.limit=threadpool_limits(1)
        barrier.wait(timeout=60)
    def one(position,indices):
        profile=fit_profile() if collect else None
        row=dict(position=position,fit_done=0,score_done=0,error=None,stage='base_setup')
        try:
            with profile if collect else nullcontext():
                runtime=_runtime()
                if not _native_one(runtime):raise RuntimeError('Parallel base worker requires native thread count one.')
                row['stage']='base_fit'
                with _stage('base_fit'):learner=model._make_estimator(kernel_kws).fit(X[indices],y[indices])
                row['fit_done']=1;row['stage']='base_validation'
                with _stage('base_validation'):
                    if not np.array_equal(learner.classes_,classes):raise ValueError('All DWD learners must use the same ordered binary labels.')
                row['stage']='base_scoring'
                with _stage('base_scoring'):scores=_binary_scores(learner,X)
                row['score_done']=1
                if _runtime()!=runtime:raise RuntimeError('Native runtime changed during parallel base work.')
                row.update(model=learner,scores=scores,runtime=runtime)
        except BaseException as error:row['error']=error
        finally:
            if collect:row['profile']=profile.as_dict()
        return row
    rows=[];failure=None;caller_failure=None;futures=[];collected=set()
    try:
        with _stage('parallel_base_work'),threadpool_limits(1),ThreadPoolExecutor(max_workers=workers,initializer=initialize) as pool:
            futures=[pool.submit(one,i,indices) for i,indices in enumerate(subsets,1)]
            try:
                for f in futures:
                    try:
                        # Windows pending Python signals need periodic returns
                        # to Python while native numerical work is in flight.
                        while not f.done():wait((f,),timeout=.05)
                        row=f.result()
                    except CancelledError:continue
                    rows.append(row);collected.add(f)
                    if row['error'] is not None and failure is None:
                        failure=row
                        for pending in futures:pending.cancel()
            except BaseException as error:
                caller_failure=error
                for pending in futures:pending.cancel()
            # All started work is joined before restoration and exception propagation.
    finally:
        # A caller interruption does not erase completed work or invent zero
        # counts. Harvest remaining started tasks only after the pool has joined.
        for f in futures:
            if f not in collected and f.done() and not f.cancelled():
                try:rows.append(f.result())
                except BaseException:pass
        rows.sort(key=lambda r:r['position'])
        if collect:_record_parallel_profiles([r['profile'] for r in rows if 'profile' in r])
    if caller_failure is not None:
        context=dict(family='ensemble_dwd',stage='base_work',base_position=None,
            base_fit_calls_completed=sum(r['fit_done'] for r in rows),base_score_calls_completed=sum(r['score_done'] for r in rows),
            cached_base_learners=0,final_fit_calls_completed=0,base_work_source='current')
        attach_failure_context(caller_failure,context,y=y,classes=classes)
        raise caller_failure
    if failure is not None:
        error=failure['error']
        context=dict(family='ensemble_dwd',stage=failure['stage'],base_position=failure['position'],
            base_fit_calls_completed=sum(r['fit_done'] for r in rows),base_score_calls_completed=sum(r['score_done'] for r in rows),
            cached_base_learners=0,final_fit_calls_completed=0,base_work_source='current')
        attach_failure_context(error,context,y=y,classes=classes,subset=subsets[failure['position']-1])
        raise error
    if _runtime()!=before:raise RuntimeError('Native runtime was not restored after parallel base work.')
    models=[r['model'] for r in rows]
    score_sum=np.zeros(len(y),dtype=np.float64)
    with _stage('base_consensus'),np.errstate(over='ignore',invalid='ignore'):
        for row in rows:score_sum+=row['scores']
        consensus=score_sum/model.M if np.isfinite(score_sum).all() else _overflow_safe_consensus(models,X,model.prediction_batch_size)
    with _stage('selection_sort'):order=np.argsort(np.abs(consensus),kind='stable')
    with _stage('base_summary'):
        return dict(consensus=consensus,order=order,models=models if model.store_base_estimators else None,
            diagnostics=[_base_diagnostics(m) for m in models],n_iter=np.asarray([m.n_iter_ for m in models],dtype=int),
            converged=np.asarray([m.converged_ for m in models],dtype=bool),
            execution=dict(requested_jobs=model.base_jobs,effective_jobs=workers,native_threads=1,policy='ordered-native-one-v1',
                           worker_runtimes=[r['runtime'] for r in rows],restored=True))
