import pickle,tempfile,threading,unittest,time,_thread
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits,threadpool_info
from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD,OvOClassifier
from ensemble_dwd.profiling import fit_profile
from ensemble_dwd._failure_context import failure_context
import ensemble_dwd._parallel_base as parallel

def data(n=160,p=12):
    rng=np.random.RandomState(20261019);y=np.tile([0,1],n//2)
    return rng.normal(size=(n,p))*np.where(y[:,None]==0,.7,1.3),y
def model(**kw):
    args=dict(M=5,theta=.2,sv_selection_ratio=.6,lambd=.01,gamma=.05,random_state=51,
        max_iter=10000,stopping='optimality',acceleration='restart',base_jobs=4)
    args.update(kw);return EnsembleKernelDWD(**args)
def fitted_names(m):return [k for k in m.__dict__ if k.endswith('_')]

class ParallelBaseIntegrationTests(unittest.TestCase):
    def setUp(self):self.X,self.y=data();self.limiter=threadpool_limits(4);self.runtime=threadpool_info()
    def tearDown(self):self.assertEqual(threadpool_info(),self.runtime);self.limiter.restore_original_limits()
    def test_numerical_identity_clone_pickle_and_refit(self):
        m=model();fresh=clone(m);self.assertEqual(fresh.base_jobs,4)
        a=m.fit(self.X,self.y);b=model(base_jobs=1).fit(self.X,self.y)
        np.testing.assert_array_equal(a.selected_indices_,b.selected_indices_)
        np.testing.assert_array_equal(a.predict(self.X),b.predict(self.X))
        np.testing.assert_allclose(a.decision_function(self.X),b.decision_function(self.X),rtol=1e-10,atol=1e-10)
        self.assertTrue(all(a.converged_));self.assertEqual(a.base_execution_['effective_jobs'],4)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(a)).predict(self.X),a.predict(self.X))
        old=a.final_estimator_;a.fit(self.X,self.y);self.assertIsNot(old,a.final_estimator_)
        self.assertEqual(len(set(np.concatenate(a.subset_indices_))),len(np.concatenate(a.subset_indices_)))
    def test_old_unfitted_serialized_defaults(self):
        m=model();del m.base_jobs;m=pickle.loads(pickle.dumps(m));self.assertEqual(m.base_jobs,1)
        m.fit(self.X,self.y);self.assertEqual(m.base_execution_['effective_jobs'],1)
    def test_cache_miss_hit_separate_resource_key_fresh_final(self):
        with tempfile.TemporaryDirectory() as d:
            a=model(cache_dir=d).fit(self.X,self.y);self.assertFalse(a.cache_hit_)
            with fit_profile() as profile:b=model(cache_dir=d).fit(self.X,self.y)
            self.assertTrue(b.cache_hit_);self.assertEqual(a.cache_key_,b.cache_key_)
            self.assertIsNot(a.final_estimator_,b.final_estimator_)
            self.assertEqual(profile.as_dict()['counts']['reused_base_learners'],5)
            self.assertEqual(profile.as_dict()['counts']['base_fit_completed'],0)
            c=model(cache_dir=d,base_jobs=1).fit(self.X,self.y)
            self.assertFalse(c.cache_hit_);self.assertNotEqual(c.cache_key_,a.cache_key_)
            np.testing.assert_array_equal(a.predict(self.X),b.predict(self.X))
    def test_profile_counts_wall_and_worker_time_are_separate(self):
        with fit_profile() as p:m=model().fit(self.X,self.y)
        r=p.as_dict();self.assertEqual(r['counts']['base_fit_completed'],5)
        self.assertEqual(r['counts']['base_scoring_completed'],5);self.assertEqual(r['counts']['final_fit_completed'],1)
        self.assertEqual(len(r['parallel_worker_profiles']),5)
        self.assertNotIn('base_fit',r['stages'])
        self.assertIn('parallel_base_work',r['stages'])
        self.assertLessEqual(sum(s['self_seconds'] for s in r['stages'].values()),r['context_wall_seconds']*1.01)
    def test_storage_false_and_jobs_greater_than_M(self):
        m=model(base_jobs=12,store_base_estimators=False).fit(self.X,self.y)
        self.assertIsNone(m.estimators_);self.assertEqual(m.base_execution_['effective_jobs'],5)
    def test_custom_factory_preserved_sequentially(self):
        calls=[]
        class Custom(EnsembleKernelDWD):
            def _make_estimator(self,kws):calls.append(threading.get_ident());return super()._make_estimator(kws)
        m=Custom(**model().get_params()).fit(self.X,self.y)
        self.assertEqual(m.base_execution_['effective_jobs'],1)
        self.assertEqual(calls,[threading.get_ident()]*6)
    def test_callable_kernel_preserved_sequentially(self):
        from sklearn.metrics.pairwise import rbf_kernel
        def kern(X,Y):return rbf_kernel(X,Y,gamma=.05)
        m=model(kernel=kern).fit(self.X,self.y);self.assertEqual(m.base_execution_['effective_jobs'],1)
    def test_background_outer_thread_preserves_caller_runtime(self):
        box=[]
        def work():
            # Native one is essential on the known AOCL foreign-thread path.
            limiter=threadpool_limits(1)
            try:box.append(model().fit(self.X,self.y))
            finally:limiter.restore_original_limits()
        t=threading.Thread(target=work);t.start();t.join()
        self.assertEqual(len(box),1);self.assertEqual(box[0].base_execution_['effective_jobs'],1)
    def test_multiclass_clones_and_refits(self):
        X,y=data(180);y=np.arange(180)%3
        m=OvOClassifier(model(sv_selection_ratio=1.),n_jobs=1).fit(X,y)
        self.assertEqual(len(m.estimators_),3)
        self.assertTrue(all(x.base_jobs==4 for x in m.estimators_))
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(m)).predict(X),m.predict(X))
    def test_worker_errors_interruptions_metadata_and_cleanup(self):
        original=KernGDWD.fit
        for typ in (ValueError,KeyboardInterrupt):
            counts=[0];lock=threading.Lock();before={t.ident for t in threading.enumerate()};err=None
            def failing(instance,X,y,*args,**kw):
                with lock:counts[0]+=1;number=counts[0]
                if number==1:raise typ('injected worker failure')
                return original(instance,X,y,*args,**kw)
            m=model()
            with fit_profile() as p:
                try:
                    with patch.object(KernGDWD,'fit',failing):m.fit(self.X,self.y)
                except BaseException as e:err=e
            self.assertIsInstance(err,typ);self.assertEqual(fitted_names(m),[])
            self.assertEqual({t.ident for t in threading.enumerate()}-before,set())
            context=failure_context(err);self.assertEqual(context['stage'],'base_fit')
            self.assertGreaterEqual(context['base_position'],1);self.assertEqual(context['final_fit_calls_completed'],0)
            self.assertEqual(context['base_fit_calls_completed'],p.as_dict()['counts']['base_fit_completed'])
    def test_failed_cache_not_published(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(KernGDWD,'fit',side_effect=ValueError('fail')):
                with self.assertRaises(ValueError):model(cache_dir=d).fit(self.X,self.y)
            self.assertEqual(list(Path(d).rglob('*.joblib')),[])
    def test_parent_interruption_harvests_completed_workers(self):
        from concurrent.futures import Future
        original=Future.result;interrupted=[False];before={t.ident for t in threading.enumerate()}
        def stop_once(f,*a,**kw):
            if threading.current_thread() is threading.main_thread() and not interrupted[0]:
                interrupted[0]=True
                original(f,*a,**kw)
                raise KeyboardInterrupt('injected parent interruption')
            return original(f,*a,**kw)
        m=model();error=None
        with fit_profile() as p:
            try:
                with patch.object(Future,'result',stop_once):m.fit(self.X,self.y)
            except KeyboardInterrupt as e:error=e
        self.assertIsNotNone(error);self.assertEqual(fitted_names(m),[])
        self.assertEqual({t.ident for t in threading.enumerate()}-before,set())
        context=failure_context(error);self.assertEqual(context['stage'],'base_work')
        self.assertEqual(context['base_fit_calls_completed'],p.as_dict()['counts']['base_fit_completed'])
        self.assertGreater(context['base_fit_calls_completed'],0)
    def test_runtime_drift_refuses_without_fitted_state(self):
        original=parallel._runtime;local=threading.local()
        def drift():
            answer=original()
            if threading.current_thread() is not threading.main_thread():
                local.n=getattr(local,'n',0)+1
                if local.n>1:answer[0]=dict(answer[0],version='injected mutation')
            return answer
        m=model()
        with patch.object(parallel,'_runtime',drift):
            with self.assertRaisesRegex(RuntimeError,'runtime changed'):m.fit(self.X,self.y)
        self.assertEqual(fitted_names(m),[])
    def test_bad_jobs_rejected(self):
        for jobs in (True,0,-1,1.5,None):
            with self.assertRaises(ValueError):model(base_jobs=jobs).fit(self.X,self.y)
    def test_real_pending_python_interrupt_cancels_queued_work(self):
        started=threading.Event();lock=threading.Lock();count=[];old=KernGDWD.fit
        def slow(learner,*args,**kwargs):
            if threading.current_thread() is not threading.main_thread():
                with lock:
                    count.append(1)
                    if len(count)==4:started.set()
                time.sleep(.25)
            return old(learner,*args,**kwargs)
        def interrupt():
            if started.wait(timeout=5):_thread.interrupt_main()
        watcher=threading.Thread(target=interrupt);watcher.start();m=model()
        try:
            with patch.object(KernGDWD,'fit',slow),fit_profile() as p:
                with self.assertRaises(KeyboardInterrupt) as got:m.fit(self.X,self.y)
        finally:watcher.join()
        self.assertEqual(fitted_names(m),[])
        self.assertEqual(len(count),4)
        self.assertEqual(p.as_dict()['counts']['base_fit_completed'],4)
        self.assertEqual(failure_context(got.exception)['base_fit_calls_completed'],4)
    def test_explicit_backends_and_alternative_kernels(self):
        for backend,kernel in [('cholesky','rbf'),('spectral','rbf'),('lbfgs','rbf'),('cholesky','linear'),('cholesky','poly')]:
            with self.subTest(backend=backend,kernel=kernel):
                settings=dict(backend=backend,kernel=kernel,stopping='fixed',max_iter=10,acceleration=None)
                a=model(**settings).fit(self.X,self.y)
                b=model(base_jobs=1,**settings).fit(self.X,self.y)
                np.testing.assert_array_equal(a.selected_indices_,b.selected_indices_)
                np.testing.assert_array_equal(a.predict(self.X),b.predict(self.X))
                np.testing.assert_allclose(a.consensus_scores_,b.consensus_scores_,atol=1e-9,rtol=1e-9)
                self.assertEqual(a.base_execution_['effective_jobs'],4)


