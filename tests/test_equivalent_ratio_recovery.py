"""Automatic retained-row equivalence composes with durable pair recovery."""
from pathlib import Path
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as cp
import pair_recovery as pr

def read(path): return json.loads(cp._io_path(path).read_text(encoding='utf-8'))
def payloads(folder): return [read(p)['payload'] for p in sorted(cp._io_path(folder/'cv-checkpoints/folds').glob('*.json'))]
def hashes(folder):
    folder=cp._io_path(folder)
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*.json')}
def strip_ratio(value):
    if isinstance(value,dict): return {k:strip_ratio(v) for k,v in value.items() if k!='sv_selection_ratio'}
    if isinstance(value,list): return [strip_ratio(v) for v in value]
    if isinstance(value,tuple): return tuple(strip_ratio(v) for v in value)
    return value

def pipeline():
    model=EnsembleKernelDWD(M=1,theta=1.,gamma=.2,random_state=42,
        sv_selection_ratio=.88,store_base_estimators=True,lambd=.1,max_iter=4,stopping='fixed')
    return Pipeline([('scale',StandardScaler()),('model',OvOClassifier(model,n_jobs=1))])

class RatioSensitive(ClassifierMixin,BaseEstimator):
    def __init__(self,sv_selection_ratio=.88): self.sv_selection_ratio=sv_selection_ratio
    def fit(self,X,y): self.classes_=np.unique(y); self.n_features_in_=X.shape[1]; return self
    def predict(self,X): return np.full(len(X),self.classes_[0 if self.sv_selection_ratio<.9 else -1])

class EquivalentRatioRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(912)
        cls.X=np.vstack([rng.normal(0,.12,(16,2))+c for c in [(-2,-1),(2,-1),(0,2),(0,-3)]])
        cls.y=np.repeat(np.arange(4),16)
        cls.folds=list(StratifiedKFold(2,shuffle=True,random_state=42).split(cls.X,cls.y))
        cls.grid={'model__estimator__sv_selection_ratio':[.88,.93]}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ratio-recovery-')
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.limit = threadpool_limits(1)
        self.addCleanup(self.limit.restore_original_limits)

    def session(self,name,*,reuse='auto',pair=True,resume=False,model=None,jobs=1):
        folder=self.output/name
        (folder/'failures').mkdir(parents=True,exist_ok=True)
        return cp.CheckpointedCV(folder,pipeline() if model is None else model,self.X,self.y,self.folds,
            jobs=jobs,error_policy='raise',reuse_equivalent_folds=reuse,pair_recovery=pair,resume=resume,
            provenance={'test_fixture':'equivalent-ratio-recovery','data_seed':912})

    def run_cv(self,name,grid=None,**kw):
        with self.session(name,**kw) as cv: return cv.search(self.grid if grid is None else grid)

    @contextmanager
    def capture(self,models):
        original=cp.eligible_accuracy
        def score(evaluator,X,y):
            result=original(evaluator,X,y)
            if evaluator.eligible_:
                model=evaluator.estimator_
                models.append({'state':strip_ratio(pr.snapshot(model)),
                    'prediction':model.predict(self.X).tolist(),
                    'scores':[e.decision_function(model.named_steps['scale'].transform(self.X)).tolist()
                        for e in model.named_steps['model'].estimators_]})
            return result
        with patch.object(cp,'eligible_accuracy',score): yield

    @contextmanager
    def interrupt(self):
        original=pr.FoldContext._pair_finished
        def finish(context,event):
            original(context,event)
            if context.fold==1 and len(context.events)==2:
                raise KeyboardInterrupt('controlled after two durable pair models')
        with patch.object(pr.FoldContext,'_pair_finished',finish): yield

    def test_01_exact_proof_all_pairs_and_controls(self):
        model=pipeline()
        for labels in (self.y[self.folds[0][0]],np.repeat(np.arange(4),[7,8,9,10])):
            seen={}
            for ratio in (.01,.125,.3,.5,.75,.8,.81,.82,.87,.88,.89,.9,.91,.92,.93,.94,.99,1.):
                proof=cp._fold_equivalence(clone(model).set_params(model__estimator__sv_selection_ratio=ratio),'a'*64,0,labels)
                self.assertIsNotNone(proof)
                counts=tuple(int((sum(labels==a)+sum(labels==b))*ratio) for a in np.unique(labels) for b in np.unique(labels) if a<b)
                if counts in seen: self.assertEqual(proof['key'],seen[counts])
                else:
                    self.assertNotIn(proof['key'],seen.values()); seen[counts]=proof['key']
        base=cp._fold_equivalence(model,'a'*64,0,self.y[self.folds[0][0]])
        for identity,fold in [('b'*64,0),('a'*64,1)]:
            self.assertNotEqual(base['key'],cp._fold_equivalence(model,identity,fold,self.y[self.folds[0][0]])['key'])
        for params in ({'model__estimator__random_state':43},{'model__estimator__gamma':.3},
                       {'model__estimator__lambd':.2},{'scale__with_std':False},
                       {'model__estimator__max_iter':5}):
            self.assertNotEqual(base['key'],cp._fold_equivalence(clone(model).set_params(**params),'a'*64,0,self.y[self.folds[0][0]])['key'])
        for ratio in (np.float64(.88),np.float32(.88)):
            self.assertIsNone(cp._fold_equivalence(clone(model).set_params(model__estimator__sv_selection_ratio=ratio),'a'*64,0,self.y))
        with patch.object(Pipeline,'get_params',side_effect=AssertionError('altered method queried')):
            self.assertIsNone(cp._fold_equivalence(model,'a'*64,0,self.y))

    def test_03_combined_parity_alias_provenance_and_final_refit(self):
        baseline,automatic=[],[]
        with self.capture(baseline): plain=self.run_cv('plain',reuse=False,pair=False)
        with self.capture(automatic): reused=self.run_cv('auto')
        self.assertEqual(len(baseline),4); self.assertEqual(len(automatic),2)
        self.assertEqual(baseline[0],baseline[2]); self.assertEqual(baseline[1],baseline[3])
        self.assertEqual(baseline[:2],automatic)
        for key in ('params','mean_test_score','std_test_score','rank_test_score','split0_test_score','split1_test_score'):
            np.testing.assert_equal(plain.cv_results_[key],reused.cv_results_[key])
        self.assertEqual(plain.best_index_,reused.best_index_)
        rows=payloads(self.output/'auto'); self.assertEqual(len(rows),4)
        aliases=[r for r in rows if 'equivalent_reuse' in r]
        reps=[r for r in rows if 'equivalent_reuse' not in r]
        self.assertEqual(len(aliases),2)
        for row in aliases:
            self.assertEqual((row['fit_time'],row['score_time']),(0.,0.))
            self.assertNotIn('pair_recovery',row)
            self.assertIn('source_receipt_sha256',row['equivalent_reuse'])
        self.assertEqual(sum(r['pair_recovery']['fitted_pairs'] for r in reps),12)
        before=hashes(self.output/'auto/cv-checkpoints/folds')
        with patch.object(pr,'fit_pipeline',side_effect=AssertionError('completed fold refitted')):
            self.run_cv('auto',resume=True)
        self.assertEqual(before,hashes(self.output/'auto/cv-checkpoints/folds'))
        self.assertFalse(list(cp._io_path(self.output/'auto').rglob('*.pkl')))
        # The CV proof does not apply to full-data retention counts.
        full=[]
        for ratio in (.88,.93):
            model=clone(pipeline()).set_params(model__estimator__sv_selection_ratio=ratio).fit(self.X,self.y)
            full.append(model)
        self.assertEqual([int(32*r) for r in (.88,.93)],[28,29])
        self.assertEqual([len(m.named_steps['model'].estimators_[0].selected_indices_) for m in full],[28,29])
        self.assertNotEqual(cp._fold_equivalence(full[0],'a'*64,0,self.y)['key'],
                            cp._fold_equivalence(full[1],'a'*64,0,self.y)['key'])

    def test_04_interruption_resume_partial_pairs_and_orphan_folds(self):
        observed=[]
        with self.capture(observed),self.interrupt(),self.assertRaises(KeyboardInterrupt):
            self.run_cv('interrupted')
        completed=hashes(self.output/'interrupted/cv-checkpoints/folds')
        self.assertEqual(len(completed),2)
        self.assertEqual(len(list(cp._io_path(self.output/'interrupted').rglob('*.pkl'))),2)
        self.assertGreater(len(list(cp._io_path(self.output/'interrupted/cv-checkpoints/pair-evidence').glob('*.json'))),0)
        with self.capture(observed): result=self.run_cv('interrupted',resume=True)
        after=hashes(self.output/'interrupted/cv-checkpoints/folds')
        self.assertTrue(all(after[k]==v for k,v in completed.items()))
        rows=payloads(self.output/'interrupted')
        active=[r for r in rows if r['fold']==1 and 'equivalent_reuse' not in r][0]
        self.assertEqual(active['pair_recovery']['recovered_pairs'],2)
        self.assertEqual(active['pair_recovery']['fitted_pairs'],4)
        expected=[]
        with self.capture(expected): self.run_cv('resume-reference',reuse=False,pair=False)
        self.assertEqual(observed,expected[:2])
        self.assertFalse(list(cp._io_path(self.output/'interrupted').rglob('*.pkl')))

    def test_05_custom_estimator_is_not_aliased(self):
        model=Pipeline([('scale',StandardScaler()),('model',RatioSensitive())])
        result=self.run_cv('custom',model=model,pair=False,grid={'model__sv_selection_ratio':[.88,.93]})
        rows=payloads(self.output/'custom')
        self.assertEqual(len(rows),4)
        self.assertTrue(all('equivalent_reuse' not in r for r in rows))
        self.assertTrue(all(r.get('equivalent_fold') is None for r in rows))

    def test_06_reordered_grid_adopts_partial_candidate(self):
        with self.interrupt(),self.assertRaises(KeyboardInterrupt):
            self.run_cv('reordered',grid={'model__estimator__sv_selection_ratio':[.93]})
        before=hashes(self.output/'reordered/cv-checkpoints/folds')
        self.run_cv('reordered',resume=True)
        after=hashes(self.output/'reordered/cv-checkpoints/folds')
        self.assertTrue(all(after[k]==v for k,v in before.items()))
        rows=payloads(self.output/'reordered')
        active=[r for r in rows if r['fold']==1 and 'equivalent_reuse' not in r][0]
        self.assertEqual(active['pair_recovery']['recovered_pairs'],2)
        self.assertEqual(active['pair_recovery']['fitted_pairs'],4)
        self.assertFalse(list(cp._io_path(self.output/'reordered').rglob('*.pkl')))

    def test_07_alias_reclaims_its_own_partial_pair_store(self):
        for index,ratio in enumerate((.93,.88)):
            with self.interrupt(),self.assertRaises(KeyboardInterrupt):
                self.run_cv('two-partials',resume=index>0,grid={'model__estimator__sv_selection_ratio':[ratio]})
        self.assertEqual(len(list(cp._io_path(self.output/'two-partials').rglob('*.pkl'))),4)
        self.run_cv('two-partials',resume=True)
        self.assertFalse(list(cp._io_path(self.output/'two-partials').rglob('*.pkl')))
        rows=payloads(self.output/'two-partials')
        rep=[r for r in rows if r['fold']==1 and 'equivalent_reuse' not in r][0]
        self.assertEqual(rep['pair_recovery']['recovered_pairs'],2)
        self.assertEqual(rep['pair_recovery']['fitted_pairs'],4)
        alias=[r for r in rows if r['fold']==1 and 'equivalent_reuse' in r][0]
        cleanup=read(self.output/'two-partials/cv-checkpoints/pair-cleanup'/(alias['task_sha256']+'.json'))['payload']
        self.assertNotEqual(cleanup['status'],'cleanup_deferred')
        self.assertNotIn('pair_recovery',alias)

    def test_08_parallel_groups_forward_recovery_and_commit_before_aliases(self):
        self.run_cv('parallel',jobs=2)
        rows=payloads(self.output/'parallel')
        self.assertEqual(len(rows),4)
        reps=[r for r in rows if 'equivalent_reuse' not in r]
        self.assertEqual(len(reps),2)
        self.assertEqual(sum(r['pair_recovery']['fitted_pairs'] for r in reps),12)
        self.assertEqual(len({r['pid'] for r in reps}),2)
        self.assertFalse(list(cp._io_path(self.output/'parallel').rglob('*.pkl')))

    def _sidecar_run(self, name, *, block_reports=False, interrupt=False):
        contexts, models = [], []
        original_init = pr.FoldContext.__init__
        original_mkdir = Path.mkdir
        original_finish = pr.FoldContext._pair_finished

        def initialize(context, *args, **kwargs):
            original_init(context, *args, **kwargs)
            contexts.append(context)

        def mkdir(path, *args, **kwargs):
            if block_reports and path.name in ('pair-evidence', 'pair-cleanup'):
                raise PermissionError('injected optional-report directory failure')
            return original_mkdir(path, *args, **kwargs)

        def finish(context, event):
            original_finish(context, event)
            if interrupt and context.fold == 1 and len(context.events) == 2:
                raise KeyboardInterrupt('preserve the primary interruption')

        with patch.object(pr.FoldContext, '__init__', initialize), \
                patch.object(Path, 'mkdir', mkdir), \
                patch.object(pr.FoldContext, '_pair_finished', finish), self.capture(models):
            if interrupt:
                with self.assertRaisesRegex(KeyboardInterrupt, 'preserve the primary interruption'):
                    self.run_cv(name, grid={}, reuse=False)
            else:
                self.run_cv(name, grid={}, reuse=False)

        folder = self.output / name
        records = payloads(folder)
        self.assertEqual(len(records), 1 if interrupt else 2)
        for record in records:
            path = folder / 'cv-checkpoints/folds' / (record['task_sha256'] + '.json')
            cp._valid_fold(cp._io_path(path), record['identity_sha256'], record['task_sha256'],
                           fold=record['fold'],
                           validation_size=len(self.folds[record['fold']][1]))
        for directory in ('pair-evidence', 'pair-cleanup'):
            sidecars = list(cp._io_path(folder / 'cv-checkpoints' / directory).glob('*.json'))
            self.assertEqual(len(sidecars), 0 if block_reports else 2)
        errors = [error for context in contexts for error in context.reporting_errors]
        self.assertEqual(bool(errors), block_reports)
        if block_reports:
            self.assertTrue(all('PermissionError' in str(error) for error in errors))
        return models

    def test_pair_sidecar_parents_are_created_and_reporting_failure_is_nonblocking(self):
        normal = self._sidecar_run('sidecars')
        denied = self._sidecar_run('sidecars-denied', block_reports=True)
        self.assertEqual(normal, denied)

    def test_sidecar_directory_failure_preserves_primary_interruption(self):
        self._sidecar_run('sidecars-interrupted', block_reports=True, interrupt=True)

    def test_confirmation_reuses_only_its_own_higher_budget_folds_with_recovery(self):
        from adaptive_search import AdaptiveSearchConfig
        from numerical_confirmation import NumericalConfirmationConfig, run_numerical_confirmation
        from search_support import bare_parameters

        original = self.run_cv('confirmation')
        result = deepcopy(original.cv_results_)
        result['params'] = [bare_parameters(parameters) for parameters in result['params']]
        rows = [{'stage': 'cv', 'stage_index': index, 'params': parameters}
                for index, parameters in enumerate(result['params'])]
        before = hashes(self.output / 'confirmation/cv-checkpoints/folds')
        report = run_numerical_confirmation(
            pipeline=pipeline(), results=result, rows=rows, searches={'cv': original},
            folds=self.folds, X=self.X, y=self.y, run_dir=self.output / 'confirmation',
            config=NumericalConfirmationConfig(max_iter=6, top_k=2),
            search_config=AdaptiveSearchConfig(regularization='lambd', prefix='model__estimator__'),
            jobs=1, error_policy='raise', resume=False, cv_seeds=None, seed_parameter=None,
            reuse_equivalent_folds='auto', parent_identity=payloads(self.output / 'confirmation')[0]['identity_sha256'],
            pair_recovery=True)
        self.assertEqual(before, hashes(self.output / 'confirmation/cv-checkpoints/folds'))
        self.assertEqual(report['checkpoint_summary']['executed_folds'], 2)
        self.assertEqual(report['checkpoint_summary']['equivalent_reused_folds'], 2)
        self.assertFalse(report['held_out_scored'])
        self.assertFalse(report['final_refit_performed'])
        original_rows = payloads(self.output / 'confirmation')
        higher_rows = payloads(self.output / 'confirmation/numerical-confirmation')
        self.assertTrue({row['identity_sha256'] for row in original_rows}.isdisjoint(
            {row['identity_sha256'] for row in higher_rows}))
        sources = {row['task_sha256'] for row in higher_rows if 'equivalent_reuse' not in row}
        for row in higher_rows:
            if 'equivalent_reuse' in row:
                self.assertIn(row['equivalent_reuse']['source_task_sha256'], sources)


if __name__ == '__main__':
    unittest.main()
