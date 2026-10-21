"""DWD action forwarding, old dependency refusal, and compact/cache metadata."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.base import clone
from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD
import ensemble_dwd.estimator as module

class LinearActionForwarding(unittest.TestCase):
    def test_old_dependency_and_old_pickle_parameter(self):
        old = EnsembleKernelDWD()
        del old.linear_action
        self.assertEqual(clone(old).linear_action, 'factor')
        def old_dependency(**kwargs):
            if 'linear_action' in kwargs:
                raise TypeError('unexpected keyword linear_action')
            return KernGDWD(**kwargs)
        with patch.object(module, 'KernGDWD', old_dependency):
            self.assertEqual(EnsembleKernelDWD()._make_estimator({'gamma': .3}).linear_action, 'factor')
            with self.assertRaises(ImportError) as caught:
                EnsembleKernelDWD(linear_action='inverse')._make_estimator({'gamma': .3})
            self.assertIsInstance(caught.exception.__cause__, TypeError)

    def test_actual_lean_cache_preserves_policy_and_no_inverse_state(self):
        X = np.array([[0.,1.],[1.,0.],[1.,1.],[2.,0.],[0.,2.],[2.,2.]])
        y = np.array([-1.,1.,-1.,1.,-1.,1.])
        options = dict(M=1, theta=1., sv_selection_ratio=1., lambd=.1,
                       gamma=.3, max_iter=3, stopping='fixed', random_state=2,
                       store_base_estimators=False)
        with tempfile.TemporaryDirectory() as directory:
            factor = EnsembleKernelDWD(**options, cache_dir=directory).fit(X, y)
            inverse = EnsembleKernelDWD(**options, cache_dir=directory, linear_action='inverse').fit(X, y)
            cached = EnsembleKernelDWD(**options, cache_dir=directory, linear_action='inverse').fit(X, y)
            self.assertNotEqual(factor.cache_key_, inverse.cache_key_)
            self.assertFalse(inverse.cache_hit_)
            self.assertTrue(cached.cache_hit_)
            self.assertEqual(inverse.requested_linear_action_, 'inverse')
            self.assertEqual(inverse.linear_action_, 'inverse')
            for model in (inverse, cached):
                self.assertIsNone(model.estimators_)
                diag = model.base_diagnostics_[0]['solver_diagnostics']
                self.assertEqual(diag['requested_linear_action'], 'inverse')
                self.assertEqual(diag['linear_action'], 'inverse')
                self.assertTrue(diag['linear_system_diagnostics']['inverse_ever_built'])
                self.assertNotIn('_full_inverse_action', vars(model.final_estimator_))
