"""Automatic policy composition and concrete dependency forwarding."""
import unittest
import numpy as np
from sklearn.base import clone
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier


class AutomaticResidualPolicyTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(21818).normal(size=(36, 4))
        self.y = np.tile([-3, 7], 18)

    def model(self, **options):
        params = dict(M=2, theta=.5, sv_selection_ratio=.6, lambd=.1,
                      gamma=.2, max_iter=4, stopping='fixed', random_state=31)
        params.update(options)
        return EnsembleKernelDWD(**params)

    def assert_policy(self, model, expected):
        self.assertEqual(model.residual_check_order_, expected)
        for learner in [*model.estimators_, model.final_estimator_]:
            self.assertEqual(learner.residual_check_order, expected)
            self.assertEqual(learner.diagnostics_['residual_check_order'], expected)

    def test_nine_policy_combinations_match_concrete_controls(self):
        for affine in ('auto', 'standard', 'joint'):
            for residual in ('auto', 'refinement_first', 'adaptive'):
                with self.subTest(affine=affine, residual=residual):
                    a = 'joint' if affine == 'auto' else affine
                    r = 'adaptive' if residual == 'auto' else residual
                    model = self.model(affine_computation=affine,
                                       residual_check_order=residual).fit(self.X, self.y)
                    explicit = self.model(affine_computation=a,
                                          residual_check_order=r).fit(self.X, self.y)
                    self.assert_policy(model, r)
                    self.assertEqual(model.requested_residual_check_order_, residual)
                    self.assertEqual(model.affine_computation_, a)
                    for field in ('selected_indices_', 'consensus_scores_'):
                        np.testing.assert_array_equal(getattr(model, field), getattr(explicit, field))
                    for field in ('dual_coef_', 'intercept_', 'objective_history_'):
                        np.testing.assert_array_equal(getattr(model.final_estimator_, field),
                                                      getattr(explicit.final_estimator_, field))
                    np.testing.assert_array_equal(model.decision_function(self.X),
                                                  explicit.decision_function(self.X))

    def test_all_backends_forward_concrete_dependency_policy(self):
        for backend in ('auto', 'cholesky', 'spectral', 'lbfgs'):
            with self.subTest(backend=backend):
                expected = 'adaptive' if backend in ('auto', 'cholesky') else 'refinement_first'
                self.assert_policy(self.model(backend=backend).fit(self.X, self.y), expected)

    def test_ovo_and_clone_preserve_requested_defaults(self):
        wrapper = clone(OvOClassifier(self.model(M=1, theta=1.), n_jobs=1))
        for field in ('affine_computation', 'residual_check_order'):
            self.assertEqual(wrapper.get_params()['estimator__' + field], 'auto')
        wrapper.fit(self.X, np.repeat(['a', 'b', 'c'], 12))
        for model in wrapper.estimators_:
            self.assert_policy(model, 'adaptive')
            self.assertEqual(model.requested_residual_check_order_, 'auto')
