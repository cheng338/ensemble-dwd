"""The final DWD objective averages loss over its actual selected rows."""

from decimal import Decimal, localcontext
import unittest

import numpy as np

from ensemble_dwd import EnsembleKernelDWD


def independent_rbf(X, gamma):
    with localcontext() as context:
        context.prec = 90
        g = Decimal.from_float(gamma)
        return np.array([[float((-g * sum(
            (Decimal.from_float(float(a)) - Decimal.from_float(float(b))) ** 2
            for a, b in zip(row, other))).exp()) for other in X] for row in X])


class SelectedObjectiveTests(unittest.TestCase):
    def test_conflicting_duplicates_use_selected_mean_and_free_intercept(self):
        X = np.array([[-2.], [-1.], [-1.], [-.3], [.3], [1.], [1.], [2.], [2.]])
        y = np.array([-1, -1, 1, -1, -1, 1, 1, -1, -1])
        for stopping, budget in (("fixed", 3), ("optimality", 6000)):
            with self.subTest(stopping=stopping):
                model = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=.67,
                    lambd=.04, gamma=.7, random_state=7, stopping=stopping,
                    max_iter=budget, tol=2e-8).fit(X, y)
                selected = model.selected_indices_
                self.assertEqual(len(selected), 6)
                self.assertLess(len(selected), len(y))
                final = model.final_estimator_
                K = independent_rbf(X[selected], .7)
                coefficients, intercept = final.dual_coef_[0], final.intercept_[0]
                margins = y[selected] * (K @ coefficients + intercept)
                # q=1 normalized DWD loss, coded independently of DWD helpers.
                losses = np.array([1 - u if u <= .5 else .25 / u for u in margins])
                penalty = .04 * (coefficients @ K @ coefficients)
                selected_objective = float(sum(losses) / len(selected) + penalty)
                wrong_original_denominator = float(sum(losses) / len(y) + penalty)
                self.assertAlmostEqual(final.final_objective_, selected_objective, delta=1e-12)
                self.assertGreater(abs(selected_objective - wrong_original_denominator), .01)
                if stopping == "fixed":
                    self.assertEqual(final.n_iter_, budget)
                    self.assertFalse(final.converged_)
                else:
                    self.assertTrue(final.converged_)


if __name__ == "__main__":
    unittest.main()
