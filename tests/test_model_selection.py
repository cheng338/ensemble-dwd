import unittest

from sklearn.model_selection import ParameterGrid
from ensemble_dwd.model_selection import mnist_param_grid, mnist_svm_param_grid


class ParameterGridTests(unittest.TestCase):
    def test_exact_broad_grid(self):
        grid = mnist_param_grid()
        self.assertEqual(set(grid), {"lambd", "gamma", "theta", "sv_selection_ratio"})
        self.assertEqual(grid["lambd"], [2.0 ** k for k in
                                       [-30, -26, -22, -18, -14, -10, -6, -2, 0]])
        self.assertEqual(grid["gamma"], [10.0 ** k for k in range(-8, 1)])
        self.assertEqual(grid["theta"], [0.1])
        self.assertEqual(grid["sv_selection_ratio"], [0.1, 0.15, 0.2, 0.25, 0.3])

    def test_pipeline_prefix_and_fresh_lists(self):
        plain = mnist_param_grid()
        prefixed = mnist_param_grid("ensemble__")
        self.assertEqual(prefixed, {"ensemble__" + k: v for k, v in plain.items()})
        plain["gamma"].clear()
        self.assertEqual(len(mnist_param_grid()["gamma"]), 9)
        self.assertEqual(len(prefixed["ensemble__gamma"]), 9)

    def test_invalid_prefix(self):
        for prefix in [None, 1, ["ensemble__"]]:
            with self.subTest(prefix=prefix), self.assertRaises(TypeError):
                mnist_param_grid(prefix)

    def test_svm_grid_matches_geometry_and_candidate_budget(self):
        dwd = mnist_param_grid()
        svm = mnist_svm_param_grid()
        self.assertEqual(set(svm), {"C", "gamma", "theta", "sv_selection_ratio"})
        self.assertEqual(svm["C"], [10.0 ** k for k in range(-4, 5)])
        self.assertEqual(len(svm["C"]), len(dwd["lambd"]))
        for name in ("gamma", "theta", "sv_selection_ratio"):
            self.assertEqual(svm[name], dwd[name])
        self.assertEqual(len(ParameterGrid(svm)), 405)
        self.assertEqual(len(ParameterGrid(svm)), len(ParameterGrid(dwd)))

    def test_svm_prefix_and_isolation(self):
        grid = mnist_svm_param_grid("ensemble__")
        self.assertEqual(grid, {"ensemble__" + k: v
                                for k, v in mnist_svm_param_grid().items()})
        for values in grid.values():
            values.clear()
        self.assertEqual(len(mnist_svm_param_grid()["gamma"]), 9)
        self.assertEqual(len(mnist_param_grid()["gamma"]), 9)
        for prefix in (None, 1, ["ensemble__"]):
            with self.subTest(prefix=prefix), self.assertRaises(TypeError):
                mnist_svm_param_grid(prefix)


if __name__ == "__main__":
    unittest.main()
