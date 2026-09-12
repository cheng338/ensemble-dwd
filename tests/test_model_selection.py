import unittest

from sklearn.model_selection import ParameterGrid
from ensemble_dwd.model_selection import mnist_param_grid, mnist_svm_param_grid


class ParameterGridTests(unittest.TestCase):
    def test_exact_default_grids(self):
        dwd = mnist_param_grid()
        svm = mnist_svm_param_grid()
        self.assertEqual(dwd["lambd"], [2.0 ** k for k in (-30, -22, -14, -6, 0)])
        self.assertEqual(svm["C"], [10.0 ** k for k in (-4, -2, 0, 2, 4)])
        for grid in (dwd, svm):
            self.assertEqual(grid["gamma"], [10.0 ** k for k in (-8, -6, -4, -2, 0)])
            self.assertEqual(grid["theta"], [0.1])
            self.assertEqual(grid["sv_selection_ratio"], [0.1, 0.15, 0.2, 0.25, 0.3])
            self.assertEqual(len(ParameterGrid(grid)), 125)

    def test_exact_broad_grid(self):
        grid = mnist_param_grid(broad=True)
        self.assertEqual(set(grid), {"lambd", "gamma", "theta", "sv_selection_ratio"})
        self.assertEqual(grid["lambd"], [2.0 ** k for k in
                                       [-30, -26, -22, -18, -14, -10, -6, -2, 0]])
        self.assertEqual(grid["gamma"], [10.0 ** k for k in range(-8, 1)])
        self.assertEqual(grid["theta"], [0.1])
        self.assertEqual(grid["sv_selection_ratio"], [0.1, 0.15, 0.2, 0.25, 0.3])
        self.assertEqual(len(ParameterGrid(grid)), 405)

    def test_default_is_subset_of_broad_with_same_endpoints(self):
        for factory in (mnist_param_grid, mnist_svm_param_grid):
            small, broad = factory(), factory(broad=True)
            for name, values in small.items():
                with self.subTest(factory=factory.__name__, parameter=name):
                    self.assertTrue(set(values).issubset(broad[name]))
                    self.assertEqual((values[0], values[-1]), (broad[name][0], broad[name][-1]))

    def test_pipeline_prefix_and_fresh_lists(self):
        for factory in (mnist_param_grid, mnist_svm_param_grid):
            for broad in (False, True):
                with self.subTest(factory=factory.__name__, broad=broad):
                    expected = factory(broad=broad)
                    plain = factory(broad=broad)
                    prefixed = factory("ensemble__", broad=broad)
                    self.assertEqual(prefixed, {"ensemble__" + k: v for k, v in expected.items()})
                    for values in plain.values():
                        values.clear()
                    self.assertEqual(factory(broad=broad), expected)
                    self.assertEqual(prefixed, {"ensemble__" + k: v for k, v in expected.items()})
                    for values in prefixed.values():
                        values.clear()
                    self.assertEqual(factory(broad=broad), expected)

    def test_invalid_prefix(self):
        for factory in (mnist_param_grid, mnist_svm_param_grid):
            for prefix in (None, 1, ["ensemble__"]):
                with self.subTest(factory=factory.__name__, prefix=prefix), self.assertRaises(TypeError):
                    factory(prefix)

    def test_svm_grid_matches_geometry_and_candidate_budget(self):
        for broad, count in ((False, 125), (True, 405)):
            with self.subTest(broad=broad):
                dwd = mnist_param_grid(broad=broad)
                svm = mnist_svm_param_grid(broad=broad)
                self.assertEqual(set(dwd), {"lambd", "gamma", "theta", "sv_selection_ratio"})
                self.assertEqual(set(svm), {"C", "gamma", "theta", "sv_selection_ratio"})
                c_exponents = range(-4, 5) if broad else range(-4, 5, 2)
                self.assertEqual(svm["C"], [10.0 ** k for k in c_exponents])
                self.assertEqual(len(svm["C"]), len(dwd["lambd"]))
                for name in ("gamma", "theta", "sv_selection_ratio"):
                    self.assertEqual(svm[name], dwd[name])
                self.assertEqual(len(ParameterGrid(svm)), count)
                self.assertEqual(len(ParameterGrid(dwd)), count)

    def test_broad_requires_boolean_keyword(self):
        for factory in (mnist_param_grid, mnist_svm_param_grid):
            for broad in (None, 0, 1, "yes", []):
                with self.subTest(factory=factory.__name__, broad=broad), self.assertRaises(TypeError):
                    factory(broad=broad)
            with self.subTest(factory=factory.__name__), self.assertRaises(TypeError):
                factory("", True)


if __name__ == "__main__":
    unittest.main()
