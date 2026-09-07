import unittest

from ensemble_dwd.model_selection import mnist_param_grid


class ParameterGridTests(unittest.TestCase):
    def test_exact_broad_grid(self):
        grid = mnist_param_grid()
        self.assertEqual(set(grid), {"lambd", "gamma", "theta", "sv_selection_ratio"})
        self.assertEqual(grid["lambd"], [2.0 ** k for k in
                                       [-30, -26, -22, -18, -14, -10, -6, -2, 0]])
        self.assertEqual(grid["gamma"], [10.0 ** k for k in range(-8, 1)])
        self.assertEqual(grid["theta"], [0.1, 0.2])
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


if __name__ == "__main__":
    unittest.main()
