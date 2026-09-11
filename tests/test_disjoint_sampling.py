"""Exact disjoint-sampling and expected-failure contracts for both ensembles."""

from collections import Counter
import itertools
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError

from ensemble_dwd import (EnsembleKernelDWD, EnsembleKernelSVM,
                          IneligibleSelectionError, IneligibleSubsetError,
                          OvOClassifier)
from ensemble_dwd._cache import _cached_base_stage
from ensemble_dwd.estimator import _sample_subsets


class PermutationRNG:
    """Supply exactly one known proposal; a redraw is a test failure."""

    def __init__(self, permutation):
        self.values = np.asarray(permutation, dtype=int)
        self.calls = 0

    def permutation(self, n):
        self.calls += 1
        if self.calls != 1 or n != len(self.values):
            raise AssertionError('The sampler must request one full permutation.')
        return self.values


class RecordingLearner:
    """A zero-margin learner exposing both DWD and SVC diagnostic contracts."""

    fits = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, X, y):
        self.X = np.array(X, copy=True)
        self.classes_ = np.unique(y)
        self.n_iter_ = 1 if 'lambd' in self.parameters else np.array([1])
        self.converged_ = True
        self.fit_status_ = 0
        self.n_support_ = np.ones(2, dtype=int)
        type(self).fits.append(self)
        return self

    def decision_function(self, X):
        return np.zeros(len(X))

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


FAMILIES = ((EnsembleKernelDWD, 'ensemble_dwd.estimator', 'KernGDWD'),
            (EnsembleKernelSVM, 'ensemble_dwd.svm', 'SVC'))


class DisjointSamplingTests(unittest.TestCase):
    def test_one_permutation_exact_blocks_unused_remainder_and_input_immutability(self):
        y = np.arange(14) % 2
        before = y.copy()
        rng = PermutationRNG(np.arange(14))
        blocks = _sample_subsets(y, 4, 3, rng)
        self.assertEqual(rng.calls, 1)
        assert_array_equal(np.concatenate(blocks), np.arange(12))
        self.assertEqual(len(np.unique(np.concatenate(blocks))), 12)
        self.assertTrue(all(len(block) == 4 for block in blocks))
        assert_array_equal(y, before)
        blocks[0][0] = 99
        self.assertEqual(rng.values[0], 0)

    def test_exhaustive_uniform_proposals_preserve_every_valid_allocation(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        counts = Counter()
        rejected = 0
        for proposal in itertools.permutations(range(6)):
            expected = [proposal[:2], proposal[2:4]]
            valid = all(len(set(y[list(block)])) == 2 for block in expected)
            rng = PermutationRNG(proposal)
            if not valid:
                with self.assertRaises(IneligibleSubsetError):
                    _sample_subsets(y, 2, 2, rng)
                rejected += 1
            else:
                actual = _sample_subsets(y, 2, 2, rng)
                assert_array_equal(np.concatenate(actual), proposal[:4])
                counts[tuple(tuple(sorted(block)) for block in actual)] += 1
            self.assertEqual(rng.calls, 1)
        expected_allocations = set()
        for first in itertools.combinations(range(6), 2):
            for second in itertools.combinations(sorted(set(range(6)) - set(first)), 2):
                if all(len(set(y[list(block)])) == 2 for block in (first, second)):
                    expected_allocations.add((first, second))
        self.assertEqual(set(counts), expected_allocations)
        # Two row orderings per block and two unused-row orderings.
        self.assertEqual(set(counts.values()), {math.factorial(2) ** 3})
        self.assertEqual(len(counts), 36)
        self.assertEqual(rejected, 432)
        self.assertEqual(sum(counts.values()) + rejected, math.factorial(6))

    def test_labels_do_not_change_a_valid_proposal(self):
        first = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        second = np.array([0, 1, 1, 1, 0, 1, 0, 0])
        a = _sample_subsets(first, 3, 2, PermutationRNG(np.arange(8)))
        b = _sample_subsets(second, 3, 2, PermutationRNG(np.arange(8)))
        assert_array_equal(np.concatenate(a), np.concatenate(b))

    def test_structural_infeasibility_fails_before_random_draw(self):
        cases = [(np.arange(10) % 2, 3, 4, 'distinct'),
                 (np.array([0] * 6 + [1] * 2), 2, 3, 'each class'),
                 (np.array([0, 1]), 1, 1, 'at least two')]
        for y, size, M, message in cases:
            with self.subTest(size=size, M=M):
                rng = PermutationRNG(np.arange(len(y)))
                with self.assertRaisesRegex(IneligibleSubsetError, message):
                    _sample_subsets(y, size, M, rng)
                self.assertEqual(rng.calls, 0)

    def test_realized_single_class_blocks_are_not_redrawn(self):
        rng = PermutationRNG(np.arange(8))
        with self.assertRaisesRegex(IneligibleSubsetError, r'positions \[1, 2\]'):
            _sample_subsets(np.array([0] * 4 + [1] * 4), 2, 2, rng)
        self.assertEqual(rng.calls, 1)

    def test_subset_error_is_backward_compatible_with_selection_error_handlers(self):
        self.assertTrue(issubclass(IneligibleSubsetError, IneligibleSelectionError))
        self.assertTrue(issubclass(IneligibleSubsetError, ValueError))


class DisjointEstimatorTests(unittest.TestCase):
    def setUp(self):
        RecordingLearner.fits = []

    def test_both_families_share_identical_seeded_disjoint_blocks(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.arange(20) % 2
        expected = np.random.RandomState(17).permutation(20)[:18]
        observed = []
        for factory, module, learner in FAMILIES:
            with self.subTest(family=factory.__name__):
                model = factory(M=3, theta=.3, sv_selection_ratio=1., gamma=.2, random_state=17)
                with patch(module + '.' + learner, RecordingLearner):
                    model.fit(X, y)
                rows = np.concatenate(model.subset_indices_)
                assert_array_equal(rows, expected)
                self.assertEqual(len(np.unique(rows)), 18)
                observed.append(rows)
        assert_array_equal(*observed)

    def test_floor_minimum_and_capacity_are_applied_to_each_actual_input(self):
        cases = [(11, 5, .2, 2), (10, 5, .1, 2), (4, 1, 1., 4)]
        for factory, module, learner in FAMILIES:
            for n, M, theta, size in cases:
                with self.subTest(family=factory.__name__, n=n, M=M, theta=theta):
                    X, y = np.arange(2 * n, dtype=float).reshape(n, 2), np.arange(n) % 2
                    model = factory(M=M, theta=theta, sv_selection_ratio=1., gamma=.2)
                    with patch(module + '.' + learner, RecordingLearner), patch(
                            module + '._private_random_state', return_value=PermutationRNG(np.arange(n))):
                        model.fit(X, y)
                    self.assertEqual([len(block) for block in model.subset_indices_], [size] * M)
                    self.assertEqual(len(np.unique(np.concatenate(model.subset_indices_))), M * size)
            model = factory(M=5, theta=.3, sv_selection_ratio=1., gamma=.2)
            with patch.object(model, '_make_estimator') as fit, patch.object(model, '_base_result') as cache:
                with self.assertRaisesRegex(IneligibleSubsetError, '15 distinct'):
                    model.fit(np.arange(22, dtype=float).reshape(11, 2), np.arange(11) % 2)
                fit.assert_not_called()
                cache.assert_not_called()

    def test_each_subset_failure_clears_a_prior_fit_before_cache_or_learner_work(self):
        X, y = np.arange(24, dtype=float).reshape(12, 2), np.arange(12) % 2
        failed_cases = [({'M': 3, 'theta': .5}, y, 'distinct'),
                        ({}, np.array([0] * 11 + [1]), 'each class'),
                        ({}, np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]), 'single-class')]
        for factory, module, learner in FAMILIES:
            for options, bad_y, message in failed_cases:
                with self.subTest(family=factory.__name__, message=message):
                    model = factory(M=2, theta=.25, sv_selection_ratio=1., gamma=.2)
                    with patch(module + '.' + learner, RecordingLearner), patch(
                            module + '._private_random_state', return_value=PermutationRNG(np.arange(12))):
                        model.fit(X, y)
                    model.set_params(**options)
                    with patch.object(model, '_make_estimator') as fit, patch.object(
                            model, '_base_result') as cache, patch(
                            module + '._private_random_state', return_value=PermutationRNG(np.arange(12))):
                        with self.assertRaisesRegex(IneligibleSubsetError, message):
                            model.fit(X, bad_y)
                        fit.assert_not_called()
                        cache.assert_not_called()
                    self.assertFalse(any(name.endswith('_') for name in vars(model)))
                    with self.assertRaises(NotFittedError):
                        model.predict(X)

    def test_sampling_identifier_separates_old_and_new_base_cache_entries(self):
        X, y = np.arange(24, dtype=float).reshape(12, 2), np.arange(12) % 2
        subsets = [np.arange(3), np.arange(3, 6)]
        for factory, module, _ in FAMILIES:
            with self.subTest(family=factory.__name__), tempfile.TemporaryDirectory() as directory:
                model = factory(M=2, theta=.25, sv_selection_ratio=1., gamma=.2, cache_dir=directory)
                captured = {}

                def capture(cache_dir, key_data, compute):
                    captured.update(key_data)
                    return {}, False, 'captured'

                args = (X, y, np.unique(y), subsets,
                        {'gamma': .2} if factory is EnsembleKernelDWD else .2)
                with patch(module + '._runtime_provenance', return_value={'source': 'held fixed'}), patch(
                        module + '._cached_base_stage', side_effect=capture):
                    model._base_result(*args)
                self.assertEqual(captured['sampling'], 'disjoint-uniform-permutation-v1')
                legacy = {key: value for key, value in captured.items() if key != 'sampling'}
                old, _, old_key = _cached_base_stage(directory, legacy,
                    lambda: {'policy': 'old', 'fit_status': np.zeros(2, dtype=int)})
                new = {'policy': 'new', 'fit_status': np.zeros(2, dtype=int)}
                with patch(module + '._runtime_provenance', return_value={'source': 'held fixed'}), patch.object(
                        model, '_base_stage', return_value=new) as compute:
                    first, hit, key, _ = model._base_result(*args)
                    second, second_hit, second_key, _ = model._base_result(*args)
                self.assertEqual(old['policy'], 'old')
                self.assertEqual(first['policy'], second['policy'])
                self.assertEqual(first['policy'], 'new')
                self.assertFalse(hit)
                self.assertNotEqual(key, old_key)
                self.assertTrue(second_hit)
                self.assertEqual(key, second_key)
                compute.assert_called_once()

    def test_disjointness_is_within_each_multiclass_pair(self):
        X, y = np.arange(24, dtype=float).reshape(12, 2), np.repeat(['a', 'b', 'c'], 4)
        for factory, module, learner in FAMILIES:
            with self.subTest(family=factory.__name__):
                base = factory(M=2, theta=.5, sv_selection_ratio=1., gamma=.2, random_state=42)
                with patch(module + '.' + learner, RecordingLearner):
                    model = OvOClassifier(base, n_jobs=1).fit(X, y)
                for child in model.estimators_:
                    self.assertEqual(len(np.unique(np.concatenate(child.subset_indices_))), 8)
                # A training row legitimately belongs to more than one class pair.
                self.assertEqual(sum(0 in rows for rows in model.pair_indices_), 2)


if __name__ == '__main__':
    unittest.main()
