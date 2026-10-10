"""Exact, incremental close-coordinate disclosure without fits or candidate changes."""
import ast
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import struct
import sys
import unittest

import numpy as np
from joblib import hash as parameter_hash

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
from near_coordinates import THRESHOLD, axis_report, close_distinct, console_notice, disclose


def oracle(values, previous=()):
    old = set(previous)
    ordered = sorted(set(values) | old)
    return sum(0 < Fraction(b) - Fraction(a) <= THRESHOLD * Fraction(b)
               for i, a in enumerate(ordered) for b in ordered[i + 1:]
               if a not in old or b not in old)


class DisclosureTests(unittest.TestCase):
    def test_known_gamma_example(self):
        self.assertTrue(close_distinct(4.033432887412458e-7, 4.0334328874124586e-7))

    def test_equal_values_excluded(self):
        self.assertFalse(close_distinct(1., 1.))
        self.assertEqual(axis_report([1., 1., 1.])['close_distinct_pair_count'], 0)

    def test_all_pairs_not_just_neighbors(self):
        values = [1., math.nextafter(1., math.inf), math.nextafter(math.nextafter(1., math.inf), math.inf)]
        self.assertEqual(axis_report(values)['close_distinct_pair_count'], 3)

    def test_no_transitive_clustering(self):
        self.assertEqual(axis_report([1., 1.+8e-15, 1.+16e-15])['close_distinct_pair_count'], 2)

    def test_extreme_and_subnormal(self):
        smallest = math.ulp(0.)
        biggest = sys.float_info.max
        values = [smallest, 2*smallest, 3*smallest, sys.float_info.min,
                  math.nextafter(sys.float_info.min, 0.), 1., math.nextafter(1., math.inf),
                  math.nextafter(biggest, 0.), biggest]
        self.assertFalse(close_distinct(smallest, 2*smallest))
        self.assertTrue(close_distinct(math.nextafter(biggest, 0.), biggest))
        self.assertEqual(axis_report(values)['close_distinct_pair_count'], oracle(values))

    def test_invalid_rejected_without_replacement(self):
        for bad in (0., -1., math.inf, -math.inf, math.nan, True, '1'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                axis_report([bad])

    def test_details_bounded(self):
        values = [1. + i * math.ulp(1.) for i in range(40)]
        for limit in (0, 1, 6, 100):
            facts = axis_report(values, limit)
            self.assertEqual(facts['close_distinct_pair_count'], 780)
            self.assertEqual(len(facts['details']), min(limit, 780))
            self.assertEqual(facts['details_omitted'], 780-min(limit, 780))

    def test_axis_closeness_is_not_candidate_closeness(self):
        rows = [{'gamma': 1., 'lambd': 1.}, {'gamma': math.nextafter(1., math.inf), 'lambd': 100.}]
        result = disclose(rows, ('gamma', 'lambd'))
        self.assertEqual(result['axes']['gamma']['close_distinct_pair_count'], 1)
        self.assertEqual(result['conditional_pair_count'], 0)

    def test_non_axis_context_remains_distinct(self):
        rows = [{'gamma': 1., 'lambd': 2., 'seed': 1},
                {'gamma': math.nextafter(1., math.inf), 'lambd': 2., 'seed': 2}]
        self.assertEqual(disclose(rows, ('gamma', 'lambd'))['conditional_pair_count'], 0)

    def test_multi_axis_closeness_explicitly_outside_conditional_count(self):
        rows = [{'gamma': 1., 'lambd': 1.}, {'gamma': math.nextafter(1., math.inf), 'lambd': math.nextafter(1., math.inf)}]
        facts = disclose(rows, ('gamma', 'lambd'))
        self.assertEqual(facts['conditional_pair_count'], 0)
        self.assertEqual(sum(x['close_distinct_pair_count'] for x in facts['axes'].values()), 2)
        self.assertEqual(facts['whole_candidates']['close_distinct_pair_count'], 1)

    def test_bounded_whole_candidate_counts_unknown_when_unchecked(self):
        rows = [{'gamma': 1.+i*math.ulp(1.)} for i in range(40)]
        facts = disclose(rows, ('gamma',), max_pair_checks=100)
        self.assertEqual(facts['conditional_pair_count'], 780)
        self.assertEqual(facts['whole_candidates']['status'], 'not_checked_pair_limit')
        self.assertIsNone(facts['whole_candidates']['close_distinct_pair_count'])

    def test_whole_candidate_context_and_exact_duplicate(self):
        rows = [{'gamma': 1., 'seed': 1}, {'gamma': 1., 'seed': 1},
                {'gamma': math.nextafter(1., math.inf), 'seed': 1},
                {'gamma': math.nextafter(1., math.inf), 'seed': 2}]
        facts = disclose(rows, ('gamma',))
        self.assertEqual(facts['whole_candidates']['close_distinct_pair_count'], 1)

    def test_dense_whole_candidate_details_bounded(self):
        rows = [{'gamma': 1.+i*math.ulp(1.)} for i in range(40)]
        facts = disclose(rows, ('gamma',), max_details=3)
        self.assertEqual(facts['whole_candidates']['close_distinct_pair_count'], 780)
        self.assertEqual(len(facts['whole_candidates']['details']), 3)
        self.assertEqual(facts['whole_candidates']['details_omitted'], 777)

    def test_inputs_identity_order_unmodified(self):
        rows = [{'gamma': 1., 'lambd': 2., 'extra': [1, 2]},
                {'gamma': math.nextafter(1., math.inf), 'lambd': 2., 'extra': [1, 2]}]
        before = copy.deepcopy(rows)
        ids = [id(row) for row in rows]
        keys = [parameter_hash(row) for row in rows]
        facts = disclose(rows, ('gamma', 'lambd'))
        self.assertEqual(rows, before)
        self.assertEqual(ids, [id(row) for row in rows])
        self.assertEqual(keys, [parameter_hash(row) for row in rows])
        self.assertEqual(facts['conditional_pair_count'], 1)
        self.assertEqual(len(rows), 2)


    def test_exact_candidate_duplicates_not_close_distinct_pairs(self):
        rows = [{'gamma': 1.}, {'gamma': 1.}, {'gamma': math.nextafter(1., math.inf)}]
        facts = disclose(rows, ('gamma',))
        self.assertEqual(facts['candidate_count'], 3)
        self.assertEqual(facts['candidate_unique_identity_count'], 2)
        self.assertEqual(facts['conditional_pair_count'], 1)

    def test_random_binary64_oracle(self):
        rng = random.Random(42)
        for _ in range(200):
            bits = rng.randrange(1, 0x7feffffffffffff0)
            neighbors = [struct.unpack('>d', struct.pack('>Q', bits + rng.randrange(0, 12)))[0] for _ in range(14)]
            wide = [struct.unpack('>d', struct.pack('>Q', rng.randrange(1, 0x7ff0000000000000)))[0] for _ in range(6)]
            values = neighbors + wide
            self.assertEqual(axis_report(values)['close_distinct_pair_count'], oracle(values))

    def test_threshold_boundary(self):
        first = float(Fraction(1)-THRESHOLD)
        values = [math.nextafter(first, 0.), first, math.nextafter(first, 1.), 1.]
        for a in values[:-1]:
            self.assertEqual(close_distinct(a, 1.), Fraction(1)-Fraction(a) <= THRESHOLD)

    def test_live_helper_generated_path_example(self):
        path = EXAMPLES / 'adaptive_search.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        chosen = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('_outside', '_midpoint', '_rounded_geometric_midpoint')]
        self.assertEqual(len(chosen), 3)
        scope = {'math': math, 'struct': struct, 'Fraction': Fraction}
        exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), 'exec'), scope)
        # The current expansion and midpoint preserve represented distinctions.
        generated = scope['_outside']([1e-7], 'high')
        self.assertNotEqual(generated, 1e-6)
        self.assertTrue(close_distinct(generated, 1e-6))
        lower, upper = 1e-7, 1e-5
        midpoint = scope['_midpoint'](lower, upper)
        self.assertTrue(lower < midpoint < upper)
        self.assertTrue(midpoint == 1e-6 or close_distinct(midpoint, 1e-6))


class IncrementalDisclosureTests(unittest.TestCase):
    def setUp(self):
        self.a = 1.
        self.b = math.nextafter(self.a, math.inf)
        self.c = math.nextafter(self.b, math.inf)

    def test_old_old_pairs_never_repeat(self):
        previous = [{"gamma": self.a}, {"gamma": self.b}]
        report = disclose(previous, ("gamma",), previous=previous)
        self.assertEqual(report["axes"]["gamma"]["close_distinct_pair_count"], 0)
        self.assertEqual(report["conditional_pair_count"], 0)
        self.assertEqual(report["whole_candidates"]["potential_pair_checks"], 0)
        self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], 0)
        self.assertIsNone(console_notice(report))

    def test_new_old_and_new_new_pairs_are_counted(self):
        report = disclose([{"gamma": self.b}, {"gamma": self.c}], ("gamma",),
                          previous=[{"gamma": self.a}])
        self.assertEqual(report["axes"]["gamma"]["close_distinct_pair_count"], 3)
        self.assertEqual(report["conditional_pair_count"], 3)
        self.assertEqual(report["whole_candidates"]["potential_pair_checks"], 3)
        self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], 3)
        self.assertEqual(report["new_candidate_unique_identity_count"], 2)

    def test_incremental_details_exclude_old_pairs_with_old_upper(self):
        report = axis_report([self.b], previous=[self.a, self.c])
        self.assertEqual(report["close_distinct_pair_count"], 2)
        pairs = {(row["lower"], row["upper"]) for row in report["details"]}
        self.assertEqual(pairs, {(self.a, self.b), (self.b, self.c)})
        self.assertEqual(report["details_omitted"], 0)

    def test_history_is_preserved_and_generators_work(self):
        previous = [{"gamma": self.a, "array": np.array([1, 2])}]
        current = [{"gamma": self.b, "array": np.array([1, 2])}]
        original = parameter_hash((previous, current))
        report = disclose(iter(current), iter(["gamma"]), previous=iter(previous))
        self.assertEqual(parameter_hash((previous, current)), original)
        self.assertEqual(report["conditional_pair_count"], 1)
        self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], 1)
        self.assertEqual(report["input_identity"],
                         parameter_hash((("gamma",), previous, current, 6, 250000)))
        json.dumps(report, allow_nan=False)

    def test_old_global_coordinates_can_form_new_context_pairs(self):
        history = [{"gamma": self.a, "seed": 1}, {"gamma": self.b, "seed": 2}]
        report = disclose([{"gamma": self.b, "seed": 1}], ("gamma",), previous=history)
        self.assertEqual(report["axes"]["gamma"]["close_distinct_pair_count"], 0)
        self.assertEqual(report["conditional_pair_count"], 1)
        self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], 1)
        self.assertIsNotNone(console_notice(report))

    def test_multiple_new_axes_are_whole_pairs_only(self):
        history = [{"gamma": self.a, "lambd": self.a}]
        report = disclose([{"gamma": self.b, "lambd": self.b}],
                          ("gamma", "lambd"), previous=history)
        self.assertEqual(report["conditional_pair_count"], 0)
        self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], 1)
        self.assertEqual(report["whole_candidates"]["details"][0]["differing_axes"],
                         ["gamma", "lambd"])

    def test_cap_counts_only_pairs_with_new_candidates(self):
        previous = [{"gamma": float(index)} for index in range(1, 1001)]
        report = disclose([{"gamma": self.b}], ("gamma",), previous=previous,
                          max_pair_checks=1000)
        self.assertEqual(report["whole_candidates"]["potential_pair_checks"], 1000)
        self.assertEqual(report["whole_candidates"]["status"], "complete")
        self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], 1)
        refused = disclose([{"gamma": self.b}], ("gamma",), previous=previous,
                           max_pair_checks=999)
        self.assertEqual(refused["conditional_pair_count"], 1)
        self.assertEqual(refused["whole_candidates"]["status"], "not_checked_pair_limit")
        self.assertIsNone(refused["whole_candidates"]["close_distinct_pair_count"])
        self.assertIn("unknown (comparison limit)", console_notice(refused))

    def test_duplicate_current_and_historical_identities(self):
        old = {"gamma": self.a}
        new = {"gamma": self.b}
        report = disclose([old, old, new, new], ("gamma",), previous=[old, old])
        self.assertEqual(report["candidate_count"], 4)
        self.assertEqual(report["candidate_unique_identity_count"], 2)
        self.assertEqual(report["new_candidate_unique_identity_count"], 1)
        self.assertEqual(report["whole_candidates"]["potential_pair_checks"], 1)
        self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], 1)

    def test_typed_identities_remain_distinct_from_coordinate_identity(self):
        previous = [{"gamma": float(self.a)}]
        current = [{"gamma": np.float64(self.a)}, {"gamma": self.b}]
        self.assertNotEqual(parameter_hash(previous[0]), parameter_hash(current[0]))
        report = disclose(current, ("gamma",), previous=previous)
        self.assertEqual(report["candidate_unique_identity_count"], 2)
        self.assertEqual(report["new_candidate_unique_identity_count"], 2)
        self.assertEqual(report["conditional_pair_count"], 1)
        self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], 2)

    def test_limits_validated_on_empty_inputs(self):
        for argument in ("max_details", "max_pair_checks"):
            for bad in (-1, 1., True, np.bool_(False), "6"):
                with self.subTest(argument=argument, bad=bad), self.assertRaises(ValueError):
                    disclose([], [], **{argument: bad})
        report = disclose([], [], max_details=np.int64(0), max_pair_checks=np.int32(0))
        self.assertEqual(report["whole_candidates"]["status"], "complete")
        json.dumps(report)

    def test_axes_and_missing_coordinates_validated(self):
        for axes in ("gamma", ["gamma", "gamma"], [""], [1]):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                disclose([], axes)
        with self.assertRaises(ValueError):
            disclose([{}], ["gamma"])
        with self.assertRaises(ValueError):
            disclose([[]], ["gamma"])
        with self.assertRaises(ValueError):
            disclose([], ["gamma"], previous=[{}])

    def test_binary64_scalar_validation(self):
        for value in (np.float32(1.25), np.float64(1.25), np.int64(2), Fraction(5, 4)):
            with self.subTest(value=value):
                self.assertEqual(axis_report([value])["unique_coordinate_count"], 1)
        for value in (np.array(1.), np.array([1.]), np.bool_(True),
                      2**53 + 1, np.int64(2**53 + 1), Fraction(1, 3)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                axis_report([value])
        if np.finfo(np.longdouble).nmant > 52:
            lossy = np.nextafter(np.longdouble(1.), np.longdouble(2.))
            with self.assertRaises(ValueError):
                axis_report([lossy])

    def test_axis_counts_match_incremental_exhaustive_oracle(self):
        rng = random.Random(513)
        for _ in range(60):
            values = [1. + rng.randrange(100) * math.ulp(1.) for _ in range(25)]
            history = [1. + rng.randrange(100) * math.ulp(1.) for _ in range(25)]
            report = axis_report(values, max_details=2, previous=history)
            self.assertEqual(report["close_distinct_pair_count"], oracle(values, history))
            self.assertLessEqual(len(report["details"]), 2)
            self.assertTrue(all(row["lower"] not in history or row["upper"] not in history
                                for row in report["details"]))

    def test_notice_silent_for_no_pairs_even_with_unknown_whole_count(self):
        report = disclose([{"gamma": 1.}, {"gamma": 100.}], ("gamma",), max_pair_checks=0)
        self.assertIsNone(report["whole_candidates"]["close_distinct_pair_count"])
        self.assertIsNone(console_notice(report))

    def test_input_identity_changes_with_stage_context_and_limits(self):
        current = [{"gamma": self.b}]
        baseline = disclose(current, ("gamma",))["input_identity"]
        variants = [disclose(current, ("gamma",), previous=[{"gamma": self.a}]),
                    disclose(current, ("gamma",), max_details=5),
                    disclose(current, ("gamma",), max_pair_checks=1)]
        self.assertTrue(all(item["input_identity"] != baseline for item in variants))

    def test_empty_stage_never_reports_historical_pairs(self):
        report = disclose([], ("gamma",), previous=[{"gamma": self.a}, {"gamma": self.b}])
        self.assertEqual(report["axes"]["gamma"]["close_distinct_pair_count"], 0)
        self.assertEqual(report["whole_candidates"]["potential_pair_checks"], 0)
        self.assertIsNone(console_notice(report))

    def test_incremental_whole_and_conditional_counts_match_pair_oracle(self):
        rng = random.Random(907)
        for _ in range(30):
            rows = [{"gamma": 1. + rng.randrange(100) * math.ulp(1.),
                     "lambd": 1. + rng.randrange(100) * math.ulp(1.),
                     "seed": rng.randrange(3)} for _ in range(30)]
            history, current = rows[:15], rows[10:]
            old_keys = {parameter_hash(row) for row in history}
            combined = list({parameter_hash(row): row for row in history + current}.items())
            expected_whole, expected_conditional, potential = 0, 0, 0
            for index, (first_key, first) in enumerate(combined):
                for second_key, second in combined[index + 1:]:
                    if (first_key in old_keys and second_key in old_keys) or first["seed"] != second["seed"]:
                        continue
                    potential += 1
                    differing = [axis for axis in ("gamma", "lambd") if first[axis] != second[axis]]
                    close = bool(differing) and all(
                        0 < abs(Fraction(first[axis]) - Fraction(second[axis]))
                        <= THRESHOLD * Fraction(max(first[axis], second[axis]))
                        for axis in differing)
                    expected_whole += close
                    expected_conditional += close and len(differing) == 1
            report = disclose(current, ("gamma", "lambd"), previous=history)
            self.assertEqual(report["whole_candidates"]["potential_pair_checks"], potential)
            self.assertEqual(report["whole_candidates"]["close_distinct_pair_count"], expected_whole)
            self.assertEqual(report["conditional_pair_count"], expected_conditional)

