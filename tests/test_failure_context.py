"""Failure facts remain small, optional, and independent of exception behavior."""

import json
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from ensemble_dwd._failure_context import (
    attach_failure_context, copy_failure_context, failure_context,
    failure_context_scope,
)


def context(**changes):
    result = dict(family="ensemble_dwd", stage="base_fit", base_position=2,
                  base_fit_calls_completed=1, base_score_calls_completed=1,
                  cached_base_learners=0, final_fit_calls_completed=0,
                  base_work_source="current")
    result.update(changes)
    return result


class Hostile:
    calls = 0

    def fail(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("user hook executed")

    __str__ = __repr__ = __iter__ = __len__ = __eq__ = __int__ = __float__ = fail


class HostileDict(dict):
    def items(self):
        raise AssertionError("dict hook executed")

    def __iter__(self):
        raise AssertionError("dict hook executed")


class HostileList(list):
    def __iter__(self):
        raise AssertionError("list hook executed")


class HostileException(RuntimeError):
    def __getattribute__(self, key):
        if key in ("ensemble_failure_context", "__dict__"):
            raise AssertionError("exception attribute hook executed")
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        raise AssertionError("exception assignment hook executed")


class HostileMetaclass(type):
    calls = 0

    def fail(cls, *args):
        HostileMetaclass.calls += 1
        raise AssertionError("metaclass hook executed")

    __eq__ = __hash__ = fail


class HostileTyped(metaclass=HostileMetaclass):
    pass


class FailureContextTests(unittest.TestCase):
    def setUp(self):
        Hostile.calls = 0
        self.y = np.array(["a", "b", "a", "b", "a"])
        self.classes = np.array(["a", "b"])

    def test_counts_align_to_class_order_and_integer_indices(self):
        error = ValueError("original")
        attach_failure_context(error, context(), y=self.y, classes=self.classes,
                               subset=np.array([1, 2]), selected=np.array([4, 2]))
        record = failure_context(error)
        self.assertEqual(record["classes"], ["a", "b"])
        self.assertEqual(record["training_class_counts"], [3, 2])
        self.assertEqual(record["subset_class_counts"], [1, 1])
        self.assertEqual(record["selected_class_counts"], [2, 0])
        self.assertEqual(record["schema"], 1)
        self.assertEqual(record["scope"], "binary_ensemble_fit")

    def test_reversed_class_order_is_preserved(self):
        error = ValueError()
        attach_failure_context(error, context(), y=self.y, classes=self.classes[::-1])
        self.assertEqual(failure_context(error)["training_class_counts"], [2, 3])

    def test_numeric_and_object_primitive_labels(self):
        for y in (np.array([-1, 1, -1]), np.array([False, True, False]),
                  np.array(["left", "right", "left"], dtype=object)):
            with self.subTest(dtype=y.dtype):
                error = RuntimeError()
                attach_failure_context(error, context(), y=y, classes=np.unique(y))
                record = failure_context(error)
                self.assertEqual(record["training_class_counts"], [2, 1])
                self.assertEqual(json.loads(json.dumps(record)), record)

    def test_oversize_labels_are_unknown_without_truncation(self):
        y = np.array(["x" * 200, "y"])
        error = ValueError()
        attach_failure_context(error, context(), y=y, classes=y)
        self.assertEqual(failure_context(error)["classes"], [None, "y"])
        self.assertEqual(failure_context(error)["training_class_counts"], [1, 1])

    def test_unknown_counts_do_not_become_zero(self):
        error = RuntimeError()
        attach_failure_context(error, context(base_fit_calls_completed=None,
                                              cached_base_learners=None,
                                              base_work_source="unknown"))
        record = failure_context(error)
        self.assertIsNone(record["base_fit_calls_completed"])
        self.assertIsNone(record["cached_base_learners"])
        self.assertNotIn("training_class_counts", record)

    def test_existing_inner_facts_win_and_outer_adds_missing(self):
        error = RuntimeError()
        attach_failure_context(error, context(base_position=None))
        attach_failure_context(error, context(stage="final_fit", base_position=7),
                               y=self.y, classes=self.classes)
        record = failure_context(error)
        self.assertEqual(record["stage"], "base_fit")
        self.assertIsNone(record["base_position"])
        self.assertEqual(record["training_class_counts"], [3, 2])

    def test_malformed_existing_value_is_replaced(self):
        error = RuntimeError()
        error.ensemble_failure_context = {"stage": "invented"}
        attach_failure_context(error, context())
        self.assertEqual(failure_context(error)["stage"], "base_fit")

    def test_fresh_copies_cannot_mutate_exception_record(self):
        error = RuntimeError()
        attach_failure_context(error, context(), y=self.y, classes=self.classes)
        first = failure_context(error)
        first["classes"][0] = "changed"
        first["training_class_counts"][0] = 100
        self.assertEqual(failure_context(error)["classes"], ["a", "b"])
        self.assertEqual(failure_context(error)["training_class_counts"], [3, 2])

    def test_original_identity_message_cause_and_args_are_preserved(self):
        cause = OSError("cause")
        error = RuntimeError("message", 9)
        error.__cause__ = cause
        try:
            raise error
        except RuntimeError as caught:
            old_traceback = caught.__traceback__
            attach_failure_context(caught, context())
            self.assertIs(caught, error)
            self.assertIs(caught.__traceback__, old_traceback)
            self.assertIs(caught.__cause__, cause)
            self.assertEqual(caught.args, ("message", 9))

    def test_interruptions_keep_identity(self):
        for error in (KeyboardInterrupt("pause"), SystemExit(4)):
            with self.subTest(kind=type(error)):
                attach_failure_context(error, context())
                self.assertEqual(failure_context(error)["stage"], "base_fit")
                self.assertIn(error.args, (("pause",), (4,)))

    def test_copy_preserves_bounded_context_and_pickle(self):
        source = ValueError("inner")
        target = RuntimeError("wrapper")
        target.__cause__ = source
        attach_failure_context(source, context(), y=self.y, classes=self.classes)
        copy_failure_context(source, target)
        self.assertEqual(failure_context(source), failure_context(target))
        restored = pickle.loads(pickle.dumps(target))
        self.assertEqual(failure_context(restored), failure_context(source))
        self.assertEqual(restored.args, ("wrapper",))

    def test_copy_preserves_target_existing_inner_record(self):
        source, target = ValueError(), RuntimeError()
        attach_failure_context(source, context(stage="final_fit"))
        attach_failure_context(target, context(stage="base_fit"))
        copy_failure_context(source, target)
        self.assertEqual(failure_context(target)["stage"], "base_fit")

    def test_reused_exception_gets_current_fit_facts(self):
        error = RuntimeError("reused")
        with failure_context_scope():
            attach_failure_context(error, context(stage="final_fit", base_position=None,
                                                  base_fit_calls_completed=3))
        with failure_context_scope():
            attach_failure_context(error, context(stage="base_fit", base_position=1,
                                                  base_fit_calls_completed=0))
        record = failure_context(error)
        self.assertEqual(record["stage"], "base_fit")
        self.assertEqual(record["base_position"], 1)
        self.assertEqual(record["base_fit_calls_completed"], 0)
        self.assertEqual(error.args, ("reused",))

    def test_current_scope_retains_inner_facts(self):
        error = RuntimeError()
        with failure_context_scope():
            attach_failure_context(error, context(stage="base_scoring"))
            attach_failure_context(error, context(stage="base_work"),
                                   y=self.y, classes=self.classes)
        record = failure_context(error)
        self.assertEqual(record["stage"], "base_scoring")
        self.assertEqual(record["training_class_counts"], [3, 2])

    def test_nested_propagation_preserves_inner_stage_and_exception(self):
        error = RuntimeError("nested")
        with failure_context_scope():
            try:
                with failure_context_scope():
                    attach_failure_context(error, context(stage="base_scoring"))
                    raise error
            except RuntimeError as caught:
                self.assertIs(caught, error)
                attach_failure_context(caught, context(stage="base_work"),
                                       y=self.y, classes=self.classes)
        record = failure_context(error)
        self.assertEqual(record["stage"], "base_scoring")
        self.assertEqual(record["training_class_counts"], [3, 2])

    def test_subsequent_nested_fit_resets_previous_failure(self):
        error = RuntimeError()
        with failure_context_scope():
            for stage in ("final_fit", "base_fit"):
                try:
                    with failure_context_scope():
                        attach_failure_context(error, context(stage=stage))
                        raise error
                except RuntimeError:
                    attach_failure_context(error, context(stage="base_work"))
                self.assertEqual(failure_context(error)["stage"], stage)

    def test_scope_marker_is_pickle_safe_but_not_part_of_record(self):
        error = RuntimeError("pickled")
        with failure_context_scope():
            attach_failure_context(error, context(stage="final_fit"))
        restored = pickle.loads(pickle.dumps(error))
        self.assertEqual(failure_context(restored), failure_context(error))
        self.assertNotIn("_ensemble_failure_context_owner", failure_context(restored))
        with failure_context_scope():
            attach_failure_context(restored, context(stage="base_fit"))
        self.assertEqual(failure_context(restored)["stage"], "base_fit")

    def test_copy_after_scope_exit_retains_source_current_facts(self):
        source, target = ValueError(), RuntimeError()
        with failure_context_scope():
            attach_failure_context(source, context(stage="base_scoring"))
        with failure_context_scope():
            copy_failure_context(source, target)
            attach_failure_context(target, context(stage="base_work"))
        self.assertEqual(failure_context(target)["stage"], "base_scoring")

    def test_scope_restores_parent_after_interrupt(self):
        error = KeyboardInterrupt("pause")
        with failure_context_scope():
            try:
                with failure_context_scope():
                    attach_failure_context(error, context(stage="base_scoring"))
                    raise error
            except KeyboardInterrupt as caught:
                self.assertIs(caught, error)
                attach_failure_context(caught, context(stage="base_work"))
        self.assertEqual(failure_context(error)["stage"], "base_scoring")
        with failure_context_scope():
            attach_failure_context(error, context(stage="final_fit"))
        self.assertEqual(failure_context(error)["stage"], "final_fit")

    def test_does_not_follow_cause(self):
        inner, outer = ValueError(), RuntimeError()
        attach_failure_context(inner, context())
        outer.__cause__ = inner
        self.assertIsNone(failure_context(outer))

    def test_hostile_exception_attribute_hooks_are_not_run(self):
        error = HostileException("failure")
        attach_failure_context(error, context())
        self.assertEqual(failure_context(error)["stage"], "base_fit")
        copy_failure_context(error, HostileException("wrapper"))

    def test_hostile_values_and_container_subclasses_are_rejected(self):
        error = RuntimeError()
        for value in (Hostile(), HostileDict(context()), {"classes": HostileList([0, 1])}):
            error.ensemble_failure_context = value
            self.assertIsNone(failure_context(error))
        attach_failure_context(error, HostileDict(context()))
        attach_failure_context(error, context(classes=HostileList([0, 1])))
        self.assertIsNone(failure_context(error))
        self.assertEqual(Hostile.calls, 0)

    def test_hostile_object_labels_do_not_run_comparison_or_repr(self):
        y = np.empty(1, dtype=object)
        y[0] = Hostile()
        error = RuntimeError()
        attach_failure_context(error, context(), y=y, classes=np.array([0, 1]))
        self.assertIsNone(failure_context(error)["training_class_counts"])
        attach_failure_context(error, context(), y=y, classes=y)
        self.assertEqual(Hostile.calls, 0)

    def test_user_metaclass_hash_and_equality_are_not_run(self):
        HostileMetaclass.calls = 0
        value = HostileTyped()
        error = RuntimeError()
        attach_failure_context(error, context(classes=[value]))
        self.assertIsNone(failure_context(error))
        y = np.empty(1, dtype=object)
        y[0] = value
        attach_failure_context(error, context(), y=y, classes=[0, 1])
        self.assertIsNone(failure_context(error)["training_class_counts"])
        attach_failure_context(error, context(), y=y, classes=y)
        self.assertEqual(HostileMetaclass.calls, 0)

    def test_invalid_field_values_are_rejected(self):
        changes = (
            dict(schema=True), dict(schema=2), dict(scope="other"),
            dict(family="other"), dict(stage="x" * 10000),
            dict(base_position=0), dict(base_fit_calls_completed=-1),
            dict(base_fit_calls_completed=True), dict(base_fit_calls_completed=2 ** 100),
            dict(base_fit_calls_completed=np.int64(1)), dict(final_fit_calls_completed=2),
            dict(base_work_source="maybe"), dict(classes=[0, 1, 2]),
            dict(classes=[float("inf")]), dict(classes=[2 ** 100]),
            dict(classes=["x" * 161]), dict(classes=[Hostile()]),
            dict(classes=[0, 1], training_class_counts=[1]),
            dict(training_class_counts=[1, 2]), dict(classes=[0, 1], subset_class_counts=[-1, 2]),
            dict(selected_class_counts=np.array([1, 2])), dict(extra="not allowed"),
        )
        for changeset in changes:
            with self.subTest(field=next(iter(changeset))):
                error = RuntimeError()
                attach_failure_context(error, context(**changeset))
                self.assertIsNone(failure_context(error))
        self.assertEqual(Hostile.calls, 0)

    def test_bad_indices_and_unmatched_classes_are_unknown(self):
        for indices in (np.array([-1]), np.array([5]), np.array([True, False]),
                        [0, 1], np.array([[0, 1]])):
            error = RuntimeError()
            attach_failure_context(error, context(), y=self.y, classes=self.classes,
                                   subset=indices)
            self.assertIsNone(failure_context(error)["subset_class_counts"])
        error = RuntimeError()
        attach_failure_context(error, context(), y=self.y, classes=["c", "d"])
        self.assertIsNone(failure_context(error)["training_class_counts"])

    def test_metadata_failures_never_replace_original(self):
        error = ValueError("original")
        for failure in (RuntimeError("metadata"), KeyboardInterrupt("metadata")):
            with patch("ensemble_dwd._failure_context._raw_classes", side_effect=failure):
                self.assertIsNone(attach_failure_context(error, context(), classes=self.classes))
            self.assertEqual(error.args, ("original",))

    def test_non_exception_input_is_ignored_without_hooks(self):
        arbitrary = Hostile()
        attach_failure_context(arbitrary, context())
        copy_failure_context(arbitrary, RuntimeError())
        self.assertIsNone(failure_context(arbitrary))
        self.assertEqual(Hostile.calls, 0)


if __name__ == "__main__":
    unittest.main()
