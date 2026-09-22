"""Input-contract tests: every rejection must be a locatable ValidationError."""

import math
import unittest

from rt_analysis import CriticalSection, Task, ValidationError, analyze


def base_task(**overrides):
    fields = dict(name="t0", wcet=1, period=4, deadline=4, jitter=0)
    fields.update(overrides)
    return Task(**fields)


class TestScalarValidation(unittest.TestCase):
    def test_nan_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze([base_task(wcet=math.nan)])
        self.assertIn("wcet", str(ctx.exception))
        self.assertIn("t0", str(ctx.exception))

    def test_positive_infinity_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze([base_task(period=math.inf)])
        self.assertIn("period", str(ctx.exception))

    def test_negative_infinity_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([base_task(jitter=-math.inf)])

    def test_bool_rejected_for_integer_fields(self):
        for field in ("wcet", "period", "deadline", "jitter"):
            with self.assertRaises(ValidationError) as ctx:
                analyze([base_task(**{field: True})])
            self.assertIn(field, str(ctx.exception))
            self.assertIn("bool", str(ctx.exception))

    def test_non_integral_float_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([base_task(wcet=2.5)])

    def test_integral_float_accepted_as_int(self):
        # 4.0 is a finite integral number, not a correction of invalid data.
        report = analyze([base_task(period=4.0)])
        self.assertTrue(report.guaranteed)

    def test_non_numeric_type_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([base_task(wcet="1")])

    def test_deadline_must_not_exceed_period(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze([base_task(deadline=5, period=4)])
        self.assertIn("deadline", str(ctx.exception))

    def test_zero_or_negative_values_rejected(self):
        for field in ("wcet", "period", "deadline"):
            with self.assertRaises(ValidationError):
                analyze([base_task(**{field: 0})])
        with self.assertRaises(ValidationError):
            analyze([base_task(jitter=-1)])

    def test_error_message_locates_task_index(self):
        tasks = [base_task(), base_task(name="bad", wcet=0)]
        with self.assertRaises(ValidationError) as ctx:
            analyze(tasks)
        message = str(ctx.exception)
        self.assertIn("index 1", message)
        self.assertIn("bad", message)


class TestStructuralValidation(unittest.TestCase):
    def test_empty_task_set_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([])

    def test_more_than_twenty_tasks_rejected(self):
        tasks = [Task(name=f"t{i}", wcet=1, period=100, deadline=100) for i in range(21)]
        with self.assertRaises(ValidationError) as ctx:
            analyze(tasks)
        self.assertIn("20", str(ctx.exception))

    def test_exactly_twenty_tasks_accepted(self):
        tasks = [Task(name=f"t{i}", wcet=1, period=100, deadline=100) for i in range(20)]
        self.assertTrue(analyze(tasks).guaranteed)

    def test_duplicate_names_rejected(self):
        tasks = [base_task(), base_task()]
        with self.assertRaises(ValidationError) as ctx:
            analyze(tasks)
        self.assertIn("t0", str(ctx.exception))

    def test_non_task_element_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([{"name": "t0", "wcet": 1, "period": 4, "deadline": 4}])

    def test_non_iterable_rejected(self):
        with self.assertRaises(ValidationError):
            analyze(42)

    def test_empty_name_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([base_task(name="")])

    def test_max_iterations_validated(self):
        with self.assertRaises(ValidationError):
            analyze([base_task()], max_iterations=0)
        with self.assertRaises(ValidationError):
            analyze([base_task()], max_iterations=True)


class TestCriticalSectionValidation(unittest.TestCase):
    def test_zero_length_section_rejected(self):
        task = base_task(critical_sections=(CriticalSection("bus", 0),))
        with self.assertRaises(ValidationError) as ctx:
            analyze([task])
        self.assertIn("critical_sections[0]", str(ctx.exception))

    def test_bool_length_rejected(self):
        task = base_task(critical_sections=(CriticalSection("bus", True),))
        with self.assertRaises(ValidationError):
            analyze([task])

    def test_empty_resource_name_rejected(self):
        task = base_task(critical_sections=(CriticalSection("", 1),))
        with self.assertRaises(ValidationError):
            analyze([task])

    def test_sections_must_fit_in_wcet(self):
        task = base_task(
            wcet=2,
            critical_sections=(CriticalSection("a", 1), CriticalSection("b", 2)),
        )
        with self.assertRaises(ValidationError) as ctx:
            analyze([task])
        self.assertIn("wcet", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
