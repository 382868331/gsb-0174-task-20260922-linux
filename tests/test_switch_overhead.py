"""Switch-overhead semantics in response-time analysis."""

import unittest

from rt_analysis import (
    ContextSwitchCost,
    CriticalSection,
    SwitchOverhead,
    Task,
    ValidationError,
    analyze,
)


class TestSwitchOverhead(unittest.TestCase):
    def test_zero_overhead_is_original_analysis(self):
        tasks = [
            Task("high", wcet=1, period=5, deadline=5),
            Task("low", wcet=2, period=10, deadline=10),
        ]

        self.assertEqual(analyze(tasks), analyze(tasks, switch_overhead=0))
        self.assertEqual(
            analyze(tasks, switch_overhead=0),
            analyze(tasks, switch_overhead=SwitchOverhead()),
        )

    def test_own_first_entry_is_charged_even_without_interference(self):
        report = analyze(
            [Task("solo", wcet=2, period=5, deadline=3)],
            switch_overhead=SwitchOverhead(first_entry=1, preemption=9, resumption=9),
        )

        task = report.tasks[0]
        self.assertTrue(report.guaranteed)
        self.assertEqual(task.response_time, 3)
        self.assertEqual(task.iterations[0].switch_overhead, 1)

    def test_blocking_remains_in_recurrence_with_overhead(self):
        tasks = [
            Task("high", wcet=1, period=5, deadline=5,
                 critical_sections=(CriticalSection("bus", 1),)),
            Task("low", wcet=3, period=20, deadline=20,
                 critical_sections=(CriticalSection("bus", 3),)),
        ]

        report = analyze(tasks, switch_overhead=SwitchOverhead(first_entry=1))

        self.assertEqual(report.tasks[0].blocking.amount, 3)
        self.assertEqual(report.tasks[0].response_time, 5)

    def test_overhead_inside_fixed_point_can_create_more_preemptions(self):
        # high: C=1, T=5; low: C=1, T=10, D=10.
        # Every scheduling event costs 1.  Each high job therefore adds
        # C_h=1 plus first-entry/preemption/resumption = 3.
        # R0 = C_low + first_entry = 2
        # R0 -> 1 + 1 + 3 = 6 (one high release)
        # R=6 spans two high releases -> 1 + 2*1 + 1 + 2*3 = 10 (stable).
        tasks = [
            Task("high", wcet=1, period=5, deadline=5),
            Task("low", wcet=1, period=10, deadline=10),
        ]

        report = analyze(tasks, switch_overhead=1)
        low = report.tasks[1]

        self.assertTrue(report.guaranteed)
        self.assertEqual(low.response_time, 10)
        self.assertEqual([r.value for r in low.iterations], [2, 6, 10])
        self.assertEqual([r.next_value for r in low.iterations], [6, 10, 10])
        self.assertEqual([r.switch_overhead for r in low.iterations], [4, 7, 7])

    def test_separate_overhead_components(self):
        tasks = [
            Task("high", wcet=1, period=20, deadline=20),
            Task("low", wcet=2, period=40, deadline=10),
        ]

        report = analyze(
            tasks,
            switch_overhead=SwitchOverhead(first_entry=1, preemption=2, resumption=3),
        )
        low = report.tasks[1]

        # R0 = 2 + 1; one high job contributes C=1 and 1 + 2 + 3 overhead.
        # R' = 2 + 1 + (1 + 1 + 2 + 3) = 10.
        self.assertEqual([r.value for r in low.iterations], [3, 10])
        self.assertEqual([r.switch_overhead for r in low.iterations], [7, 7])
        self.assertEqual(low.response_time, 10)

    def test_context_switch_cost_alias(self):
        tasks = [Task("solo", wcet=2, period=5, deadline=5)]

        by_name = analyze(tasks, context_switch_cost=1)
        by_object = analyze(tasks, context_switch_cost=ContextSwitchCost(1, 1, 1))

        self.assertEqual(by_name, by_object)
        self.assertIs(by_name.context_switch_cost, by_name.switch_overhead)

    def test_conflicting_overhead_kwargs_rejected(self):
        with self.assertRaises(ValidationError):
            analyze(
                [Task("solo", wcet=1, period=2, deadline=2)],
                context_switch_cost=1,
                switch_overhead=1,
            )

    def test_invalid_overhead(self):
        tasks = [Task("solo", wcet=1, period=2, deadline=2)]

        for invalid in (
            -1,
            SwitchOverhead(first_entry=-1),
            SwitchOverhead(preemption=-1),
            SwitchOverhead(resumption=-1),
            True,
            "1",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValidationError):
                    analyze(tasks, switch_overhead=invalid)


if __name__ == "__main__":
    unittest.main()
