"""Analysis-semantics tests.

Every expected value below is hand-derived from the recurrence
    R = C_i + B_i + sum_h ceil((R + J_h) / T_h) * C_h
and written down literally; no expectation is computed by calling the code
under test.
"""

import unittest
from fractions import Fraction

from rt_analysis import CriticalSection, Task, analyze


def main_example():
    """Three tasks sharing resource 'bus'; 'log' is private to the lowest task.

    Hand computation (priority order = list order):
      U = 2/8 + 3/12 + 4/20 = 7/10 <= 1
      ceilings: bus -> 0 (used by all), log -> 2 (only t2)
      B0 = 3 (t2's bus section), B1 = 3 (t2's bus section), B2 = 0
      t0: limit D-J = 6; R0 = 2+3 = 5, no higher tasks -> R = 5
      t1: limit 12; R0 = 3+3 = 6
          R=6: ceil((6+1)/8)=1 -> 3+3+1*2 = 8
          R=8: ceil(9/8)=2     -> 3+3+2*2 = 10
          R=10: ceil(11/8)=2   -> 10 (stable) -> R = 10
      t2: limit 20; R0 = 4
          R=4:  ceil(5/8)=1,  ceil(4/12)=1  -> 4+2+3 = 9
          R=9:  ceil(10/8)=2, ceil(9/12)=1  -> 4+4+3 = 11
          R=11: ceil(12/8)=2, ceil(11/12)=1 -> 11 (stable) -> R = 11
    """
    return [
        Task("t0", wcet=2, period=8, deadline=7, jitter=1,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("t1", wcet=3, period=12, deadline=12,
             critical_sections=(CriticalSection("bus", 2),)),
        Task("t2", wcet=4, period=20, deadline=20,
             critical_sections=(CriticalSection("bus", 3), CriticalSection("log", 1))),
    ]


class TestMainExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analyze(main_example())

    def test_overall_guaranteed(self):
        self.assertTrue(self.report.guaranteed)
        self.assertTrue(self.report.utilization_ok)

    def test_exact_rational_utilization(self):
        self.assertEqual(self.report.utilization, Fraction(7, 10))

    def test_response_times(self):
        self.assertEqual([t.response_time for t in self.report.tasks], [5, 10, 11])
        self.assertTrue(all(t.status == "schedulable" for t in self.report.tasks))

    def test_blocking_terms_and_sources(self):
        b0, b1, b2 = (t.blocking for t in self.report.tasks)
        self.assertEqual((b0.amount, b0.source_task, b0.source_resource), (3, "t2", "bus"))
        self.assertEqual((b1.amount, b1.source_task, b1.source_resource), (3, "t2", "bus"))
        self.assertEqual((b2.amount, b2.source_task, b2.source_resource), (0, None, None))

    def test_private_resource_does_not_block(self):
        # 'log' is used only by t2, so its ceiling (2) never blocks t0/t1.
        for task in self.report.tasks[:2]:
            self.assertNotEqual(task.blocking.source_resource, "log")

    def test_iteration_chain_for_t1(self):
        rounds = self.report.tasks[1].iterations
        self.assertEqual([r.value for r in rounds], [6, 8, 10])
        self.assertEqual([r.next_value for r in rounds], [8, 10, 10])
        # Interference counts: ceil((R + J_t0) / T_t0) with J_t0 = 1, T_t0 = 8.
        self.assertEqual([dict(r.interference)["t0"] for r in rounds], [1, 2, 2])

    def test_iteration_chain_for_t2(self):
        rounds = self.report.tasks[2].iterations
        self.assertEqual([r.value for r in rounds], [4, 9, 11])
        self.assertEqual([dict(r.interference)["t0"] for r in rounds], [1, 2, 2])
        self.assertEqual([dict(r.interference)["t1"] for r in rounds], [1, 1, 1])


class TestJitterInducedMiss(unittest.TestCase):
    """Same task set, only the higher-priority jitter changes.

    h: C=2, T=4, D=4; l: C=2, T=8, D=5 (limit D-J = 5).
    J_h = 0: R0=2 -> ceil(2/4)=1 -> 4; R=4 -> ceil(4/4)=1 -> 4 (stable, ok).
    J_h = 2: R0=2 -> ceil(4/4)=1 -> 4; R=4 -> ceil(6/4)=2 -> 6 > 5 (miss).
    """

    def make(self, jitter):
        return [
            Task("h", wcet=2, period=4, deadline=4, jitter=jitter),
            Task("l", wcet=2, period=8, deadline=5),
        ]

    def test_schedulable_without_jitter(self):
        report = analyze(self.make(0))
        self.assertTrue(report.guaranteed)
        self.assertEqual(report.tasks[1].response_time, 4)

    def test_miss_with_jitter(self):
        report = analyze(self.make(2))
        low = report.tasks[1]
        self.assertEqual(low.status, "deadline_miss")
        self.assertIsNone(low.response_time)
        self.assertEqual([r.next_value for r in low.iterations], [4, 6])
        self.assertFalse(report.guaranteed)

    def test_utilization_below_one_is_not_sufficient(self):
        report = analyze(self.make(2))
        self.assertTrue(report.utilization_ok)  # U = 3/4 <= 1 ...
        self.assertFalse(report.guaranteed)     # ... yet not guaranteeable


class TestLowPriorityLongLock(unittest.TestCase):
    def make(self, deadline_t0):
        return [
            Task("t0", wcet=1, period=5, deadline=deadline_t0,
                 critical_sections=(CriticalSection("bus", 1),)),
            Task("t1", wcet=1, period=10, deadline=10),
            Task("t2", wcet=3, period=20, deadline=20,
                 critical_sections=(CriticalSection("bus", 3),)),
        ]

    def test_blocking_term_comes_from_lowest_task(self):
        report = analyze(self.make(5))
        blocking = report.tasks[0].blocking
        self.assertEqual(blocking.amount, 3)
        self.assertEqual((blocking.source_task, blocking.source_resource), ("t2", "bus"))
        # R_0 = C + B = 1 + 3 = 4 <= 5: schedulable despite the long lock.
        self.assertEqual(report.tasks[0].response_time, 4)
        self.assertTrue(report.guaranteed)

    def test_long_lock_can_cause_miss(self):
        report = analyze(self.make(3))
        top = report.tasks[0]
        self.assertEqual(top.status, "deadline_miss")  # R = 4 > D-J = 3
        self.assertFalse(report.guaranteed)

    def test_ceiling_filters_irrelevant_resources(self):
        # 'x' is used only below t0, so t0 cannot be blocked by it under PCP.
        tasks = [
            Task("t0", wcet=1, period=5, deadline=5),
            Task("t1", wcet=5, period=10, deadline=10,
                 critical_sections=(CriticalSection("x", 5),)),
            Task("t2", wcet=5, period=20, deadline=20,
                 critical_sections=(CriticalSection("x", 5),)),
        ]
        report = analyze(tasks)
        self.assertEqual(report.tasks[0].blocking.amount, 0)
        self.assertEqual(report.tasks[1].blocking.amount, 5)
        self.assertEqual(report.tasks[1].blocking.source_task, "t2")


class TestZeroBlockingAndExactDeadline(unittest.TestCase):
    def test_zero_blocking_without_resources(self):
        tasks = [
            Task("a", wcet=1, period=4, deadline=4),
            Task("b", wcet=2, period=8, deadline=8),
        ]
        report = analyze(tasks)
        for task in report.tasks:
            self.assertEqual(task.blocking.amount, 0)
            self.assertIsNone(task.blocking.source_task)
            self.assertIsNone(task.blocking.source_resource)
        # b: R0=2 -> ceil(2/4)=1 -> 3; R=3 -> ceil(3/4)=1 -> 3 (stable).
        self.assertEqual(report.tasks[1].response_time, 3)
        self.assertTrue(report.guaranteed)

    def test_exact_deadline_boundary_is_schedulable(self):
        # b: R0=3 -> ceil(3/4)=1 -> 4; R=4 -> ceil(4/4)=1 -> 4 = D-J exactly.
        tasks = [
            Task("a", wcet=1, period=4, deadline=4),
            Task("b", wcet=3, period=8, deadline=4),
        ]
        report = analyze(tasks)
        low = report.tasks[1]
        self.assertEqual(low.status, "schedulable")
        self.assertEqual(low.response_time, 4)
        self.assertTrue(report.guaranteed)

    def test_one_past_deadline_misses(self):
        tasks = [
            Task("a", wcet=1, period=4, deadline=4),
            Task("b", wcet=3, period=8, deadline=3),  # D-J = 3 < R = 4
        ]
        report = analyze(tasks)
        self.assertEqual(report.tasks[1].status, "deadline_miss")
        self.assertFalse(report.guaranteed)


class TestUtilization(unittest.TestCase):
    def test_utilization_above_one_is_reported_not_raised(self):
        report = analyze([Task("heavy", wcet=3, period=2, deadline=2)])
        self.assertFalse(report.utilization_ok)
        self.assertEqual(report.utilization, Fraction(3, 2))
        self.assertFalse(report.guaranteed)
        self.assertIn("utilization", report.summary)

    def test_exact_utilization_boundary(self):
        # U = 2/4 + 2/4 = 1 exactly: allowed, and this set converges.
        tasks = [
            Task("a", wcet=2, period=4, deadline=4),
            Task("b", wcet=2, period=4, deadline=4),
        ]
        report = analyze(tasks)
        self.assertEqual(report.utilization, Fraction(1, 1))
        self.assertTrue(report.utilization_ok)
        # b: R0=2 -> ceil(2/4)=1 -> 4; R=4 -> ceil(4/4)=1 -> 4 (stable, == D).
        self.assertEqual(report.tasks[1].response_time, 4)
        self.assertTrue(report.guaranteed)


class TestIterationBudget(unittest.TestCase):
    def test_budget_exhaustion_is_distinct_from_miss(self):
        # b needs two rounds: R0=2 -> 3 -> 3.  One round is not enough.
        tasks = [
            Task("a", wcet=1, period=4, deadline=4),
            Task("b", wcet=2, period=20, deadline=20),
        ]
        report = analyze(tasks, max_iterations=1)
        low = report.tasks[1]
        self.assertEqual(low.status, "iteration_budget_exhausted")
        self.assertIsNone(low.response_time)
        self.assertNotEqual(low.status, "deadline_miss")
        self.assertFalse(report.guaranteed)
        # With enough budget the same set provably converges to R = 3.
        self.assertEqual(analyze(tasks).tasks[1].response_time, 3)


class TestContextSwitchCost(unittest.TestCase):
    def test_zero_cost_preserves_original_analysis(self):
        report = analyze(main_example(), context_switch_cost=0)
        self.assertEqual([t.response_time for t in report.tasks], [5, 10, 11])

    def test_initial_dispatch_is_charged_once(self):
        report = analyze([Task("only", wcet=3, period=10, deadline=10)],
                         context_switch_cost=2)
        task = report.tasks[0]
        self.assertEqual(task.response_time, 5)
        self.assertEqual(task.iterations[0].next_value, 5)

    def test_blocking_is_retained_with_initial_dispatch(self):
        tasks = [
            Task("high", wcet=1, period=5, deadline=5,
                 critical_sections=(CriticalSection("bus", 1),)),
            Task("low", wcet=3, period=20, deadline=20,
                 critical_sections=(CriticalSection("bus", 3),)),
        ]
        report = analyze(tasks, context_switch_cost=1)
        high = report.tasks[0]
        self.assertEqual(high.blocking.amount, 3)
        self.assertEqual(high.response_time, 5)  # C=1 + B=3 + initial S=1

    def test_switch_cost_feeds_back_into_interference_count(self):
        # Formula with S=1:
        # R0 = C_l + S = 3
        # Each interference contributes C_h + 2S = 3.
        # R=3 -> 1 release  -> 6
        # R=6 -> 2 releases -> 9
        # R=9 -> 3 releases -> 12 (fixed point)
        tasks = [
            Task("high", wcet=1, period=4, deadline=4),
            Task("low", wcet=2, period=20, deadline=20),
        ]
        report = analyze(tasks, context_switch_cost=1)
        low = report.tasks[1]
        self.assertEqual(low.status, "schedulable")
        self.assertEqual(low.response_time, 12)
        self.assertEqual([r.value for r in low.iterations], [3, 6, 9, 12])
        self.assertEqual([r.next_value for r in low.iterations], [6, 9, 12, 12])

    def test_extra_preemptions_can_cause_deadline_miss(self):
        tasks = [
            Task("high", wcet=1, period=4, deadline=4),
            Task("low", wcet=2, period=20, deadline=11),
        ]
        report = analyze(tasks, context_switch_cost=1)
        low = report.tasks[1]
        self.assertEqual(low.status, "deadline_miss")
        self.assertIsNone(low.response_time)
        self.assertEqual([r.next_value for r in low.iterations], [6, 9, 12])


if __name__ == "__main__":
    unittest.main()
