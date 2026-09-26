"""Tests for the single added model: periodic last-Q supply windows.

The minimum supply function and its inverse are checked against an
independent brute-force enumeration over every arrival phase (not derived
from the code under test).  Response-time bounds are written down literally
from the hand recurrence
    demand(R) = C + B + sum_h ceil((R + J_h)/T_h) * C_h
    R = supply_time_for(demand(R))
and the integer small-period examples additionally cross-check every
arrival phase with the discrete scheduler (analytic bound >= observed worst
response; a proven miss really has a late job).
"""

import unittest
from fractions import Fraction

from rt_analysis import (
    SupplyWindow,
    Task,
    analyze,
    minimum_supply,
    simulate,
    supply_time_for,
)
from rt_analysis import ValidationError


# Independent reference: worst phase over a finite (>= 2P) availability belt.
def _brute_minimum_supply(period, window, t):
    available = [s % period >= period - window for s in range(4 * period)]
    return min(
        sum(1 for s in range(phase, phase + t) if available[s])
        for phase in range(period)
    )


class TestMinimumSupplyFunction(unittest.TestCase):
    def test_matches_phase_enumeration(self):
        for period, window in ((5, 3), (8, 2), (10, 1), (7, 6), (3, 1), (4, 4)):
            sw = SupplyWindow(period, window)
            for t in range(0, 3 * period + 1):
                self.assertEqual(
                    minimum_supply(t, sw),
                    _brute_minimum_supply(period, window, t),
                    (period, window, t),
                )

    def test_first_blackout(self):
        # With P=5, Q=3 the processor is dark for the leading 2 slots:
        # nothing is guaranteed for t <= 2; one slot by t=3.
        sw = SupplyWindow(5, 3)
        self.assertEqual([minimum_supply(t, sw) for t in range(6)],
                         [0, 0, 0, 1, 2, 3])

    def test_not_the_rate_approximation(self):
        # t*Q/P = 1.2 at t=2 would falsely promise service; the true
        # worst-phase guarantee over the first blackout is 0.
        sw = SupplyWindow(5, 3)
        self.assertEqual(minimum_supply(2, sw), 0)

    def test_inverse_is_exact(self):
        for period, window in ((5, 3), (8, 2), (10, 1), (7, 6)):
            sw = SupplyWindow(period, window)
            for demand in range(1, 25):
                t = supply_time_for(demand, sw)
                self.assertGreaterEqual(minimum_supply(t, sw), demand)
                self.assertLess(minimum_supply(t - 1, sw), demand)

    def test_inverse_hand_values(self):
        sw = SupplyWindow(5, 3)
        # demand 1 can be served only at slot 3 (first open slot): t=3.
        # demand 3 needs the full first window: t=5; demand 4 waits out the
        # next two-slot blackout: t=8.
        self.assertEqual([supply_time_for(d, sw) for d in range(1, 5)],
                         [3, 4, 5, 8])

    def test_full_window_is_always_available(self):
        sw = SupplyWindow(4, 4)
        for t in range(20):
            self.assertEqual(minimum_supply(t, sw), t)
        for d in range(1, 10):
            self.assertEqual(supply_time_for(d, sw), d)

    def test_none_supply_matches_full_window(self):
        for t in range(10):
            self.assertEqual(minimum_supply(t, None), t)


class TestIterationRecords(unittest.TestCase):
    """Schedulable example, hand-derived.

    Supply P=5, Q=3 (open slots 2,3,4 of each period); h: C=1, T=8;
    l: C=2, T=16, D=16.
      h: demand 1 -> supply_time_for(1)=3 <= 8: R_h = 3 (stable at round 1).
      l: start R = supply_time_for(2) = 4
         R=4: ceil(4/8)=1 -> demand 3, sbf(4)=2, R'=5
         R=5: ceil(5/8)=1 -> demand 3, sbf(5)=3, R'=5 -> R_l = 5.
    """

    @classmethod
    def setUpClass(cls):
        cls.tasks = [Task("h", 1, 8, 8), Task("l", 2, 16, 16)]
        cls.report = analyze(cls.tasks, supply=SupplyWindow(5, 3))

    def test_guaranteed(self):
        self.assertTrue(self.report.guaranteed)
        self.assertEqual(self.report.utilization, Fraction(1, 8) + Fraction(2, 16))

    def test_response_times(self):
        self.assertEqual(
            [t.response_time for t in self.report.tasks], [3, 5]
        )

    def test_every_round_carries_demand_supply_and_stop_reason(self):
        low = self.report.tasks[1]
        self.assertEqual(
            [(r.value, r.demand, r.supply, r.next_value) for r in low.iterations],
            [(4, 3, 2, 5), (5, 3, 3, 5)],
        )
        self.assertIsNone(low.iterations[0].stop_reason)
        self.assertIn("fixed point", low.iterations[-1].stop_reason)


class TestDeadlineMissVsBudget(unittest.TestCase):
    def test_low_utilization_still_misses_short_window(self):
        # U = 1/8 + 2/20 = 9/40, far below 1, yet a single unit of work can
        # only finish at the first open slot: supply P=8, Q=2 opens slots
        # 6 and 7, so even the highest task's response is 7 > D = 4.
        tasks = [Task("h", 1, 8, 4), Task("l", 2, 20, 20)]
        report = analyze(tasks, supply=SupplyWindow(8, 2))
        self.assertTrue(report.utilization_ok)
        self.assertFalse(report.guaranteed)
        high = report.tasks[0]
        self.assertEqual(high.status, "deadline_miss")
        self.assertIsNone(high.response_time)
        self.assertEqual(high.iterations[-1].next_value, 7)
        self.assertIn("D-J", high.iterations[-1].stop_reason)

    def test_budget_exhaustion_is_still_distinct(self):
        tasks = [Task("h", 1, 4, 4), Task("l", 2, 20, 20)]
        report = analyze(tasks, supply=SupplyWindow(5, 3), max_iterations=1)
        low = report.tasks[1]
        self.assertEqual(low.status, "iteration_budget_exhausted")
        self.assertIsNone(low.response_time)
        self.assertIn("budget", low.iterations[-1].stop_reason)
        # With the normal budget the same set converges to R = 8:
        # R=4 -> demand 3 -> 5; R=5 -> 2 interferers -> demand 4 -> 8;
        # R=8 -> demand 4 -> 8 (stable).
        full = analyze(tasks, supply=SupplyWindow(5, 3))
        self.assertEqual(full.tasks[1].response_time, 8)


class TestDegenerationQEqualsP(unittest.TestCase):
    def test_q_equals_p_reproduces_baseline(self):
        tasks = [Task("a", 2, 8, 8), Task("b", 3, 12, 12)]
        baseline = analyze(tasks)
        full_window = analyze(tasks, supply=SupplyWindow(6, 6))
        for base, win in zip(baseline.tasks, full_window.tasks):
            self.assertEqual(base.response_time, win.response_time)
            self.assertEqual(base.status, win.status)
        # Rounds also carry demand/supply fields, equal to the plain demand.
        for rec in full_window.tasks[1].iterations:
            self.assertEqual(rec.supply, rec.value)
            self.assertEqual(rec.demand, rec.next_value)


class TestPhaseEnumerationCrossCheck(unittest.TestCase):
    """Integer small-period schedules over every arrival phase."""

    def cross_check(self, tasks, supply, period):
        report = analyze(tasks, supply=supply)
        horizon = 3 * period
        result = simulate(
            tasks, horizon, supply=supply, phases=range(period)
        )
        worst = result.max_response()
        for tr in report.tasks:
            if tr.status == "schedulable":
                self.assertLessEqual(worst[tr.name], tr.response_time)
        return report, result, worst

    def test_schedulable_example_is_tight(self):
        tasks = [Task("h", 1, 8, 8), Task("l", 2, 16, 16)]
        report, result, worst = self.cross_check(tasks, SupplyWindow(5, 3), 40)
        self.assertTrue(report.guaranteed)
        # The analytic bound is really attained by some arrival phase.
        self.assertEqual(worst["h"], 3)
        self.assertEqual(worst["l"], 5)

    def test_extremely_short_window_misses_and_really_runs_late(self):
        # Q=1, P=8: only one open slot per period.  U = 3/8 <= 1, but the
        # single task needs 2 slots and its deadline (3) falls inside the
        # first blackout; some phase makes it finish after slot 8.
        tasks = [Task("solo", 2, 16, 3)]
        report, result, worst = self.cross_check(tasks, SupplyWindow(8, 1), 16)
        self.assertEqual(report.tasks[0].status, "deadline_miss")
        self.assertGreater(worst["solo"], 3)
        # Worst of the 16 phases: a job released at slot 8 finishes at 24.
        self.assertEqual(worst["solo"], 16)

    def test_phases_collapse_without_supply(self):
        tasks = [Task("a", 1, 4, 4), Task("b", 2, 8, 8)]
        result = simulate(tasks, 16, phases=range(4))
        # Phases 1..3 produce genuinely shifted additional jobs; with no
        # supply window the plain schedule is exactly the baseline one.
        baseline = simulate(tasks, 16)
        self.assertEqual(
            sorted((j.task, j.release, j.response) for j in baseline.jobs),
            sorted((j.task, j.release, j.response)
                   for j in result.jobs if j.release % 4 == 0),
        )


class TestSupplyValidation(unittest.TestCase):
    def test_zero_window_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze([Task("t", 1, 4, 4)], supply=SupplyWindow(4, 0))
        self.assertIn("supply.window", str(ctx.exception))

    def test_window_larger_than_period_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([Task("t", 1, 4, 4)], supply=SupplyWindow(4, 5))

    def test_bool_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([Task("t", 1, 4, 4)], supply=SupplyWindow(True, 2))

    def test_wrong_type_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([Task("t", 1, 4, 4)], supply=(4, 2))

    def test_bad_phase_rejected(self):
        with self.assertRaises(ValidationError):
            simulate([Task("t", 1, 4, 4)], 8, supply=SupplyWindow(4, 2),
                     phases=(0, -1))
        with self.assertRaises(ValidationError):
            simulate([Task("t", 1, 4, 4)], 8, phases=())


if __name__ == "__main__":
    unittest.main()
