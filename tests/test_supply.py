"""Supply-window tests: minimum supply function, windowed response-time
bounds, and phase-enumeration simulation references.

Every expected value is hand-derived from
    sbf(t) = (u // P) * Q + min(Q, u mod P),  u = max(0, t - (P-Q))
    R    = supply_inverse(C_i + B_i + sum_h ceil((R + J_h)/T_h) * C_h)
and written down literally; nothing is computed by calling the code under
test.
"""

import unittest
from fractions import Fraction

from rt_analysis import (
    SupplyWindow,
    Task,
    ValidationError,
    analyze,
    simulate,
    simulate_phases,
    supply_bound,
    supply_inverse,
)

W41 = SupplyWindow(period=4, budget=1)   # blackout 3, window [3,4) etc.
W42 = SupplyWindow(period=4, budget=2)   # blackout 2, window [2,4) etc.
W44 = SupplyWindow(period=4, budget=4)   # degenerate: always available


class TestSupplyBound(unittest.TestCase):
    def test_blackout_gives_zero_supply(self):
        # P=4, Q=1: any interval of length <= 3 can miss the window entirely.
        self.assertEqual(supply_bound(0, W41), 0)
        self.assertEqual(supply_bound(3, W41), 0)

    def test_window_shape(self):
        # u = t-3; sbf = (u//4) + min(1, u%4): 0,1,1,1,1,2,2,2,2,3 for t=3..12
        expected = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3]
        self.assertEqual([supply_bound(t, W41) for t in range(3, 13)], expected)

    def test_full_period_always_supplies_q(self):
        # Any interval of length exactly P contains exactly Q window units.
        self.assertEqual(supply_bound(4, W41), 1)
        self.assertEqual(supply_bound(4, W42), 2)
        self.assertEqual(supply_bound(8, W42), 4)

    def test_degenerate_full_window_is_identity(self):
        for t in range(0, 17):
            self.assertEqual(supply_bound(t, W44), t)

    def test_not_average_rate(self):
        # Averaging C/(Q/P) would give 2 for one unit under P=4,Q=2; the
        # worst-case blackout forces 3 ([0,3) can contain only the unit at 2).
        self.assertEqual(supply_inverse(1, W42), 3)
        self.assertEqual(supply_bound(2, W42), 0)

    def test_inverse_values(self):
        # P=4, Q=2: t = 2 + (x-1)//2*4 + (x-1)%2 + 1 -> 3,4,7,8,11,12
        self.assertEqual([supply_inverse(x, W42) for x in range(1, 7)],
                         [3, 4, 7, 8, 11, 12])

    def test_inverse_degenerate_is_identity(self):
        for x in range(1, 10):
            self.assertEqual(supply_inverse(x, W44), x)

    def test_inverse_roundtrip(self):
        for x in range(1, 20):
            t = supply_inverse(x, W41)
            self.assertGreaterEqual(supply_bound(t, W41), x)
            self.assertLess(supply_bound(t - 1, W41), x)


class TestWindowedAnalysis(unittest.TestCase):
    def test_degenerate_matches_baseline(self):
        tasks = [
            Task("hi", wcet=2, period=8, deadline=7, jitter=1),
            Task("lo", wcet=3, period=12, deadline=12),
        ]
        base = analyze(tasks)
        win = analyze(tasks, supply=W44)
        self.assertEqual([r.response_time for r in win.tasks],
                        [r.response_time for r in base.tasks])
        self.assertEqual([r.status for r in win.tasks],
                        [r.status for r in base.tasks])
        for wr, br in zip(win.tasks, base.tasks):
            self.assertEqual([(i.value, i.next_value) for i in wr.iterations],
                             [(i.value, i.next_value) for i in br.iterations])

    def test_single_task_pays_blackout(self):
        # P=4, Q=1 -> supply_inverse(x) = 4x; C=2 alone needs R = 8.
        report = analyze([Task("a", wcet=2, period=8, deadline=8)], supply=W41)
        task = report.tasks[0]
        self.assertEqual(task.status, "schedulable")
        self.assertEqual(task.response_time, 8)
        (it,) = task.iterations
        self.assertEqual((it.value, it.demand, it.supply, it.next_value),
                         (8, 2, 2, 8))

    def test_low_utilization_still_misses(self):
        # U = 2/8 + 3/12 = 1/2, yet the blackout makes 'lo' miss.
        # lo: r0 = inv(3) = 12; ceil(12/8)=2 -> demand 3+2*2=7 -> inv(7)=28
        #     28 > D-J = 12 -> deadline_miss (proven, not budget).
        tasks = [
            Task("hi", wcet=2, period=8, deadline=8),
            Task("lo", wcet=3, period=12, deadline=12),
        ]
        report = analyze(tasks, supply=W41)
        self.assertEqual(report.utilization, Fraction(1, 2))
        self.assertTrue(report.utilization_ok)
        self.assertFalse(report.guaranteed)
        hi, lo = report.tasks
        self.assertEqual(hi.status, "schedulable")
        self.assertEqual(hi.response_time, 8)
        self.assertEqual(lo.status, "deadline_miss")
        self.assertIsNone(lo.response_time)
        (it,) = lo.iterations
        self.assertEqual((it.value, it.demand, it.supply, it.next_value),
                         (12, 7, 3, 28))
        self.assertEqual(dict(it.interference), {"hi": 2})

    def test_multi_round_convergence(self):
        # P=2, Q=1 -> supply_inverse(x) = 2x.
        # lo: r0 = inv(2) = 4; ceil(4/8)=1 -> demand 4 -> 8;
        #     ceil(8/8)=1 -> demand 4 -> 8 (stable) -> R = 8.
        tasks = [
            Task("hi", wcet=2, period=8, deadline=8),
            Task("lo", wcet=2, period=16, deadline=16),
        ]
        report = analyze(tasks, supply=SupplyWindow(period=2, budget=1))
        lo = report.tasks[1]
        self.assertEqual(lo.status, "schedulable")
        self.assertEqual(lo.response_time, 8)
        self.assertEqual([(i.value, i.demand, i.next_value) for i in lo.iterations],
                         [(4, 4, 8), (8, 4, 8)])

    def test_budget_exhausted_distinct_from_miss(self):
        tasks = [
            Task("hi", wcet=2, period=8, deadline=8),
            Task("lo", wcet=2, period=16, deadline=16),
        ]
        report = analyze(tasks, supply=SupplyWindow(period=2, budget=1),
                         max_iterations=1)
        lo = report.tasks[1]
        self.assertEqual(lo.status, "iteration_budget_exhausted")
        self.assertIsNone(lo.response_time)
        self.assertNotEqual(lo.status, "deadline_miss")


class TestSupplyValidation(unittest.TestCase):
    def test_budget_above_period_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([Task("a", 1, 4, 4)], supply=SupplyWindow(period=4, budget=5))

    def test_zero_budget_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([Task("a", 1, 4, 4)], supply=SupplyWindow(period=4, budget=0))

    def test_bool_budget_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([Task("a", 1, 4, 4)], supply=SupplyWindow(period=4, budget=True))

    def test_non_window_supply_rejected(self):
        with self.assertRaises(ValidationError):
            analyze([Task("a", 1, 4, 4)], supply="always")


class TestWindowedSimulation(unittest.TestCase):
    def test_degenerate_window_matches_baseline_schedule(self):
        tasks = [
            Task("hi", wcet=2, period=8, deadline=8),
            Task("lo", wcet=3, period=12, deadline=12),
        ]
        plain = simulate(tasks, 24)
        windowed = simulate(tasks, 24, supply=W44)
        self.assertEqual(plain.jobs, windowed.jobs)

    def test_observed_response_matches_bound(self):
        # P=4, Q=1, phase 0: job of 'a' (C=2, T=8) released at 0 executes at
        # t=3 and t=7 -> completes at 8 -> response 8, equal to the bound.
        tasks = [Task("a", wcet=2, period=8, deadline=8)]
        result = simulate(tasks, 16, supply=W41)
        self.assertEqual(result.max_response(), {"a": 8})
        bound = analyze(tasks, supply=W41).tasks[0].response_time
        self.assertEqual(result.max_response()["a"], bound)

    def test_phase_sweep_covers_alignment_cycle(self):
        # lcm(P=4, T=8) = 8 distinct phases; worst observed response is 8.
        tasks = [Task("a", wcet=2, period=8, deadline=8)]
        sweep = simulate_phases(tasks, 16, supply=W41)
        self.assertEqual(sweep.phases, tuple(range(8)))
        self.assertEqual(sweep.worst_response(), {"a": 8})
        self.assertEqual(sweep.worst_phase("a"), 0)
        for run in sweep.runs:
            self.assertLessEqual(run.max_response()["a"], 8)

    def test_phase_sweep_never_exceeds_bound(self):
        tasks = [
            Task("hi", wcet=2, period=8, deadline=8),
            Task("lo", wcet=2, period=16, deadline=16),
        ]
        window = SupplyWindow(period=2, budget=1)
        report = analyze(tasks, supply=window)
        sweep = simulate_phases(tasks, 32, supply=window)
        observed = sweep.worst_response()
        for task_report in report.tasks:
            self.assertLessEqual(observed[task_report.name],
                                 task_report.response_time)

    def test_low_utilization_miss_leaves_incomplete_jobs(self):
        # Same set as the analysis miss above: under P=4,Q=1 the 'lo' job
        # cannot finish within its period anywhere in the horizon.
        tasks = [
            Task("hi", wcet=2, period=8, deadline=8),
            Task("lo", wcet=3, period=12, deadline=12),
        ]
        result = simulate(tasks, 12, supply=W41)
        self.assertIn(("lo", 0), result.incomplete)

    def test_phase_sweep_requires_window(self):
        with self.assertRaises(ValidationError):
            simulate_phases([Task("a", 1, 4, 4)], 8, supply=None)


if __name__ == "__main__":
    unittest.main()

