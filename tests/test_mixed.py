"""Mixed-criticality analysis semantics (LO/HI modes, two budgets).

Every expected value below is hand-derived from the conservative
fixed-priority recurrences:

  LO mode (all tasks, higher-priority interference at C(LO)):
    R = C_i(LO) + sum_h ceil((R + J_h) / T_h) * C_h(LO)
  HI mode (high-criticality tasks only):
    R = C_i(HI)
        + sum_{h HI} ceil((R + J_h) / T_h) * C_h(HI)
        + sum_{l LO} ceil((R + J_l) / T_l) * C_l(LO)

The HI recurrence counts every higher-priority task over the whole window at
its largest budget (LO jobs dropped after a real switch are included), which
is provably safe for every switch time but not tight.  No expected number is
produced by calling the code under test.
"""

import math
import unittest
from fractions import Fraction

from rt_analysis import (
    CRITICALITY_HI,
    CRITICALITY_LO,
    MCTask,
    Task,
    ValidationError,
    analyze,
    analyze_mc,
)


def mc_example():
    """Priority order h1(HI) > L(LO) > h2(HI).

    Hand computation (J = 0):
      U(LO) = 1/10 + 1/10 + 2/20 = 3/10
      U(HI) = 2/10 + 3/20 = 7/20
      h1 (HI): R(LO) = 1; R(HI) = 2
      L  (LO): R0 = 1 -> ceil(1/10)*1 = 1 -> 2; R = 2 -> 1 -> 2
      h2 (HI):
        LO: R0 = 2 -> (h1:ceil(2/10)=1, L:ceil(2/10)=1) -> 2+1+1 = 4;
            R = 4 -> (1,1) -> 4
        HI: R0 = 3 -> (h1:ceil(3/10)=1*2, L:ceil(3/10)=1*1) -> 3+2+1 = 6;
            R = 6 -> (1,1) -> 6
    """
    return [
        MCTask("h1", wcet_lo=1, period=10, deadline=10,
               criticality=CRITICALITY_HI, wcet_hi=2),
        MCTask("L", wcet_lo=1, period=10, deadline=10,
               criticality=CRITICALITY_LO),
        MCTask("h2", wcet_lo=2, period=20, deadline=20,
               criticality=CRITICALITY_HI, wcet_hi=3),
    ]


class TestMCAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analyze_mc(mc_example())

    def test_overall_guaranteed(self):
        self.assertTrue(self.report.guaranteed)
        self.assertTrue(self.report.utilization_lo_ok)
        self.assertTrue(self.report.utilization_hi_ok)

    def test_utilizations_are_exact_and_mode_scoped(self):
        self.assertEqual(self.report.utilization_lo, Fraction(3, 10))
        self.assertEqual(self.report.utilization_hi, Fraction(7, 20))

    def test_response_time_bounds(self):
        by_name = {t.name: t for t in self.report.tasks}
        self.assertEqual(by_name["h1"].response_time, 2)   # max(1, 2)
        self.assertEqual(by_name["L"].response_time, 2)
        self.assertEqual(by_name["h2"].response_time, 6)   # max(4, 6)
        self.assertTrue(all(t.status == "schedulable" for t in self.report.tasks))

    def test_lo_mode_iteration_chains(self):
        by_name = {t.name: t for t in self.report.tasks}
        lo_l = by_name["L"].lo
        self.assertEqual([r.value for r in lo_l.iterations], [1, 2])
        self.assertEqual([r.next_value for r in lo_l.iterations], [2, 2])
        lo_h2 = by_name["h2"].lo
        self.assertEqual([r.value for r in lo_h2.iterations], [2, 4])
        self.assertEqual([r.next_value for r in lo_h2.iterations], [4, 4])
        self.assertEqual([dict(r.lo_interference)["h1"] for r in lo_h2.iterations], [1, 1])
        self.assertEqual([dict(r.lo_interference)["L"] for r in lo_h2.iterations], [1, 1])

    def test_hi_mode_iteration_chain(self):
        hi = {t.name: t.hi for t in self.report.tasks}["h2"]
        self.assertEqual([r.value for r in hi.iterations], [3, 6])
        self.assertEqual([r.next_value for r in hi.iterations], [6, 6])
        self.assertEqual([dict(r.hi_interference)["h1"] for r in hi.iterations], [1, 1])
        self.assertEqual([dict(r.lo_interference)["L"] for r in hi.iterations], [1, 1])

    def test_low_task_has_no_hi_report(self):
        low = next(t for t in self.report.tasks if t.name == "L")
        self.assertIsNone(low.hi)
        self.assertEqual(low.criticality, CRITICALITY_LO)


class TestJitter(unittest.TestCase):
    def make(self, jitter_h1):
        # Two HI tasks only; h1's jitter moves its interfering releases left.
        return [
            MCTask("h1", wcet_lo=1, period=4, deadline=4,
                   criticality=CRITICALITY_HI, wcet_hi=2, jitter=jitter_h1),
            MCTask("h2", wcet_lo=1, period=8, deadline=8,
                   criticality=CRITICALITY_HI, wcet_hi=2),
        ]

    def test_hi_mode_miss_induced_by_higher_priority_jitter(self):
        # HI rounds for h2 with J(h1) = 5:
        # R0 = 2 -> ceil(7/4)=2 -> 2+4 = 6
        # R = 6 -> ceil(11/4)=3 -> 2+6 = 8
        # R = 8 -> ceil(13/4)=4 -> 2+8 = 10 > D-J = 8 (proven miss).
        report = analyze_mc(self.make(5))
        h2 = {t.name: t for t in report.tasks}["h2"]
        self.assertEqual(h2.hi.status, "deadline_miss")
        self.assertIsNone(h2.response_time)
        self.assertEqual([r.next_value for r in h2.hi.iterations], [6, 8, 10])
        self.assertFalse(report.guaranteed)
        # LO mode still converges: R(LO) = 1 + ceil((R+5)/4)*1 -> 3 (stable).
        self.assertEqual(h2.lo.status, "schedulable")
        self.assertEqual(h2.lo.bound, 3)

    def test_smaller_jitter_is_exactly_schedulable(self):
        # J(h1) = 3: 2 -> ceil(5/4)*2=4 -> 6; 6 -> ceil(9/4)*2=6 -> 8 == D.
        report = analyze_mc(self.make(3))
        h2 = {t.name: t for t in report.tasks}["h2"]
        self.assertEqual(h2.hi.status, "schedulable")
        self.assertEqual(h2.hi.bound, 8)


class TestExactBoundary(unittest.TestCase):
    def make(self, wcet_hi_h2):
        return [
            MCTask("h1", wcet_lo=1, period=6, deadline=6,
                   criticality=CRITICALITY_HI, wcet_hi=2),
            MCTask("L", wcet_lo=2, period=6, deadline=6,
                   criticality=CRITICALITY_LO),
            MCTask("h2", wcet_lo=1, period=12, deadline=12,
                   criticality=CRITICALITY_HI, wcet_hi=wcet_hi_h2),
        ]

    def test_bound_equal_to_deadline_is_schedulable(self):
        # Chi(h2) = 4: R0 = 4 -> ceil(4/6)=1: 4+2+2 = 8;
        # R = 8 -> ceil(8/6)=2: 4+4+4 = 12; R = 12 -> 2 -> 12 == D.
        report = analyze_mc(self.make(4))
        h2 = {t.name: t for t in report.tasks}["h2"]
        self.assertEqual(h2.hi.status, "schedulable")
        self.assertEqual(h2.hi.bound, 12)
        self.assertEqual(h2.response_time, 12)
        self.assertTrue(report.guaranteed)

    def test_bound_one_past_deadline_misses(self):
        # Chi(h2) = 5: R0 = 5 -> 1 -> 9; R = 9 -> 2: 5+4+4 = 13 > 12.
        report = analyze_mc(self.make(5))
        h2 = {t.name: t for t in report.tasks}["h2"]
        self.assertEqual(h2.hi.status, "deadline_miss")
        self.assertIsNone(h2.response_time)
        self.assertFalse(report.guaranteed)


class TestLOModeMiss(unittest.TestCase):
    def test_lo_mode_miss_makes_set_unguarnteed(self):
        # The LO task alone needs 7 units against D-J = 6: proven LO miss.
        tasks = [
            MCTask("h1", wcet_lo=1, period=6, deadline=6,
                   criticality=CRITICALITY_HI, wcet_hi=2),
            MCTask("L", wcet_lo=7, period=6, deadline=6,
                   criticality=CRITICALITY_LO),
            MCTask("h2", wcet_lo=1, period=12, deadline=12,
                   criticality=CRITICALITY_HI, wcet_hi=5),
        ]
        report = analyze_mc(tasks)
        by_name = {t.name: t for t in report.tasks}
        self.assertEqual(by_name["L"].status, "deadline_miss")
        self.assertIsNone(by_name["L"].response_time)
        self.assertFalse(report.guaranteed)
        # The HI-mode iteration is still computed independently (it does not
        # need the LO task to converge).
        self.assertIsNotNone(by_name["h2"].hi)
        self.assertTrue(by_name["h2"].hi.iterations)


class TestBudgetExhaustion(unittest.TestCase):
    def make(self):
        # Every multi-task iteration here needs at least two rounds.
        return [
            MCTask("hi", wcet_lo=1, period=4, deadline=4,
                   criticality=CRITICALITY_HI, wcet_hi=1),
            MCTask("lo", wcet_lo=2, period=20, deadline=20,
                   criticality=CRITICALITY_LO),
            MCTask("hi2", wcet_lo=1, period=20, deadline=20,
                   criticality=CRITICALITY_HI, wcet_hi=2),
        ]

    def test_exhaustion_is_unknown_not_miss(self):
        report = analyze_mc(self.make(), max_iterations=1)
        by_name = {t.name: t for t in report.tasks}
        self.assertEqual(by_name["lo"].status, "iteration_budget_exhausted")
        self.assertIsNone(by_name["lo"].response_time)
        self.assertEqual(by_name["hi2"].status, "iteration_budget_exhausted")
        self.assertFalse(report.guaranteed)
        # With the default budget:
        #   lo: R0=2 -> ceil(2/4)*1=1 -> 3 (stable)
        #   hi2 LO: 1 + hi*1 + lo*2 = 4 (stable); hi2 HI:
        #   R0=2 -> hi ceil(2/4)*1=1 + lo ceil(2/20)*2=2 -> 5;
        #   R=5 -> hi ceil(5/4)=2*1 + lo*1*2 -> 6; R=6 -> (2,1) -> 6.
        full = analyze_mc(self.make())
        self.assertTrue(full.guaranteed)
        self.assertEqual({t.name: t.response_time for t in full.tasks},
                         {"hi": 1, "lo": 3, "hi2": 6})


class TestConservatismRegression(unittest.TestCase):
    def test_lo_interference_is_counted_over_the_whole_window(self):
        # Regression for a tighter-but-unsound HI term that charges each
        # higher-priority LO task only once via its own R(LO).  Set:
        #   L(LO): C=2, T=4 ; H(HI): C(LO)=2, C(HI)=4, T=8, D=8.
        # Tight once-only term would give:
        #   R = 4 + ceil(R_L(LO)/4)*2 = 4 + ceil(4/4)*2 = 6.
        # But an overrun H can still be preempted by LO jobs released around
        # the switch: the integer-time reference schedule observes a response
        # of 8 for H@0.  The conservative recurrence must therefore report 8:
        #   R0 = 4 -> ceil(4/4)*2 = 2 -> 6; R = 6 -> ceil(6/4)*2 = 4 -> 8.
        tasks = [
            MCTask("L", wcet_lo=2, period=4, deadline=4,
                   criticality=CRITICALITY_LO),
            MCTask("H", wcet_lo=2, period=8, deadline=8,
                   criticality=CRITICALITY_HI, wcet_hi=4),
        ]
        report = analyze_mc(tasks)
        h = report.tasks[1]
        self.assertEqual(h.lo.bound, 4)
        self.assertEqual(h.hi.bound, 8)
        self.assertEqual([r.value for r in h.hi.iterations], [4, 6, 8])
        self.assertEqual([r.next_value for r in h.hi.iterations], [6, 8, 8])
        self.assertEqual(
            [dict(r.lo_interference)["L"] for r in h.hi.iterations], [1, 2, 2]
        )
        self.assertTrue(report.guaranteed)


class TestUtilization(unittest.TestCase):
    def test_hi_utilization_counts_high_tasks_at_hi_budget(self):
        tasks = [
            MCTask("h1", wcet_lo=1, period=4, deadline=4,
                   criticality=CRITICALITY_HI, wcet_hi=3),
            MCTask("l1", wcet_lo=1, period=4, deadline=4,
                   criticality=CRITICALITY_LO),
        ]
        report = analyze_mc(tasks)
        self.assertEqual(report.utilization_lo, Fraction(1, 2))
        self.assertEqual(report.utilization_hi, Fraction(3, 4))

    def test_hi_overload_reported_not_raised(self):
        # LO-feasible set (U(LO) = 1/2) whose HI workload alone exceeds 1.
        tasks = [
            MCTask("h", wcet_lo=2, period=4, deadline=4,
                   criticality=CRITICALITY_HI, wcet_hi=5),
        ]
        report = analyze_mc(tasks)
        self.assertTrue(report.utilization_lo_ok)
        self.assertFalse(report.utilization_hi_ok)
        self.assertFalse(report.guaranteed)
        self.assertIn("HI", report.summary)


class TestAllLowCriticalityMatchesPlainAnalysis(unittest.TestCase):
    def test_bounds_equal_plain_fixed_priority_analysis(self):
        # Supplementary interface-boundary cross-check: an all-LO MC set is
        # an ordinary task set with C = C(LO).
        plain = [
            Task("a", wcet=1, period=4, deadline=4),
            Task("b", wcet=2, period=8, deadline=8),
        ]
        mc = [
            MCTask("a", wcet_lo=1, period=4, deadline=4, criticality=CRITICALITY_LO),
            MCTask("b", wcet_lo=2, period=8, deadline=8, criticality=CRITICALITY_LO),
        ]
        plain_report = analyze(plain)
        mc_report = analyze_mc(mc)
        for got, expected in zip(mc_report.tasks, plain_report.tasks):
            self.assertEqual(got.response_time, expected.response_time)  # 1 and 3
        self.assertEqual(mc_report.utilization_lo, plain_report.utilization)
        self.assertEqual(mc_report.utilization_hi, Fraction(0))


class TestMCTaskValidation(unittest.TestCase):
    def base(self, **overrides):
        fields = dict(name="t", wcet_lo=1, period=4, deadline=4,
                      criticality=CRITICALITY_HI, wcet_hi=2, jitter=0)
        fields.update(overrides)
        return MCTask(**fields)

    def test_bad_criticality_string(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([self.base(criticality="HI")])
        self.assertIn("criticality", str(ctx.exception))

    def test_high_task_requires_hi_budget(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([self.base(wcet_hi=None)])
        self.assertIn("wcet_hi", str(ctx.exception))

    def test_low_task_must_not_set_hi_budget(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc(
                [MCTask("l", wcet_lo=1, period=4, deadline=4,
                        criticality=CRITICALITY_LO, wcet_hi=1)]
            )
        self.assertIn("wcet_hi", str(ctx.exception))

    def test_hi_budget_below_lo_budget_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([self.base(wcet_hi=1, wcet_lo=2)])
        self.assertIn("wcet_hi", str(ctx.exception))

    def test_equal_budgets_accepted(self):
        report = analyze_mc([self.base(wcet_hi=1)])
        self.assertTrue(report.guaranteed)

    def test_scalar_rules_are_inherited(self):
        with self.assertRaises(ValidationError):
            analyze_mc([self.base(wcet_lo=True)])
        with self.assertRaises(ValidationError):
            analyze_mc([self.base(wcet_hi=2.5)])
        with self.assertRaises(ValidationError):
            analyze_mc([self.base(wcet_lo=math.nan)])
        with self.assertRaises(ValidationError):
            analyze_mc([self.base(deadline=5, period=4)])
        with self.assertRaises(ValidationError):
            analyze_mc([self.base(wcet_lo=0)])
        with self.assertRaises(ValidationError):
            analyze_mc([self.base(jitter=-1)])

    def test_name_and_set_rules(self):
        with self.assertRaises(ValidationError):
            analyze_mc([])
        with self.assertRaises(ValidationError):
            analyze_mc(42)
        with self.assertRaises(ValidationError):
            analyze_mc([{"name": "x"}])
        with self.assertRaises(ValidationError):
            analyze_mc([
                MCTask(f"t{i}", 1, 100, 100, CRITICALITY_LO) for i in range(21)
            ])
        with self.assertRaises(ValidationError):
            analyze_mc([self.base(name="a"), self.base(name="a")])

    def test_max_iterations_validated(self):
        with self.assertRaises(ValidationError):
            analyze_mc([self.base()], max_iterations=0)
        with self.assertRaises(ValidationError):
            analyze_mc([self.base()], max_iterations=True)


if __name__ == "__main__":
    unittest.main()
