"""EDF demand-bound tests with independent simulation / enumeration cross-checks.

Independent references used here (never derived from the analyzed output):

* test_checkpoint_compression: scans EVERY integer t in (0, H] with a
  locally written dbf loop and compares the earliest violation to the
  checkpoint-only analysis;
* TestEdfSimulationCrossCheck: builds small-integer sporadic sets and runs
  an independently specified EDF event simulation, checking the two
  analyses agree direction-wise (schedulable => no observed miss; any set
  the demand bound rejects that the simulation also misses is observed
  concretely).
"""

from fractions import Fraction
import math
import unittest

from rt_analysis import (
    EdfTask,
    EdfSimResult,
    analyze_edf,
    simulate_edf,
)
from rt_analysis import ValidationError
from rt_analysis.edf import demand_bound, _validate_edf_tasks


def ref_dbf(tasks, t):
    """Independent literal implementation of dbf(t)."""
    total = Fraction(0)
    for c, period, deadline in tasks:
        n = (t - deadline) // period + 1
        if n > 0:
            total += n * c
    return total


class TestEdfValidation(unittest.TestCase):
    def test_deadline_above_period_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_edf([EdfTask("a", 1, 3, 4)])
        self.assertIn("deadline", str(ctx.exception))

    def test_zero_period_or_deadline_rejected(self):
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, 0, 0)])
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, 2, 0)])

    def test_bool_rejected(self):
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", True, 2, 2)])
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, True, 2)])

    def test_negative_and_nan_rejected(self):
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", -Fraction(1, 2), 2, 2)])
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", math.nan, 2, 2)])

    def test_fractional_float_rejected_use_fraction(self):
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 0.5, 2, 2)])

    def test_empty_duplicate_non_task(self):
        with self.assertRaises(ValidationError):
            analyze_edf([])
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, 2, 2), EdfTask("a", 1, 2, 2)])
        with self.assertRaises(ValidationError):
            analyze_edf([("a", 1, 2, 2)])

    def test_budget_validation(self):
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, 2, 2)], max_hyperperiod=0)
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, 2, 2)], max_checkpoints=True)


class TestEdfBasicVerdicts(unittest.TestCase):
    def test_schedulable_set(self):
        tasks = [
            EdfTask("a", Fraction(1, 2), 2, 2),
            EdfTask("b", 2, 4, 4),
        ]
        report = analyze_edf(tasks)
        self.assertEqual(report.status, "schedulable")
        self.assertEqual(report.utilization, Fraction(3, 4))
        self.assertEqual(report.hyperperiod, 4)
        self.assertIsNone(report.first_violation)
        self.assertEqual(report.checked_up_to, 4)
        # checkpoints in (0, 4]: {2, 4}
        self.assertEqual(report.checkpoints_checked, 2)

    def test_utilization_above_one_is_unschedulable_not_raised(self):
        report = analyze_edf([EdfTask("h", 3, 2, 2)])
        self.assertEqual(report.status, "unschedulable")
        self.assertEqual(report.utilization, Fraction(3, 2))
        self.assertIsNone(report.first_violation)
        self.assertEqual(report.hyperperiod, None)

    def test_utilization_exactly_one_is_schedulable_when_deadlines_allow(self):
        tasks = [
            EdfTask("a", 2, 4, 4),
            EdfTask("b", 1, 2, 2),
        ]
        self.assertEqual(analyze_edf(tasks).utilization, Fraction(1, 1))
        self.assertEqual(analyze_edf(tasks).status, "schedulable")

    def test_utilization_exactly_one_can_still_miss_with_short_deadlines(self):
        # U = 1/2 + 1/2 = 1, but deadlines are shorter than periods.
        tasks = [
            EdfTask("a", 2, 4, 2),
            EdfTask("b", 2, 4, 2),
        ]
        report = analyze_edf(tasks)
        self.assertEqual(report.status, "unschedulable")
        self.assertEqual(report.first_violation, 2)
        self.assertEqual(report.utilization, Fraction(1, 1))

    def test_zero_wcet_tasks_are_schedulable(self):
        tasks = [
            EdfTask("z0", 0, 3, 1),
            EdfTask("z1", Fraction(0), 7, 7),
            EdfTask("a", 1, 4, 4),
        ]
        report = analyze_edf(tasks)
        self.assertEqual(report.status, "schedulable")
        self.assertEqual(report.hyperperiod, 84)
        # Zero-cost tasks contribute no demand anywhere.
        normalized = _validate_edf_tasks(tasks)
        self.assertEqual(demand_bound(normalized, 50), Fraction(12))

    def test_low_utilization_short_deadline_failure(self):
        # U = 2/5 + 3/6 = 9/10 < 1, but dbf(4) = 2 + 3 = 5 > 4.
        tasks = [
            EdfTask("a", 2, 5, 2),
            EdfTask("b", 3, 6, 4),
        ]
        report = analyze_edf(tasks)
        self.assertEqual(report.status, "unschedulable")
        self.assertEqual(report.first_violation, 4)
        self.assertEqual(report.utilization, Fraction(9, 10))
        demands = {d.name: (d.jobs, d.demand) for d in report.violation_demands}
        self.assertEqual(demands["a"], (1, Fraction(2)))
        self.assertEqual(demands["b"], (1, Fraction(3)))
        # checkpoints before 4 are t = 2 (dbf(2) = 2 <= 2): earliest hit is 4.
        self.assertEqual(report.checkpoints_checked, 2)
        # EDF really misses a deadline for this set in the synchronous trace.
        sim = simulate_edf(tasks, report.hyperperiod)
        self.assertTrue(sim.any_miss())
        first_miss = next(j for j in sim.jobs if j.missed)
        self.assertEqual((first_miss.task, first_miss.release), ("b", Fraction(0)))


class TestCheckpointCompression(unittest.TestCase):
    """Checking only D+kT checkpoints equals checking every integer t."""

    def enumerate_cases(self):
        # Small-integer sporadic sets: C in 0..2, T,D in 1..3, D<=T.
        cases = []
        for c1 in range(3):
            for t1 in range(1, 4):
                for d1 in range(1, t1 + 1):
                    for c2 in range(3):
                        for t2 in range(1, 4):
                            for d2 in range(1, t2 + 1):
                                cases.append(
                                    (
                                        EdfTask("a", c1, t1, d1),
                                        EdfTask("b", c2, t2, d2),
                                    )
                                )
        return cases

    def test_earliest_violation_matches_full_integer_scan(self):
        checked = 0
        for edf_tasks in self.enumerate_cases():
            report = analyze_edf(list(edf_tasks))
            tasks = [(Fraction(t.wcet), t.period, t.deadline) for t in edf_tasks]
            h = report.hyperperiod if report.hyperperiod is not None else 1
            h = math.lcm(tasks[0][1], tasks[1][1])
            full_violation = None
            for t in range(1, h + 1):
                if ref_dbf(tasks, t) > t:
                    full_violation = t
                    break
            util = sum((c / p for c, p, _ in tasks), Fraction(0))
            if util > 1:
                self.assertEqual(report.status, "unschedulable")
                continue
            if full_violation is None:
                self.assertEqual(report.status, "schedulable", edf_tasks)
                self.assertIsNone(report.first_violation)
            else:
                self.assertEqual(report.status, "unschedulable", edf_tasks)
                self.assertEqual(report.first_violation, full_violation)
            checked += 1
        self.assertGreater(checked, 100)


class TestEdfSimulationCrossCheck(unittest.TestCase):
    def test_schedulable_sets_never_miss_in_simulation(self):
        # One- and two-task small sets: demand-bound schedulable => the
        # synchronous EDF trace over the hyperperiod meets every deadline.
        cases = 0
        for c1 in range(3):
            for t1 in range(1, 4):
                for d1 in range(1, t1 + 1):
                    for c2 in range(3):
                        for t2 in range(1, 4):
                            for d2 in range(1, t2 + 1):
                                tasks = [
                                    EdfTask("a", c1, t1, d1),
                                    EdfTask("b", c2, t2, d2),
                                ]
                                report = analyze_edf(tasks)
                                if report.status != "schedulable":
                                    continue
                                sim = simulate_edf(tasks, report.hyperperiod)
                                self.assertEqual(
                                    sim.incomplete, (), f"unfinished jobs in {tasks}"
                                )
                                self.assertFalse(
                                    sim.any_miss(), f"miss despite proof for {tasks}"
                                )
                                cases += 1
        self.assertGreater(cases, 10)

    def test_simulation_handles_fractional_wcet_exactly(self):
        # U = 3/4 + 1/3 = 13/12 > 1 -> rejected; but check a feasible
        # fractional schedule executes jobs at exact rational times.
        tasks = [EdfTask("a", Fraction(1, 2), 2, 2), EdfTask("b", Fraction(1, 2), 3, 3)]
        report = analyze_edf(tasks)
        self.assertEqual(report.status, "schedulable")
        sim = simulate_edf(tasks, 6)
        self.assertFalse(sim.any_miss())
        first = next(j for j in sim.jobs if j.task == "a")
        self.assertEqual(first.completion, Fraction(1, 2))
        self.assertEqual(first.response, Fraction(1, 2))

    def test_simulate_validation(self):
        with self.assertRaises(ValidationError):
            simulate_edf([EdfTask("a", 1, 2, 2)], 0)
        with self.assertRaises(ValidationError):
            simulate_edf([], 5)


class TestEdfBudgets(unittest.TestCase):
    def test_hyperperiod_budget_gives_not_proven(self):
        # lcm(1009, 1013) is over a million.
        tasks = [
            EdfTask("a", 1, 1009, 1009),
            EdfTask("b", 1, 1013, 1013),
        ]
        report = analyze_edf(tasks, max_hyperperiod=100)
        self.assertEqual(report.status, "not_proven")
        self.assertFalse(report.schedulable)
        self.assertIsNone(report.first_violation)

    def test_checkpoint_budget_gives_not_proven(self):
        # lcm = 20; D=2 gives 5 + 4 - 1 = 8 distinct checkpoints.
        tasks = [
            EdfTask("a", 1, 4, 2),
            EdfTask("b", 1, 5, 2),
        ]
        report = analyze_edf(tasks, max_checkpoints=7)
        self.assertEqual(report.status, "not_proven")
        self.assertEqual(report.hyperperiod, 20)
        # not_proven must never be confused with unschedulable:
        # with the default budget the same set is proven schedulable.
        self.assertEqual(analyze_edf(tasks).status, "schedulable")

    def test_budget_exhausted_even_when_utilization_is_one(self):
        tasks = [
            EdfTask("a", 1, 1009, 1009),
            EdfTask("b", 1, 1013, 1013),
        ]
        # U = 1/1009 + 1/1013 < 1; the set is likely fine but is unproven
        # under a tiny hyperperiod budget — verdict stays inconclusive.
        report = analyze_edf(tasks, max_hyperperiod=10)
        self.assertEqual(report.status, "not_proven")
        self.assertTrue(report.utilization < 1)


if __name__ == "__main__":
    unittest.main()
