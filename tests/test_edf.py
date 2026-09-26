
"""EDF demand-bound analysis tests.

Expected values are hand-derived from dbf(t) = sum max(0, floor((t-D)/T)+1)*C.
Cross-checks use (a) a brute-force dbf scan over every integer t in (0, H]
to validate the key-point compression, and (b) an independent discrete-time
EDF simulation over enumerated short sporadic arrival sequences — neither
calls analyze_edf to obtain its expectations.
"""

import itertools
import unittest
from fractions import Fraction

from rt_analysis import (
    EdfTask,
    ValidationError,
    analyze_edf,
    dbf,
    key_points,
    simulate_edf,
)


def brute_force_first_violation(tasks, horizon):
    """Independent check: scan EVERY integer t in (0, horizon]."""
    for t in range(1, horizon + 1):
        if dbf(tasks, t) > t:
            return t
    return None


def sporadic_release_sets(period, horizon):
    """All release sets in [0, horizon) containing 0, min gap >= period."""
    candidates = list(range(1, horizon))
    sets = []
    for size in range(0, len(candidates) + 1):
        for combo in itertools.combinations(candidates, size):
            releases = (0,) + combo
            if all(b - a >= period for a, b in zip(releases, releases[1:])):
                sets.append(releases)
    return sets


class DbfTests(unittest.TestCase):
    def test_hand_derived_values(self):
        tasks = [EdfTask("a", 2, 4, 2), EdfTask("b", 3, 6, 5)]
        # a: jobs at t>=2,5(=2+... ) -> floor((t-2)/4)+1 for t>=2
        # b: jobs at t>=5 -> floor((t-5)/6)+1 for t>=5
        self.assertEqual(dbf(tasks, 0), 0)
        self.assertEqual(dbf(tasks, 1), 0)
        self.assertEqual(dbf(tasks, 2), 2)   # a: 1 job
        self.assertEqual(dbf(tasks, 4), 2)   # a: floor(2/4)+1 = 1
        self.assertEqual(dbf(tasks, 5), 5)   # a: 1 job (2) + b: 1 job (3)
        self.assertEqual(dbf(tasks, 6), 7)   # a: 2 jobs (4) + b: 1 job (3)

    def test_hand_derived_values_exact(self):
        tasks = [EdfTask("a", 2, 4, 2), EdfTask("b", 3, 6, 5)]
        self.assertEqual(dbf(tasks, 11), 12)
        self.assertEqual(dbf(tasks, 10), 9)  # a: 3 jobs=6, b: 1 job=3

    def test_rational_wcet(self):
        tasks = [EdfTask("a", Fraction(3, 2), 4, 3)]
        self.assertEqual(dbf(tasks, 3), Fraction(3, 2))
        self.assertEqual(dbf(tasks, 7), Fraction(6, 2))

    def test_zero_wcet_contributes_nothing(self):
        tasks = [EdfTask("z", 0, 3, 1), EdfTask("a", 2, 5, 4)]
        self.assertEqual(dbf(tasks, 100), dbf([tasks[1]], 100))

    def test_key_points(self):
        tasks = [EdfTask("a", 1, 4, 2), EdfTask("b", 1, 6, 3)]
        self.assertEqual(key_points(tasks, 12), (2, 3, 6, 9, 10))


class AnalyzeEdfTests(unittest.TestCase):
    def test_low_utilization_short_deadline_failure(self):
        # U = 2/4 + 2/6 = 5/6 < 1, yet dbf(3) = 2 + 2 = 4 > 3.
        tasks = [EdfTask("a", 2, 4, 2), EdfTask("b", 2, 6, 3)]
        report = analyze_edf(tasks)
        self.assertEqual(report.status, "unschedulable")
        self.assertEqual(report.utilization, Fraction(5, 6))
        self.assertIsNotNone(report.violation)
        self.assertEqual(report.violation.time, 3)
        self.assertEqual(report.violation.demand, 4)
        per_task = {d.name: d.demand for d in report.violation.per_task}
        self.assertEqual(per_task, {"a": 2, "b": 2})
        self.assertEqual(report.hyperperiod, 12)
        self.assertEqual(report.checked_until, 12)

    def test_utilization_exactly_one_schedulable(self):
        tasks = [
            EdfTask("a", 1, 2, 2),
            EdfTask("b", 1, 3, 3),
            EdfTask("c", 1, 6, 6),
        ]
        report = analyze_edf(tasks)
        self.assertEqual(report.utilization, 1)
        self.assertEqual(report.status, "schedulable")
        self.assertIsNone(report.violation)
        self.assertEqual(report.hyperperiod, 6)

    def test_utilization_above_one_unschedulable_without_checkpoints(self):
        tasks = [EdfTask("a", 3, 4, 4), EdfTask("b", 2, 3, 3)]
        report = analyze_edf(tasks)
        self.assertEqual(report.status, "unschedulable")
        self.assertGreater(report.utilization, 1)
        self.assertIsNone(report.violation)
        self.assertEqual(report.checkpoints, ())

    def test_zero_execution_task_set(self):
        tasks = [EdfTask("z", 0, 2, 1), EdfTask("a", 1, 4, 4)]
        report = analyze_edf(tasks)
        self.assertEqual(report.status, "schedulable")
        self.assertEqual(report.utilization, Fraction(1, 4))

    def test_hyperperiod_budget_gives_unproven(self):
        tasks = [EdfTask("a", 1, 7, 7), EdfTask("b", 1, 11, 11)]
        report = analyze_edf(tasks, max_hyperperiod=10)
        self.assertEqual(report.status, "unproven")
        self.assertIsNone(report.violation)
        self.assertIsNone(report.hyperperiod)

    def test_checkpoint_budget_gives_unproven(self):
        tasks = [EdfTask("a", 1, 2, 1), EdfTask("b", 1, 3, 1)]
        report = analyze_edf(tasks, max_checkpoints=3)
        self.assertEqual(report.status, "unproven")
        self.assertEqual(report.hyperperiod, 6)
        self.assertIsNone(report.violation)

    def test_keypoint_compression_matches_brute_force(self):
        sets = [
            [EdfTask("a", 2, 4, 2), EdfTask("b", 2, 6, 3)],
            [EdfTask("a", 1, 2, 2), EdfTask("b", 1, 3, 3), EdfTask("c", 1, 6, 6)],
            [EdfTask("a", 1, 3, 2), EdfTask("b", 2, 5, 4), EdfTask("c", 1, 4, 3)],
            [EdfTask("a", Fraction(1, 2), 2, 1), EdfTask("b", 1, 5, 5)],
            [EdfTask("a", 0, 3, 1), EdfTask("b", 3, 7, 6), EdfTask("c", 2, 5, 2)],
        ]
        for tasks in sets:
            with self.subTest(tasks=tasks):
                report = analyze_edf(tasks)
                horizon = report.hyperperiod
                self.assertIsNotNone(horizon)
                expected = brute_force_first_violation(tasks, horizon)
                if expected is None:
                    self.assertEqual(report.status, "schedulable")
                    self.assertIsNone(report.violation)
                else:
                    self.assertEqual(report.status, "unschedulable")
                    self.assertEqual(report.violation.time, expected)


class EdfSimulationCrossCheckTests(unittest.TestCase):
    def test_schedulable_set_survives_all_short_arrival_sequences(self):
        tasks = [EdfTask("a", 1, 3, 2), EdfTask("b", 2, 5, 5)]
        self.assertEqual(analyze_edf(tasks).status, "schedulable")
        horizon = 10
        sets_a = sporadic_release_sets(3, horizon)
        sets_b = sporadic_release_sets(5, horizon)
        checked = 0
        for ra in sets_a:
            for rb in sets_b:
                result = simulate_edf(
                    tasks, horizon, arrivals={"a": ra, "b": rb}
                )
                self.assertEqual(result.misses, (), msg=f"arrivals {ra} {rb}")
                checked += 1
        self.assertGreater(checked, 50)  # the enumeration is non-trivial

    def test_failing_set_has_a_concrete_missing_sequence(self):
        # Same hand-derived set as the analysis failure test: synchronous
        # releases at 0 make job 'b' (C=2, D=3) miss at t=3 under EDF.
        tasks = [EdfTask("a", 2, 4, 2), EdfTask("b", 2, 6, 3)]
        self.assertEqual(analyze_edf(tasks).status, "unschedulable")
        result = simulate_edf(tasks, 6, arrivals={"a": (0,), "b": (0,)})
        self.assertEqual(result.misses, (("b", 0, 3),))

    def test_simulation_respects_sporadic_constraint(self):
        tasks = [EdfTask("a", 1, 4, 4)]
        with self.assertRaises(ValidationError):
            simulate_edf(tasks, 10, arrivals={"a": (0, 2)})  # gap 2 < T=4

    def test_simulation_rejects_nonintegral_wcet(self):
        with self.assertRaises(ValidationError):
            simulate_edf([EdfTask("a", Fraction(1, 2), 2, 2)], 4)


class EdfValidationTests(unittest.TestCase):
    def test_invalid_inputs(self):
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", -1, 4, 4)])
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", True, 4, 4)])
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", float("nan"), 4, 4)])
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, 0, 1)])
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, 4, 5)])  # D > T
        with self.assertRaises(ValidationError):
            analyze_edf([EdfTask("a", 1, 4, 4), EdfTask("a", 1, 5, 5)])
        with self.assertRaises(ValidationError):
            analyze_edf([])
        with self.assertRaises(ValidationError):
            dbf([EdfTask("a", 1, 4, 4)], -1)


if __name__ == "__main__":
    unittest.main()

