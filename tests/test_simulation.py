"""Simulation cross-check: concrete preemptive PCP schedules must respect the bounds.

The simulator generates real schedules on a small time axis; every observed
job response time must be <= the hand-computed bound.  The hand-computed
bounds are written literally below (independent reference); agreement of
``analyze`` with those literals is checked separately in test_analysis.py,
and a round-trip check against ``analyze`` is included here only as a
supplement.
"""

import unittest

from rt_analysis import CriticalSection, Task, analyze, simulate
from rt_analysis import ValidationError


def bus_example():
    """Hand bounds (J=0): B=(3,3,0) -> R_a=5, R_b=8, R_c=11.

    a: R0=2+3=5, no higher tasks -> 5
    b: R0=3+3=6 -> ceil(6/8)=1 -> 8;  R=8 -> ceil(8/8)=1 -> 8
    c: R0=4 -> (1,1) -> 9;  R=9 -> (2,1) -> 11;  R=11 -> (2,1) -> 11
    """
    return (
        [
            Task("a", wcet=2, period=8, deadline=8,
                 critical_sections=(CriticalSection("bus", 1),)),
            Task("b", wcet=3, period=12, deadline=12,
                 critical_sections=(CriticalSection("bus", 2),)),
            Task("c", wcet=4, period=20, deadline=20,
                 critical_sections=(CriticalSection("bus", 3),)),
        ],
        {"a": 5, "b": 8, "c": 11},
        120,  # lcm(8, 12, 20)
    )


def blocking_example():
    """Engineered so 'low' is inside its critical section when 'high' is released.

    Timeline: high runs [0,1); mid [1,4); low starts at 4 and holds 'bus'
    over [4,10); high's second job (released at 8) is blocked during [8,10),
    runs at 10 and completes at 11 — observed response 3 > C=1 proves
    blocking.  Hand bounds: B_high=6 -> R_high=1+6=7; B_mid=6 -> R_mid=11
    (9->11->11); B_low=0 -> R_low=12 (7->11->12->12).
    """
    return (
        [
            Task("high", wcet=1, period=8, deadline=8,
                 critical_sections=(CriticalSection("bus", 1),)),
            Task("mid", wcet=3, period=12, deadline=12),
            Task("low", wcet=7, period=24, deadline=24,
                 critical_sections=(CriticalSection("bus", 6),)),
        ],
        {"high": 7, "mid": 11, "low": 12},
        24,  # lcm(8, 12, 24)
    )


class TestSimulationRespectsBounds(unittest.TestCase):
    def check(self, tasks, bounds, horizon):
        result = simulate(tasks, horizon)
        self.assertEqual(result.incomplete, (), "all jobs must finish within the horizon")
        for job in result.jobs:
            self.assertLessEqual(
                job.response,
                bounds[job.task],
                f"job {job.task} released at {job.release} took {job.response} "
                f"> bound {bounds[job.task]}",
            )
        return result

    def test_bus_example(self):
        tasks, bounds, horizon = bus_example()
        result = self.check(tasks, bounds, horizon)
        # The bound for 'c' is tight: its first job really takes 11 units
        # (3 blocked + 2 interfered by 'a' + 3 by 'b' + own 4 - 1 overlap-free).
        first_c = next(j for j in result.jobs if j.task == "c")
        self.assertEqual(first_c.response, 11)

    def test_blocking_example_actually_blocks(self):
        tasks, bounds, horizon = blocking_example()
        result = self.check(tasks, bounds, horizon)
        worst = result.max_response()
        # high's own WCET is 1; an observed response of 3 proves the
        # lower-priority lock really delayed it in the generated schedule.
        self.assertEqual(worst["high"], 3)
        self.assertGreater(worst["high"], 1)

    def test_round_trip_with_analyze(self):
        # Supplementary only: analyze must reproduce the hand bounds, and the
        # simulator must respect analyze's bounds.
        for tasks, bounds, horizon in (bus_example(), blocking_example()):
            report = analyze(tasks)
            self.assertTrue(report.guaranteed)
            analytic = {t.name: t.response_time for t in report.tasks}
            self.assertEqual(analytic, bounds)
            result = simulate(tasks, horizon)
            for job in result.jobs:
                self.assertLessEqual(job.response, analytic[job.task])


class TestSimulateValidation(unittest.TestCase):
    def test_invalid_horizon(self):
        tasks, _, _ = bus_example()
        with self.assertRaises(ValidationError):
            simulate(tasks, 0)
        with self.assertRaises(ValidationError):
            simulate(tasks, True)

    def test_invalid_tasks(self):
        with self.assertRaises(ValidationError):
            simulate([], 10)


if __name__ == "__main__":
    unittest.main()
