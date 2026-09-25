"""Priority-assignment tests.

Expected values are hand-derived from the same recurrence as
test_analysis.py and written literally.  Independently of the search
implementation, an exhaustive permutation enumeration (itertools, using the
public ``analyze`` only as the feasibility oracle for each *order*) verifies
every ``assigned`` order and every ``no_feasible_order`` verdict — the
search code under test never computes its own reference.
"""

import itertools
import random
import unittest
from fractions import Fraction

from rt_analysis import (
    CriticalSection,
    Task,
    ValidationError,
    analyze,
    assign_priorities,
)


def feasible_orders(tasks):
    """Independent reference: all permutations that ``analyze`` guarantees."""
    return [
        perm
        for perm in itertools.permutations(tasks)
        if analyze(perm).guaranteed
    ]


def reordering_example():
    """Low utilization (U = 2/5), yet the input order fails through blocking.

    Input order [comm, ctrl, log] (priority order for analyze):
      ceilings: bus -> 0 (comm).  ctrl uses no resource but sits below a
      'bus' user, so log's 5-unit section blocks it: B_ctrl = 5.
      ctrl: limit D-J = 2; R0 = 1+5 = 6; round 0: ceil(6/16)=1 ->
            R' = 1+5+2 = 8 > 2  -> deadline_miss.
    Greedy placement (candidates in name order comm < ctrl < log):
      level 2: comm placed (order [ctrl, log, comm]):
               B=0; R: 2 -> 2+1+6=9 -> 2+2+6=10 -> 10 (stable) <= 16
      level 1: ctrl rejected ([log, ctrl, comm]: B=1, R0=2,
               ceil(2/40)=1 -> 1+1+6=8 > 2);
               log placed ([ctrl, log, comm]): B=1 (comm's section,
               ceiling(bus)=1); R: 7 -> 7+1=8 -> 8 (stable) <= 40
      level 0: ctrl placed ([ctrl, log, comm]): ceiling(bus)=1 > 0 so
               B=0; R = 1 <= 2
    Final order (ctrl, log, comm): R = (1, 8, 10), all within D-J.
    """
    return [
        Task("comm", wcet=2, period=16, deadline=16,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("ctrl", wcet=1, period=8, deadline=2),
        Task("log", wcet=6, period=40, deadline=40,
             critical_sections=(CriticalSection("bus", 5),)),
    ]


def hopeless_example():
    """Low utilization (U = 9/50) but no order works — blocking either way.

    [urgent, worker]: B_urgent = 4 (worker's section, ceiling(bus)=0);
        R0 = 1+4 = 5 > D-J = 2 -> miss.
    [worker, urgent]: B_urgent = 0 but worker interferes:
        R0 = 1; ceil(1/50)=1 -> 1+4 = 5 > 2 -> miss.
    """
    return [
        Task("urgent", wcet=1, period=10, deadline=2,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("worker", wcet=4, period=50, deadline=50,
             critical_sections=(CriticalSection("bus", 4),)),
    ]


class TestReorderingExample(unittest.TestCase):
    def test_input_order_fails_through_blocking_at_low_utilization(self):
        report = analyze(reordering_example())
        self.assertEqual(report.utilization, Fraction(2, 5))
        self.assertTrue(report.utilization_ok)
        self.assertFalse(report.guaranteed)
        ctrl = report.tasks[1]
        self.assertEqual(ctrl.status, "deadline_miss")
        self.assertEqual(
            (ctrl.blocking.amount, ctrl.blocking.source_task,
             ctrl.blocking.source_resource),
            (5, "log", "bus"),
        )
        # Hand-derived round: R0 = 6, one interfering release of comm -> 8 > 2.
        self.assertEqual([(r.value, r.next_value) for r in ctrl.iterations],
                         [(6, 8)])

    def test_assignment_finds_hand_derived_order(self):
        result = assign_priorities(reordering_example())
        self.assertEqual(result.status, "assigned")
        self.assertEqual(result.order, ("ctrl", "log", "comm"))
        # The returned evidence is a full analyze() report for that order.
        evidence = result.report
        self.assertTrue(evidence.guaranteed)
        self.assertEqual([t.name for t in evidence.tasks],
                         ["ctrl", "log", "comm"])
        self.assertEqual([t.response_time for t in evidence.tasks], [1, 8, 10])
        # Blocking recomputed for the NEW order: ctrl is now above every
        # 'bus' user, so its blocking dropped from 5 to 0.
        self.assertEqual(evidence.tasks[0].blocking.amount, 0)
        self.assertEqual(
            (evidence.tasks[1].blocking.amount,
             evidence.tasks[1].blocking.source_task),
            (1, "comm"),
        )
        self.assertEqual(evidence.tasks[2].blocking.amount, 0)

    def test_per_round_evidence_matches_hand_derivation(self):
        evidence = assign_priorities(reordering_example()).report
        ctrl, log, comm = evidence.tasks
        self.assertEqual([(r.value, r.next_value) for r in ctrl.iterations],
                         [(1, 1)])
        self.assertEqual([(r.value, r.next_value) for r in log.iterations],
                         [(7, 8), (8, 8)])
        self.assertEqual([(r.value, r.next_value) for r in comm.iterations],
                         [(2, 9), (9, 10), (10, 10)])
        self.assertEqual([dict(r.interference)["ctrl"] for r in comm.iterations],
                         [1, 2, 2])
        self.assertEqual([dict(r.interference)["log"] for r in comm.iterations],
                         [1, 1, 1])

    def test_found_order_is_confirmed_by_independent_enumeration(self):
        tasks = reordering_example()
        result = assign_priorities(tasks)
        feasible = feasible_orders(tasks)
        self.assertTrue(any(
            tuple(t.name for t in perm) == result.order for perm in feasible
        ))

    def test_search_is_deterministic(self):
        first = assign_priorities(reordering_example())
        second = assign_priorities(reordering_example())
        self.assertEqual(first.order, second.order)
        self.assertEqual(first.nodes_explored, second.nodes_explored)
        self.assertEqual(first.placements, second.placements)


class TestHopelessExample(unittest.TestCase):
    def test_no_feasible_order_after_exhaustive_search(self):
        result = assign_priorities(hopeless_example())
        self.assertEqual(result.status, "no_feasible_order")
        self.assertIsNone(result.order)
        self.assertIsNone(result.report)
        self.assertIn("model", result.summary)
        # The verdict must not claim real-world infeasibility.
        self.assertNotIn("no real-world schedule exists.", result.summary)

    def test_verdict_confirmed_by_independent_enumeration(self):
        self.assertEqual(feasible_orders(hopeless_example()), [])

    def test_utilization_is_low(self):
        report = analyze(hopeless_example())
        self.assertEqual(report.utilization, Fraction(9, 50))
        self.assertTrue(report.utilization_ok)

    def test_tiny_budget_yields_incomplete_not_a_verdict(self):
        result = assign_priorities(hopeless_example(), node_budget=1)
        self.assertEqual(result.status, "incomplete")
        self.assertIsNone(result.order)
        result = assign_priorities(hopeless_example(), node_budget=4)
        self.assertEqual(result.status, "incomplete")
        # With the default budget the same set is proven exhaustively.
        self.assertEqual(
            assign_priorities(hopeless_example()).status, "no_feasible_order"
        )


class TestCrossCheckAgainstEnumeration(unittest.TestCase):
    """Seeded pseudo-random sets: the search verdict must match brute force."""

    def make_sets(self):
        rng = random.Random(20260922)
        sets = []
        for _ in range(40):
            n = rng.randint(1, 5)
            tasks = []
            for i in range(n):
                period = rng.randint(4, 40)
                wcet = rng.randint(1, period)
                deadline = rng.randint(1, period)
                jitter = rng.randint(0, 2)
                sections = []
                if wcet >= 2 and rng.random() < 0.5:
                    sections.append(
                        CriticalSection(f"r{rng.randint(0, 1)}",
                                        rng.randint(1, wcet - 1))
                    )
                tasks.append(Task(
                    f"t{i}", wcet=wcet, period=period, deadline=deadline,
                    jitter=jitter, critical_sections=tuple(sections),
                ))
            sets.append(tasks)
        return sets

    def test_search_matches_exhaustive_enumeration(self):
        for tasks in self.make_sets():
            with self.subTest(tasks=[t.name for t in tasks]):
                result = assign_priorities(tasks)
                feasible = feasible_orders(tasks)
                if result.status == "assigned":
                    self.assertTrue(feasible)
                    self.assertTrue(result.report.guaranteed)
                    self.assertEqual(
                        sorted(result.order), sorted(t.name for t in tasks)
                    )
                    self.assertTrue(any(
                        tuple(t.name for t in perm) == result.order
                        for perm in feasible
                    ))
                elif result.status == "no_feasible_order":
                    self.assertEqual(feasible, [])
                else:
                    self.fail(
                        f"small sets must never be inconclusive, "
                        f"got {result.status}"
                    )


class TestAssignmentContract(unittest.TestCase):
    def test_eight_tasks_accepted(self):
        tasks = [Task(f"t{i}", wcet=1, period=100, deadline=100)
                 for i in range(8)]
        result = assign_priorities(tasks)
        self.assertEqual(result.status, "assigned")
        self.assertEqual(len(result.order), 8)

    def test_nine_tasks_rejected_but_analyze_limit_unchanged(self):
        tasks = [Task(f"t{i}", wcet=1, period=100, deadline=100)
                 for i in range(9)]
        with self.assertRaises(ValidationError) as ctx:
            assign_priorities(tasks)
        self.assertIn("8", str(ctx.exception))
        # The old analyze() limit of 20 tasks is untouched.
        twenty = [Task(f"t{i}", wcet=1, period=100, deadline=100)
                  for i in range(20)]
        self.assertTrue(analyze(twenty).guaranteed)

    def test_utilization_above_one_needs_no_search(self):
        tasks = [Task("heavy", wcet=3, period=2, deadline=2)]
        result = assign_priorities(tasks)
        self.assertEqual(result.status, "no_feasible_order")
        self.assertEqual(result.nodes_explored, 0)

    def test_single_trivial_task(self):
        result = assign_priorities([Task("solo", wcet=1, period=4, deadline=4)])
        self.assertEqual(result.status, "assigned")
        self.assertEqual(result.order, ("solo",))
        self.assertEqual(result.report.tasks[0].response_time, 1)

    def test_node_budget_validated(self):
        with self.assertRaises(ValidationError):
            assign_priorities(hopeless_example(), node_budget=0)
        with self.assertRaises(ValidationError):
            assign_priorities(hopeless_example(), node_budget=True)

    def test_invalid_tasks_rejected(self):
        with self.assertRaises(ValidationError):
            assign_priorities([])
        with self.assertRaises(ValidationError):
            assign_priorities([Task("bad", wcet=0, period=4, deadline=4)])

    def test_placement_trail_is_recorded(self):
        result = assign_priorities(reordering_example())
        self.assertTrue(result.placements)
        self.assertTrue(all(p.phase == "greedy" for p in result.placements))
        # ctrl is tried and rejected at level 1 before log is placed there.
        rejected = [p for p in result.placements if p.outcome == "rejected"]
        self.assertEqual([(p.level, p.candidate) for p in rejected],
                         [(1, "ctrl")])


if __name__ == "__main__":
    unittest.main()
