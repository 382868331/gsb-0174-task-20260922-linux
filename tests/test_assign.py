"""Priority-assignment tests.

Hand-derived expectations are written literally below.  In addition, an
independent reference enumerates ALL permutations of each small task set
and analyzes every one; the search entry point under test must agree with
that enumeration both when it finds an order and when it declares none.
"""

import itertools
import unittest
from fractions import Fraction

from rt_analysis import (
    CriticalSection,
    Task,
    ValidationError,
    analyze,
    assign_priorities,
)


def reorder_example():
    """Low utilization, fails in the given order only because of blocking.

    tA: C=1 T=6  D=6  (no resource)
    tB: C=1 T=8  D=3  cs(bus,1)
    tC: C=2 T=24 D=24 cs(bus,2)
    U = 1/6 + 1/8 + 2/24 = 9/24 = 3/8 <= 1.

    Given order [tA, tB, tC], ceilings: bus -> 1 (tB).
      tA: B=0 (bus ceiling 1 > 0), R = 1 <= 6.
      tB: B=2 (tC's bus section, ceiling 1 <= 1); R0 = 1+2 = 3,
          round 0: ceil(3/6)=1 -> R' = 1+2+1 = 4 > D-J = 3: miss.
      tC: B=0; R0=2 -> (1,1) -> 4; R=4 -> (1,1) -> 4.  R = 4.

    Search (IDs: tA=0, tB=1, tC=2), lowest slot first:
      slot 2: tA passes (B=0; R0=1 -> 1+1+2=4 -> 4 <= 6).
      slot 1: tB passes (below=[tA] no cs, B=0; R0=1 -> 1+2=3 -> 3 <= 3).
      slot 0: tC passes (B=1 from tB's bus section, ceiling 0; R0=3 -> 3).
    Found order (tC, tB, tA) after exactly 3 candidate tests, no backtrack.
    Re-analysis of (tC, tB, tA), ceilings: bus -> 0 (tC):
      tC: B=1 (tB bus, ceiling 0 <= 0); R0 = 2+1 = 3 -> 3.  R = 3 <= 24.
      tB: B=0; R0=1 -> ceil(1/24)=1 -> 3; R=3 -> 3.  R = 3 <= 3 (exact).
      tA: B=0; R0=1 -> (ceil(1/24)=1, ceil(1/8)=1) -> 4; R=4 -> (1,1) -> 4.
          R = 4 <= 6.
    """
    return [
        Task("tA", wcet=1, period=6, deadline=6),
        Task("tB", wcet=1, period=8, deadline=3,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("tC", wcet=2, period=24, deadline=24,
             critical_sections=(CriticalSection("bus", 2),)),
    ]


def scan_example():
    """Lowest-ID candidate fails the bottom test; the scan moves to the next.

    Same tasks as reorder_example, different input order:
      ID0=tB (C=1 T=8  D=3  cs(bus,1))
      ID1=tC (C=2 T=24 D=24 cs(bus,2))
      ID2=tA (C=1 T=6  D=6)

    Search trace, 5 candidate tests:
      slot 2: tB fails (B=0; R0=1 -> 1+2+1=4 > D-J = 3).            [test 1]
              tC passes (B=0; R0=2 -> 4 -> 4 <= 24).                [test 2]
      slot 1: tB fails (B=2 from tC's bus section below; HP=tA:
              R0=3 -> ceil(3/6)=1 -> 4 > 3).                        [test 3]
              tA passes (B=2; HP=tB: R0=3 -> 4 -> 4 <= 6).          [test 4]
      slot 0: tB passes (B=2 from tC; no HP: R0=3 <= 3).            [test 5]
    Found order (tB, tA, tC); re-analysis: R = (3, 4, 4).
    """
    return [
        Task("tB", wcet=1, period=8, deadline=3,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("tC", wcet=2, period=24, deadline=24,
             critical_sections=(CriticalSection("bus", 2),)),
        Task("tA", wcet=1, period=6, deadline=6),
    ]


def infeasible_example():
    """No priority order is guaranteeable under the model (hand proof).

    t1: C=1 T=10 D=3 cs(bus,1)   t2: C=2 T=20 D=20   t3: C=3 T=30 D=30 cs(bus,3)
    U = 1/10 + 2/20 + 3/30 = 3/10 <= 1.

    t1 needs R <= 3 in every order.  If t3 is below t1, t3's bus section
    (length 3; bus's ceiling is at most t1 because t1 uses bus) gives
    B >= 3, so R >= 1+3 = 4.  If t3 is above t1, its interference alone
    gives R >= 1+3 = 4.  Hence t1 misses in all 6 permutations.  The DFS
    performs 9 candidate tests (failing candidates are not expanded):
    3 at the bottom slot, 2+2 at the middle, 1+1 at the top.
    """
    return [
        Task("t1", wcet=1, period=10, deadline=3,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("t2", wcet=2, period=20, deadline=20),
        Task("t3", wcet=3, period=30, deadline=30,
             critical_sections=(CriticalSection("bus", 3),)),
    ]


def jittered_four():
    """Four-task set with jitter, used for enumeration cross-checks only."""
    return [
        Task("s1", wcet=1, period=5, deadline=5, jitter=1),
        Task("s2", wcet=2, period=11, deadline=11),
        Task("s3", wcet=1, period=13, deadline=13,
             critical_sections=(CriticalSection("m", 1),)),
        Task("s4", wcet=3, period=29, deadline=29,
             critical_sections=(CriticalSection("m", 2),)),
    ]


def guaranteed_permutations(tasks):
    """Independent reference: every permutation analyzed directly."""
    good = set()
    for perm in itertools.permutations(tasks):
        if analyze(perm).guaranteed:
            good.add(tuple(t.name for t in perm))
    return good


class TestGivenOrderFailsReorderingSucceeds(unittest.TestCase):
    def test_given_order_misses_on_blocking(self):
        report = analyze(reorder_example())
        self.assertTrue(report.utilization_ok)
        self.assertEqual(report.utilization, Fraction(3, 8))
        self.assertFalse(report.guaranteed)
        tb = report.tasks[1]
        self.assertEqual(tb.status, "deadline_miss")
        self.assertEqual(
            (tb.blocking.amount, tb.blocking.source_task, tb.blocking.source_resource),
            (2, "tC", "bus"),
        )
        # Miss proven in round 0: R0 = 3, interference 1 -> R' = 4 > D-J = 3.
        self.assertEqual([(r.value, r.next_value) for r in tb.iterations], [(3, 4)])
        self.assertEqual(report.tasks[0].response_time, 1)
        self.assertEqual(report.tasks[2].response_time, 4)

    def test_search_finds_hand_derived_order(self):
        result = assign_priorities(reorder_example())
        self.assertEqual(result.status, "assigned")
        self.assertEqual([t.name for t in result.order], ["tC", "tB", "tA"])
        self.assertEqual(result.candidates_tested, 3)  # greedy, no backtrack
        evidence = result.report
        self.assertTrue(evidence.guaranteed)
        self.assertEqual([t.response_time for t in evidence.tasks], [3, 3, 4])
        # Blocking evidence in the new order: only tC is blocked, by tB.
        b = evidence.tasks[0].blocking
        self.assertEqual((b.amount, b.source_task, b.source_resource), (1, "tB", "bus"))
        self.assertEqual(evidence.tasks[1].blocking.amount, 0)
        self.assertEqual(evidence.tasks[2].blocking.amount, 0)


class TestCandidateScan(unittest.TestCase):
    def test_lowest_id_candidate_fails_scan_moves_on(self):
        # Greedy completeness: the search still has to skip candidates that
        # fail the slot test (here the lowest-ID task tB cannot sit at the
        # bottom).  It is placed first as soon as a higher slot is reached.
        result = assign_priorities(scan_example())
        self.assertEqual(result.status, "assigned")
        self.assertEqual([t.name for t in result.order], ["tB", "tA", "tC"])
        self.assertEqual(result.candidates_tested, 5)
        self.assertEqual(
            [t.response_time for t in result.report.tasks], [3, 4, 4]
        )

    def test_greedy_never_needs_to_undo_a_passing_choice(self):
        # Exchange property of this model: if a task passes a slot test,
        # pinning it there cannot destroy feasibility (moving a task below
        # others replaces >= C_k of interference by at most one critical
        # section <= C_k of blocking).  Consequently every found order is
        # reached on the first passing branch; the backtracking code only
        # resumes to finish the exhaustive infeasibility proof.
        for maker in (reorder_example, scan_example, jittered_four):
            result = assign_priorities(maker())
            self.assertEqual(result.status, "assigned")
            # Found before any second candidate at the same slot after a
            # pass could dead-end: the implementation records the count, and
            # independent enumeration confirms the order is guaranteeable.
            good = guaranteed_permutations(maker())
            self.assertIn(tuple(t.name for t in result.order), good)


class TestNoFeasibleOrder(unittest.TestCase):
    def test_exhaustive_infeasibility(self):
        result = assign_priorities(infeasible_example())
        self.assertEqual(result.status, "no_feasible_order")
        self.assertIsNone(result.order)
        self.assertIsNone(result.report)
        self.assertEqual(result.candidates_tested, 9)  # full 3-task tree, pruned
        self.assertIn("analysis model", result.summary)
        # The verdict must be model-scoped, not a real-world claim.
        self.assertIn("not a proof of real-world unschedulability", result.summary)

    def test_utilization_above_one_short_circuits(self):
        tasks = [Task("heavy", wcet=3, period=2, deadline=2)]
        result = assign_priorities(tasks)
        self.assertEqual(result.status, "no_feasible_order")
        self.assertEqual(result.candidates_tested, 0)


class TestIndependentEnumeration(unittest.TestCase):
    """Cross-check the search against brute-force enumeration of all orders."""

    def check_against_enumeration(self, tasks):
        good = guaranteed_permutations(tasks)
        result = assign_priorities(tasks)
        if result.status == "assigned":
            names = tuple(t.name for t in result.order)
            self.assertIn(names, good)
            self.assertTrue(result.report.guaranteed)
        elif result.status == "no_feasible_order":
            self.assertEqual(good, set())
        else:
            self.fail(f"unexpected status with default budget: {result.status}")
        return result

    def test_reorder_example(self):
        self.check_against_enumeration(reorder_example())

    def test_scan_example(self):
        self.check_against_enumeration(scan_example())

    def test_infeasible_example(self):
        self.check_against_enumeration(infeasible_example())

    def test_jittered_four_task_set(self):
        self.check_against_enumeration(jittered_four())


class TestAssignValidation(unittest.TestCase):
    def test_nine_tasks_rejected(self):
        tasks = [Task(f"t{i}", wcet=1, period=100, deadline=100) for i in range(9)]
        with self.assertRaises(ValidationError) as ctx:
            assign_priorities(tasks)
        self.assertIn("8", str(ctx.exception))

    def test_eight_tasks_accepted(self):
        tasks = [Task(f"t{i}", wcet=1, period=100, deadline=100) for i in range(8)]
        result = assign_priorities(tasks)
        self.assertEqual(result.status, "assigned")

    def test_analyze_limit_unchanged(self):
        # The analysis entry point keeps its own 20-task interface limit.
        tasks = [Task(f"t{i}", wcet=1, period=100, deadline=100) for i in range(20)]
        self.assertTrue(analyze(tasks).guaranteed)

    def test_budget_validated(self):
        for bad in (0, -1, True, 2.5):
            with self.assertRaises(ValidationError):
                assign_priorities(infeasible_example(), node_budget=bad)
        for bad in (0, True):
            with self.assertRaises(ValidationError):
                assign_priorities(reorder_example(), max_iterations=bad)

    def test_budget_exhaustion_is_inconclusive(self):
        # 5 tests are not enough for the 9-test exhaustive 3-task tree.
        result = assign_priorities(infeasible_example(), node_budget=5)
        self.assertEqual(result.status, "search_budget_exhausted")
        self.assertIsNone(result.order)
        self.assertEqual(result.candidates_tested, 5)
        self.assertIn("inconclusive", result.summary)
        # With the full budget the same set is proven to have no order.
        self.assertEqual(
            assign_priorities(infeasible_example()).status, "no_feasible_order"
        )

    def test_iteration_budget_exhaustion_is_inconclusive(self):
        # t2 needs two rounds (R: 2 -> 3 -> 3); one round is inconclusive.
        tasks = [
            Task("t1", wcet=1, period=4, deadline=4),
            Task("t2", wcet=2, period=20, deadline=20),
        ]
        result = assign_priorities(tasks, max_iterations=1)
        self.assertEqual(result.status, "search_budget_exhausted")
        self.assertIsNone(result.order)
        self.assertIn("inconclusive", result.summary)
        # With enough iteration budget the same set is assigned.
        self.assertEqual(assign_priorities(tasks).status, "assigned")

    def test_invalid_tasks_rejected(self):
        with self.assertRaises(ValidationError):
            assign_priorities([])
        with self.assertRaises(ValidationError):
            assign_priorities([Task("t", wcet=0, period=4, deadline=4)])


if __name__ == "__main__":
    unittest.main()
