"""Deterministic demo: priority assignment with worst-case response-time evidence.

Runs entirely offline with the standard library only.  It exercises the new
``assign_priorities`` entry point on three hand-built sets:

1. A low-utilization set that FAILS in the given order purely because of
   blocking, and succeeds after reordering — with the final per-round
   response-time evidence printed in full, plus a concrete simulator run
   respecting the bounds.
2. A set for which exhaustive search proves NO order is guaranteeable
   under the analysis model (model-scoped statement, not a claim about
   real-world scheduling), cross-checked by independent enumeration of
   every permutation.
3. A reduced node budget that stops the same search early:
   "inconclusive", strictly distinct from the proven no-order verdict.
"""

import itertools
import time

from rt_analysis import (
    CriticalSection,
    Task,
    analyze,
    assign_priorities,
    simulate,
)


def blocking_failure_set():
    # U = 1/6 + 1/8 + 2/24 = 3/8: low utilization, yet the given priority
    # order is not guaranteeable because tC's bus section blocks tB.
    return [
        Task("tA", wcet=1, period=6, deadline=6),
        Task("tB", wcet=1, period=8, deadline=3,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("tC", wcet=2, period=24, deadline=24,
             critical_sections=(CriticalSection("bus", 2),)),
    ]


def no_order_set():
    # t1 (D=3) is blocked for 3 by t3's bus section if t3 is below it, and
    # suffers 3 units of interference if t3 is above it: R >= 4 either way.
    return [
        Task("t1", wcet=1, period=10, deadline=3,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("t2", wcet=2, period=20, deadline=20),
        Task("t3", wcet=3, period=30, deadline=30,
             critical_sections=(CriticalSection("bus", 3),)),
    ]


def print_report_chain(report, tasks):
    by_name = {t.name: t for t in tasks}
    for task in report.tasks:
        spec = by_name[task.name]
        b = task.blocking
        if b.amount:
            src = f"B={b.amount} (from '{b.source_task}' on '{b.source_resource}')"
        else:
            src = "B=0"
        print(f"  task '{task.name}' ({src}), limit D-J={spec.deadline - spec.jitter}")
        for it in task.iterations:
            interference = ", ".join(
                f"{name}: ceil((R+J)/T)={n}" for name, n in it.interference
            ) or "no higher-priority tasks"
            print(f"    round {it.round}: R={it.value} -> [{interference}] -> R'={it.next_value}")
        print(f"      => {task.status}: R={task.response_time}; {task.detail}")


def enumerate_guaranteeable(tasks):
    """Independent reference: analyze every permutation directly."""
    good = []
    for perm in itertools.permutations(tasks):
        if analyze(perm).guaranteed:
            good.append(tuple(t.name for t in perm))
    return good


def main():
    started = time.monotonic()

    print("=== Part 1: low utilization, given order fails on blocking ===")
    tasks = blocking_failure_set()
    given = analyze(tasks)
    print(f"utilization = {given.utilization} (~{float(given.utilization):.4f})")
    print("given-order analysis:")
    print_report_chain(given, tasks)
    print(f"verdict: guaranteed={given.guaranteed} ({given.summary})")

    print("\n--- deterministic search: lowest priority slot, candidates by task ID ---")
    result = assign_priorities(tasks)
    print(f"status: {result.status}")
    print(f"candidate tests: {result.candidates_tested} (budget {result.node_budget})")
    print("found order (highest priority first):",
          " > ".join(t.name for t in result.order))
    print("final per-round response-time evidence for the found order:")
    print_report_chain(result.report, result.order)
    print(f"verdict: guaranteed={result.report.guaranteed}")

    print("\n--- concrete simulator run over lcm(6, 8, 24) = 24 units ---")
    sim = simulate(result.order, 24)
    worst = sim.max_response()
    for task in result.report.tasks:
        observed = worst.get(task.name, 0)
        print(f"  {task.name}: observed worst response {observed} <= bound {task.response_time}")
    assert sim.incomplete == ()
    assert all(worst[t.name] <= t.response_time for t in result.report.tasks)

    print("\n--- independent cross-check: enumerate all 6 permutations ---")
    good = enumerate_guaranteeable(tasks)
    found = tuple(t.name for t in result.order)
    print(f"  guaranteeable permutations ({len(good)}): {sorted(good)}")
    print(f"  found order {found} is among them: {found in good}")

    print("\n=== Part 2: no feasible order under the analysis model ===")
    tasks2 = no_order_set()
    t0 = time.monotonic()
    result2 = assign_priorities(tasks2)
    dt = time.monotonic() - t0
    print(f"status: {result2.status} after {result2.candidates_tested} candidate tests "
          f"({dt * 1000:.1f} ms)")
    print(f"summary: {result2.summary}")
    good2 = enumerate_guaranteeable(tasks2)
    print(f"independent enumeration of all 6 permutations: {len(good2)} guaranteeable")

    print("\n=== Part 3: budget exhaustion is inconclusive, not 'no order' ===")
    result3 = assign_priorities(tasks2, node_budget=5)
    print(f"status with node_budget=5: {result3.status} "
          f"(tests performed: {result3.candidates_tested})")
    print(f"summary: {result3.summary}")

    print(f"\ntotal demo time: {time.monotonic() - started:.2f} s")


if __name__ == "__main__":
    main()
