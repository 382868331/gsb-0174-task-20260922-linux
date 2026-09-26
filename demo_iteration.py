"""Iteration demo: EDF demand-bound analysis + independent EDF simulation.

Runs entirely offline with the Python 3.14 standard library.  Every value
below is computed on the spot (no sleeps, no pre-recorded output):

1. a normal schedulable set with rational WCETs, printing H, checkpoints,
   per-task demands and the range (0, H] that was actually verified;
2. boundary cases for real: U exactly 1 schedulable, U exactly 1 with
   short deadlines unschedulable, low utilization with short deadlines
   unschedulable (cross-checked by an independent EDF simulation that
   observes the concrete miss), zero-execution tasks, and a
   budget-exceeded "not proven" result.

Run:  python demo_iteration.py   (well under 90 s)
"""

from fractions import Fraction

from rt_analysis import EdfTask, analyze_edf, simulate_edf
from rt_analysis.edf import demand_bound, _validate_edf_tasks


def describe(tasks):
    return ", ".join(
        f"{t.name}(C={Fraction(t.wcet)}, T={t.period}, D={t.deadline})" for t in tasks
    )


def show(title, tasks, **kwargs):
    print(f"--- {title}")
    print(f"    tasks: {describe(tasks)}")
    report = analyze_edf(tasks, **kwargs)
    print(f"    utilization U = {report.utilization} (~{float(report.utilization):.4f})")
    print(f"    status = {report.status}; hyperperiod H = {report.hyperperiod}")
    print(f"    checkpoints checked = {report.checkpoints_checked}, "
          f"verified range = (0, {report.checked_up_to}]")
    if report.first_violation is not None:
        print(f"    earliest violation t = {report.first_violation}")
        for d in report.violation_demands:
            print(f"      task {d.name}: jobs={d.jobs}, demand={d.demand}")
    print(f"    {report.summary}")
    return report


def main():
    print("=== Part 1: normal schedulable EDF set (rational C allowed) ===")
    normal = [
        EdfTask("ctrl", Fraction(1, 2), 2, 2),
        EdfTask("sense", 2, 5, 5),
        EdfTask("log", Fraction(3, 2), 10, 8),
    ]
    report = show("normal set", normal)
    normalized = _validate_edf_tasks(normal)
    for t in (2, 5, 8, 10, report.hyperperiod):
        print(f"    dbf({t:>3}) = {demand_bound(normalized, t)} <= {t}")
    sim = simulate_edf(normal, report.hyperperiod)
    print(f"    independent EDF simulation over (0, {report.hyperperiod}]: "
          f"{len(sim.jobs)} jobs completed, deadline misses = {sim.any_miss()}, "
          f"incomplete = {len(sim.incomplete)}")

    print()
    print("=== Part 2: boundary results, computed for real ===")

    show("U exactly 1, deadlines equal periods -> schedulable", [
        EdfTask("a", 2, 4, 4),
        EdfTask("b", 1, 2, 2),
    ])

    show("U exactly 1 but deadlines shorter than periods -> unschedulable", [
        EdfTask("a", 2, 4, 2),
        EdfTask("b", 2, 4, 2),
    ])

    tight = [
        EdfTask("a", 2, 5, 2),
        EdfTask("b", 3, 6, 4),
    ]
    tight_report = show("U = 9/10 < 1 but short deadlines -> unschedulable", tight)
    tight_sim = simulate_edf(tight, tight_report.hyperperiod)
    misses = [j for j in tight_sim.jobs if j.missed]
    print(f"    independent EDF simulation observes {len(misses)} deadline miss(es); "
          f"first: task {misses[0].task} released at {misses[0].release}, "
          f"deadline {misses[0].deadline}, completed {misses[0].completion}")

    show("zero-execution tasks contribute no demand -> schedulable", [
        EdfTask("idle0", 0, 3, 1),
        EdfTask("idle1", Fraction(0), 7, 3),
        EdfTask("work", 1, 4, 4),
    ])

    show("U > 1 -> infeasible directly", [
        EdfTask("heavy", 3, 2, 2),
    ])

    show("hyperperiod over explicit budget -> not proven (not unschedulable)", [
        EdfTask("a", 1, 1009, 1009),
        EdfTask("b", 1, 1013, 1013),
    ], max_hyperperiod=100)

    show("checkpoint count over explicit budget -> not proven", [
        EdfTask("a", 1, 4, 2),
        EdfTask("b", 1, 5, 2),
    ], max_checkpoints=7)

    print()
    print("Why H is enough (not an arbitrary cut):")
    print("    dbf(t + H) = dbf(t) + H*U. With U <= 1, dbf(t) <= t on (0, H]")
    print("    implies dbf(t + k*H) <= t + k*H for every k >= 0, so no violation")
    print("    can first appear beyond the hyperperiod. Only the jump points")
    print("    D_i + k*T_i in (0, H] need checking: between two such points dbf")
    print("    is constant while the available supply t keeps growing.")


if __name__ == "__main__":
    main()

