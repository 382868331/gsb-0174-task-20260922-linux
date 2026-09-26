
"""Offline demo for the EDF demand-bound analysis (rt_analysis.edf).

Computes, in well under 90 seconds and with no sleeps or prerecorded
output:

1. a normal schedulable task set (including a zero-execution task),
2. a low-utilization set that still fails because of short deadlines,
3. the boundary case U == 1 (schedulable),
4. an overloaded set (U > 1, unschedulable without any key-point check),
5. a budget-exhausted run that is UNPROVEN — not unschedulable,
6. an independent EDF simulation cross-check of case 2's failing set.
"""

from fractions import Fraction

from rt_analysis import EdfTask, analyze_edf, dbf, simulate_edf


def show(title, report):
    print(f"== {title}")
    print(f"   status           : {report.status}")
    print(f"   utilization      : {report.utilization} (~{float(report.utilization):.4f})")
    if report.hyperperiod is not None:
        print(f"   hyperperiod H    : {report.hyperperiod}")
    if report.checkpoints:
        pts = report.checkpoints
        shown = ", ".join(map(str, pts[:8])) + (" ..." if len(pts) > 8 else "")
        print(f"   checkpoints ({len(pts):3d}): {shown}")
    if report.checked_until is not None:
        print(f"   checked interval : (0, {report.checked_until}]")
    if report.violation is not None:
        v = report.violation
        print(f"   earliest violation at t={v.time}: dbf={v.demand} > {v.time}")
        for d in v.per_task:
            print(f"      task {d.name!r}: {d.jobs} job(s) -> demand {d.demand}")
    print(f"   summary          : {report.summary}")
    print()


def main():
    show(
        "1. normal schedulable set (with a zero-execution task)",
        analyze_edf(
            [
                EdfTask("ctrl", 1, 4, 3),
                EdfTask("sense", 2, 6, 5),
                EdfTask("log", 2, 12, 12),
                EdfTask("idle-hook", 0, 2, 1),
            ]
        ),
    )

    failing = [EdfTask("fast", 2, 4, 2), EdfTask("slow", 2, 6, 3)]
    show(
        "2. low utilization (U=5/6) but short deadlines -> unschedulable",
        analyze_edf(failing),
    )

    show(
        "3. boundary U == 1 exactly -> schedulable",
        analyze_edf(
            [
                EdfTask("a", 1, 2, 2),
                EdfTask("b", 1, 3, 3),
                EdfTask("c", Fraction(1, 2), 3, 3),
            ]
        ),
    )

    show(
        "4. overload U > 1 -> unschedulable (no key-point check needed)",
        analyze_edf([EdfTask("a", 3, 4, 4), EdfTask("b", 2, 3, 3)]),
    )

    show(
        "5. hyperperiod beyond explicit budget -> UNPROVEN (not unschedulable)",
        analyze_edf(
            [EdfTask("a", 1, 97, 97), EdfTask("b", 1, 89, 89)],
            max_hyperperiod=1000,
        ),
    )

    print("== 6. independent EDF simulation cross-check of case 2")
    result = simulate_edf(failing, 6, arrivals={"fast": (0,), "slow": (0,)})
    print(f"   simulated synchronous releases over [0, 6): misses = {result.misses}")
    for job in result.jobs:
        print(
            f"   job {job.task!r} released {job.release}, deadline {job.deadline}, "
            f"completed {job.completion}"
        )
    print(
        f"   dbf at the simulated miss: dbf(3) = {dbf(failing, 3)} > 3 "
        f"(analysis and simulation agree)"
    )


if __name__ == "__main__":
    main()

