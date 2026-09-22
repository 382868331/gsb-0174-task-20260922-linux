"""Deterministic demo: worst-case response-time bounds for a fixed-priority set.

Runs entirely offline with the standard library only.  Part 1 analyzes a
three-task set and prints the full derivation chain (utilization, resource
ceilings, blocking terms with sources, every iteration round with
interference counts, and the verdict).  Part 2 triggers two rejection /
boundary paths for real: an invalid input rejected with a locatable error,
and a task set whose utilization exceeds 1 (reported as not guaranteeable,
not raised as a format error).
"""

from fractions import Fraction

from rt_analysis import CriticalSection, Task, ValidationError, analyze


def demo_task_set():
    # Priority order = list order.  'bus' is shared by all three tasks;
    # 'log' is private to the lowest-priority task.
    return [
        Task("ctrl", wcet=2, period=8, deadline=7, jitter=1,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("sense", wcet=3, period=12, deadline=12,
             critical_sections=(CriticalSection("bus", 2),)),
        Task("logger", wcet=4, period=20, deadline=20,
             critical_sections=(CriticalSection("bus", 3),
                                CriticalSection("log", 1))),
    ]


def print_chain(report):
    util = report.utilization
    print(f"utilization U = sum(C/T) = {util} (~{float(util):.4f}) "
          f"-> {'<= 1, necessary condition holds' if report.utilization_ok else '> 1, cannot be guaranteed'}")
    for task in report.tasks:
        b = task.blocking
        if b.amount:
            src = f"from task '{b.source_task}' on resource '{b.source_resource}'"
        else:
            src = "no lower-priority critical section can block it"
        print(f"\ntask '{task.name}': B = {b.amount} ({src})")
        for it in task.iterations:
            interference = ", ".join(
                f"{name}: ceil(({it.value}+J)/T)={n}" for name, n in it.interference
            ) or "no higher-priority tasks"
            print(f"  round {it.round}: R={it.value} -> [{interference}] "
                  f"-> R'={it.next_value}")
        print(f"  => {task.status}: {task.detail}")
    print(f"\nverdict: guaranteed={report.guaranteed}")
    print(f"summary: {report.summary}")


def main():
    print("=== Part 1: normal analysis with full derivation chain ===")
    report = analyze(demo_task_set())
    print_chain(report)

    print("\n=== Part 2a: invalid input is rejected with a locatable error ===")
    bad = Task("ctrl", wcet=2, period=8, deadline=7, jitter=float("nan"))
    try:
        analyze([bad])
    except ValidationError as exc:
        print(f"ValidationError caught as designed: {exc}")

    print("\n=== Part 2b: utilization > 1 is reported, not raised ===")
    overloaded = analyze([Task("heavy", wcet=3, period=2, deadline=2)])
    print(f"utilization = {overloaded.utilization} (> 1)")
    print(f"guaranteed={overloaded.guaranteed}; summary: {overloaded.summary}")


if __name__ == "__main__":
    main()
