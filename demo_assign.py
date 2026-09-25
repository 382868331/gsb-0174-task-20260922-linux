"""Deterministic demo for the priority-assignment entry point.

Runs entirely offline with the standard library only (well under 90 s;
typically a few milliseconds).

Part 1: a task set with low utilization (U = 2/5) whose input order fails
through priority-ceiling blocking; ``assign_priorities`` finds a different
order and the final per-round response-time evidence is printed.

Part 2: a task set with even lower utilization (U = 9/50) for which the
search proves — exhaustively, within the analysis model — that no priority
order can be guaranteed.
"""

from rt_analysis import CriticalSection, Task, analyze, assign_priorities


def reordering_example():
    # ctrl never touches 'bus', but in the input order it sits below a 'bus'
    # user, so log's 5-unit critical section blocks it past its deadline.
    return [
        Task("comm", wcet=2, period=16, deadline=16,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("ctrl", wcet=1, period=8, deadline=2),
        Task("log", wcet=6, period=40, deadline=40,
             critical_sections=(CriticalSection("bus", 5),)),
    ]


def hopeless_example():
    # urgent's deadline (2) admits neither worker's 4-unit blocking (urgent
    # on top) nor worker's 4-unit interference (worker on top).
    return [
        Task("urgent", wcet=1, period=10, deadline=2,
             critical_sections=(CriticalSection("bus", 1),)),
        Task("worker", wcet=4, period=50, deadline=50,
             critical_sections=(CriticalSection("bus", 4),)),
    ]


def print_report(report):
    util = report.utilization
    print(f"  utilization U = {util} (~{float(util):.4f})")
    for task in report.tasks:
        b = task.blocking
        src = (f"from '{b.source_task}' on '{b.source_resource}'"
               if b.amount else "no blocking")
        print(f"  task '{task.name}': B = {b.amount} ({src})")
        for it in task.iterations:
            interference = ", ".join(
                f"{name}: ceil(({it.value}+J)/T)={n}" for name, n in it.interference
            ) or "no higher-priority tasks"
            print(f"    round {it.round}: R={it.value} -> [{interference}] "
                  f"-> R'={it.next_value}")
        print(f"    => {task.status}: {task.detail}")
    print(f"  guaranteed={report.guaranteed}")


def main():
    print("=== Part 1: low utilization, input order fails through blocking;")
    print("    a reordered priority assignment is found and verified ===")
    tasks = reordering_example()
    print("input order [comm, ctrl, log] analyzed as-is:")
    print_report(analyze(tasks))

    result = assign_priorities(tasks)
    print(f"\nsearch: status={result.status}, "
          f"nodes_explored={result.nodes_explored}")
    for p in result.placements:
        print(f"  [{p.phase}] level {p.level}: try '{p.candidate}' "
              f"-> {p.outcome} ({p.detail})")
    print(f"\nfound order (highest priority first): {result.order}")
    print("final per-round response-time evidence for the found order:")
    print_report(result.report)

    print("\n=== Part 2: low utilization, no feasible order under the model ===")
    tasks = hopeless_example()
    print("input order [urgent, worker] analyzed as-is:")
    print_report(analyze(tasks))
    result = assign_priorities(tasks)
    print(f"\nsearch: status={result.status}, "
          f"nodes_explored={result.nodes_explored}")
    print(f"summary: {result.summary}")


if __name__ == "__main__":
    main()
