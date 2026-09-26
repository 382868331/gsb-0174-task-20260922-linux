"""Deterministic offline demo: response-time bounds under a periodic
supply window (last Q of every P units executable).

Runs entirely offline with the standard library only, in well under 90
seconds.  No sleep, no network, no prerecorded results: every number is
computed live from the library.

Part 1 analyzes a two-task set under a P=4, Q=1 window and prints every
iteration round with its demand, guaranteed supply and stop reason.
Part 2 shows that Q == P degenerates exactly to the always-available
baseline.  Part 3 demonstrates that a low-utilization set can still miss
its deadline once the blackout is charged (and that averaging C/(Q/P)
would wrongly pass it).  Part 4 enumerates every distinct arrival phase
on a small integer time axis as a concrete-schedule reference.
"""

from rt_analysis import (
    SupplyWindow,
    Task,
    analyze,
    simulate_phases,
    supply_bound,
    supply_inverse,
)

WINDOW = SupplyWindow(period=4, budget=2)  # blackout of 2, then 2 executable units
TIGHT = SupplyWindow(period=4, budget=1)   # extreme: only the last unit runs


def print_report(report):
    util = report.utilization
    print(f"utilization U = {util} (~{float(util):.4f}) "
          f"-> {'<= 1' if report.utilization_ok else '> 1'}")
    for task in report.tasks:
        print(f"task '{task.name}': B = {task.blocking.amount}")
        for it in task.iterations:
            interference = ", ".join(
                f"{name} x{n}" for name, n in it.interference
            ) or "no higher-priority tasks"
            print(f"  round {it.round}: R={it.value:>3}  supply(R)={it.supply:>3}"
                  f"  [{interference}]  demand={it.demand:>3}"
                  f"  -> R'={it.next_value}")
        print(f"  => {task.status}: {task.detail}")
    print(f"verdict: guaranteed={report.guaranteed}")
    print(f"summary: {report.summary}")


def main():
    print("=== Part 1: windowed analysis, P=4 Q=2 (last 2 units of each period) ===")
    print("supply_bound(t) for t=0..12:",
          [supply_bound(t, WINDOW) for t in range(13)])
    print("supply_inverse(x) for x=1..4:",
          [supply_inverse(x, WINDOW) for x in range(1, 5)])
    tasks = [
        Task("hi", wcet=2, period=8, deadline=8),
        Task("lo", wcet=2, period=16, deadline=16),
    ]
    print_report(analyze(tasks, supply=WINDOW))

    print("\n=== Part 2: Q == P degenerates to the baseline model ===")
    full = SupplyWindow(period=4, budget=4)
    base = analyze(tasks)
    deg = analyze(tasks, supply=full)
    print("baseline R:", [r.response_time for r in base.tasks])
    print("Q=P window R:", [r.response_time for r in deg.tasks])
    assert [r.response_time for r in base.tasks] == [r.response_time for r in deg.tasks]
    print("identical bounds, as required")

    print("\n=== Part 3: utilization 1/2, yet 'lo' misses its deadline ===")
    tight = [
        Task("hi", wcet=2, period=8, deadline=8),
        Task("lo", wcet=3, period=12, deadline=12),
    ]
    report = analyze(tight, supply=TIGHT)
    print_report(report)
    naive = 3 / (TIGHT.budget / TIGHT.period)
    print(f"(naive C/(Q/P) for 'lo' alone would say {naive:.0f} <= 12: "
          f"the blackout makes that unsafe)")

    print("\n=== Part 4: phase enumeration as scheduling reference ===")
    sweep = simulate_phases(tasks, 32, supply=WINDOW)
    print(f"enumerated {len(sweep.phases)} distinct phases "
          f"(lcm of window period and task periods)")
    for name in ("hi", "lo"):
        print(f"  task '{name}': worst observed response "
              f"{sweep.worst_response()[name]} at phase {sweep.worst_phase(name)}")
    bound = analyze(tasks, supply=WINDOW)
    for task_report in bound.tasks:
        observed = sweep.worst_response()[task_report.name]
        assert observed <= task_report.response_time
        print(f"  observed {observed} <= bound {task_report.response_time} "
              f"for '{task_report.name}'")


if __name__ == "__main__":
    main()
