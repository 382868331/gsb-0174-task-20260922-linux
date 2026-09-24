"""Deterministic demo: fixed-priority response-time bounds, plain and mixed.

Runs entirely offline with the standard library only.

* Part 1 analyzes a plain three-task set and prints the full derivation
  chain (utilization, resource ceilings, blocking terms with sources, every
  iteration round, and the verdict).
* Part 2 triggers plain rejection / boundary paths: invalid input rejected
  with a locatable error, and utilization > 1 reported rather than raised.
* Part 3 analyzes a mixed-criticality set (LO/HI modes, two budgets) and
  prints both per-mode derivation chains.
* Part 4 shows the mixed-criticality boundary (bound == deadline is
  guaranteed; one more time unit is a proven miss), a real mode switch in
  the tick simulator (low jobs dropped after the switch), and an actual
  invalid-input rejection.
"""

from rt_analysis import (
    CRITICALITY_HI,
    CRITICALITY_LO,
    MCTask,
    OVERRUN_ALL,
    CriticalSection,
    Task,
    ValidationError,
    analyze,
    analyze_mc,
    simulate_mc,
)


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


def mc_task_set():
    # Priority order = list order: h1 (high) > L (low) > h2 (high).
    return [
        MCTask("h1", wcet_lo=1, period=10, deadline=10,
               criticality=CRITICALITY_HI, wcet_hi=2),
        MCTask("L", wcet_lo=1, period=10, deadline=10,
               criticality=CRITICALITY_LO),
        MCTask("h2", wcet_lo=2, period=20, deadline=20,
               criticality=CRITICALITY_HI, wcet_hi=3),
    ]


def print_mc_chain(report):
    print(f"U(LO) = {report.utilization_lo} (~{float(report.utilization_lo):.4f}), "
          f"U(HI) = {report.utilization_hi} (~{float(report.utilization_hi):.4f})")
    for task in report.tasks:
        print(f"\ntask '{task.name}' ({task.criticality})")
        for md in (task.lo, task.hi):
            if md is None:
                continue
            for it in md.iterations:
                hi_part = ", ".join(f"{n}:{name}@C(HI)" for name, n in it.hi_interference)
                lo_part = ", ".join(f"{n}:{name}@C(LO)" for name, n in it.lo_interference)
                counts = "; ".join(p for p in (hi_part, lo_part) if p) or "none"
                print(f"  {md.mode.upper()} round {it.round}: R={it.value} -> "
                      f"[{counts}] -> R'={it.next_value}")
            print(f"  => {md.mode.upper()} {md.status}: {md.detail}")
        print(f"  overall: {task.status}, guaranteed bound = {task.response_time}")
    print(f"\nverdict: guaranteed={report.guaranteed}")
    print(f"summary: {report.summary}")


def main():
    print("=== Part 1: plain analysis with full derivation chain ===")
    print_chain(analyze(demo_task_set()))

    print("\n=== Part 2a: invalid plain input is rejected with a locatable error ===")
    bad = Task("ctrl", wcet=2, period=8, deadline=7, jitter=float("nan"))
    try:
        analyze([bad])
    except ValidationError as exc:
        print(f"ValidationError caught as designed: {exc}")

    print("\n=== Part 2b: utilization > 1 is reported, not raised ===")
    overloaded = analyze([Task("heavy", wcet=3, period=2, deadline=2)])
    print(f"utilization = {overloaded.utilization} (> 1)")
    print(f"guaranteed={overloaded.guaranteed}; summary: {overloaded.summary}")

    print("\n=== Part 3: mixed-criticality analysis (LO/HI modes, two budgets) ===")
    print_mc_chain(analyze_mc(mc_task_set()))

    print("\n=== Part 4a: mixed-criticality deadline boundary ===")
    # Dedicated tight set: h1(HI): C(LO)=1 C(HI)=2 T=6; L(LO): C=2 T=6;
    # h2(HI): C(LO)=1 T=12 D=12, HI budget varied.
    def boundary_set(chi):
        return [
            MCTask("h1", wcet_lo=1, period=6, deadline=6,
                   criticality=CRITICALITY_HI, wcet_hi=2),
            MCTask("L", wcet_lo=2, period=6, deadline=6,
                   criticality=CRITICALITY_LO),
            MCTask("h2", wcet_lo=1, period=12, deadline=12,
                   criticality=CRITICALITY_HI, wcet_hi=chi),
        ]

    at_bound = analyze_mc(boundary_set(4))
    print(f"wcet_hi(h2)=4: R(HI)={at_bound.tasks[2].hi.bound} == D=12 -> "
          f"{at_bound.tasks[2].hi.status}, guaranteed={at_bound.guaranteed}")
    past_bound = analyze_mc(boundary_set(5))
    print(f"wcet_hi(h2)=5: {past_bound.tasks[2].hi.detail} -> "
          f"guaranteed={past_bound.guaranteed}")

    print("\n=== Part 4b: real mode switch in the tick simulator ===")
    sim = simulate_mc(mc_task_set(), 40, overruns=(("h2", 0),))
    print(f"h2@0 exceeds its LO budget; mode switches at t={sim.mode_switch}")
    h2 = next(j for j in sim.jobs if j.task == "h2" and j.release == 0)
    print(f"h2@0 completes at t={h2.completion} (response {h2.response}, "
          f"<= analytic HI bound 6)")
    print(f"low-criticality jobs never started after the switch: {sim.abandoned}")

    print("\n=== Part 4c: invalid mixed-criticality input is rejected ===")
    bad_mc = MCTask("L", wcet_lo=1, period=4, deadline=4,
                    criticality=CRITICALITY_LO, wcet_hi=1)
    try:
        analyze_mc([bad_mc])
    except ValidationError as exc:
        print(f"ValidationError caught as designed: {exc}")
    try:
        simulate_mc(mc_task_set(), 10, overruns=(("L", 0),))
    except ValidationError as exc:
        print(f"simulator rejection as designed: {exc}")
    print(f"(OVERRUN_ALL selector = {OVERRUN_ALL!r})")


if __name__ == "__main__":
    main()
