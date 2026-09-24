"""Deterministic demo: worst-case response-time bounds for fixed-priority sets.

Runs entirely offline with the standard library only.  Parts 1-2 cover the
single-criticality resource/PCP model: a full derivation chain and two
rejection / boundary paths.  Parts 3-5 cover the mixed-criticality (LO/HI)
extension: three per-task bounds (LO, HI, AMC-rtb transition), a mode-switch
simulation instance, and real boundary/error paths (a proven HI-mode deadline
miss with the LO bound still known, an invalid task rejected, and the
no-shared-resources scope enforced).
"""

from rt_analysis import (
    CRITICALITY_HI,
    CRITICALITY_LO,
    MC_BOUND_KNOWN,
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


def mc_task_set():
    # h: HI  (C_LO=1, C_HI=2); m: LO (C=3); l: HI (C_LO=2, C_HI=4).
    return [
        Task("h", wcet=1, period=6, deadline=6,
             criticality=CRITICALITY_HI, wcet_hi=2),
        Task("m", wcet=3, period=12, deadline=12,
             criticality=CRITICALITY_LO),
        Task("l", wcet=2, period=24, deadline=24,
             criticality=CRITICALITY_HI, wcet_hi=4),
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


def print_mc(report):
    print(f"LO utilization = {report.utilization_lo} "
          f"(~{float(report.utilization_lo):.4f}); "
          f"HI utilization = {report.utilization_hi} "
          f"(~{float(report.utilization_hi):.4f})")
    for t in report.tasks:
        print(f"\ntask '{t.name}' ({t.criticality}):")
        for b in (t.lo, t.hi, t.amc_rtb):
            if b is None:
                print("     N/A: not scheduled in HI mode")
                continue
            shown = (
                f"R={b.value}" if b.status == MC_BOUND_KNOWN
                else f"R=unknown ({b.status})"
            )
            print(f"  {b.kind:>7}: {shown} <= D-J={b.limit}  [{b.status}]")
            for it in b.iterations:
                interference = ", ".join(
                    f"{name}:{n}" for name, n in it.interference
                ) or "none"
                print(f"           round {it.round}: R={it.value} "
                      f"+ [{interference}] -> {it.next_value}")
        print(f"           task guaranteed={t.guaranteed}")
    print(f"\nverdict: guaranteed={report.guaranteed}")
    print(f"summary: {report.summary}")


def main():
    print("=== Part 1: normal PCP analysis with full derivation chain ===")
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

    print("\n=== Part 3: mixed-criticality LO/HI analysis (normal result) ===")
    mc_report = analyze_mc(mc_task_set())
    print_mc(mc_report)

    print("\n=== Part 4: concrete mode-switch instance (switch at t=3) ===")
    sim = simulate_mc(mc_task_set(), 24, switch_time=3)
    print(f"LO jobs dropped at the switch: {sim.dropped}")
    for job in sim.jobs:
        tag = "HI budget" if job.switched else "LO budget"
        print(f"  {job.task:>1} released {job.release:>2} completed {job.completion:>2} "
              f"response={job.response} ({tag})")
    worst = sim.max_response()
    print(f"observed maxima: h={worst['h']} (<= AMC bound 2), "
          f"l={worst['l']} (<= AMC bound 11); m has no completed HI-mode jobs")

    print("\n=== Part 5a: HI-mode boundary — exact deadline vs one past it ===")
    ok = analyze_mc([Task("x", wcet=1, period=8, deadline=3,
                          criticality=CRITICALITY_HI, wcet_hi=3)])
    print(f"D-J=3 == R_HI=3 -> {ok.tasks[0].hi.status} "
          f"({ok.tasks[0].hi.value}), guaranteed={ok.guaranteed}")
    miss = analyze_mc([Task("x", wcet=1, period=8, deadline=2,
                            criticality=CRITICALITY_HI, wcet_hi=3)])
    t = miss.tasks[0]
    print(f"D-J=2 <  R_HI=3 -> LO bound {t.lo.status}={t.lo.value}, "
          f"HI bound {t.hi.status}, AMC bound {t.amc_rtb.status}; "
          f"guaranteed={miss.guaranteed}")

    print("\n=== Part 5b: invalid mixed-criticality input is rejected ===")
    try:
        analyze_mc([Task("x", wcet=3, period=8, deadline=8,
                         criticality=CRITICALITY_HI, wcet_hi=2)])
    except ValidationError as exc:
        print(f"ValidationError caught as designed: {exc}")

    print("\n=== Part 5c: shared resources are outside the MC scope ===")
    try:
        analyze_mc([Task("x", wcet=2, period=8, deadline=8,
                         criticality=CRITICALITY_HI, wcet_hi=3,
                         critical_sections=(CriticalSection("bus", 1),))])
    except ValidationError as exc:
        print(f"ValidationError caught as designed: {exc}")


if __name__ == "__main__":
    main()
