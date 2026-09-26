"""Offline demo for the periodic last-Q supply-window response-time check.

Everything here is computed on the spot with the standard library only:
no network, accounts, keys, sleep, or pre-recorded results.

Scenarios:
  1. Short windows, schedulable set: every iteration prints its demand,
     minimum guaranteed supply, next candidate, and the final stop reason.
     Integer small-period simulation then enumerates every arrival phase as
     an independent scheduling reference and prints the observed worst
     response next to the analytic bound.
  2. Utilization well below 1 but a deadline is still missed because the
     first supply window opens too late (proven miss, not budget exhaustion).
  3. Q = P degenerates to the original always-available model.
"""

from math import gcd

from rt_analysis import (
    SupplyWindow,
    Task,
    analyze,
    minimum_supply,
    simulate,
    supply_time_for,
)


def lcm(a, b):
    return a * b // gcd(a, b)


def print_rounds(report):
    for tr in report.tasks:
        print(f"task '{tr.name}' (status={tr.status}):")
        for it in tr.iterations:
            inter = ", ".join(f"{name}x{n}" for name, n in it.interference) \
                or "no higher-priority tasks"
            print(
                f"  round {it.round}: R={it.value} | interference: {inter} "
                f"| demand={it.demand} | sbf({it.value})={it.supply} "
                f"| next R={it.next_value}"
            )
        print(f"  stop reason: {it.stop_reason}")
        print(f"  detail: {tr.detail}")


def phase_cross_check(tasks, supply, bound_by_name):
    period = supply.period
    for task in tasks:
        period = lcm(period, task.period)
    horizon = 2 * period
    result = simulate(tasks, horizon, supply=supply, phases=range(period))
    worst = result.max_response()
    print(f"  enumerated {period} arrival phase(s) over horizon {horizon}; "
          f"observed worst response vs analytic bound:")
    for name, bound in bound_by_name.items():
        observed = worst.get(name)
        if bound is None:
            verdict = "analytic miss (jobs can run past their deadline)"
        else:
            verdict = f"{observed} <= bound {bound}"
        print(f"    {name}: {verdict}")


def main():
    print("=== Supply function sanity: P=5, Q=3 (open slots 2,3,4) ===")
    sw = SupplyWindow(5, 3)
    print("  sbf(t) for t=0..9 =", [minimum_supply(t, sw) for t in range(10)])
    print("  supply_time_for(demand) for demand=1..4 =",
          [supply_time_for(d, sw) for d in range(1, 5)])
    print("  (note sbf(2)=0: t*Q/P=1.2 would be unsound; worst phase counts)")

    print("\n=== Scenario 1: short windows, set still schedulable ===")
    tasks = [Task("h", 1, 8, 8), Task("l", 2, 16, 16)]
    report = analyze(tasks, supply=sw)
    print(f"  utilization={report.utilization}; guaranteed={report.guaranteed}")
    print_rounds(report)
    phase_cross_check(
        tasks, sw, {t.name: t.response_time for t in report.tasks}
    )

    print("\n=== Scenario 2: low utilization but deadline miss ===")
    # U = 1/8 + 2/20 = 9/40, yet the first open slot is slot 6 of each
    # period of length 8, so even one unit of work cannot finish by D=4.
    miss_tasks = [Task("h", 1, 8, 4), Task("l", 2, 20, 20)]
    miss_sw = SupplyWindow(8, 2)
    miss_report = analyze(miss_tasks, supply=miss_sw)
    print(f"  utilization={miss_report.utilization} (<= 1); "
          f"guaranteed={miss_report.guaranteed}")
    print_rounds(miss_report)

    print("\n=== Scenario 3: Q = P degenerates to the baseline model ===")
    full = SupplyWindow(7, 7)
    base_tasks = [Task("a", 2, 8, 8), Task("b", 3, 12, 12)]
    plain = analyze(base_tasks)
    with_window = analyze(base_tasks, supply=full)
    print("  plain analyze:    ",
          [(t.name, t.response_time) for t in plain.tasks])
    print("  with SupplyWindow(7,7):",
          [(t.name, t.response_time) for t in with_window.tasks])
    assert [t.response_time for t in plain.tasks] == \
           [t.response_time for t in with_window.tasks]
    print("  identical bounds confirmed.")


if __name__ == "__main__":
    main()
