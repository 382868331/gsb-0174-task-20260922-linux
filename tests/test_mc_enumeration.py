"""Independent small-scale enumeration cross-check.

A deliberately independent, minimal tick scheduler is written *in this file*
(it shares no scheduling code with rt_analysis.simulate_mc) and driven over a
small deterministic grid of task sets and overrun scenarios.  Sound,
one-directional check: whenever analyze_mc guarantees a set, every completed
job released inside the first hyperperiod, in every enumerated scenario, must
respect that task's overall bound.  Expected values are never taken from the
implementation; the reference is the independently written scheduler below
plus the analytic fixed-priority critical-instant argument.
"""

import unittest
from itertools import product
from math import gcd

from rt_analysis import CRITICALITY_HI, MCTask, analyze_mc


# ---------------------------------------------------------------------------
# Independent reference scheduler.
# A raw task is (name, c_lo, c_hi, period, is_high); an overrun HI job gets
# execution requirement c_hi.  When a job executes past its c_lo budget the
# system switches and no low job ever executes again.
# ---------------------------------------------------------------------------


def reference_schedule(raw_tasks, horizon, overruns):
    """Return {(task name, release): observed response} for finished jobs."""
    jobs = []
    for idx, (_name, c_lo, c_hi, period, _is_high) in enumerate(raw_tasks):
        release = 0
        while release < horizon:
            requirement = c_hi if (idx, release) in overruns else c_lo
            jobs.append({"i": idx, "rel": release, "prog": 0,
                         "need": requirement, "done": False})
            release += period

    mode_lo = True
    responses = {}
    for tick in range(horizon):
        chosen = None
        for k in sorted(range(len(jobs)), key=lambda e: jobs[e]["i"]):
            job = jobs[k]
            if job["done"] or job["rel"] > tick:
                continue
            if not mode_lo and not raw_tasks[job["i"]][4]:
                continue  # low tasks do not run after the switch
            chosen = job
            break
        if chosen is None:
            continue
        idx = chosen["i"]
        c_lo, c_hi, is_high = raw_tasks[idx][1], raw_tasks[idx][2], raw_tasks[idx][4]
        chosen["prog"] += 1
        if mode_lo and is_high and chosen["need"] == c_hi and chosen["prog"] > c_lo:
            mode_lo = False
        if chosen["prog"] == chosen["need"]:
            chosen["done"] = True
            responses[(raw_tasks[idx][0], chosen["rel"])] = (
                tick + 1 - chosen["rel"]
            )
    return responses


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

PERIODS = (4, 8)
LO_BUDGETS = (1, 2)
HI_DELTAS = (0, 2)  # c_hi = c_lo + delta
PATTERNS = (("lo", "hi"), ("hi", "lo"), ("hi", "hi"))


def generate_sets():
    seen = set()
    for pat in PATTERNS:
        for t0, t1 in product(PERIODS, PERIODS):
            for c0, c1 in product(LO_BUDGETS, LO_BUDGETS):
                for d0, d1 in product(HI_DELTAS, HI_DELTAS):
                    specs = []
                    for crit, t, c, d in zip(pat, (t0, t1), (c0, c1), (d0, d1)):
                        specs.append((crit, c, c + d if crit == "hi" else 0, t))
                    key = tuple(specs)
                    if key in seen:
                        continue
                    seen.add(key)
                    raw = [
                        (f"t{i}", c_lo, c_hi, t, crit == "hi")
                        for i, (crit, c_lo, c_hi, t) in enumerate(specs)
                    ]
                    mc = [
                        MCTask(
                            f"t{i}", wcet_lo=c_lo, period=t, deadline=t,
                            criticality=CRITICALITY_HI if crit == "hi" else "lo",
                            wcet_hi=c_hi if crit == "hi" else None,
                        )
                        for i, (crit, c_lo, c_hi, t) in enumerate(specs)
                    ]
                    yield raw, mc


class TestEnumerationRespectsBounds(unittest.TestCase):
    def test_grid(self):
        guaranteed_sets = 0
        scenarios = 0
        switch_scenarios = 0
        for raw, mc in generate_sets():
            report = analyze_mc(mc)
            if not report.guaranteed:
                continue
            guaranteed_sets += 1
            bound = {t.name: t.response_time for t in report.tasks}
            p0, p1 = mc[0].period, mc[1].period
            window = p0 * p1 // gcd(p0, p1)
            horizon = 2 * window + 8  # margin so early jobs can finish

            cases = [frozenset()]  # pure LO schedule
            for idx, (_n, _cl, ch, t, is_high) in enumerate(raw):
                if is_high and ch > _cl:
                    release = 0
                    while release < window:
                        cases.append(frozenset({(idx, release)}))
                        release += t
            # Every HI job with headroom overruns simultaneously.
            cases.append(frozenset(
                (idx, r)
                for idx, (_n, _cl, ch, t, is_high) in enumerate(raw)
                if is_high and ch > _cl
                for r in range(0, horizon, t)
            ))

            for overruns in cases:
                scenarios += 1
                if overruns:
                    switch_scenarios += 1
                responses = reference_schedule(raw, horizon, overruns)
                for (name, release), response in responses.items():
                    # The horizon margin gives jobs released in the first
                    # hyperperiod room to finish; unfinished ones are simply
                    # absent from the map (one-directional, sound check).
                    if release < window:
                        self.assertLessEqual(
                            response, bound[name],
                            f"set={raw} overruns={set(overruns)} "
                            f"job={name}@{release} response={response} "
                            f"> bound={bound[name]}",
                        )
        # Sanity on the enumeration itself: it covered real switch scenarios.
        self.assertGreater(guaranteed_sets, 20)
        self.assertGreater(switch_scenarios, 20)


if __name__ == "__main__":
    unittest.main()
