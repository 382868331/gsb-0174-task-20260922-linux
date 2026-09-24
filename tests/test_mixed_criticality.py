"""Mixed-criticality (LO/HI) analysis tests.

Every expected value below is hand-derived from the per-mode recurrences
    LO      R = C_i(LO)  + sum_{h in hp}        ceil((R+J_h)/T_h) * C_h(LO)
    HI      R = C_i(HI)  + sum_{h in hp(HI)}    ceil((R+J_h)/T_h) * C_h(HI)
    AMC-rtb R = C_i(HI)  + sum_{h in hp}        ceil((R+J_h)/T_h) * C_h(mode of h)
and written down literally; no expectation is produced by calling the code
under test.  Simulator schedules are enumerated by hand on a small time axis.
"""

import unittest
from fractions import Fraction

from rt_analysis import (
    CRITICALITY_HI,
    CRITICALITY_LO,
    MC_BOUND_KNOWN,
    MC_BOUND_MISS,
    MC_BOUND_UNKNOWN,
    Task,
    ValidationError,
    analyze,
    analyze_mc,
    simulate_mc,
)


def mc_example():
    """Priority order = list order.

    h: HI, C_LO=1, C_HI=2, T=6,  D=6
    m: LO, C=3,        T=12, D=12
    l: HI, C_LO=2, C_HI=4, T=24, D=24

    Hand derivation (all J = 0):
      U_LO = 1/6 + 3/12 + 2/24 = 1/2
      U_HI = 2/6 + 4/24        = 1/2
      h: LO R=1; HI R=2; AMC-rtb R=2 (no higher-priority tasks)
      m: LO only: R0=3 -> ceil(3/6)*1=1 -> 4; R=4 -> 1 -> 4 (stable)
      l: LO:   R0=2 -> ceil(2/6)*1 + ceil(2/12)*3 = 1+3=4 -> 6
                   R=6 -> ceil(6/6)*1 + ceil(6/12)*3 = 4 -> 6 (stable)
         HI:   only h at C_HI=2: R0=4 -> ceil(4/6)*2=2 -> 6
                   R=6 -> ceil(6/6)*2=2 -> 6 (stable)
         AMC:  own 4; h at C_HI=2,T=6; m at C_LO=3,T=12:
                   R0=4 -> ceil(4/6)*2 + ceil(4/12)*3 = 2+3=5 -> 9
                   R=9 -> ceil(9/6)*2 + ceil(9/12)*3 = 4+3=7 -> 11
                   R=11-> ceil(11/6)*2 + ceil(11/12)*3= 4+3=7 -> 11 (stable)
    """
    return [
        Task("h", wcet=1, period=6, deadline=6,
             criticality=CRITICALITY_HI, wcet_hi=2),
        Task("m", wcet=3, period=12, deadline=12,
             criticality=CRITICALITY_LO),
        Task("l", wcet=2, period=24, deadline=24,
             criticality=CRITICALITY_HI, wcet_hi=4),
    ]


class TestMCNormalAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analyze_mc(mc_example())

    def test_overall_guaranteed(self):
        self.assertTrue(self.report.guaranteed)
        self.assertTrue(self.report.utilization_ok)

    def test_dual_utilizations_exact(self):
        self.assertEqual(self.report.utilization_lo, Fraction(1, 2))
        self.assertEqual(self.report.utilization_hi, Fraction(1, 2))

    def test_lo_bounds(self):
        bounds = {t.name: t.lo for t in self.report.tasks}
        for name, expected in (("h", 1), ("m", 4), ("l", 6)):
            self.assertEqual(bounds[name].status, MC_BOUND_KNOWN, name)
            self.assertEqual(bounds[name].value, expected, name)

    def test_hi_bounds(self):
        h, m, l = self.report.tasks
        self.assertEqual(h.hi.status, MC_BOUND_KNOWN)
        self.assertEqual(h.hi.value, 2)
        self.assertIsNone(m.hi)  # LO tasks are not scheduled in HI mode
        self.assertEqual(l.hi.status, MC_BOUND_KNOWN)
        self.assertEqual(l.hi.value, 6)

    def test_amc_rtb_bounds(self):
        h, m, l = self.report.tasks
        self.assertEqual(h.amc_rtb.value, 2)
        self.assertIsNone(m.amc_rtb)
        self.assertEqual(l.amc_rtb.status, MC_BOUND_KNOWN)
        self.assertEqual(l.amc_rtb.value, 11)

    def test_amc_iteration_chain_for_l(self):
        amc = next(t for t in self.report.tasks if t.name == "l").amc_rtb
        self.assertEqual([r.value for r in amc.iterations], [4, 9, 11])
        self.assertEqual([r.next_value for r in amc.iterations], [9, 11, 11])
        counts = [dict(r.interference) for r in amc.iterations]
        self.assertEqual([c["h"] for c in counts], [1, 2, 2])
        self.assertEqual([c["m"] for c in counts], [1, 1, 1])

    def test_hi_bound_uses_only_hi_interference(self):
        # In the steady HI recurrence m (LO) never appears as a contributor.
        hi_l = next(t for t in self.report.tasks if t.name == "l").hi
        for rnd in hi_l.iterations:
            self.assertEqual([name for name, _ in rnd.interference], ["h"])

    def test_report_shape_mirrors_priority_order(self):
        self.assertEqual([t.name for t in self.report.tasks], ["h", "m", "l"])
        self.assertEqual(
            [t.criticality for t in self.report.tasks],
            [CRITICALITY_HI, CRITICALITY_LO, CRITICALITY_HI],
        )
        self.assertTrue(all(t.guaranteed for t in self.report.tasks))


class TestMCBoundariesAndUnknown(unittest.TestCase):
    def test_exact_hi_deadline_boundary_is_known(self):
        # Single HI task: HI bound R = C_HI = 3 == D-J exactly -> known.
        task = Task("x", wcet=1, period=8, deadline=3,
                    criticality=CRITICALITY_HI, wcet_hi=3)
        report = analyze_mc([task])
        self.assertEqual(report.tasks[0].hi.value, 3)
        self.assertEqual(report.tasks[0].amc_rtb.value, 3)
        self.assertTrue(report.guaranteed)

    def test_one_past_hi_deadline_misses_but_lo_holds(self):
        # LO budget fits (R=1), HI budget misses (R=3 > D-J=2): the two
        # modes are judged independently and the miss is a proven miss.
        task = Task("x", wcet=1, period=8, deadline=2,
                    criticality=CRITICALITY_HI, wcet_hi=3)
        report = analyze_mc([task])
        t = report.tasks[0]
        self.assertEqual(t.lo.status, MC_BOUND_KNOWN)
        self.assertEqual(t.lo.value, 1)
        self.assertEqual(t.hi.status, MC_BOUND_MISS)
        self.assertIsNone(t.hi.value)
        self.assertEqual(t.amc_rtb.status, MC_BOUND_MISS)
        self.assertFalse(t.guaranteed)
        self.assertFalse(report.guaranteed)
        self.assertTrue(report.utilization_ok)  # U_HI = 3/8 <= 1, still misses

    def test_unknown_is_distinct_from_miss_and_known(self):
        # l's AMC-rtb bound needs three rounds (4 -> 9 -> 11 -> 11).  With a
        # one-round budget the first candidate (9) is still within D-J=24,
        # so it is neither a proven miss nor a converged bound: unknown.
        report = analyze_mc(mc_example(), max_iterations=1)
        l = next(t for t in report.tasks if t.name == "l")
        self.assertEqual(l.amc_rtb.status, MC_BOUND_UNKNOWN)
        self.assertIsNone(l.amc_rtb.value)
        self.assertNotIn(l.amc_rtb.status, (MC_BOUND_KNOWN, MC_BOUND_MISS))
        # h's bounds converge in a single round and stay known.
        h = next(t for t in report.tasks if t.name == "h")
        self.assertEqual(h.hi.status, MC_BOUND_KNOWN)
        # With the normal budget the same task set is fully guaranteed.
        self.assertTrue(analyze_mc(mc_example()).guaranteed)

    def test_equal_lo_and_hi_budgets_accepted(self):
        task = Task("x", wcet=2, period=8, deadline=8,
                    criticality=CRITICALITY_HI, wcet_hi=2)
        report = analyze_mc([task])
        self.assertEqual(report.tasks[0].lo.value, 2)
        self.assertEqual(report.tasks[0].hi.value, 2)
        self.assertEqual(report.tasks[0].amc_rtb.value, 2)


class TestMCHighUtilizationReportedNotRaised(unittest.TestCase):
    def test_hi_utilization_above_one_reported(self):
        # C_HI/T = 3/2 > 1 while C_LO/T = 1/2: the HI mode cannot be
        # guaranteed but this is not a format error.
        task = Task("x", wcet=1, period=2, deadline=2,
                    criticality=CRITICALITY_HI, wcet_hi=3)
        report = analyze_mc([task])
        self.assertFalse(report.utilization_ok)
        self.assertEqual(report.utilization_lo, Fraction(1, 2))
        self.assertEqual(report.utilization_hi, Fraction(3, 2))
        self.assertFalse(report.guaranteed)


class TestMCValidation(unittest.TestCase):
    def test_hi_task_without_wcet_hi_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([Task("x", wcet=1, period=4, deadline=4,
                             criticality=CRITICALITY_HI)])
        self.assertIn("wcet_hi", str(ctx.exception))
        self.assertIn("x", str(ctx.exception))

    def test_lo_task_with_wcet_hi_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([Task("x", wcet=1, period=4, deadline=4,
                             criticality=CRITICALITY_LO, wcet_hi=2)])
        self.assertIn("LO-criticality", str(ctx.exception))

    def test_wcet_hi_smaller_than_wcet_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([Task("x", wcet=3, period=8, deadline=8,
                             criticality=CRITICALITY_HI, wcet_hi=2)])
        self.assertIn("wcet_hi", str(ctx.exception))

    def test_bad_criticality_label_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([Task("x", wcet=1, period=4, deadline=4,
                             criticality="MED", wcet_hi=2)])
        self.assertIn("criticality", str(ctx.exception))

    def test_bool_wcet_hi_rejected(self):
        with self.assertRaises(ValidationError):
            analyze_mc([Task("x", wcet=1, period=4, deadline=4,
                             criticality=CRITICALITY_HI, wcet_hi=True)])

    def test_shared_resources_out_of_scope_rejected(self):
        from rt_analysis import CriticalSection

        task = Task("x", wcet=2, period=8, deadline=8,
                    criticality=CRITICALITY_HI, wcet_hi=3,
                    critical_sections=(CriticalSection("bus", 1),))
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([task])
        self.assertIn("shared resources", str(ctx.exception))

    def test_lo_only_set_rejected_for_mc(self):
        with self.assertRaises(ValidationError) as ctx:
            analyze_mc([Task("x", wcet=1, period=4, deadline=4)])
        self.assertIn("HI-criticality", str(ctx.exception))

    def test_max_iterations_validated(self):
        task = Task("x", wcet=1, period=4, deadline=4,
                    criticality=CRITICALITY_HI, wcet_hi=2)
        with self.assertRaises(ValidationError):
            analyze_mc([task], max_iterations=0)

    def test_failure_leaves_no_partial_state(self):
        # analyze_mc is a pure function: a failing call cannot affect a
        # subsequent valid call on the same (immutable) task objects.
        tasks = mc_example()
        with self.assertRaises(ValidationError):
            analyze_mc(tasks, max_iterations=0)
        self.assertTrue(analyze_mc(tasks).guaranteed)


class TestBackwardCompatibility(unittest.TestCase):
    def test_analyze_still_accepts_extended_tasks(self):
        # Existing entry point keeps working; it uses the LO budget only.
        report = analyze(mc_example())
        self.assertEqual([t.response_time for t in report.tasks], [1, 4, 6])
        self.assertTrue(report.guaranteed)

    def test_plain_lo_task_defaults(self):
        # criticality/wcet_hi default to a plain LO task.
        report = analyze_mc(
            [Task("hi", wcet=1, period=4, deadline=4,
                  criticality=CRITICALITY_HI, wcet_hi=1),
             Task("lo", wcet=1, period=8, deadline=8)]
        )
        lo = next(t for t in report.tasks if t.name == "lo")
        self.assertEqual(lo.criticality, CRITICALITY_LO)
        self.assertIsNone(lo.hi)
        self.assertIsNone(lo.amc_rtb)
        self.assertEqual(lo.lo.value, 2)  # R0=1 -> ceil(1/4)*1=1 -> 2


class TestMCSimulation(unittest.TestCase):
    def test_pure_lo_run_respects_hand_lo_bounds(self):
        # No switch: HI jobs execute their LO budget only.
        result = simulate_mc(mc_example(), 24)  # lcm(6, 12, 24)
        self.assertEqual(result.incomplete, ())
        self.assertEqual(result.dropped, ())
        worst = result.max_response()
        # Literal hand bounds for the generated schedule:
        self.assertEqual(worst["h"], 1)
        self.assertEqual(worst["m"], 4)
        self.assertEqual(worst["l"], 6)  # tight against the LO bound
        self.assertFalse(any(j.switched for j in result.jobs))

    def test_mode_switch_drops_lo_and_respects_amc_bounds(self):
        # Hand-enumerated schedule with switch at t=3 (see module docstring
        # derivation): m@0 is mid-execution and gets dropped; l@0 is upgraded
        # to its HI budget and finishes at t=9 (response 9 <= AMC bound 11);
        # every post-switch h job takes 2 (<= HI bound 2).
        result = simulate_mc(mc_example(), 24, switch_time=3)
        self.assertEqual(result.incomplete, ())
        self.assertEqual(result.dropped, (("m", 0),))

        m_jobs = [j for j in result.jobs if j.task == "m"]
        self.assertEqual(m_jobs, [])  # m@0 dropped; no new LO jobs released
        self.assertNotIn("m", result.max_response())

        l0 = next(j for j in result.jobs if j.task == "l" and j.release == 0)
        self.assertTrue(l0.switched)
        self.assertEqual(l0.response, 9)
        self.assertLessEqual(l0.response, 11)

        h_switched = [j for j in result.jobs if j.task == "h" and j.switched]
        self.assertTrue(h_switched)
        for job in h_switched:
            self.assertEqual(job.response, 2)

    def test_switch_at_release_instant_starts_no_lo_job(self):
        # m releases at 0 and 12; a switch exactly at 12 admits no m job at 12.
        result = simulate_mc(mc_example(), 24, switch_time=12)
        self.assertEqual(
            sorted(j.release for j in result.jobs if j.task == "m"), [0]
        )

    def test_sim_respects_analytic_bounds_generic(self):
        # Supplementary round trip across several switch instants: every
        # observed completed HI job stays within its applicable bound.
        report = analyze_mc(mc_example())
        lo_b = {t.name: t.lo.value for t in report.tasks}
        amc_b = {t.name: t.amc_rtb.value for t in report.tasks if t.amc_rtb}
        for switch in (None, 1, 3, 6, 12):
            result = simulate_mc(mc_example(), 24, switch_time=switch)
            self.assertEqual(result.incomplete, (), switch)
            for job in result.jobs:
                bound = amc_b.get(job.task) if job.switched else lo_b[job.task]
                self.assertLessEqual(job.response, bound, (switch, job))

    def test_sim_validation(self):
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 0)
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 24, switch_time=0)
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 24, switch_time=25)
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 24, switch_time=True)
        with self.assertRaises(ValidationError):
            simulate_mc([Task("x", wcet=1, period=4, deadline=4)], 10)


if __name__ == "__main__":
    unittest.main()
