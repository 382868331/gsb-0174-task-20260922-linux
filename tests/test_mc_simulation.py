"""Mixed-criticality simulator cross-check.

The simulator generates real schedules on a small integer time axis.  Every
expected response / switch time / abandoned set below is derived by hand on
the timeline (and written literally); the analytic bounds used for the
"observed <= bound" checks are the hand-derived values documented in
test_mixed.mc_example, not numbers produced by the implementation.
"""

import unittest

from rt_analysis import (
    CRITICALITY_HI,
    CRITICALITY_LO,
    MCTask,
    OVERRUN_ALL,
    ValidationError,
    simulate_mc,
)


def mc_example():
    """Same set as test_mixed.mc_example; hand bounds:

    R(LO): h1 = 1, L = 2, h2 = 4.
    R(HI): h1 = 2, h2 = 6.
    """
    return [
        MCTask("h1", wcet_lo=1, period=10, deadline=10,
               criticality=CRITICALITY_HI, wcet_hi=2),
        MCTask("L", wcet_lo=1, period=10, deadline=10,
               criticality=CRITICALITY_LO),
        MCTask("h2", wcet_lo=2, period=20, deadline=20,
               criticality=CRITICALITY_HI, wcet_hi=3),
    ]


LO_BOUNDS = {"h1": 1, "L": 2, "h2": 4}
HI_BOUNDS = {"h1": 2, "h2": 6}


class TestLOModeSchedule(unittest.TestCase):
    def test_no_overrun_is_plain_fixed_priority_schedule(self):
        result = simulate_mc(mc_example(), 40)
        self.assertIsNone(result.mode_switch)
        self.assertEqual(result.abandoned, ())
        self.assertEqual(result.incomplete, ())
        # Hand timeline, first hyperperiod: h1@0 [0,1); L@0 [1,2);
        # h2@0 [2,4); releases at 10 (h1, L) and 20 (all three) repeat.
        self.assertEqual(result.max_response(), {"h1": 1, "L": 2, "h2": 4})
        for job in result.jobs:
            self.assertFalse(job.overrun)
            self.assertLessEqual(job.response, LO_BOUNDS[job.task])


class TestModeSwitchSchedule(unittest.TestCase):
    def test_overrun_switches_mode_and_abandons_low_jobs(self):
        # h2's job released at 0 overruns to its HI budget of 3.
        result = simulate_mc(mc_example(), 40, overruns=(("h2", 0),))

        # Hand timeline: h1@0 [0,1); L@0 [1,2); h2@0 executes [2,4), its
        # two LO units; needing a third unit at t=4 exceeds the LO budget ->
        # the mode switches at 4; h2 runs [4,5), response 5.
        self.assertEqual(result.mode_switch, 4)
        h2_first = next(j for j in result.jobs if j.task == "h2" and j.release == 0)
        self.assertTrue(h2_first.overrun)
        self.assertEqual((h2_first.completion, h2_first.response), (5, 5))
        self.assertLessEqual(h2_first.response, HI_BOUNDS["h2"])

        # L@0 finished just before the switch; L releases at 10, 20, 30 never
        # run again and are abandoned (not counted as incomplete).
        l_jobs = [j for j in result.jobs if j.task == "L"]
        self.assertEqual([(j.release, j.completion) for j in l_jobs], [(0, 2)])
        self.assertEqual(result.abandoned,
                         (("L", 10), ("L", 20), ("L", 30)))
        self.assertEqual(result.incomplete, ())

    def test_post_switch_high_job_runs_at_hi_budget(self):
        # h2@0 triggers the switch; h1's job released at 10 also overruns and
        # needs 2 units in HI mode.
        result = simulate_mc(
            mc_example(), 40, overruns=(("h2", 0), ("h1", 10))
        )
        h1_at10 = next(j for j in result.jobs if j.task == "h1" and j.release == 10)
        self.assertTrue(h1_at10.overrun)
        # No higher-priority job interferes: response 2 == R(HI) of h1.
        self.assertEqual(h1_at10.response, HI_BOUNDS["h1"])
        h2_first = next(j for j in result.jobs if j.task == "h2" and j.release == 0)
        self.assertEqual(h2_first.response, 5)
        self.assertLessEqual(h2_first.response, HI_BOUNDS["h2"])
        self.assertEqual(result.abandoned,
                         (("L", 10), ("L", 20), ("L", 30)))

    def test_all_selector_switches_on_first_overrun(self):
        result = simulate_mc(mc_example(), 20, overruns=OVERRUN_ALL)
        self.assertIsNotNone(result.mode_switch)
        flagged = {(j.task, j.release) for j in result.jobs if j.overrun}
        self.assertIn(("h2", 0), flagged)
        self.assertIn(("h1", 10), flagged)
        # Every job completed after the switch is high-criticality.
        self.assertTrue(all(
            j.completion <= result.mode_switch or j.criticality == CRITICALITY_HI
            for j in result.jobs
        ))


class TestHorizonEdge(unittest.TestCase):
    def test_overrun_cut_short_by_horizon_is_incomplete(self):
        # h2@0 fills its LO budget over [2,4) and the switch is at t=4; the
        # horizon ends exactly then with one HI unit outstanding, so the HI
        # job is incomplete (not abandoned).
        result = simulate_mc(mc_example(), 4, overruns=(("h2", 0),))
        self.assertEqual(result.mode_switch, 4)
        self.assertEqual(result.incomplete, (("h2", 0),))
        self.assertEqual(result.abandoned, ())

    def test_low_job_released_but_unstarted_is_abandoned_at_switch(self):
        # Priority H > L, both released at 0.  H overruns: [0,1) finishes its
        # LO budget and switches at t=1, completes [1,3).  L@0 never started
        # and is abandoned by the switch (distinct from incomplete).
        tasks = [
            MCTask("H", wcet_lo=1, period=10, deadline=10,
                   criticality=CRITICALITY_HI, wcet_hi=3),
            MCTask("L", wcet_lo=4, period=20, deadline=20,
                   criticality=CRITICALITY_LO),
        ]
        result = simulate_mc(tasks, 8, overruns=(("H", 0),))
        self.assertEqual(result.mode_switch, 1)
        self.assertEqual(result.abandoned, (("L", 0),))
        self.assertEqual(result.incomplete, ())
        h = next(j for j in result.jobs if j.task == "H")
        self.assertEqual((h.completion, h.response), (3, 3))


class TestMCSimulateValidation(unittest.TestCase):
    def test_invalid_horizon(self):
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 0)
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), True)

    def test_invalid_tasks_propagate(self):
        with self.assertRaises(ValidationError):
            simulate_mc([], 10)
        with self.assertRaises(ValidationError):
            simulate_mc(
                [MCTask("l", 1, 4, 4, CRITICALITY_LO, wcet_hi=1)], 10
            )

    def test_unknown_overrun_task(self):
        with self.assertRaises(ValidationError) as ctx:
            simulate_mc(mc_example(), 10, overruns=(("ghost", 0),))
        self.assertIn("ghost", str(ctx.exception))

    def test_low_task_cannot_overrun(self):
        with self.assertRaises(ValidationError) as ctx:
            simulate_mc(mc_example(), 10, overruns=(("L", 0),))
        self.assertIn("low-criticality", str(ctx.exception))

    def test_release_must_match_period(self):
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 10, overruns=(("h2", 5),))
        with self.assertRaises(ValidationError) as ctx:
            simulate_mc(mc_example(), 10, overruns=(("h2", -20),))
        self.assertIn("horizon", str(ctx.exception))

    def test_release_outside_horizon(self):
        with self.assertRaises(ValidationError) as ctx:
            simulate_mc(mc_example(), 10, overruns=(("h2", 20),))
        self.assertIn("horizon", str(ctx.exception))

    def test_no_headroom_job_cannot_be_named(self):
        tasks = [
            MCTask("h", 1, 4, 4, CRITICALITY_HI, wcet_hi=1),
        ]
        with self.assertRaises(ValidationError) as ctx:
            simulate_mc(tasks, 10, overruns=(("h", 0),))
        self.assertIn("wcet_hi == wcet_lo", str(ctx.exception))

    def test_bad_overruns_structure(self):
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 10, overruns="banana")
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 10, overruns=(("h2",),))
        with self.assertRaises(ValidationError):
            simulate_mc(mc_example(), 10, overruns=42)


if __name__ == "__main__":
    unittest.main()
