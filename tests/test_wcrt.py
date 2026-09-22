"""wcrt.analyze 的测试。

期望值来源：
  1) 纸笔手算的整数递推（写在每个断言附近）；
  2) tests/timeline.py 中独立实现的离散时间轴 ICPP 调度器（不导入 wcrt）。
timeline 只用于"实际实例响应 <= 分析界"的上界核对与 miss 复现，
不作为往返一致的唯一证据。
"""

import math
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wcrt  # noqa: E402
from wcrt import (  # noqa: E402
    AnalysisResult,
    BudgetExhaustedError,
    Resource,
    Task,
    TaskSetError,
    analyze,
)
from timeline import simulate  # noqa: E402  (tests 目录由 discover 加入 sys.path)


def bounds(result):
    return [it.bound for it in result.iterations]


def classic_tasks():
    # 无资源经典三任务；手算见各断言。
    return [
        Task("t1", priority=1, c=1, t=4, d=4),
        Task("t2", priority=2, c=2, t=6, d=6),
        Task("t3", priority=3, c=3, t=12, d=12),
    ]


class TestHandComputedRecurrence(unittest.TestCase):
    def setUp(self):
        self.res = analyze(classic_tasks())

    def test_utilization_exact(self):
        # 1/4 + 2/6 + 3/12 = 5/6
        self.assertEqual(self.res.utilization_exact, Fraction(5, 6))
        self.assertFalse(self.res.utilization_over)
        self.assertTrue(self.res.guaranteed)

    def test_t1_highest_priority(self):
        r = self.res.task("t1")
        # R0 = C = 1，无干扰，确认轮仍为 1
        self.assertEqual(bounds(r), [1, 1])
        self.assertEqual(r.response_time, 1)
        self.assertEqual(r.blocking, 0)
        self.assertIsNone(r.blocking_source)
        self.assertEqual(r.deadline_limit, 4)

    def test_t2_recurrence(self):
        r = self.res.task("t2")
        # R0=2; R1=2+ceil(2/4)*1=3; R2=2+ceil(3/4)*1=3 稳定
        self.assertEqual(bounds(r), [2, 3, 3])
        self.assertEqual(r.response_time, 3)
        it1 = r.iterations[1]
        self.assertEqual(it1.interference, {"t1": 1})
        self.assertEqual(it1.interference_workload, 1)
        it2 = r.iterations[2]
        self.assertEqual(it2.interference, {"t1": 1})

    def test_t3_recurrence_full_chain(self):
        r = self.res.task("t3")
        # R0=3
        # R1=3+ceil(3/4)*1+ceil(3/6)*2 =3+1+2=6
        # R2=3+ceil(6/4)*1+ceil(6/6)*2 =3+2+2=7
        # R3=3+ceil(7/4)*1+ceil(7/6)*2 =3+2+4=9
        # R4=3+ceil(9/4)*1+ceil(9/6)*2 =3+3+4=10
        # R5=3+ceil(10/4)*1+ceil(10/6)*2=3+3+4=10 稳定
        self.assertEqual(bounds(r), [3, 6, 7, 9, 10, 10])
        self.assertEqual(r.response_time, 10)
        self.assertEqual(r.iterations[1].interference, {"t1": 1, "t2": 1})
        self.assertEqual(r.iterations[2].interference, {"t1": 2, "t2": 1})
        self.assertEqual(r.iterations[3].interference, {"t1": 2, "t2": 2})
        self.assertEqual(r.iterations[4].interference, {"t1": 3, "t2": 2})
        self.assertEqual(r.iterations[5].interference, {"t1": 3, "t2": 2})
        self.assertEqual(r.iterations[3].interference_workload, 6)
        self.assertEqual(r.iterations[0].blocking, 0)
        self.assertIsNone(r.iterations[0].blocking_source)


class TestTimelineUpperBound(unittest.TestCase):
    """独立时间轴：临界瞬时同步释放下，实际响应不得超过分析界。"""

    def test_observed_responses_within_bounds(self):
        res = analyze(classic_tasks())
        specs = [
            {"name": "t1", "prio": 1, "c": 1, "t": 4, "d": 4},
            {"name": "t2", "prio": 2, "c": 2, "t": 6, "d": 6},
            {"name": "t3", "prio": 3, "c": 3, "t": 12, "d": 12},
        ]
        records = simulate(specs, ceilings={}, horizon=24)
        worst = {}
        for rec in records:
            worst[rec.task] = max(worst.get(rec.task, -1), rec.response)
        # 临界瞬时（全部 t=0 释放）下界是紧的：首作业逐个达到界。
        self.assertEqual(worst["t1"], 1)
        self.assertEqual(worst["t2"], 3)
        self.assertEqual(worst["t3"], 10)
        for name, observed in worst.items():
            self.assertLessEqual(observed, res.task(name).response_time)

    def test_long_low_priority_lock_blocks_high_priority(self):
        # R 被 hi 与 lo 共用 => 给定天花板 = 1。
        tasks = [
            Task("hi", 1, c=1, t=8, d=8, sections=[("R", 1)]),
            Task("mid", 2, c=2, t=10, d=10),
            Task("lo", 3, c=4, t=20, d=20, sections=[("R", 3)]),
        ]
        res = analyze(tasks, [Resource("R", ceiling=1)])
        hi, mid, lo = res.task("hi"), res.task("mid"), res.task("lo")
        # B_hi = 3 来自 lo 的 (R,3)
        self.assertEqual(hi.blocking, 3)
        self.assertEqual(hi.blocking_source, ("lo", "R", 3))
        self.assertEqual(hi.response_time, 4)  # R0=1+3=4，无高优先级干扰
        # mid 不使用 R，但天花板 1 <= P_mid=2，ICPP 下同样被挡：B_mid=3
        self.assertEqual(mid.blocking, 3)
        self.assertEqual(mid.blocking_source, ("lo", "R", 3))
        # R0=5; R1=5+ceil(5/8)=6; R2=6+ceil(6/8)=6
        self.assertEqual(bounds(mid), [5, 6, 6])
        self.assertEqual(lo.blocking, 0)

        # 时间轴：lo 在 t=0 持锁运行到 3，hi 在 t=1 释放被天花板挡住。
        specs = [
            {"name": "hi", "prio": 1, "c": 1, "t": 8, "d": 8,
             "sections": [("R", 1)]},
            {"name": "mid", "prio": 2, "c": 2, "t": 10, "d": 10},
            {"name": "lo", "prio": 3, "c": 4, "t": 20, "d": 20,
             "sections": [("R", 3)]},
        ]
        records = simulate(
            specs, {"R": 1}, horizon=12, phases={"hi": 1, "mid": 2}
        )
        first_hi = next(r for r in records if r.task == "hi")
        self.assertEqual(first_hi.response, 3)       # 3 完成 - 1 释放
        self.assertEqual(first_hi.ceiling_blocked, 2)  # t=1,2 被挡
        self.assertLessEqual(first_hi.response, hi.response_time)
        # 该作业实际满足截止期，但保守界 4 仍给出保证（D=8）。
        self.assertTrue(hi.guaranteed)

    def test_long_lock_conservative_rejection(self):
        # lo 的 6 tick 长锁使 hi 的界 R=C+B=2+6=8 > D-J=7 => miss。
        tasks = [
            Task("hi", 1, c=2, t=10, d=7, sections=[("R", 1)]),
            Task("lo", 2, c=7, t=20, d=20, sections=[("R", 6)]),
        ]
        res = analyze(tasks, [Resource("R", 1)])
        hi = res.task("hi")
        self.assertEqual(hi.status, "miss")
        self.assertIsNone(hi.response_time)
        self.assertEqual(hi.last_bound, 8)
        self.assertEqual(hi.blocking_source, ("lo", "R", 6))
        self.assertFalse(res.guaranteed)

        # 时间轴上界仍然成立：phase=1 时 hi 被挡 5 tick，观测响应 7 <= 8。
        specs = [
            {"name": "hi", "prio": 1, "c": 2, "t": 10, "d": 7,
             "sections": [("R", 1)]},
            {"name": "lo", "prio": 2, "c": 7, "t": 20, "d": 20,
             "sections": [("R", 6)]},
        ]
        records = simulate(specs, {"R": 1}, horizon=24, phases={"hi": 1})
        first_hi = next(r for r in records if r.task == "hi")
        self.assertEqual(first_hi.response, 7)
        self.assertLessEqual(first_hi.response, hi.last_bound)


class TestJitter(unittest.TestCase):
    def _tasks(self):
        return [
            Task("hi", 1, c=7, t=10, d=10, j=4),
            Task("lo", 2, c=5, t=20, d=20),
        ]

    def test_jitter_causes_miss_chain(self):
        res = analyze(self._tasks())
        hi, lo = res.task("hi"), res.task("lo")
        # hi 自身：D-J = 6，R=7 > 6，初始轮即 miss
        self.assertEqual(hi.deadline_limit, 6)
        self.assertEqual(hi.status, "miss")
        self.assertEqual(bounds(hi), [7])
        # lo：R0=5
        #  R1=5+ceil((5+4)/10)*7 =12
        #  R2=5+ceil((12+4)/10)*7=19
        #  R3=5+ceil((19+4)/10)*7=26 >20 => miss（J=0 时该集合可保证，见下）
        self.assertEqual(bounds(lo), [5, 12, 19, 26])
        self.assertEqual(lo.status, "miss")
        self.assertIsNone(lo.response_time)
        self.assertFalse(res.guaranteed)

    def test_same_set_without_jitter_is_guaranteed(self):
        tasks = [
            Task("hi", 1, c=7, t=10, d=10),
            Task("lo", 2, c=5, t=20, d=20),
        ]
        res = analyze(tasks)
        lo = res.task("lo")
        # 5 -> 12 -> 19 -> 19 稳定，恰好 <= 20
        self.assertEqual(bounds(lo), [5, 12, 19, 19])
        self.assertTrue(res.guaranteed)

    def test_timeline_reproduces_jitter_miss(self):
        specs = [
            {"name": "hi", "prio": 1, "c": 7, "t": 10, "d": 10, "j": 4},
            {"name": "lo", "prio": 2, "c": 5, "t": 20, "d": 20},
        ]
        # 对抗相位（标准抖动 RTA 的最坏情况）：hi 首个作业在 -J 到达、
        # 延迟满 J 恰好在 0 释放（积压作业），之后作业到达即释放：
        # 释放序列 0, 6, 16, 26。lo 在 0 临界瞬时释放。
        records = simulate(
            specs,
            {},
            horizon=30,
            jitter=lambda name, k: 4 if name == "hi" and k == 0 else 0,
            phases={"hi": -4},
        )
        first_lo = next(r for r in records if r.task == "lo")
        # hi 释放序列 0,6,16,26；占用槽 [0,7) [7,14) [16,23)，
        # lo 的 5 个执行槽为 14,15,23,24,25，完成于 26。
        self.assertEqual(first_lo.response, 26)    # 独立调度器实跑
        self.assertGreater(first_lo.response, 20)  # 时间轴上确实错过截止期
        self.assertLessEqual(first_lo.response, 26)  # 且不超过分析 miss 界


class TestZeroBlockingAndExactDeadline(unittest.TestCase):
    def test_zero_blocking_with_unused_ceiling_resource(self):
        # hi 独自使用 R：天花板=1，没有更低优先级临界区，B_hi=0。
        tasks = [Task("hi", 1, c=2, t=5, d=5, sections=[("R", 1)])]
        res = analyze(tasks, [Resource("R", 1)])
        hi = res.task("hi")
        self.assertEqual(hi.blocking, 0)
        self.assertIsNone(hi.blocking_source)
        self.assertEqual(hi.response_time, 2)

    def test_response_exactly_at_deadline(self):
        # hi C=4 T=8；lo C=6 T=16 D=16：
        # R0=6; R1=6+ceil(6/8)*4=10; R2=6+ceil(10/8)*4=14;
        # R3=6+ceil(14/8)*4=14 稳定，R == D（恰好截止 => 可保证）。
        tasks = [
            Task("hi", 1, c=4, t=8, d=8),
            Task("lo", 2, c=6, t=16, d=16),
        ]
        res = analyze(tasks)
        lo = res.task("lo")
        self.assertEqual(bounds(lo), [6, 10, 14, 14])
        self.assertEqual(lo.response_time, 14)
        self.assertTrue(lo.guaranteed)

        # 时间轴临界瞬时同样得到 14。
        specs = [
            {"name": "hi", "prio": 1, "c": 4, "t": 8, "d": 8},
            {"name": "lo", "prio": 2, "c": 6, "t": 16, "d": 16},
        ]
        records = simulate(specs, {}, horizon=32)
        first_lo = next(r for r in records if r.task == "lo")
        self.assertEqual(first_lo.response, 14)
        self.assertEqual(first_lo.response, lo.response_time)

    def test_initial_bound_equals_limit_is_guaranteed(self):
        # 单任务 C=D=3：R0=3 == D-J=3，可保证（不越界）。
        res = analyze([Task("solo", 1, c=3, t=3, d=3)])
        r = res.task("solo")
        self.assertTrue(r.guaranteed)
        self.assertEqual(r.response_time, 3)
        self.assertEqual(bounds(r), [3, 3])

    def test_initial_bound_one_over_limit_misses(self):
        # C=4 > D=3：R0=4 > 3，初始轮即 miss。
        res = analyze([Task("solo", 1, c=4, t=5, d=3)])
        r = res.task("solo")
        self.assertFalse(r.guaranteed)
        self.assertEqual(r.status, "miss")
        self.assertIsNone(r.response_time)
        self.assertEqual(bounds(r), [4])


class TestUtilizationSemantics(unittest.TestCase):
    def test_utilization_over_one_is_not_format_error(self):
        tasks = [
            Task("a", 1, c=6, t=8, d=8),
            Task("b", 2, c=5, t=8, d=8),
        ]
        res = analyze(tasks)  # U=11/8 > 1：不抛异常
        self.assertIsInstance(res, AnalysisResult)
        self.assertTrue(res.utilization_over)
        self.assertEqual(res.utilization_exact, Fraction(11, 8))
        self.assertFalse(res.guaranteed)
        # 逐任务结果仍然给出。
        self.assertEqual([r.task for r in res.tasks], ["a", "b"])

    def test_utilization_under_one_is_not_sufficient(self):
        # U=4/6=2/3 < 1，但 D=3 < C=4，必然 miss。
        res = analyze([Task("a", 1, c=4, t=6, d=3)])
        self.assertFalse(res.utilization_over)
        self.assertFalse(res.guaranteed)
        self.assertEqual(res.task("a").status, "miss")


class TestInputValidation(unittest.TestCase):
    def test_nan_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            analyze([Task("a", 1, c=float("nan"), t=5, d=5)])

    def test_infinity_rejected(self):
        with self.assertRaisesRegex(ValueError, "Infinity"):
            analyze([Task("a", 1, c=4, t=float("inf"), d=4)])
        with self.assertRaisesRegex(ValueError, "Infinity"):
            analyze([Task("a", 1, c=4, t=5, d=4, j=float("-inf"))])

    def test_bool_rejected_for_integer_params(self):
        for kw in (
            dict(c=True),
            dict(t=False),
            dict(d=True),
            dict(j=False),
            dict(priority=True),
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(TypeError):
                    analyze([Task("a", 1, c=1, t=5, d=5, **kw)])

    def test_float_integer_rejected(self):
        with self.assertRaises(TypeError):
            analyze([Task("a", 1, c=2.0, t=5, d=5)])

    def test_d_greater_than_t(self):
        with self.assertRaisesRegex(TaskSetError, "d <= t"):
            analyze([Task("a", 1, c=1, t=5, d=6)])

    def test_nonpositive_and_negative(self):
        with self.assertRaises(TaskSetError):
            analyze([Task("a", 1, c=0, t=5, d=5)])
        with self.assertRaises(TaskSetError):
            analyze([Task("a", 1, c=1, t=0, d=0)])
        with self.assertRaisesRegex(TaskSetError, "j"):
            analyze([Task("a", 1, c=1, t=5, d=5, j=-1)])

    def test_duplicate_names_and_priorities(self):
        with self.assertRaisesRegex(TaskSetError, "任务名重复"):
            analyze([
                Task("a", 1, c=1, t=5, d=5),
                Task("a", 2, c=1, t=6, d=6),
            ])
        with self.assertRaisesRegex(TaskSetError, "优先级必须唯一"):
            analyze([
                Task("a", 1, c=1, t=5, d=5),
                Task("b", 1, c=1, t=6, d=6),
            ])

    def test_too_many_tasks(self):
        tasks = [Task(f"t{i}", i, c=1, t=100, d=100) for i in range(21)]
        with self.assertRaisesRegex(TaskSetError, "上限 20"):
            analyze(tasks)
        # 恰好 20 个合法。
        ok = [Task(f"t{i}", i, c=1, t=100, d=100) for i in range(20)]
        self.assertTrue(analyze(ok).guaranteed)

    def test_unknown_resource_and_overlong_sections(self):
        with self.assertRaisesRegex(TaskSetError, "未在 resources"):
            analyze(
                [Task("a", 1, c=2, t=5, d=5, sections=[("X", 1)])],
                [Resource("R", 1)],
            )
        with self.assertRaisesRegex(TaskSetError, "临界区长度之和"):
            analyze(
                [Task("a", 1, c=2, t=5, d=5, sections=[("R", 2), ("S", 1)])],
                [Resource("R", 1), Resource("S", 1)],
            )

    def test_section_length_must_be_positive(self):
        with self.assertRaisesRegex(TaskSetError, "正整数"):
            analyze(
                [Task("a", 1, c=3, t=5, d=5, sections=[("R", 0)])],
                [Resource("R", 1)],
            )

    def test_ceiling_mismatch_and_dangling(self):
        tasks = [
            Task("hi", 1, c=1, t=5, d=5, sections=[("R", 1)]),
            Task("lo", 2, c=2, t=9, d=9, sections=[("R", 1)]),
        ]
        with self.assertRaisesRegex(TaskSetError, "天花板"):
            analyze(tasks, [Resource("R", ceiling=2)])
        with self.assertRaisesRegex(TaskSetError, "没有任何任务使用"):
            analyze(
                [Task("a", 1, c=1, t=5, d=5)],
                [Resource("R", 1)],
            )

    def test_duplicate_resource(self):
        with self.assertRaisesRegex(TaskSetError, "资源名重复"):
            analyze(
                [Task("a", 1, c=2, t=5, d=5, sections=[("R", 1)])],
                [Resource("R", 1), Resource("R", 1)],
            )

    def test_wrong_container_and_element_types(self):
        with self.assertRaises(TypeError):
            analyze("not-a-sequence")
        with self.assertRaises(TypeError):
            analyze([object()])
        with self.assertRaises(TypeError):
            analyze(
                [Task("a", 1, c=2, t=5, d=5)],
                resources=[("R", 1)],
            )

    def test_empty_name_rejected(self):
        # 名称在 analyze() 校验时报可定位错误（dataclass 本身不校验）。
        with self.assertRaises(TypeError):
            analyze([Task("", 1, c=1, t=5, d=5)])
        with self.assertRaises(TypeError):
            analyze([Task("a", 1, c=1, t=5, d=5, sections=[("", 1)])],
                    [Resource("", 1)])


class TestBudgetVsMiss(unittest.TestCase):
    def test_budget_exhaustion_is_distinct_from_miss(self):
        # 该集合本可正常收敛（t3 需要 5 轮确认）；人为把硬预算压到 2，
        # 必须抛 BudgetExhaustedError 而不是误报 miss。
        saved = wcrt.MAX_ITERATIONS_HARD
        wcrt.MAX_ITERATIONS_HARD = 2
        try:
            with self.assertRaises(BudgetExhaustedError):
                analyze(classic_tasks())
        finally:
            wcrt.MAX_ITERATIONS_HARD = saved
        # 恢复后正常可保证，证明两种结论严格分开。
        self.assertTrue(analyze(classic_tasks()).guaranteed)


class TestImmutabilityAndStructure(unittest.TestCase):
    def test_result_frozen_and_sorted_by_priority(self):
        res = analyze(list(reversed(classic_tasks())))
        self.assertEqual([r.task for r in res.tasks], ["t1", "t2", "t3"])
        with self.assertRaises(Exception):
            res.guaranteed = False  # type: ignore[misc]

    def test_iteration_indices(self):
        res = analyze(classic_tasks())
        for r in res.tasks:
            self.assertEqual(
                [it.index for it in r.iterations],
                list(range(len(r.iterations))),
            )


class TestRandomizedTimelineCrossCheck(unittest.TestCase):
    """固定种子的随机小规模交叉核对（补充证据，非期望值来源）。

    对每个可保证任务：临界瞬时（全体 t=0 释放、无抖动）下时间轴实跑的
    首个作业响应，无资源时必须恰好等于分析界，有资源时不得超过分析界。
    """

    def test_seeded_sets(self):
        import random

        rng = random.Random(20260922)
        checked = 0
        for _ in range(60):
            n = rng.randint(1, 4)
            prios = list(range(1, n + 1))
            use_res = rng.random() < 0.5
            tasks, specs = [], []
            users_of_r = []
            for i in range(n):
                t = rng.randint(4, 12)
                c = rng.randint(1, 3)
                sections = []
                spec_sec = []
                if use_res and c >= 2 and rng.random() < 0.6:
                    z = rng.randint(1, c - 1)
                    sections, spec_sec = [("R", z)], [("R", z)]
                    users_of_r.append(prios[i])
                tasks.append(
                    Task(f"t{i+1}", prios[i], c=c, t=t, d=t, sections=sections)
                )
                specs.append(
                    {"name": f"t{i+1}", "prio": prios[i], "c": c, "t": t,
                     "d": t, "sections": spec_sec}
                )
            # 给定天花板必须与使用者最高优先级一致；悬空资源不合法。
            resources = [Resource("R", min(users_of_r))] if users_of_r else []
            res = analyze(tasks, resources)
            if not res.guaranteed:
                continue  # miss 集合不在此核对（miss 界是停止值，非上界）
            records = simulate(
                specs, {"R": resources[0].ceiling} if resources else {},
                horizon=40,
            )
            first_resp = {}
            for rec in records:
                if rec.task not in first_resp:
                    first_resp[rec.task] = rec.response
            for r in res.tasks:
                observed = first_resp[r.task]
                if not use_res:
                    self.assertEqual(observed, r.response_time)
                else:
                    self.assertLessEqual(observed, r.response_time)
                checked += 1
        self.assertGreater(checked, 10)  # 确保确实核对了足量实例


if __name__ == "__main__":
    unittest.main()
