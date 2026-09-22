"""固定输入演示：真实运行 wcrt.analyze 并打印推导链。

运行：python demo.py
内容：
  1) 正常结果：3 任务 + 1 个优先级天花板资源，逐轮打印
     R_{k+1}=C+B+Σ ceil((R_k+J_h)/T_h)·C_h 的干扰次数、阻塞来源与判定；
     并用 tests/timeline.py 的独立时间轴调度器跑一个具体实例，
     证明观测响应不超过分析界（实例不超过界）。
  2) 抖动导致 miss：同一集合提高高优先级任务抖动 J，逐轮打印到越界。
  3) 实际触发的拒绝边界：构造含 NaN 的输入，真实调用并捕获错误。

无 sleep、无预录结果、无网络。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wcrt import Resource, Task, analyze  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from timeline import simulate  # noqa: E402


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_chain(result):
    print(f"任务 {result.task}: 判定阈值 D-J = {result.deadline_limit}，"
          f"阻塞 B = {result.blocking}", end="")
    if result.blocking_source:
        tname, rname, length = result.blocking_source
        print(f"（来源：低优先级任务 {tname} 在资源 {rname} 上的 {length} 临界区）")
    else:
        print("（零阻塞）")
    for it in result.iterations:
        if it.index == 0:
            print(f"  R0 = C + B = {it.bound}")
            continue
        terms = ", ".join(f"{n}×{name}" for name, n in it.interference.items()) or "无"
        flag = ""
        if it.bound > result.deadline_limit:
            flag = "  -> 超过 D-J，证明 miss"
        elif it.bound == result.iterations[it.index - 1].bound:
            flag = "  -> 与上轮相等，迭代稳定，可保证"
        print(f"  R{it.index} = C + B + Σ ceil((R{it.index-1}+J)/T)·C "
              f"= {it.bound}（干扰作业数 {{{terms}}}，合计 {it.interference_workload}）{flag}")
    print(f"  结论：{result.status}"
          + (f"，最坏响应时间界 R = {result.response_time}" if result.guaranteed
             else f"，最后界 {result.last_bound} > {result.deadline_limit}"))


def main():
    # ------------------------------------------------------------------
    # 1) 正常可保证集合：3 任务，R 被最高与最低优先级任务共用（天花板=1）。
    # ------------------------------------------------------------------
    hr("场景 1：正常结果（固定优先级 + 优先级天花板资源 R，禁止嵌套）")
    resources = [Resource("R", ceiling=1)]
    tasks = [
        Task("tau_hi", priority=1, c=1, t=8,  d=8,
             sections=[("R", 1)]),
        Task("tau_md", priority=2, c=2, t=10, d=10),
        Task("tau_lo", priority=3, c=4, t=24, d=24,
             sections=[("R", 3)]),
    ]
    res = analyze(tasks, resources)
    print(f"精确利用率 U = {res.utilization_exact} ≈ {res.utilization:.6f}"
          f"（>1={res.utilization_over}；U≤1 只是必要条件，不作充分保证）")
    for r in res.tasks:
        print_chain(r)
    print(f"\n整集结论：guaranteed = {res.guaranteed}")

    # 独立时间轴：低优先级 t=0 持锁，hi 在 t=1 释放被天花板挡到 t=3。
    specs = [
        {"name": "tau_hi", "prio": 1, "c": 1, "t": 8, "d": 8,
         "sections": [("R", 1)]},
        {"name": "tau_md", "prio": 2, "c": 2, "t": 10, "d": 10},
        {"name": "tau_lo", "prio": 3, "c": 4, "t": 24, "d": 24,
         "sections": [("R", 3)]},
    ]
    records = simulate(specs, {"R": 1}, horizon=26,
                       phases={"tau_hi": 1, "tau_md": 2})
    print("\n独立时间轴实跑（hi 相位=1，mid 相位=2；逐 tick ICPP 调度）：")
    for rec in records[:4]:
        print(f"  {rec.task}: 释放={rec.release} 完成={rec.completion} "
              f"实际响应={rec.response} 天花板阻挡={rec.ceiling_blocked} tick")
    hi_obs = next(r for r in records if r.task == "tau_hi")
    hi_bound = res.task("tau_hi").response_time
    print(f"核对：hi 实例实际响应 {hi_obs.response} <= 分析界 {hi_bound}"
          f"（{hi_obs.response <= hi_bound}）")

    # ------------------------------------------------------------------
    # 2) 抖动导致 miss：把 hi 的 J 提到 5。
    # ------------------------------------------------------------------
    hr("场景 2：释放抖动 J 导致 miss（其余参数不变）")
    tasks_j = [
        Task("tau_hi", priority=1, c=1, t=8, d=8, j=5,
             sections=[("R", 1)]),
        Task("tau_md", priority=2, c=2, t=10, d=10, j=5),
        Task("tau_lo", priority=3, c=4, t=24, d=24,
             sections=[("R", 3)]),
    ]
    res_j = analyze(tasks_j, resources)
    for r in res_j.tasks:
        print_chain(r)
    print(f"\n整集结论：guaranteed = {res_j.guaranteed}（利用率仍为 "
          f"{res_j.utilization_exact}，miss 来自抖动干扰，而非利用率）")

    # ------------------------------------------------------------------
    # 3) 实际触发的拒绝/异常边界：NaN 输入。
    # ------------------------------------------------------------------
    hr("场景 3：实际触发的输入拒绝（执行时间传 NaN）")
    bad = [Task("bad", priority=1, c=float("nan"), t=8, d=8)]
    try:
        analyze(bad)
    except (TypeError, ValueError) as exc:
        print(f"analyze() 真实抛出 {type(exc).__name__}: {exc}")
        print("输入未被静默纠正；analyze 为纯函数，无任何部分状态被修改。")
    else:
        raise AssertionError("NaN 必须被拒绝")


if __name__ == "__main__":
    main()
