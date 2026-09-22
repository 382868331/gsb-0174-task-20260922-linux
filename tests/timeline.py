"""独立的离散时间轴抢占调度参考（不导入、不调用被测核心 wcrt）。

用途：在小规模确定性场景上真实模拟*立即优先级天花板协议(ICPP)*下的
单核抢占调度，生成每个作业的实际响应时间，供测试核对：

  * 被测分析给出的最坏响应时间界 R 是实际调度的上界（实例不超过界）；
  * 在对抗性抖动相位下确实能观察到截止期错过。

期望值由调度器逐 tick 跑出来，分析模块既不提供输入也不提供期望值。
这是独立实现的第二套语义（测试不把"往返一致"当充分证据）。

ICPP 要点（数值越小优先级越高）：
  - 作业持有资源 r 期间按 ceiling(r) 提升优先级；
  - 系统天花板 = 所有被持有资源天花板的最小值；
  - 未持锁的就绪作业只有在自身优先级严格高于系统天花板时才能运行，
    否则在释放点被天花板阻挡（直接阻塞，不进入锁等待队列，不会死锁）；
  - 禁止嵌套：任何作业至多持有一把锁，锁请求总是立即成功。

任务用普通 dict 描述（刻意不用 wcrt.Task，保持完全独立）：
  {name, prio, c, t, d, j, sections:[(resource, length)]}
资源天花板: {resource: ceiling}。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobRecord:
    task: str
    arrival: int          # 周期到达 k*T + phase
    release: int          # 实际释放 = arrival + jitter
    completion: int       # 完成时刻
    ceiling_blocked: int  # 释放后因系统天花板阻挡而等待的 tick 数

    @property
    def response(self) -> int:
        """从实际释放时刻到完成的响应时间（分析以 release 为零点）。"""
        return self.completion - self.release

    @property
    def response_from_arrival(self) -> int:
        """从名义到达时刻到完成（含抖动的端到端口径，仅展示用）。"""
        return self.completion - self.arrival


def _layout(c, sections):
    """把临界区确定性地铺进长度 c 的执行程序：非临界区时间均分到各段间隙。"""
    total = sum(length for _, length in sections)
    gaps = len(sections) + 1
    base = (c - total) // gaps
    extra = (c - total) - base * gaps  # 余数加到最后一个间隙
    ticks = [None] * base
    for i, (rname, length) in enumerate(sections):
        ticks.extend([rname] * length)
        tail = base + (extra if i == len(sections) - 1 else 0)
        ticks.extend([None] * tail)
    assert len(ticks) == c
    return ticks


def simulate(task_specs, ceilings, horizon, jitter=None, phases=None):
    """在 [0, horizon) 内按周期释放作业，逐 tick 模拟 ICPP 并排空到全部完成。

    jitter: f(name, k) -> [0, J] 的整数抖动；默认 0。
    phases: {name: 首次到达偏移}；默认 0。
    返回 JobRecord 列表（按完成时刻排序）。
    """
    jitter = jitter or (lambda name, k: 0)
    phases = phases or {}

    specs = sorted(task_specs, key=lambda s: s["prio"])
    order = {s["name"]: i for i, s in enumerate(specs)}
    program = {s["name"]: _layout(s["c"], s.get("sections", [])) for s in specs}

    jobs = []
    for s in specs:
        phase = phases.get(s["name"], 0)
        k = 0
        while True:
            arrival = phase + k * s["t"]
            if arrival >= horizon:
                break
            jobs.append(
                {
                    "task": s["name"],
                    "prio": s["prio"],
                    "arrival": arrival,
                    "release": arrival + jitter(s["name"], k),
                    "cursor": 0,
                    "held": None,
                    "ceiling_blocked": 0,
                }
            )
            k += 1

    held: dict[str, dict] = {}  # resource -> holder
    records: list[JobRecord] = []
    now = 0

    def unfinished(j):
        return j["cursor"] < len(program[j["task"]])

    def system_ceiling():
        if not held:
            return None
        return min(ceilings[r] for r in held)

    def effective_prio(j):
        p = j["prio"]
        if j["held"] is not None:
            p = min(p, ceilings[j["held"]])
        return p

    while any(unfinished(j) for j in jobs):
        sysceil = system_ceiling()

        def ceiling_ineligible(j):
            return (
                j["held"] is None
                and sysceil is not None
                and j["prio"] >= sysceil
            )

        # 已释放、未完成、但被系统天花板挡在临界区外的就绪作业：本 tick 记账。
        for j in jobs:
            if j["release"] <= now and unfinished(j) and ceiling_ineligible(j):
                j["ceiling_blocked"] += 1

        eligible = [
            j
            for j in jobs
            if j["release"] <= now
            and unfinished(j)
            and not ceiling_ineligible(j)
        ]
        # 持锁者始终可运行（不会被自己的天花板挡住）；其余按（提升后的）优先级。
        if not eligible:
            # 仅当此刻没有任何已释放作业（纯空闲）才推进时间。
            now += 1
            continue
        eligible.sort(
            key=lambda j: (effective_prio(j), j["release"], order[j["task"]])
        )
        j = eligible[0]
        ticks = program[j["task"]]
        want = ticks[j["cursor"]]
        if want is not None and j["held"] is None:
            # 禁止嵌套：该作业此前不持锁；ICPP 下获取立即成功。
            assert want not in held, "嵌套/重复持锁违反契约，模拟不应到达"
            held[want] = j
            j["held"] = want
        res_before = want
        j["cursor"] += 1
        now += 1
        if j["cursor"] == len(ticks):
            if j["held"] is not None:
                held.pop(j["held"], None)
                j["held"] = None
            records.append(
                JobRecord(
                    task=j["task"],
                    arrival=j["arrival"],
                    release=j["release"],
                    completion=now,
                    ceiling_blocked=j["ceiling_blocked"],
                )
            )
        else:
            nxt = ticks[j["cursor"]]
            if res_before is not None and nxt != res_before:
                held.pop(res_before, None)
                j["held"] = None

    records.sort(key=lambda r: (r.completion, order[r.task]))
    return records
