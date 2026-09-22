"""固定优先级单核任务集的最坏响应时间（WCRT）可调度性分析。

公开接口：
    Task, Resource, Iteration, TaskResult, AnalysisResult
    analyze(tasks, resources=())
    TaskSetError, BudgetExhaustedError

约定见模块文档字符串与 README.md。所有分析运算均为精确整数/有理数运算，
不使用浮点做任何判定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# 接口边界常量
# ---------------------------------------------------------------------------

MAX_TASKS = 20
MAX_RESOURCES = 20
# 迭代的硬性保护上限：正常输入下（单调递增整数、每轮 miss 检查）不可能触及；
# 触及则抛 BudgetExhaustedError，与“已证明错过截止期”严格区分。
MAX_ITERATIONS_HARD = 10_000_000


# ---------------------------------------------------------------------------
# 异常语义
# ---------------------------------------------------------------------------


class WCRTError(Exception):
    """本库所有异常的公共基类。"""


class TaskSetError(WCRTError, ValueError):
    """任务/资源描述违反契约（可定位的输入错误）。"""


class BudgetExhaustedError(WCRTError):
    """迭代预算耗尽（搜索未能终止）。区别于已证明的截止期错过。"""


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resource:
    """一个共享资源及其*给定的*优先级天花板。

    优先级数值约定：数值越小优先级越高。ceiling 使用同一数值尺度。
    analyze() 会校验给定 ceiling 等于所有使用该资源任务中的最高优先级，
    即 min(使用者优先级)，不一致按 TaskSetError 拒绝。
    """

    name: str
    ceiling: int


@dataclass(frozen=True)
class Task:
    """一个固定优先级周期任务。

    name      唯一非空名称
    priority  整数优先级（越小越高，同集内唯一）
    c         最坏执行时间 C，正整数
    t         周期/最小到达间隔 T，正整数
    d         相对截止期 D，正整数且 D <= T
    j         释放抖动 J，非负整数（默认 0）
    sections  进入即依次执行的资源临界区列表：(资源名, 临界区长度)，
              不允许嵌套（扁平结构本身无法表达嵌套）；各段长度之和 <= C。
    """

    name: str
    priority: int
    c: int
    t: int
    d: int
    j: int = 0
    sections: Sequence[tuple[str, int]] = ()


@dataclass(frozen=True)
class Iteration:
    """响应时间递归的一轮记录。

    index                轮次，0 表示初始界 R0 = C + B
    bound                本轮界 R（整数）
    interference         各高优先级任务在本轮计入的干扰作业数
                         ceil((R_prev + J_h) / T_h)
    interference_workload 干扰总量 sum(count * C_h)
    blocking             采用的阻塞项 B
    blocking_source      阻塞来源 (低优先级任务名, 资源名, 临界区长度)，无阻塞为 None
    """

    index: int
    bound: int
    interference: dict[str, int]
    interference_workload: int
    blocking: int
    blocking_source: Optional[tuple[str, str, int]]


@dataclass(frozen=True)
class TaskResult:
    """单个任务的分析结果。

    status            'guaranteed'（迭代稳定且未越界）或 'miss'（证明界超过 D-J）
    guaranteed        status == 'guaranteed'
    response_time     稳定时的最坏响应时间界；miss 时为 None
    last_bound        最后一轮界（miss 时是超出前/当轮的下界信息，非真实 WCRT）
    deadline_limit    判定阈值 D_i - J_i（整数，可能为负）
    blocking          采用的 B_i
    blocking_source   B_i 的来源
    iterations        每轮推导记录
    """

    task: str
    status: str
    response_time: Optional[int]
    last_bound: int
    deadline_limit: int
    blocking: int
    blocking_source: Optional[tuple[str, str, int]]
    iterations: Sequence[Iteration]

    @property
    def guaranteed(self) -> bool:
        return self.status == "guaranteed"


@dataclass(frozen=True)
class AnalysisResult:
    """整集分析结果。

    guaranteed         当且仅当利用率精确 <= 1 且每个任务都有稳定保证
    utilization_exact  精确利用率（fractions.Fraction，判定用）
    utilization_over   精确利用率 > 1；为真时不可保证，但这不是输入格式错误
    tasks              按优先级从高到低排列的逐任务结果
    """

    guaranteed: bool
    utilization_exact: Fraction
    utilization_over: bool
    tasks: Sequence[TaskResult] = field(default_factory=tuple)

    @property
    def utilization(self) -> float:
        """仅供展示的浮点利用率；任何判定都不使用它。"""
        return float(self.utilization_exact)

    def task(self, name: str) -> TaskResult:
        for r in self.tasks:
            if r.task == name:
                return r
        raise KeyError(name)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _require_int(value, where: str) -> int:
    """整数参数校验：拒绝 bool；浮点 NaN/Infinity 显式拒绝；其余非 int 拒绝。"""
    if isinstance(value, bool):
        raise TypeError(f"{where}: bool 不能作为整数参数")
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"{where}: 拒绝 NaN")
        if math.isinf(value):
            raise ValueError(f"{where}: 拒绝 Infinity")
        raise TypeError(f"{where}: 需要 int，得到 float({value!r})")
    if not isinstance(value, int):
        raise TypeError(f"{where}: 需要 int，得到 {type(value).__name__}")
    return value


def _require_name(value, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{where}: 名称必须是非空 str")
    return value


def _validate(tasks: Sequence[Task], resources: Sequence[Resource]):
    if not isinstance(tasks, (list, tuple)):
        raise TypeError("tasks: 需要 list[Task] 或 tuple[Task]")
    if not isinstance(resources, (list, tuple)):
        raise TypeError("resources: 需要 list[Resource] 或 tuple[Resource]")
    if len(tasks) > MAX_TASKS:
        raise TaskSetError(f"任务数 {len(tasks)} 超过上限 {MAX_TASKS}")
    if len(resources) > MAX_RESOURCES:
        raise TaskSetError(f"资源数 {len(resources)} 超过上限 {MAX_RESOURCES}")

    # --- 资源 ---
    res_map: dict[str, Resource] = {}
    for idx, res in enumerate(resources):
        if not isinstance(res, Resource):
            raise TypeError(f"resources[{idx}]: 需要 Resource 实例")
        _require_name(res.name, f"resources[{idx}].name")
        if res.name in res_map:
            raise TaskSetError(f"resources[{idx}].name={res.name!r}: 资源名重复")
        _require_int(res.ceiling, f"resources[{idx}]({res.name}).ceiling")
        res_map[res.name] = res

    # --- 任务 ---
    seen_names: set[str] = set()
    seen_priorities: set[int] = set()
    users: dict[str, list[int]] = {name: [] for name in res_map}

    for idx, task in enumerate(tasks):
        if not isinstance(task, Task):
            raise TypeError(f"tasks[{idx}]: 需要 Task 实例")
        where = f"tasks[{idx}]"
        _require_name(task.name, f"{where}.name")
        if task.name in seen_names:
            raise TaskSetError(f"{where}.name={task.name!r}: 任务名重复")
        seen_names.add(task.name)

        p = _require_int(task.priority, f"{where}({task.name}).priority")
        if p in seen_priorities:
            raise TaskSetError(
                f"{where}({task.name}).priority={p}: 优先级必须唯一"
            )
        seen_priorities.add(p)

        c = _require_int(task.c, f"{where}({task.name}).c")
        t = _require_int(task.t, f"{where}({task.name}).t")
        d = _require_int(task.d, f"{where}({task.name}).d")
        j = _require_int(task.j, f"{where}({task.name}).j")
        if c <= 0:
            raise TaskSetError(f"{where}({task.name}).c={c}: 必须为正整数")
        if t <= 0:
            raise TaskSetError(f"{where}({task.name}).t={t}: 必须为正整数")
        if d <= 0:
            raise TaskSetError(f"{where}({task.name}).d={d}: 必须为正整数")
        if d > t:
            raise TaskSetError(
                f"{where}({task.name}): d={d} > t={t}，契约要求 d <= t"
            )
        if j < 0:
            raise TaskSetError(f"{where}({task.name}).j={j}: 必须是非负整数")

        if not isinstance(task.sections, (list, tuple)):
            raise TypeError(f"{where}({task.name}).sections: 需要 list/tuple")
        sec_total = 0
        sec_res_seen: set[str] = set()
        for k, sec in enumerate(task.sections):
            sw = f"{where}({task.name}).sections[{k}]"
            if not isinstance(sec, tuple) or len(sec) != 2:
                raise TypeError(f"{sw}: 需要 (资源名: str, 长度: int) 二元组")
            rname, rlen = sec
            _require_name(rname, f"{sw}[0]")
            rlen = _require_int(rlen, f"{sw}[1]")
            if rname not in res_map:
                raise TaskSetError(f"{sw}: 资源 {rname!r} 未在 resources 中声明")
            if rname in sec_res_seen:
                raise TaskSetError(f"{sw}: 同一任务不能重复占用资源 {rname!r}")
            sec_res_seen.add(rname)
            if rlen <= 0:
                raise TaskSetError(f"{sw}: 临界区长度 {rlen} 必须为正整数")
            sec_total += rlen
            users[rname].append(p)
        if sec_total > c:
            raise TaskSetError(
                f"{where}({task.name}): 临界区长度之和 {sec_total} 超过 c={c}"
            )

    # 给定天花板一致性：ceiling 必须等于使用者中的最高（最小）优先级。
    for rname, res in res_map.items():
        prios = users[rname]
        if not prios:
            raise TaskSetError(
                f"resource({rname}): 已声明但没有任何任务使用（拒绝悬空资源）"
            )
        expected = min(prios)
        if res.ceiling != expected:
            raise TaskSetError(
                f"resource({rname}).ceiling={res.ceiling} 与实际天花板 "
                f"{expected}（使用者最高优先级）不一致"
            )


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    # a >= 0, b > 0，调用处保证
    return (a + b - 1) // b


def _blocking_for(
    task: Task, ordered: Sequence[Task], ceilings: dict[str, int]
) -> tuple[int, Optional[tuple[str, str, int]]]:
    """优先级天花板协议下的单段阻塞 B_i。

    候选：低优先级任务 j 的临界区 (j, r, z)，且 ceiling(r) 不低于 P_i
    （数值约定下 ceiling(r) <= P_i）。注意：天花板协议（ICPP/PCP）下，
    即使 i 自身并不使用 r，只要系统天花板在 i 释放时不低于 P_i，i 就会
    在释放点被低优先级持锁者挡住——因此候选资格只看天花板，不看 i 是否
    用锁；禁止嵌套保证 i 至多被一个临界区阻塞。同值取声明顺序最先者，
    保证确定性。
    """
    best = 0
    source: Optional[tuple[str, str, int]] = None
    for lower in ordered:
        if lower.priority <= task.priority:
            continue
        for rname, rlen in lower.sections:
            if ceilings[rname] <= task.priority and rlen > best:
                best = rlen
                source = (lower.name, rname, rlen)
    return best, source


def analyze(
    tasks: Sequence[Task], resources: Sequence[Resource] = ()
) -> AnalysisResult:
    # 全部校验通过后才计算；纯函数、无任何外部状态，失败时不存在部分变更。
    _validate(tasks, resources)

    ceilings = {r.name: r.ceiling for r in resources}
    ordered = sorted(tasks, key=lambda x: x.priority)

    utilization = sum((Fraction(t.c, t.t) for t in tasks), Fraction(0))
    util_over = utilization > 1  # 精确有理数比较

    results: list[TaskResult] = []
    for task in ordered:
        hp = [h for h in ordered if h.priority < task.priority]
        blocking, source = _blocking_for(task, ordered, ceilings)

        limit = task.d - task.j  # 超过该整数即 miss
        r = task.c + blocking
        iterations = [
            Iteration(
                index=0,
                bound=r,
                interference={},
                interference_workload=0,
                blocking=blocking,
                blocking_source=source,
            )
        ]

        status = "guaranteed"
        response: Optional[int] = None
        if r > limit:
            status = "miss"

        if status == "guaranteed":
            # 未 miss 的非收敛轮使整数 R 至少 +1，且 R <= limit；
            # 故收敛轮数有界（再叠加硬保护预算）。
            budget = min(limit + 2, MAX_ITERATIONS_HARD)
            while True:
                if len(iterations) > budget:
                    raise BudgetExhaustedError(
                        f"任务 {task.name!r}: 响应时间迭代在 {budget} 轮内未收敛"
                    )
                counts = {
                    h.name: _ceil_div(r + h.j, h.t) for h in hp
                }
                workload = sum(n * h.c for h, n in zip(hp, counts.values()))
                r_next = task.c + blocking + workload
                it = Iteration(
                    index=len(iterations),
                    bound=r_next,
                    interference=dict(counts),
                    interference_workload=workload,
                    blocking=blocking,
                    blocking_source=source,
                )
                iterations.append(it)

                if r_next > limit:
                    status = "miss"
                    r = r_next
                    break
                if r_next == r:
                    response = r_next
                    break
                r = r_next

        results.append(
            TaskResult(
                task=task.name,
                status=status,
                response_time=response,
                last_bound=r,
                deadline_limit=limit,
                blocking=blocking,
                blocking_source=source,
                iterations=tuple(iterations),
            )
        )

    all_guaranteed = (not util_over) and all(r.guaranteed for r in results)
    return AnalysisResult(
        guaranteed=all_guaranteed,
        utilization_exact=utilization,
        utilization_over=util_over,
        tasks=tuple(results),
    )
