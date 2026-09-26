"""Worst-case response-time bounds for fixed-priority single-CPU task sets.

Model (explicitly conservative, see README):

* Single CPU, fixed-priority preemptive scheduling.  The position of a task
  in the input sequence IS its priority: index 0 is the highest priority.
* Each task is described by integer WCET ``C``, period ``T``, deadline
  ``D <= T`` and release jitter ``J``.
* Shared resources use the (immediate) priority ceiling protocol.  The
  ceiling of a resource is the highest priority (smallest index) among the
  tasks that use it.  Critical sections never nest.
* The blocking term ``B_i`` is the largest single critical section among
  lower-priority tasks on a resource whose ceiling is at least the priority
  of task ``i`` (i.e. a critical section that can actually block ``i``).
* The response-time bound is the least fixed point of
  ``R = C_i + B_i + sum_{h in hp(i)} ceil((R + J_h) / T_h) * C_h``
  starting from ``R = C_i + B_i``.  If a candidate exceeds ``D_i - J_i``
  the task is declared a deadline miss; if the value stabilises the task is
  guaranteed (under the model's assumptions).

* An optional :class:`SupplyWindow` (period ``P``, accessible only during
  the last ``Q`` time units of each supply period, ``0 < Q <= P``) models a
  processor that is unavailable outside those windows.  The recurrence then
  pairs the worst-case *demand* with the *minimum supply function* of the
  window and solves ``R = supply_inv(demand(R))``.  ``Q == P`` degenerates
  to the always-available model above; dividing execution by ``Q/P`` is
  deliberately NOT used (it ignores the worst-case arrival phase).

All arithmetic is exact integer / rational arithmetic; no floating point is
used anywhere in the analysis.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

__all__ = [
    "MAX_TASKS",
    "DEFAULT_MAX_ITERATIONS",
    "ValidationError",
    "CriticalSection",
    "Task",
    "SupplyWindow",
    "BlockingInfo",
    "IterationRecord",
    "TaskReport",
    "Report",
    "analyze",
]

#: Maximum number of tasks accepted by :func:`analyze` (interface bound).
MAX_TASKS = 20

#: Default upper bound on fixed-point iterations per task.  Exhausting this
#: budget is reported as ``iteration_budget_exhausted`` and is deliberately
#: distinct from a proven ``deadline_miss``.
DEFAULT_MAX_ITERATIONS = 1000

#: Per-task analysis outcomes.
STATUS_SCHEDULABLE = "schedulable"
STATUS_DEADLINE_MISS = "deadline_miss"
STATUS_BUDGET_EXHAUSTED = "iteration_budget_exhausted"


class ValidationError(ValueError):
    """Raised when the task-set input violates the contract.

    The message always locates the offending task (by index and name, when
    available) and the offending field, so callers can pinpoint the problem.
    Nothing is silently corrected.
    """


@dataclass(frozen=True)
class CriticalSection:
    """One non-nestable critical section on a named shared resource."""

    resource: str
    length: int


@dataclass(frozen=True)
class Task:
    """A sporadic/periodic fixed-priority task.

    ``wcet`` (C), ``period`` (T), ``deadline`` (D) and ``jitter`` (J) are
    integers in the model's time unit, with ``1 <= D <= T``, ``C >= 1`` and
    ``J >= 0``.  ``critical_sections`` lists the (flat, non-nested) critical
    sections the task may execute; their lengths must sum to at most ``C``.
    """

    name: str
    wcet: int
    period: int
    deadline: int
    jitter: int = 0
    critical_sections: tuple[CriticalSection, ...] = ()


@dataclass(frozen=True)
class SupplyWindow:
    """Periodic processor supply window.

    Within every supply period ``P`` the processor is available (for every
    task, including critical sections) only during the last ``Q`` time
    units of the period, i.e. slots ``[kP + P - Q, (k+1)P)`` for
    ``k = 0, 1, 2, ...``; it is unavailable during the leading ``P - Q``
    units.  ``Q == P`` means the processor is always available and the
    model degenerates to the plain fixed-priority model.
    """

    period: int
    window: int


@dataclass(frozen=True)
class BlockingInfo:
    """Blocking term B_i and its origin.

    ``source_task`` / ``source_resource`` are ``None`` exactly when
    ``amount == 0`` (zero blocking).
    """

    amount: int
    source_task: str | None
    source_resource: str | None


@dataclass(frozen=True)
class IterationRecord:
    """One round of the response-time fixed-point iteration.

    ``value`` is the candidate R used in this round, ``interference`` pairs
    each higher-priority task name with the number of interfering releases
    ``ceil((value + J_h) / T_h)``.  ``demand`` is the total execution demand
    over the interval (own ``C + B`` plus interference; always available in
    the always-on model).  ``supply`` is the minimum number of time units of
    processor service the supply window guarantees within ``value`` time
    units (equal to ``value`` when no supply window is given), and
    ``next_value`` is the smallest interval that can serve ``demand`` units
    under that minimum supply (``demand`` itself in the always-on model).
    ``stop_reason`` is set only on the final round.
    """

    round: int
    value: int
    interference: tuple[tuple[str, int], ...]
    next_value: int
    demand: int
    supply: int
    stop_reason: str | None = None


@dataclass(frozen=True)
class TaskReport:
    """Per-task analysis outcome.

    ``response_time`` is the guaranteed bound R_i when
    ``status == "schedulable"`` and ``None`` otherwise.  The bound on the
    latency relative to the task's nominal request time is
    ``response_time + jitter``.
    """

    name: str
    blocking: BlockingInfo
    iterations: tuple[IterationRecord, ...]
    response_time: int | None
    status: str
    detail: str


@dataclass(frozen=True)
class Report:
    """Result of :func:`analyze`.

    ``utilization`` is the exact rational ``sum(C_i / T_i)``.
    ``utilization_ok`` is ``utilization <= 1``; a utilization above 1 makes
    the task set unguaranteeable on one CPU but is NOT a format error, and a
    utilization at or below 1 is NOT by itself a guarantee.  ``guaranteed``
    is true only when utilization is at most 1 AND every task converged to a
    bound within ``D_i - J_i``.
    """

    utilization: Fraction
    utilization_ok: bool
    tasks: tuple[TaskReport, ...]
    guaranteed: bool
    summary: str


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _where(index: int | None, task_name: str | None) -> str:
    parts = []
    if index is not None:
        parts.append(f"index {index}")
    if task_name is not None:
        parts.append(f"name {task_name!r}")
    return " (" + ", ".join(parts) + ")" if parts else ""


def _as_int(value, *, field: str, index: int | None, task_name: str | None) -> int:
    """Coerce a numeric input to int or raise a locatable ValidationError."""
    where = _where(index, task_name)
    if isinstance(value, bool):
        raise ValidationError(
            f"task{where}: field {field!r} must be an integer, got bool {value!r}"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValidationError(
                f"task{where}: field {field!r} must be finite, got {value!r}"
            )
        if not value.is_integer():
            raise ValidationError(
                f"task{where}: field {field!r} must be integral, got {value!r}"
            )
        return int(value)
    raise ValidationError(
        f"task{where}: field {field!r} must be an integer, "
        f"got {type(value).__name__} {value!r}"
    )


def _validate_task(raw, index: int) -> Task:
    if not isinstance(raw, Task):
        raise ValidationError(
            f"task (index {index}): expected a rt_analysis.Task instance, "
            f"got {type(raw).__name__} {raw!r}"
        )
    name = raw.name
    if not isinstance(name, str) or not name:
        raise ValidationError(
            f"task (index {index}): field 'name' must be a non-empty string, "
            f"got {name!r}"
        )
    c = _as_int(raw.wcet, field="wcet", index=index, task_name=name)
    t = _as_int(raw.period, field="period", index=index, task_name=name)
    d = _as_int(raw.deadline, field="deadline", index=index, task_name=name)
    j = _as_int(raw.jitter, field="jitter", index=index, task_name=name)
    where = _where(index, name)
    if c < 1:
        raise ValidationError(f"task{where}: field 'wcet' must be >= 1, got {c}")
    if t < 1:
        raise ValidationError(f"task{where}: field 'period' must be >= 1, got {t}")
    if d < 1:
        raise ValidationError(f"task{where}: field 'deadline' must be >= 1, got {d}")
    if d > t:
        raise ValidationError(
            f"task{where}: field 'deadline' ({d}) must be <= 'period' ({t})"
        )
    if j < 0:
        raise ValidationError(f"task{where}: field 'jitter' must be >= 0, got {j}")

    if not isinstance(raw.critical_sections, (tuple, list)):
        raise ValidationError(
            f"task{where}: field 'critical_sections' must be a tuple/list of "
            f"CriticalSection, got {type(raw.critical_sections).__name__}"
        )
    sections = []
    total = 0
    for k, cs in enumerate(raw.critical_sections):
        if not isinstance(cs, CriticalSection):
            raise ValidationError(
                f"task{where}: critical_sections[{k}] must be a CriticalSection, "
                f"got {type(cs).__name__} {cs!r}"
            )
        if not isinstance(cs.resource, str) or not cs.resource:
            raise ValidationError(
                f"task{where}: critical_sections[{k}].resource must be a "
                f"non-empty string, got {cs.resource!r}"
            )
        length = _as_int(
            cs.length, field=f"critical_sections[{k}].length", index=index, task_name=name
        )
        if length < 1:
            raise ValidationError(
                f"task{where}: critical_sections[{k}].length must be >= 1, got {length}"
            )
        total += length
        sections.append(CriticalSection(resource=cs.resource, length=length))
    if total > c:
        raise ValidationError(
            f"task{where}: critical sections sum to {total}, exceeding 'wcet' ({c}); "
            f"critical sections execute inside the task's budget"
        )
    return Task(
        name=name,
        wcet=c,
        period=t,
        deadline=d,
        jitter=j,
        critical_sections=tuple(sections),
    )


def _validate_task_set(tasks: Sequence[Task]) -> tuple[Task, ...]:
    if isinstance(tasks, (str, bytes)) or not hasattr(tasks, "__iter__"):
        raise ValidationError(
            f"analyze() expects a sequence of Task, got {type(tasks).__name__}"
        )
    tasks = list(tasks)
    if not tasks:
        raise ValidationError("task set must contain at least one task")
    if len(tasks) > MAX_TASKS:
        raise ValidationError(
            f"task set has {len(tasks)} tasks, exceeding the maximum of {MAX_TASKS}"
        )
    normalized = tuple(_validate_task(raw, i) for i, raw in enumerate(tasks))
    seen: set[str] = set()
    for task in normalized:
        if task.name in seen:
            raise ValidationError(f"duplicate task name {task.name!r}")
        seen.add(task.name)
    return normalized


def _validate_supply_window(raw) -> SupplyWindow | None:
    """Validate the optional periodic supply window (same numeric contract)."""
    if raw is None:
        return None
    if not isinstance(raw, SupplyWindow):
        raise ValidationError(
            f"'supply' must be a rt_analysis.SupplyWindow instance or None, "
            f"got {type(raw).__name__} {raw!r}"
        )
    p = _as_int(raw.period, field="supply.period", index=None, task_name=None)
    q = _as_int(raw.window, field="supply.window", index=None, task_name=None)
    if p < 1:
        raise ValidationError(f"'supply.period' must be >= 1, got {p}")
    if q < 1:
        raise ValidationError(f"'supply.window' must be >= 1, got {q}")
    if q > p:
        raise ValidationError(
            f"'supply.window' ({q}) must be <= 'supply.period' ({p})"
        )
    return SupplyWindow(period=p, window=q)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _resource_ceilings(tasks: tuple[Task, ...]) -> dict[str, int]:
    """Ceiling of each resource = smallest task index (highest priority) using it."""
    ceilings: dict[str, int] = {}
    for i, task in enumerate(tasks):
        for cs in task.critical_sections:
            if cs.resource not in ceilings or i < ceilings[cs.resource]:
                ceilings[cs.resource] = i
    return ceilings


def _blocking(index: int, tasks: tuple[Task, ...], ceilings: dict[str, int]) -> BlockingInfo:
    """Largest single lower-priority critical section that can block task ``index``.

    Under the priority ceiling protocol a lower-priority task can only block
    task ``i`` through a resource whose ceiling is at least i's priority
    (ceiling index <= i).
    """
    best = 0
    source_task = None
    source_resource = None
    for j in range(index + 1, len(tasks)):
        for cs in tasks[j].critical_sections:
            if ceilings[cs.resource] <= index and cs.length > best:
                best = cs.length
                source_task = tasks[j].name
                source_resource = cs.resource
    return BlockingInfo(amount=best, source_task=source_task, source_resource=source_resource)


# ---------------------------------------------------------------------------
# Periodic supply window: minimum supply function and its inverse
# ---------------------------------------------------------------------------
#
# The processor is available only during the last Q slots of every period of
# length P: available slots are [kP + P - Q, (k+1)P).  The minimum amount of
# service available over a window of t consecutive slots is obtained with the
# window's left edge placed at the start of an unavailable stretch (the
# boundary just after a supply window ends):
#
# Writing k = t // P and r = t mod P, the worst-phased window consists of k
# full periods (each contributes exactly Q service slots) followed by r
# trailing slots, of which only the last Q can be service:
#   sbf(t) = k*Q + max(0, r - (P - Q)).
# The single leading blackout of P - Q is captured by max(0, r - (P-Q)) for
# every partial tail; for t <= P - Q the bound is 0 (nothing guaranteed).
# Q == P gives sbf(t) = t (always available).  The inverse ``supply_time_for``
# returns the smallest t >= 0 with sbf(t) >= demand, so a fixed-point of
# t -> supply_time_for(demand(t)) is a sound response-time bound.


def minimum_supply(t: int, supply: SupplyWindow | None) -> int:
    """Minimum processor service (integer slots) guaranteed over ``t`` slots.

    Worst-case arrival phase relative to the periodic supply window is
    accounted for; this is NOT ``t * Q / P``.  With ``supply is None`` (or
    ``Q == P``) every slot is available and the result is ``t``.
    """
    if t <= 0:
        return 0
    if supply is None or supply.window == supply.period:
        return t
    p = supply.period
    q = supply.window
    gap = p - q
    k, r = divmod(t, p)
    return k * q + max(0, r - gap)


def supply_time_for(demand: int, supply: SupplyWindow | None) -> int:
    """Smallest interval ``t`` with :func:`minimum_supply` ``(t) >= demand``.

    Exact integer inverse of the minimum supply function.  Before the first
    guaranteed supply window at most ``P - Q`` leading slots are unavailable;
    every further full period contributes exactly ``Q`` service slots.
    """
    if demand <= 0:
        return 0
    if supply is None or supply.window == supply.period:
        return demand
    p = supply.period
    q = supply.window
    gap = p - q
    full_periods = (demand - 1) // q
    remainder = demand - full_periods * q  # 1 <= remainder <= q
    return gap + full_periods * p + remainder


def _response_time(
    task: Task,
    hp: tuple[Task, ...],
    blocking: BlockingInfo,
    max_iterations: int,
    supply: SupplyWindow | None,
) -> TaskReport:
    limit = task.deadline - task.jitter  # miss when R exceeds D_i - J_i
    initial_demand = task.wcet + blocking.amount
    r = supply_time_for(initial_demand, supply)
    iterations: list[IterationRecord] = []
    for round_no in range(max_iterations):
        counts = tuple(
            (h.name, -(-(r + h.jitter) // h.period))  # exact ceil, integers only
            for h in hp
        )
        demand = task.wcet + blocking.amount + sum(
            n * h.wcet for h, (_, n) in zip(hp, counts)
        )
        supplied = minimum_supply(r, supply)
        nxt = supply_time_for(demand, supply)
        iterations.append(
            IterationRecord(
                round=round_no,
                value=r,
                interference=counts,
                next_value=nxt,
                demand=demand,
                supply=supplied,
            )
        )
        if nxt > limit:
            iterations[-1] = dataclasses.replace(
                iterations[-1],
                stop_reason=(
                    f"demand {demand} requires {nxt} supplied time units; "
                    f"{nxt} > D-J = {limit}"
                ),
            )
            return TaskReport(
                name=task.name,
                blocking=blocking,
                iterations=tuple(iterations),
                response_time=None,
                status=STATUS_DEADLINE_MISS,
                detail=(
                    f"round {round_no}: supply-aware response bound R={nxt} "
                    f"exceeds D-J={limit} (deadline miss proven within the model)"
                ),
            )
        if nxt == r:
            iterations[-1] = dataclasses.replace(
                iterations[-1],
                stop_reason=(
                    f"fixed point reached: minimum supply over R={r} covers "
                    f"demand {demand}, and R <= D-J = {limit}"
                ),
            )
            return TaskReport(
                name=task.name,
                blocking=blocking,
                iterations=tuple(iterations),
                response_time=r,
                status=STATUS_SCHEDULABLE,
                detail=(
                    f"converged after {round_no + 1} round(s): R={r} <= D-J={limit}"
                ),
            )
        if nxt < r:  # unreachable: the recurrence is monotone non-decreasing
            raise AssertionError(
                f"internal error: non-monotone iteration for task {task.name!r}"
            )
        r = nxt
    return TaskReport(
        name=task.name,
        blocking=blocking,
        iterations=tuple(
            [
                dataclasses.replace(
                    it,
                    stop_reason=(
                        f"iteration budget of {max_iterations} round(s) "
                        f"exhausted at R={r} (inconclusive: neither converged "
                        f"nor proven to miss)"
                    ) if it is iterations[-1] else None,
                )
                for it in iterations
            ]
        ),
        response_time=None,
        status=STATUS_BUDGET_EXHAUSTED,
        detail=(
            f"iteration budget of {max_iterations} round(s) exhausted at R={r} "
            f"(inconclusive: neither converged nor proven to miss)"
        ),
    )


def analyze(
    tasks: Sequence[Task],
    *,
    supply: SupplyWindow | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Report:
    """Compute worst-case response-time bounds for a fixed-priority task set.

    ``tasks`` is ordered by priority: ``tasks[0]`` has the highest priority.
    Raises :class:`ValidationError` (locatable) on any contract violation;
    the function is pure, so a failed call leaves no partial state behind.
    A utilization above 1 is reported in the returned :class:`Report`
    (``guaranteed == False``), not raised as an error.

    ``max_iterations`` bounds the fixed-point search per task; exhausting it
    yields status ``iteration_budget_exhausted`` (inconclusive), which is
    distinct from a proven ``deadline_miss``.
    """
    max_iterations = _as_int(
        max_iterations, field="max_iterations", index=None, task_name=None
    )
    if max_iterations < 1:
        raise ValidationError(
            f"'max_iterations' must be >= 1, got {max_iterations}"
        )
    normalized = _validate_task_set(tasks)
    supply = _validate_supply_window(supply)
    ceilings = _resource_ceilings(normalized)

    utilization = Fraction(0)
    for task in normalized:
        utilization += Fraction(task.wcet, task.period)
    utilization_ok = utilization <= 1

    reports = []
    for i, task in enumerate(normalized):
        blocking = _blocking(i, normalized, ceilings)
        reports.append(
            _response_time(task, normalized[:i], blocking, max_iterations, supply)
        )

    if not utilization_ok:
        guaranteed = False
        summary = (
            f"utilization {utilization} ({float(utilization):.4f}) > 1: "
            f"the task set cannot be guaranteed on one CPU"
        )
    elif all(r.status == STATUS_SCHEDULABLE for r in reports):
        guaranteed = True
        summary = (
            f"utilization {utilization} ({float(utilization):.4f}) <= 1 and all "
            f"{len(reports)} task(s) converged within D-J: guaranteed under the model"
        )
    else:
        guaranteed = False
        bad = ", ".join(
            f"{r.name}({r.status})" for r in reports if r.status != STATUS_SCHEDULABLE
        )
        summary = (
            f"utilization {utilization} ({float(utilization):.4f}) <= 1 but not "
            f"guaranteed: {bad}"
        )
    return Report(
        utilization=utilization,
        utilization_ok=utilization_ok,
        tasks=tuple(reports),
        guaranteed=guaranteed,
        summary=summary,
    )
