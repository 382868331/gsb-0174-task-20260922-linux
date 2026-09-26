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

* Optionally the processor is only available inside a periodic service
  window (SupplyWindow): in every period ``P`` only the last ``Q`` time
  units are executable (``0 < Q <= P``).  The analysis then uses the
  model's minimum supply function (see :func:`supply_bound`) so the
  worst-case arrival phase's supply gap is accounted for; the recurrence
  becomes ``R = supply_inverse(C_i + B_i + sum_h ...)``.  ``Q == P``
  degenerates exactly to the always-available model.

All arithmetic is exact integer / rational arithmetic; no floating point is
used anywhere in the analysis.
"""

from __future__ import annotations

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
    "supply_bound",
    "supply_inverse",
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
    """Periodic processor supply: only the last ``Q`` units of every
    period ``P`` are executable (``0 < Q <= P``).

    Time unit 0 starts a period, so within period ``k`` the processor is
    available on the closed-open interval ``[k*P + (P-Q), (k+1)*P)`` and
    unavailable during the first ``P-Q`` units.  ``Q == P`` means the
    processor is always available and degenerates to the baseline model.
    """

    period: int
    budget: int


def supply_bound(t: int, supply: SupplyWindow) -> int:
    """Minimum supply (executable time units) in any interval of length
    ``t`` for the last-``Q``-of-``P`` window.

    Worst case starts an interval at the end of a window: it then waits
    ``P-Q`` units (the longest blackout) before supply resumes.  Writing
    ``u = max(0, t - (P-Q))`` gives

        sbf(t) = (u // P) * Q + min(Q, u mod P).

    This is the exact minimum over arrival phases, not the average rate
    ``t * Q/P`` (which ignores the blackout and is unsafe).
    """
    p = supply.period
    q = supply.budget
    gap = p - q
    if t <= gap:
        return 0
    u = t - gap
    return (u // p) * q + min(q, u % p)


def supply_inverse(needed: int, supply: SupplyWindow) -> int:
    """Smallest interval length ``t`` with ``supply_bound(t) >= needed``.

    For ``needed >= 1``, with ``f = (needed - 1) // Q`` full windows
    elapsed before the window that supplies the last unit:

        t = (P-Q) + f*P + (needed - f*Q).

    For ``Q == P`` this is exactly ``needed`` (baseline identity).
    """
    if needed <= 0:
        return 0
    p = supply.period
    q = supply.budget
    full = (needed - 1) // q
    return (p - q) + full * p + (needed - full * q)


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
    ``ceil((value + J_h) / T_h)``, and ``next_value`` is the resulting new
    candidate.  ``demand`` is the execution demand ``C + B + sum(count *
    C_h)``; ``supply`` is the minimum processor supply available inside an
    interval of length ``value``.  Without a supply window both collapse
    onto the identity (``demand == next_value``, ``supply == value``);
    with one, ``next_value = supply_inverse(demand)`` so the worst-case
    supply gap is priced in.
    """

    round: int
    value: int
    interference: tuple[tuple[str, int], ...]
    next_value: int
    demand: int | None = None
    supply: int | None = None


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
    supply: SupplyWindow | None = None


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


def _validate_supply(supply) -> SupplyWindow | None:
    if supply is None:
        return None
    if not isinstance(supply, SupplyWindow):
        raise ValidationError(
            f"'supply' must be a rt_analysis.SupplyWindow or None, "
            f"got {type(supply).__name__} {supply!r}"
        )
    p = _as_int(supply.period, field="supply.period", index=None, task_name=None)
    q = _as_int(supply.budget, field="supply.budget", index=None, task_name=None)
    if p < 1:
        raise ValidationError(f"'supply.period' must be >= 1, got {p}")
    if q < 1:
        raise ValidationError(f"'supply.budget' must be >= 1, got {q}")
    if q > p:
        raise ValidationError(
            f"'supply.budget' ({q}) must be <= 'supply.period' ({p})"
        )
    return SupplyWindow(period=p, budget=q)


def _response_time(
    task: Task,
    hp: tuple[Task, ...],
    blocking: BlockingInfo,
    max_iterations: int,
    supply: SupplyWindow | None,
) -> TaskReport:
    limit = task.deadline - task.jitter  # miss when R exceeds D_i - J_i
    # Map an execution demand (units of work) to the interval length that
    # guarantees that much supply; identity when no window is given.
    if supply is None:
        def to_time(needed: int) -> int:
            return needed

        def available(t: int) -> int:
            return t
    else:
        def to_time(needed: int) -> int:
            return supply_inverse(needed, supply)

        def available(t: int) -> int:
            return supply_bound(t, supply)

    r = to_time(task.wcet + blocking.amount)
    iterations: list[IterationRecord] = []
    for round_no in range(max_iterations):
        counts = tuple(
            (h.name, -(-(r + h.jitter) // h.period))  # exact ceil, integers only
            for h in hp
        )
        demand = task.wcet + blocking.amount + sum(
            n * h.wcet for h, (_, n) in zip(hp, counts)
        )
        nxt = to_time(demand)
        iterations.append(
            IterationRecord(
                round=round_no,
                value=r,
                interference=counts,
                next_value=nxt,
                demand=demand,
                supply=available(r),
            )
        )
        if nxt > limit:
            return TaskReport(
                name=task.name,
                blocking=blocking,
                iterations=tuple(iterations),
                response_time=None,
                status=STATUS_DEADLINE_MISS,
                detail=(
                    f"round {round_no}: R={nxt} (demand {demand}) exceeds "
                    f"D-J={limit} "
                    f"(deadline miss proven within the model)"
                ),
            )
        if nxt == r:
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
        iterations=tuple(iterations),
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

    ``supply`` optionally restricts the processor to the last ``Q`` units
    of every period ``P`` (see :class:`SupplyWindow`).  The iteration then
    maps execution demand through the window's minimum supply function, so
    the worst-case arrival phase's blackout is charged to every task;
    ``Q == P`` reproduces the always-available analysis exactly.
    """
    max_iterations = _as_int(
        max_iterations, field="max_iterations", index=None, task_name=None
    )
    if max_iterations < 1:
        raise ValidationError(
            f"'max_iterations' must be >= 1, got {max_iterations}"
        )
    window = _validate_supply(supply)
    normalized = _validate_task_set(tasks)
    ceilings = _resource_ceilings(normalized)

    utilization = Fraction(0)
    for task in normalized:
        utilization += Fraction(task.wcet, task.period)
    utilization_ok = utilization <= 1

    reports = []
    for i, task in enumerate(normalized):
        blocking = _blocking(i, normalized, ceilings)
        reports.append(
            _response_time(task, normalized[:i], blocking, max_iterations, window)
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
        supply=window,
    )
