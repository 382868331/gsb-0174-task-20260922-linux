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

All arithmetic is exact integer / rational arithmetic; no floating point is
used anywhere in the analysis.

Mixed-criticality extension (:func:`analyze_mc`): two execution modes LO and
HI with two WCET budgets per HI task, no shared resources.  Three bounds are
solved per task: the plain LO-mode bound, the steady HI-mode bound, and the
AMC-rtb transition bound (LO interference at LO budget, HI interference at
HI budget, own execution at HI budget).  See README for the scoped subset.
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
    "CRITICALITY_LO",
    "CRITICALITY_HI",
    "CRITICALITY_LEVELS",
    "CriticalSection",
    "Task",
    "BlockingInfo",
    "IterationRecord",
    "TaskReport",
    "Report",
    "analyze",
    "ModeBound",
    "MCTaskReport",
    "MCReport",
    "analyze_mc",
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

#: Mixed-criticality execution modes / per-task criticality labels.
CRITICALITY_LO = "LO"
CRITICALITY_HI = "HI"
CRITICALITY_LEVELS = (CRITICALITY_LO, CRITICALITY_HI)

#: Per-mode outcomes of one mixed-criticality response-time bound.
MC_BOUND_KNOWN = "known"
MC_BOUND_MISS = "deadline_miss"
MC_BOUND_UNKNOWN = "unknown"


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

    For mixed-criticality analysis (:func:`analyze_mc`) ``criticality`` is
    :data:`CRITICALITY_LO` or :data:`CRITICALITY_HI`.  A HI task carries a
    second, high-mode execution budget ``wcet_hi`` (``C_HI >= C``); a LO
    task must leave ``wcet_hi`` as ``None`` (it has no high budget and is
    dropped after the switch).  These fields default to a plain LO task, so
    every existing ``Task(...)`` call stays valid for :func:`analyze`.
    """

    name: str
    wcet: int
    period: int
    deadline: int
    jitter: int = 0
    critical_sections: tuple[CriticalSection, ...] = ()
    criticality: str = CRITICALITY_LO
    wcet_hi: int | None = None


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
    candidate ``C + B + sum(count * C_h)``.
    """

    round: int
    value: int
    interference: tuple[tuple[str, int], ...]
    next_value: int


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


@dataclass(frozen=True)
class ModeBound:
    """One response-time bound of a task under one mixed-criticality view.

    ``kind`` is ``"LO"`` (plain low mode), ``"HI"`` (steady high mode) or
    ``"AMC-rtb"`` (the mode-switch transition).  ``status`` is one of
    :data:`MC_BOUND_KNOWN` (``value`` is the exact integer bound),
    :data:`MC_BOUND_MISS` (the bound provably exceeds ``D - J``) or
    :data:`MC_BOUND_UNKNOWN` (the iteration budget ran out before either
    conclusion).  ``iterations`` records every fixed-point round.
    """

    kind: str
    status: str
    value: int | None
    limit: int
    iterations: tuple[IterationRecord, ...]
    detail: str


@dataclass(frozen=True)
class MCTaskReport:
    """Per-task mixed-criticality outcome.

    Every task gets a :attr:`lo` bound.  HI-criticality tasks additionally
    get :attr:`hi` (steady HI mode) and :attr:`amc_rtb` (transition) bounds;
    both are ``None`` for LO tasks, which stop being scheduled after the
    switch.  ``guaranteed`` means every bound applicable to the task is
    known and within ``D - J``.
    """

    name: str
    criticality: str
    lo: ModeBound
    hi: ModeBound | None
    amc_rtb: ModeBound | None
    guaranteed: bool


@dataclass(frozen=True)
class MCReport:
    """Result of :func:`analyze_mc`.

    ``utilization_lo`` / ``utilization_hi`` are the exact rationals
    ``sum C(LO)/T`` and ``sum_{HI tasks} C(HI)/T``.  ``guaranteed`` is true
    only when both utilizations are at most 1 AND every applicable per-task
    bound is known within its deadline; LO tasks only need the LO bound, HI
    tasks need all three bounds.
    """

    utilization_lo: Fraction
    utilization_hi: Fraction
    utilization_ok: bool
    tasks: tuple[MCTaskReport, ...]
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

    criticality = raw.criticality
    if criticality not in CRITICALITY_LEVELS:
        raise ValidationError(
            f"task{where}: field 'criticality' must be one of {CRITICALITY_LEVELS}, "
            f"got {criticality!r}"
        )
    if raw.wcet_hi is None:
        c_hi = None
    else:
        c_hi = _as_int(raw.wcet_hi, field="wcet_hi", index=index, task_name=name)
        if c_hi < 1:
            raise ValidationError(
                f"task{where}: field 'wcet_hi' must be >= 1, got {c_hi}"
            )
        if c_hi < c:
            raise ValidationError(
                f"task{where}: field 'wcet_hi' ({c_hi}) must be >= 'wcet' ({c}): "
                f"the high-mode budget cannot be smaller than the low-mode budget"
            )
    if criticality == CRITICALITY_HI:
        if c_hi is None:
            raise ValidationError(
                f"task{where}: a HI-criticality task must provide 'wcet_hi' "
                f"(its high-mode execution budget)"
            )
    elif c_hi is not None:
        raise ValidationError(
            f"task{where}: a LO-criticality task must not provide 'wcet_hi' "
            f"(got {c_hi}); LO tasks have no high-mode budget and are dropped "
            f"after the mode switch"
        )

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
        criticality=criticality,
        wcet_hi=c_hi,
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


def _response_time(
    task: Task, hp: tuple[Task, ...], blocking: BlockingInfo, max_iterations: int
) -> TaskReport:
    limit = task.deadline - task.jitter  # miss when R exceeds D_i - J_i
    r = task.wcet + blocking.amount
    iterations: list[IterationRecord] = []
    for round_no in range(max_iterations):
        counts = tuple(
            (h.name, -(-(r + h.jitter) // h.period))  # exact ceil, integers only
            for h in hp
        )
        nxt = task.wcet + blocking.amount + sum(
            n * h.wcet for h, (_, n) in zip(hp, counts)
        )
        iterations.append(
            IterationRecord(round=round_no, value=r, interference=counts, next_value=nxt)
        )
        if nxt > limit:
            return TaskReport(
                name=task.name,
                blocking=blocking,
                iterations=tuple(iterations),
                response_time=None,
                status=STATUS_DEADLINE_MISS,
                detail=(
                    f"round {round_no}: R={nxt} exceeds D-J={limit} "
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
    tasks: Sequence[Task], *, max_iterations: int = DEFAULT_MAX_ITERATIONS
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
    ceilings = _resource_ceilings(normalized)

    utilization = Fraction(0)
    for task in normalized:
        utilization += Fraction(task.wcet, task.period)
    utilization_ok = utilization <= 1

    reports = []
    for i, task in enumerate(normalized):
        blocking = _blocking(i, normalized, ceilings)
        reports.append(
            _response_time(task, normalized[:i], blocking, max_iterations)
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


# ---------------------------------------------------------------------------
# Mixed-criticality analysis (LO/HI modes, AMC-rtb transition bound)
#
# Scoped subset (see README): single core, fixed priorities, NO shared
# resources, two modes with two WCET budgets per HI task.  A job of a HI task
# executing beyond its LO budget triggers the switch; after the switch LO
# tasks start no new jobs while HI tasks run at their HI budget.
# ---------------------------------------------------------------------------


def _validate_mc_task_set(tasks: Sequence[Task]) -> tuple[Task, ...]:
    normalized = _validate_task_set(tasks)
    for i, task in enumerate(normalized):
        if task.critical_sections:
            raise ValidationError(
                f"task (index {i}, name {task.name!r}): analyze_mc() is scoped "
                f"to task sets without shared resources, but the task declares "
                f"{len(task.critical_sections)} critical section(s); use analyze() "
                f"for the resource/PCP model"
            )
    if not any(t.criticality == CRITICALITY_HI for t in normalized):
        raise ValidationError(
            "analyze_mc() requires at least one HI-criticality task (a task with "
            "criticality='HI' and wcet_hi); a LO-only set has no mode switch and "
            "should be analyzed with analyze()"
        )
    return normalized


def _mc_fixed_point(
    kind: str,
    task: Task,
    own_c: int,
    contributors: tuple[tuple[str, int, int, int], ...],  # (name, budget C, J, T)
    max_iterations: int,
) -> ModeBound:
    """Solve R = own_c + sum ceil((R + J_h)/T_h) * C_h to a fixed point.

    Returns a :class:`ModeBound` classified as known / deadline_miss /
    unknown (budget exhausted).  ``contributors`` is already restricted to
    the higher-priority tasks present in the corresponding mode and carries
    the per-mode execution budget of each.
    """
    limit = task.deadline - task.jitter
    r = own_c
    iterations: list[IterationRecord] = []
    for round_no in range(max_iterations):
        counts = tuple(
            (name, -(-((r + jitter)) // period))  # exact ceil, integers only
            for name, _budget, jitter, period in contributors
        )
        nxt = own_c + sum(
            n * budget
            for (_name, budget, _j, _t), (_n, n) in zip(contributors, counts)
        )
        iterations.append(
            IterationRecord(round=round_no, value=r, interference=counts, next_value=nxt)
        )
        if nxt > limit:
            return ModeBound(
                kind=kind,
                status=MC_BOUND_MISS,
                value=None,
                limit=limit,
                iterations=tuple(iterations),
                detail=(
                    f"{kind}: round {round_no}: R={nxt} exceeds D-J={limit} "
                    f"(deadline miss proven within the model)"
                ),
            )
        if nxt == r:
            return ModeBound(
                kind=kind,
                status=MC_BOUND_KNOWN,
                value=r,
                limit=limit,
                iterations=tuple(iterations),
                detail=f"{kind}: converged after {round_no + 1} round(s): R={r} <= D-J={limit}",
            )
        if nxt < r:  # unreachable: the recurrence is monotone non-decreasing
            raise AssertionError(
                f"internal error: non-monotone {kind} iteration for task {task.name!r}"
            )
        r = nxt
    return ModeBound(
        kind=kind,
        status=MC_BOUND_UNKNOWN,
        value=None,
        limit=limit,
        iterations=tuple(iterations),
        detail=(
            f"{kind}: iteration budget of {max_iterations} round(s) exhausted at "
            f"R={r} (unknown: neither converged nor proven to miss)"
        ),
    )


def analyze_mc(
    tasks: Sequence[Task], *, max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> MCReport:
    """Analyze a dual-criticality (LO/HI) fixed-priority task set.

    Returns per-task response-time bounds, or ``unknown`` where the search
    budget is inconclusive:

    * **LO bound** (every task): all tasks execute at their LO budget —
      behaviour before any mode switch.
    * **HI bound** (HI tasks only): steady HI mode; only HI tasks remain,
      each at its ``wcet_hi`` budget.
    * **AMC-rtb bound** (HI tasks only): the mode-switch transition.  The
      analyzed job executes at its HI budget; higher-priority LO tasks
      interfere at LO budget, higher-priority HI tasks at HI budget.  LO
      tasks start no new jobs after the switch, so they receive no such
      bound (they are not scheduled in HI mode).

    Scoped to a single core and task sets without shared resources; such
    inputs raise :class:`ValidationError`, as do all other contract
    violations.  The function is pure: a failed call leaves no partial
    state.  Utilization is reported for both modes; either one above 1 makes
    the set unguaranteeable but is not a format error.
    """
    max_iterations = _as_int(
        max_iterations, field="max_iterations", index=None, task_name=None
    )
    if max_iterations < 1:
        raise ValidationError(f"'max_iterations' must be >= 1, got {max_iterations}")
    normalized = _validate_mc_task_set(tasks)

    utilization_lo = sum(
        (Fraction(t.wcet, t.period) for t in normalized), Fraction(0)
    )
    utilization_hi = sum(
        (
            Fraction(t.wcet_hi, t.period)
            for t in normalized
            if t.criticality == CRITICALITY_HI
        ),
        Fraction(0),
    )
    utilization_ok = utilization_lo <= 1 and utilization_hi <= 1

    reports: list[MCTaskReport] = []
    for i, task in enumerate(normalized):
        hp = normalized[:i]

        # LO mode: every higher-priority task interferes at its LO budget.
        lo_contrib = tuple((h.name, h.wcet, h.jitter, h.period) for h in hp)
        lo = _mc_fixed_point(
            CRITICALITY_LO, task, task.wcet, lo_contrib, max_iterations
        )

        hi = None
        amc = None
        guaranteed = lo.status == MC_BOUND_KNOWN
        if task.criticality == CRITICALITY_HI:
            # Steady HI mode: only higher-priority HI tasks, at HI budget.
            hi_contrib = tuple(
                (h.name, h.wcet_hi, h.jitter, h.period)
                for h in hp
                if h.criticality == CRITICALITY_HI
            )
            hi = _mc_fixed_point(
                CRITICALITY_HI, task, task.wcet_hi, hi_contrib, max_iterations
            )
            # AMC-rtb transition: LO interference at LO budget, HI
            # interference at HI budget; the crossing job at HI budget.
            amc_contrib = tuple(
                (
                    h.name,
                    h.wcet_hi if h.criticality == CRITICALITY_HI else h.wcet,
                    h.jitter,
                    h.period,
                )
                for h in hp
            )
            amc = _mc_fixed_point(
                "AMC-rtb", task, task.wcet_hi, amc_contrib, max_iterations
            )
            guaranteed = guaranteed and hi.status == MC_BOUND_KNOWN and amc.status == MC_BOUND_KNOWN

        reports.append(
            MCTaskReport(
                name=task.name,
                criticality=task.criticality,
                lo=lo,
                hi=hi,
                amc_rtb=amc,
                guaranteed=guaranteed,
            )
        )

    if not utilization_ok:
        why = []
        if utilization_lo > 1:
            why.append(f"LO utilization {utilization_lo} > 1")
        if utilization_hi > 1:
            why.append(f"HI utilization {utilization_hi} > 1")
        guaranteed = False
        summary = "; ".join(why) + ": the task set cannot be guaranteed on one CPU"
    elif all(r.guaranteed for r in reports):
        guaranteed = True
        summary = (
            f"LO utilization {utilization_lo} and HI utilization {utilization_hi} "
            f"<= 1; all LO/HI/AMC-rtb bounds converged within D-J: guaranteed under "
            f"the mixed-criticality model"
        )
    else:
        guaranteed = False
        bad = []
        for r in reports:
            for b in (r.lo, r.hi, r.amc_rtb):
                if b is not None and b.status != MC_BOUND_KNOWN:
                    bad.append(f"{r.name}[{b.kind}:{b.status}]")
        summary = (
            f"LO utilization {utilization_lo}, HI utilization {utilization_hi} <= 1 "
            f"but not guaranteed: {', '.join(bad)}"
        )
    return MCReport(
        utilization_lo=utilization_lo,
        utilization_hi=utilization_hi,
        utilization_ok=utilization_ok,
        tasks=tuple(reports),
        guaranteed=guaranteed,
        summary=summary,
    )
