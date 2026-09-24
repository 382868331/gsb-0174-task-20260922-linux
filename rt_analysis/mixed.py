"""Mixed-criticality fixed-priority analysis: two modes, two budgets.

This module adds the mixed-criticality (MC) iteration on top of the plain
fixed-priority analysis in :mod:`rt_analysis.core`.  Scope of this iteration
(deliberately narrow, see README):

* Single CPU, fixed-priority preemptive scheduling, **no shared resources**
  (the data model cannot express critical sections).
* Two criticality levels and two execution modes:

  - LO mode: every task executes within its low budget ``wcet_lo``.
  - HI mode: entered when a high-criticality job needs more than its low
    budget.  High tasks then run up to ``wcet_hi``; low-criticality tasks
    start no new jobs (jobs already released are abandoned at the switch,
    the standard AMC rule; see :mod:`rt_analysis.mc_simulate`).

Analysis (conservative fixed-priority bound, exact integer arithmetic, no
floating point).  Why deliberately conservative is explained below.

* LO-mode bound for every task, all higher-priority tasks counted at C_lo::

    R = C_i(LO) + sum_{h in hp(i)} ceil((R + J_h) / T_h) * C_h(LO)

* HI-mode bound for high-criticality tasks only.  The analyzed task runs at
  C(HI); every higher-priority task is counted over the *whole* response
  window at its largest applicable budget — HI tasks at C(HI), LO tasks at
  C(LO)::

    R = C_i(HI)
        + sum_{h in hp(i), HI} ceil((R + J_h) / T_h) * C_h(HI)
        + sum_{l in hp(i), LO} ceil((R + J_l) / T_l) * C_l(LO)

  Counting LO jobs that would in fact be dropped after the switch (and
  counting pre-switch HI jobs at C(HI)) only ever over-counts interference,
  so the fixed point is a sound upper bound for every possible switch time.
  A tempting tighter term that charges each LO task only once via its own
  converged LO-mode bound is NOT safe under this contract: a LO job released
  exactly when an overrunning job reaches its C(LO) can preempt it and delay
  the switch itself, interfering a second time.  A switch-time-aware
  refinement (AMC-max style) is out of scope here.

A candidate above ``D_i - J_i`` proves a deadline miss; a value that
stabilises proves schedulability; exhausting ``max_iterations`` is reported
as ``iteration_budget_exhausted`` (unknown), never confused with a miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .core import (
    DEFAULT_MAX_ITERATIONS,
    MAX_TASKS,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_DEADLINE_MISS,
    STATUS_SCHEDULABLE,
    ValidationError,
    _as_int,
)

__all__ = [
    "CRITICALITY_LO",
    "CRITICALITY_HI",
    "MCTask",
    "MCIterationRecord",
    "ModeReport",
    "MCTaskReport",
    "MCReport",
    "analyze_mc",
]

#: Low criticality / LO mode.
CRITICALITY_LO = "lo"
#: High criticality / HI mode.
CRITICALITY_HI = "hi"
_CRITICALITIES = (CRITICALITY_LO, CRITICALITY_HI)


@dataclass(frozen=True)
class MCTask:
    """A mixed-criticality sporadic/periodic fixed-priority task.

    ``wcet_lo`` (C(LO)) is the execution budget assumed in LO mode and is
    always ``>= 1``.  For a high-criticality task ``wcet_hi`` (C(HI)) is the
    budget assumed after a mode switch and must satisfy ``wcet_hi >=
    wcet_lo``; for a low-criticality task it must be ``None`` (low tasks have
    no HI-mode budget because they do not run after a switch).  ``deadline``
    satisfies ``1 <= deadline <= period``; ``jitter`` is ``>= 0``.  The
    model has no shared resources, so there are no critical sections.
    """

    name: str
    wcet_lo: int
    period: int
    deadline: int
    criticality: str
    wcet_hi: int | None = None
    jitter: int = 0

    @property
    def is_high(self) -> bool:
        return self.criticality == CRITICALITY_HI


@dataclass(frozen=True)
class MCIterationRecord:
    """One round of a response-time iteration in one mode.

    In a LO-mode round all higher-priority interference (counted at C(LO))
    is listed in ``lo_interference`` and ``hi_interference`` is empty.  In a
    HI-mode round ``hi_interference`` counts higher-priority HI tasks at
    C(HI) over the current window, and ``lo_interference`` counts
    higher-priority LO tasks at C(LO) over the same window (jobs dropped
    after a real switch are included here on purpose, so the bound stays
    conservative).  Each pair is ``(task name, release count)``.
    """

    round: int
    value: int
    hi_interference: tuple[tuple[str, int], ...]
    lo_interference: tuple[tuple[str, int], ...]
    next_value: int


@dataclass(frozen=True)
class ModeReport:
    """Outcome of the iteration for one task in one mode."""

    mode: str
    bound: int | None
    status: str
    iterations: tuple[MCIterationRecord, ...]
    detail: str


@dataclass(frozen=True)
class MCTaskReport:
    """Per-task MC outcome.

    ``lo`` is always present (every task must meet its deadline in LO mode).
    ``hi`` is present only for high-criticality tasks.  ``response_time`` is
    the single guaranteed bound: R(LO) for low tasks, max(R(LO), R(HI)) for
    high tasks, and ``None`` when the task is not guaranteed (miss/unknown).
    """

    name: str
    criticality: str
    lo: ModeReport
    hi: ModeReport | None
    response_time: int | None
    status: str
    detail: str


@dataclass(frozen=True)
class MCReport:
    """Result of :func:`analyze_mc`.

    ``utilization_lo`` counts every task at C(LO); ``utilization_hi`` counts
    high tasks only at C(HI).  A utilization above 1 makes the corresponding
    mode unguaranteeable but is not a format error, and a utilization at or
    below 1 is not treated as sufficient.
    """

    utilization_lo: Fraction
    utilization_hi: Fraction
    utilization_lo_ok: bool
    utilization_hi_ok: bool
    tasks: tuple[MCTaskReport, ...]
    guaranteed: bool
    summary: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_mc_task(raw, index: int) -> MCTask:
    if not isinstance(raw, MCTask):
        raise ValidationError(
            f"task (index {index}): expected a rt_analysis.MCTask instance, "
            f"got {type(raw).__name__} {raw!r}"
        )
    name = raw.name
    if not isinstance(name, str) or not name:
        raise ValidationError(
            f"task (index {index}): field 'name' must be a non-empty string, "
            f"got {name!r}"
        )
    if raw.criticality not in _CRITICALITIES:
        raise ValidationError(
            f"task (index {index}, name {name!r}): field 'criticality' must be "
            f"{CRITICALITY_LO!r} or {CRITICALITY_HI!r}, got {raw.criticality!r}"
        )
    c_lo = _as_int(raw.wcet_lo, field="wcet_lo", index=index, task_name=name)
    t = _as_int(raw.period, field="period", index=index, task_name=name)
    d = _as_int(raw.deadline, field="deadline", index=index, task_name=name)
    j = _as_int(raw.jitter, field="jitter", index=index, task_name=name)
    where = f" (index {index}, name {name!r})"
    if c_lo < 1:
        raise ValidationError(f"task{where}: field 'wcet_lo' must be >= 1, got {c_lo}")
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

    if raw.criticality == CRITICALITY_HI:
        if raw.wcet_hi is None:
            raise ValidationError(
                f"task{where}: high-criticality task requires 'wcet_hi' "
                f"(the HI-mode budget); got None"
            )
        c_hi = _as_int(raw.wcet_hi, field="wcet_hi", index=index, task_name=name)
        if c_hi < c_lo:
            raise ValidationError(
                f"task{where}: 'wcet_hi' ({c_hi}) must be >= 'wcet_lo' ({c_lo})"
            )
    else:
        if raw.wcet_hi is not None:
            raise ValidationError(
                f"task{where}: low-criticality task must not set 'wcet_hi' "
                f"(low tasks have no HI-mode budget); got {raw.wcet_hi!r}"
            )
        c_hi = None
    return MCTask(
        name=name,
        wcet_lo=c_lo,
        period=t,
        deadline=d,
        criticality=raw.criticality,
        wcet_hi=c_hi,
        jitter=j,
    )


def _validate_mc_set(tasks: Sequence[MCTask]) -> tuple[MCTask, ...]:
    if isinstance(tasks, (str, bytes)) or not hasattr(tasks, "__iter__"):
        raise ValidationError(
            f"analyze_mc() expects a sequence of MCTask, got {type(tasks).__name__}"
        )
    tasks = list(tasks)
    if not tasks:
        raise ValidationError("task set must contain at least one task")
    if len(tasks) > MAX_TASKS:
        raise ValidationError(
            f"task set has {len(tasks)} tasks, exceeding the maximum of {MAX_TASKS}"
        )
    normalized = tuple(_validate_mc_task(raw, i) for i, raw in enumerate(tasks))
    seen: set[str] = set()
    for task in normalized:
        if task.name in seen:
            raise ValidationError(f"duplicate task name {task.name!r}")
        seen.add(task.name)
    return normalized


# ---------------------------------------------------------------------------
# Fixed-point iterations
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _lo_mode_report(task: MCTask, hp: tuple[MCTask, ...], max_iterations: int) -> ModeReport:
    """LO-mode RTA: all higher-priority tasks interfere at their LO budget."""
    limit = task.deadline - task.jitter
    r = task.wcet_lo
    iterations: list[MCIterationRecord] = []
    for round_no in range(max_iterations):
        counts = tuple((h.name, _ceil_div(r + h.jitter, h.period)) for h in hp)
        nxt = task.wcet_lo + sum(n * h.wcet_lo for h, (_, n) in zip(hp, counts))
        iterations.append(
            MCIterationRecord(
                round=round_no, value=r, hi_interference=(),
                lo_interference=counts, next_value=nxt,
            )
        )
        if nxt > limit:
            return ModeReport(
                CRITICALITY_LO, None, STATUS_DEADLINE_MISS, tuple(iterations),
                f"LO round {round_no}: R={nxt} exceeds D-J={limit} "
                f"(deadline miss proven within the model)",
            )
        if nxt == r:
            return ModeReport(
                CRITICALITY_LO, r, STATUS_SCHEDULABLE, tuple(iterations),
                f"LO converged after {round_no + 1} round(s): R={r} <= D-J={limit}",
            )
        r = nxt
    return ModeReport(
        CRITICALITY_LO, None, STATUS_BUDGET_EXHAUSTED, tuple(iterations),
        f"LO iteration budget of {max_iterations} round(s) exhausted at R={r} "
        f"(unknown: neither converged nor proven to miss)",
    )


def _hi_mode_report(
    task: MCTask,
    hp_hi: tuple[MCTask, ...],
    hp_lo: tuple[MCTask, ...],
    max_iterations: int,
) -> ModeReport:
    """Conservative HI-mode bound for a high-criticality task."""
    limit = task.deadline - task.jitter
    r = task.wcet_hi
    iterations: list[MCIterationRecord] = []
    for round_no in range(max_iterations):
        hi_counts = tuple((h.name, _ceil_div(r + h.jitter, h.period)) for h in hp_hi)
        lo_counts = tuple((l.name, _ceil_div(r + l.jitter, l.period)) for l in hp_lo)
        nxt = (
            task.wcet_hi
            + sum(n * h.wcet_hi for h, (_, n) in zip(hp_hi, hi_counts))
            + sum(n * l.wcet_lo for l, (_, n) in zip(hp_lo, lo_counts))
        )
        iterations.append(
            MCIterationRecord(
                round=round_no, value=r,
                hi_interference=hi_counts,
                lo_interference=lo_counts,
                next_value=nxt,
            )
        )
        if nxt > limit:
            return ModeReport(
                CRITICALITY_HI, None, STATUS_DEADLINE_MISS, tuple(iterations),
                f"HI round {round_no}: R={nxt} exceeds D-J={limit} "
                f"(deadline miss proven within the model)",
            )
        if nxt == r:
            return ModeReport(
                CRITICALITY_HI, r, STATUS_SCHEDULABLE, tuple(iterations),
                f"HI converged after {round_no + 1} round(s): R={r} <= D-J={limit}",
            )
        r = nxt
    return ModeReport(
        CRITICALITY_HI, None, STATUS_BUDGET_EXHAUSTED, tuple(iterations),
        f"HI iteration budget of {max_iterations} round(s) exhausted at R={r} "
        f"(unknown: neither converged nor proven to miss)",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_mc(
    tasks: Sequence[MCTask], *, max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> MCReport:
    """Compute LO/HI response-time bounds for a mixed-criticality task set.

    ``tasks`` is ordered by priority (``tasks[0]`` highest); priorities are
    fixed and identical in both modes.  Raises :class:`ValidationError`
    (locatable) on any contract violation; the call is pure and leaves no
    partial state.  Utilization above 1 in either mode is reported on the
    returned :class:`MCReport` (``guaranteed == False``), not raised.
    """
    max_iterations = _as_int(
        max_iterations, field="max_iterations", index=None, task_name=None
    )
    if max_iterations < 1:
        raise ValidationError(f"'max_iterations' must be >= 1, got {max_iterations}")
    normalized = _validate_mc_set(tasks)

    utilization_lo = Fraction(0)
    utilization_hi = Fraction(0)
    for task in normalized:
        utilization_lo += Fraction(task.wcet_lo, task.period)
        if task.is_high:
            utilization_hi += Fraction(task.wcet_hi, task.period)
    lo_ok = utilization_lo <= 1
    hi_ok = utilization_hi <= 1

    reports: list[MCTaskReport] = []
    for i, task in enumerate(normalized):
        hp = normalized[:i]
        lo = _lo_mode_report(task, hp, max_iterations)
        if task.is_high:
            hp_hi = tuple(h for h in hp if h.is_high)
            hp_lo = tuple(h for h in hp if not h.is_high)
            hi = _hi_mode_report(task, hp_hi, hp_lo, max_iterations)
        else:
            hi = None
        reports.append(_combine(task, lo, hi))

    guaranteed = lo_ok and hi_ok and all(
        r.status == STATUS_SCHEDULABLE for r in reports
    )
    summary = _summary(utilization_lo, utilization_hi, lo_ok, hi_ok, reports)
    return MCReport(
        utilization_lo=utilization_lo,
        utilization_hi=utilization_hi,
        utilization_lo_ok=lo_ok,
        utilization_hi_ok=hi_ok,
        tasks=tuple(reports),
        guaranteed=guaranteed,
        summary=summary,
    )


def _combine(task: MCTask, lo: ModeReport, hi: ModeReport | None) -> MCTaskReport:
    """Overall per-task status from the per-mode reports."""
    if hi is None:
        status = lo.status
        bound = lo.bound if status == STATUS_SCHEDULABLE else None
        return MCTaskReport(
            name=task.name, criticality=task.criticality, lo=lo, hi=None,
            response_time=bound, status=status, detail=lo.detail,
        )
    # A proven miss in either mode dominates; unknown (budget exhausted) in
    # either mode makes the overall bound unknown.
    if STATUS_DEADLINE_MISS in (lo.status, hi.status):
        status = STATUS_DEADLINE_MISS
    elif STATUS_BUDGET_EXHAUSTED in (lo.status, hi.status):
        status = STATUS_BUDGET_EXHAUSTED
    else:
        status = STATUS_SCHEDULABLE
    if status == STATUS_SCHEDULABLE:
        bound = max(lo.bound, hi.bound)
    else:
        bound = None
    return MCTaskReport(
        name=task.name, criticality=task.criticality, lo=lo, hi=hi,
        response_time=bound, status=status,
        detail=f"{lo.detail} | {hi.detail}",
    )


def _summary(
    utilization_lo: Fraction,
    utilization_hi: Fraction,
    lo_ok: bool,
    hi_ok: bool,
    reports: Sequence[MCTaskReport],
) -> str:
    problems = []
    if not lo_ok:
        problems.append(f"LO utilization {utilization_lo} > 1")
    if not hi_ok:
        problems.append(f"HI utilization {utilization_hi} > 1")
    bad = ", ".join(
        f"{r.name}({r.status})" for r in reports if r.status != STATUS_SCHEDULABLE
    )
    if bad:
        problems.append(f"not guaranteed: {bad}")
    if problems:
        return (
            f"U(LO)={utilization_lo}, U(HI)={utilization_hi}; "
            + "; ".join(problems)
        )
    return (
        f"U(LO)={utilization_lo} ({float(utilization_lo):.4f}) <= 1, "
        f"U(HI)={utilization_hi} ({float(utilization_hi):.4f}) <= 1 and all "
        f"{len(reports)} task(s) guaranteed in their required modes: "
        f"mixed-criticality schedulable under the model"
    )
