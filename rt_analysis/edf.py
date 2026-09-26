"""Preemptive single-CPU EDF schedulability analysis via the demand bound.

Model (this module is independent of the fixed-priority analysis in
core.py):

* Single CPU, preemptive Earliest-Deadline-First scheduling.
* Each task is sporadic with minimum inter-arrival time 'period' (T),
  relative deadline 'deadline' (D) with 1 <= D <= T, and a
  non-negative rational worst-case execution time 'wcet' (C).
* No locks, no self-suspension.

The demand bound function is

    dbf(t) = sum_i max(0, floor((t - D_i) / T_i) + 1) * C_i

and the task set is schedulable under EDF iff dbf(t) <= t for all
relevant t (and U = sum(C_i / T_i) <= 1).

Why the hyperperiod H = lcm(T_i) is a sufficient finite bound (not an
arbitrary truncation): dbf(t + H) = dbf(t) + H * U, because every task
contributes exactly H / T_i extra jobs worth (H / T_i) * C_i over
any window of length H.  With U <= 1 this gives
dbf(t + H) <= dbf(t) + H.  So if dbf(t) <= t holds for every
t in (0, H], induction over k shows
dbf(t + k*H) <= t + k*H for all k: no violation can ever appear
beyond H.  Truncating at H is exact, not heuristic.

Why only the points D_i + k * T_i need checking: dbf is a
non-decreasing step function that is constant between consecutive such
points, while the supply t grows.  Hence dbf(t) <= t is hardest
to satisfy exactly at the jump points, and checking all
t = D_i + k * T_i inside (0, H] is equivalent to checking every
real t in the interval.

All arithmetic is exact integer / Fraction arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .core import MAX_TASKS, ValidationError, _as_int, _where

__all__ = [
    "DEFAULT_MAX_HYPERPERIOD",
    "DEFAULT_MAX_CHECKPOINTS",
    "EdfTask",
    "TaskDemand",
    "EdfReport",
    "demand_bound",
    "analyze_edf",
]

#: Default explicit budgets.  Exceeding either one yields status
#: "not_proven" — never silently reported as unschedulable.
DEFAULT_MAX_HYPERPERIOD = 1_000_000
DEFAULT_MAX_CHECKPOINTS = 100_000

STATUS_SCHEDULABLE = "schedulable"
STATUS_UNSCHEDULABLE = "unschedulable"
STATUS_NOT_PROVEN = "not_proven"


@dataclass(frozen=True)
class EdfTask:
    """A sporadic EDF task: rational WCET 'wcet' (C >= 0), integer
    'period' (T >= 1) and 'deadline' (D) with 1 <= D <= T."""

    name: str
    wcet: object  # int | Fraction | integral float; non-negative rational
    period: int
    deadline: int


@dataclass(frozen=True)
class TaskDemand:
    """One task's contribution to dbf at a checkpoint."""

    name: str
    jobs: int
    demand: Fraction


@dataclass(frozen=True)
class EdfReport:
    """Result of analyze_edf.

    'status' is "schedulable", "unschedulable" or
    "not_proven" (budget exceeded — inconclusive, NOT a proof of
    unschedulability).  'first_violation' is the earliest checkpoint
    t with dbf(t) > t (None unless proven unschedulable by a
    checkpoint), and 'violation_demands' lists each task's demand
    contribution at that point.  The verified interval is
    (0, checked_up_to].
    """

    utilization: Fraction
    status: str
    hyperperiod: int | None
    checkpoints_checked: int
    checked_up_to: int
    first_violation: int | None
    violation_demands: tuple[TaskDemand, ...]
    summary: str

    @property
    def schedulable(self) -> bool:
        return self.status == STATUS_SCHEDULABLE


def _as_rational(value, *, field: str, index: int, task_name: str | None) -> Fraction:
    """Coerce a non-negative rational input to an exact Fraction."""
    where = _where(index, task_name)
    if isinstance(value, bool):
        raise ValidationError(
            f"task{where}: field {field!r} must be a rational number, got bool {value!r}"
        )
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, int):
        result = Fraction(value)
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValidationError(
                f"task{where}: field {field!r} must be finite, got {value!r}"
            )
        if not value.is_integer():
            raise ValidationError(
                f"task{where}: field {field!r} as a float must be integral to stay "
                f"exact, got {value!r}; pass fractions.Fraction for fractional values"
            )
        result = Fraction(int(value))
    else:
        raise ValidationError(
            f"task{where}: field {field!r} must be an int, Fraction or integral "
            f"float, got {type(value).__name__} {value!r}"
        )
    if result < 0:
        raise ValidationError(
            f"task{where}: field {field!r} must be >= 0, got {result}"
        )
    return result


def _validate_edf_tasks(tasks: Sequence[EdfTask]) -> tuple[tuple[str, Fraction, int, int], ...]:
    if isinstance(tasks, (str, bytes)) or not hasattr(tasks, "__iter__"):
        raise ValidationError(
            f"analyze_edf() expects a sequence of EdfTask, got {type(tasks).__name__}"
        )
    tasks = list(tasks)
    if not tasks:
        raise ValidationError("task set must contain at least one task")
    if len(tasks) > MAX_TASKS:
        raise ValidationError(
            f"task set has {len(tasks)} tasks, exceeding the maximum of {MAX_TASKS}"
        )
    normalized = []
    seen: set[str] = set()
    for i, raw in enumerate(tasks):
        if not isinstance(raw, EdfTask):
            raise ValidationError(
                f"task (index {i}): expected a rt_analysis.EdfTask instance, "
                f"got {type(raw).__name__} {raw!r}"
            )
        name = raw.name
        if not isinstance(name, str) or not name:
            raise ValidationError(
                f"task (index {i}): field 'name' must be a non-empty string, got {name!r}"
            )
        if name in seen:
            raise ValidationError(f"duplicate task name {name!r}")
        seen.add(name)
        c = _as_rational(raw.wcet, field="wcet", index=i, task_name=name)
        t = _as_int(raw.period, field="period", index=i, task_name=name)
        d = _as_int(raw.deadline, field="deadline", index=i, task_name=name)
        where = _where(i, name)
        if t < 1:
            raise ValidationError(f"task{where}: field 'period' must be >= 1, got {t}")
        if d < 1:
            raise ValidationError(f"task{where}: field 'deadline' must be >= 1, got {d}")
        if d > t:
            raise ValidationError(
                f"task{where}: field 'deadline' ({d}) must be <= 'period' ({t})"
            )
        normalized.append((name, c, t, d))
    return tuple(normalized)


def demand_bound(normalized: tuple[tuple[str, Fraction, int, int], ...], t: int) -> Fraction:
    """Exact dbf(t) for a validated task set (tuple of (name, C, T, D))."""
    total = Fraction(0)
    for _name, c, period, deadline in normalized:
        jobs = (t - deadline) // period + 1
        if jobs > 0:
            total += jobs * c
    return total


def _checkpoints(
    normalized: tuple[tuple[str, Fraction, int, int], ...], hyperperiod: int
) -> list[int]:
    """Sorted distinct D_i + k*T_i inside (0, hyperperiod]."""
    points: set[int] = set()
    for _name, _c, period, deadline in normalized:
        t = deadline
        while t <= hyperperiod:
            points.add(t)
            t += period
    return sorted(points)


def analyze_edf(
    tasks: Sequence[EdfTask],
    *,
    max_hyperperiod: int = DEFAULT_MAX_HYPERPERIOD,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> EdfReport:
    """EDF schedulability check via the demand bound over one hyperperiod.

    Raises ValidationError on contract violations.  A utilization
    above 1 is reported as status == "unschedulable", not raised.  If
    the hyperperiod or the number of checkpoints exceeds the explicit
    budget, the status is "not_proven" — an inconclusive result that
    must not be read as unschedulable.
    """
    max_hyperperiod = _as_int(
        max_hyperperiod, field="max_hyperperiod", index=None, task_name=None
    )
    max_checkpoints = _as_int(
        max_checkpoints, field="max_checkpoints", index=None, task_name=None
    )
    if max_hyperperiod < 1:
        raise ValidationError(f"'max_hyperperiod' must be >= 1, got {max_hyperperiod}")
    if max_checkpoints < 1:
        raise ValidationError(f"'max_checkpoints' must be >= 1, got {max_checkpoints}")
    normalized = _validate_edf_tasks(tasks)

    utilization = sum((c / t for _n, c, t, _d in normalized), Fraction(0))
    if utilization > 1:
        return EdfReport(
            utilization=utilization,
            status=STATUS_UNSCHEDULABLE,
            hyperperiod=None,
            checkpoints_checked=0,
            checked_up_to=0,
            first_violation=None,
            violation_demands=(),
            summary=(
                f"utilization {utilization} ({float(utilization):.4f}) > 1: "
                f"infeasible on one CPU regardless of deadlines"
            ),
        )

    hyperperiod = 1
    for _n, _c, period, _d in normalized:
        hyperperiod = hyperperiod * period // math.gcd(hyperperiod, period)
        if hyperperiod > max_hyperperiod:
            return EdfReport(
                utilization=utilization,
                status=STATUS_NOT_PROVEN,
                hyperperiod=None,
                checkpoints_checked=0,
                checked_up_to=0,
                first_violation=None,
                violation_demands=(),
                summary=(
                    f"hyperperiod exceeds the budget of {max_hyperperiod}: "
                    f"inconclusive (not proven either way)"
                ),
            )

    points = _checkpoints(normalized, hyperperiod)
    if len(points) > max_checkpoints:
        return EdfReport(
            utilization=utilization,
            status=STATUS_NOT_PROVEN,
            hyperperiod=hyperperiod,
            checkpoints_checked=0,
            checked_up_to=0,
            first_violation=None,
            violation_demands=(),
            summary=(
                f"{len(points)} checkpoints in (0, {hyperperiod}] exceed the "
                f"budget of {max_checkpoints}: inconclusive (not proven either way)"
            ),
        )

    for index, t in enumerate(points):
        demand = demand_bound(normalized, t)
        if demand > t:
            per_task = tuple(
                TaskDemand(
                    name=name,
                    jobs=max(0, (t - d) // p + 1),
                    demand=max(0, (t - d) // p + 1) * c,
                )
                for name, c, p, d in normalized
            )
            return EdfReport(
                utilization=utilization,
                status=STATUS_UNSCHEDULABLE,
                hyperperiod=hyperperiod,
                checkpoints_checked=index + 1,
                checked_up_to=t,
                first_violation=t,
                violation_demands=per_task,
                summary=(
                    f"dbf({t}) = {demand} > {t}: earliest violation in "
                    f"(0, {hyperperiod}]; unschedulable under EDF"
                ),
            )

    return EdfReport(
        utilization=utilization,
        status=STATUS_SCHEDULABLE,
        hyperperiod=hyperperiod,
        checkpoints_checked=len(points),
        checked_up_to=hyperperiod,
        first_violation=None,
        violation_demands=(),
        summary=(
            f"utilization {utilization} ({float(utilization):.4f}) <= 1 and "
            f"dbf(t) <= t at all {len(points)} checkpoint(s) in (0, {hyperperiod}]: "
            f"schedulable under EDF (hyperperiod argument covers all t > H)"
        ),
    )

