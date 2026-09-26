
"""Demand-bound-function schedulability analysis for preemptive EDF.

Model (single CPU, fully preemptive EDF, no locks, no self-suspension):

* Each task is sporadic with minimum inter-arrival time T (a positive
  integer), relative deadline D (a positive integer, D <= T) and
  worst-case execution demand C (a non-negative rational).
* The demand bound function is
  dbf(t) = sum_i max(0, floor((t - D_i) / T_i) + 1) * C_i.
  The task set is schedulable under EDF iff dbf(t) <= t for all t > 0
  and the utilization U = sum(C_i / T_i) <= 1.

Finite check (why the hyperperiod H = lcm(T_i) suffices, no arbitrary
truncation):

* dbf only increases at points t = D_i + k * T_i; between two consecutive
  such points it is constant while t keeps growing, so a violation
  dbf(t) > t can first appear only at one of these key points.  Checking
  key points instead of every instant is exact, not an approximation.
* H is a multiple of every T_i, hence for every task
  dbf_i(t + H) = dbf_i(t) + (H / T_i) * C_i and therefore
  dbf(t + H) = dbf(t) + U * H.  With U <= 1, if dbf(t) <= t holds
  throughout (0, H], then for any t' = t + m*H:
  dbf(t') = dbf(t) + m * U * H <= t + m * H = t'.  So the interval
  (0, H] is a *proof-complete* finite bound, not a heuristic cutoff.

Budgets: if H exceeds max_hyperperiod or the number of key points in
(0, H] exceeds max_checkpoints, the analysis reports "unproven" —
explicitly distinct from a proven "unschedulable".

All arithmetic is exact (integers and Fraction); no floating point is
used in any decision.
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
    "STATUS_SCHEDULABLE",
    "STATUS_UNSCHEDULABLE",
    "STATUS_UNPROVEN",
    "EdfTask",
    "TaskDemand",
    "Violation",
    "EdfReport",
    "dbf",
    "key_points",
    "analyze_edf",
]

#: Default explicit budgets.  Exceeding either yields status "unproven".
DEFAULT_MAX_HYPERPERIOD = 1_000_000
DEFAULT_MAX_CHECKPOINTS = 100_000

STATUS_SCHEDULABLE = "schedulable"
STATUS_UNSCHEDULABLE = "unschedulable"
STATUS_UNPROVEN = "unproven"


@dataclass(frozen=True)
class EdfTask:
    """A sporadic EDF task: min inter-arrival 'period', relative
    'deadline' <= 'period', non-negative rational 'wcet'."""

    name: str
    wcet: object  # int | Fraction | finite float, >= 0
    period: int
    deadline: int


@dataclass(frozen=True)
class TaskDemand:
    """One task's contribution at a checked instant: job count and demand."""

    name: str
    jobs: int
    demand: Fraction


@dataclass(frozen=True)
class Violation:
    """The earliest instant with dbf(t) > t and its demand breakdown."""

    time: int
    demand: Fraction
    per_task: tuple[TaskDemand, ...]


@dataclass(frozen=True)
class EdfReport:
    """Result of analyze_edf.

    'status' is one of schedulable / unschedulable / unproven.
    'hyperperiod' is H = lcm(periods) when it was within budget (else
    None); 'checkpoints' are the key points actually checked and
    'checked_until' the end of the verified interval (0, H].
    'violation' is the earliest violating key point when one was proven.
    """

    utilization: Fraction
    status: str
    hyperperiod: int | None
    checkpoints: tuple[int, ...]
    checked_until: int | None
    violation: Violation | None
    summary: str


def _as_rational(value, *, field: str, index: int, task_name: str | None) -> Fraction:
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
        result = Fraction(value)  # exact binary value of the float
    else:
        raise ValidationError(
            f"task{where}: field {field!r} must be an int/Fraction/finite float, "
            f"got {type(value).__name__} {value!r}"
        )
    if result < 0:
        raise ValidationError(
            f"task{where}: field {field!r} must be >= 0, got {result}"
        )
    return result


def _validate_edf_task(raw, index: int) -> tuple[str, Fraction, int, int]:
    if not isinstance(raw, EdfTask):
        raise ValidationError(
            f"task (index {index}): expected a rt_analysis.EdfTask instance, "
            f"got {type(raw).__name__} {raw!r}"
        )
    name = raw.name
    if not isinstance(name, str) or not name:
        raise ValidationError(
            f"task (index {index}): field 'name' must be a non-empty string, "
            f"got {name!r}"
        )
    c = _as_rational(raw.wcet, field="wcet", index=index, task_name=name)
    t = _as_int(raw.period, field="period", index=index, task_name=name)
    d = _as_int(raw.deadline, field="deadline", index=index, task_name=name)
    where = _where(index, name)
    if t < 1:
        raise ValidationError(f"task{where}: field 'period' must be >= 1, got {t}")
    if d < 1:
        raise ValidationError(f"task{where}: field 'deadline' must be >= 1, got {d}")
    if d > t:
        raise ValidationError(
            f"task{where}: field 'deadline' ({d}) must be <= 'period' ({t})"
        )
    return name, c, t, d


def _validate_edf_task_set(
    tasks: Sequence[EdfTask],
) -> tuple[tuple[str, Fraction, int, int], ...]:
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
    normalized = tuple(_validate_edf_task(raw, i) for i, raw in enumerate(tasks))
    seen: set[str] = set()
    for name, _c, _t, _d in normalized:
        if name in seen:
            raise ValidationError(f"duplicate task name {name!r}")
        seen.add(name)
    return normalized


def dbf(tasks: Sequence[EdfTask], t) -> Fraction:
    """Exact demand bound function at integer instant t >= 0.

    dbf(t) = sum_i max(0, floor((t - D_i) / T_i) + 1) * C_i.
    """
    t = _as_int(t, field="t", index=None, task_name=None)
    if t < 0:
        raise ValidationError(f"'t' must be >= 0, got {t}")
    normalized = _validate_edf_task_set(tasks)
    total = Fraction(0)
    for _name, c, period, deadline in normalized:
        jobs = (t - deadline) // period + 1
        if jobs > 0:
            total += jobs * c
    return total


def key_points(tasks: Sequence[EdfTask], until) -> tuple[int, ...]:
    """Sorted distinct key points D_i + k * T_i lying in (0, until]."""
    until = _as_int(until, field="until", index=None, task_name=None)
    if until < 1:
        raise ValidationError(f"'until' must be >= 1, got {until}")
    normalized = _validate_edf_task_set(tasks)
    points: set[int] = set()
    for _name, _c, period, deadline in normalized:
        point = deadline
        while point <= until:
            points.add(point)
            point += period
    return tuple(sorted(points))


def analyze_edf(
    tasks: Sequence[EdfTask],
    *,
    max_hyperperiod: int = DEFAULT_MAX_HYPERPERIOD,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> EdfReport:
    """Check EDF schedulability via the demand bound function over (0, H].

    Returns an EdfReport; never raises for overload or budget exhaustion.
    'unschedulable' is returned only when utilization exceeds 1 or a
    concrete earliest violating key point is proven; 'unproven' means an
    explicit budget (max_hyperperiod / max_checkpoints) stopped the
    proof — it must not be read as unschedulable.
    """
    max_hyperperiod = _as_int(
        max_hyperperiod, field="max_hyperperiod", index=None, task_name=None
    )
    if max_hyperperiod < 1:
        raise ValidationError(f"'max_hyperperiod' must be >= 1, got {max_hyperperiod}")
    max_checkpoints = _as_int(
        max_checkpoints, field="max_checkpoints", index=None, task_name=None
    )
    if max_checkpoints < 1:
        raise ValidationError(f"'max_checkpoints' must be >= 1, got {max_checkpoints}")
    normalized = _validate_edf_task_set(tasks)

    utilization = sum((c / period for _n, c, period, _d in normalized), Fraction(0))

    if utilization > 1:
        return EdfReport(
            utilization=utilization,
            status=STATUS_UNSCHEDULABLE,
            hyperperiod=None,
            checkpoints=(),
            checked_until=None,
            violation=None,
            summary=(
                f"utilization {utilization} ({float(utilization):.4f}) > 1: "
                f"unschedulable on one CPU (overload, no key-point check needed)"
            ),
        )

    hyperperiod = math.lcm(*(period for _n, _c, period, _d in normalized))
    if hyperperiod > max_hyperperiod:
        return EdfReport(
            utilization=utilization,
            status=STATUS_UNPROVEN,
            hyperperiod=None,
            checkpoints=(),
            checked_until=None,
            violation=None,
            summary=(
                f"hyperperiod H={hyperperiod} exceeds budget "
                f"max_hyperperiod={max_hyperperiod}: unproven "
                f"(NOT a proof of unschedulability)"
            ),
        )

    points = key_points(
        [EdfTask(n, c, p, d) for n, c, p, d in normalized], hyperperiod
    )
    if len(points) > max_checkpoints:
        return EdfReport(
            utilization=utilization,
            status=STATUS_UNPROVEN,
            hyperperiod=hyperperiod,
            checkpoints=(),
            checked_until=None,
            violation=None,
            summary=(
                f"{len(points)} key points in (0, H={hyperperiod}] exceed budget "
                f"max_checkpoints={max_checkpoints}: unproven "
                f"(NOT a proof of unschedulability)"
            ),
        )

    for point in points:
        per_task = []
        total = Fraction(0)
        for name, c, period, deadline in normalized:
            jobs = (point - deadline) // period + 1
            if jobs > 0:
                demand = jobs * c
            else:
                jobs, demand = 0, Fraction(0)
            per_task.append(TaskDemand(name=name, jobs=jobs, demand=demand))
            total += demand
        if total > point:
            violation = Violation(time=point, demand=total, per_task=tuple(per_task))
            return EdfReport(
                utilization=utilization,
                status=STATUS_UNSCHEDULABLE,
                hyperperiod=hyperperiod,
                checkpoints=points,
                checked_until=hyperperiod,
                violation=violation,
                summary=(
                    f"earliest violation at t={point}: dbf={total} > {point} "
                    f"(utilization {utilization} <= 1); unschedulable under EDF"
                ),
            )

    return EdfReport(
        utilization=utilization,
        status=STATUS_SCHEDULABLE,
        hyperperiod=hyperperiod,
        checkpoints=points,
        checked_until=hyperperiod,
        violation=None,
        summary=(
            f"dbf(t) <= t at all {len(points)} key point(s) in (0, H={hyperperiod}] "
            f"and utilization {utilization} <= 1: schedulable under EDF "
            f"(H is proof-complete: dbf(t+H) = dbf(t) + U*H with U <= 1)"
        ),
    )

