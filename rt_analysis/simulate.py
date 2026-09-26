"""Discrete-time validation simulator for the analysis model.

This is a *validation aid*, not part of the guarantee: it generates concrete
schedules on a small time axis so tests can check that observed response
times never exceed the bounds computed by :func:`rt_analysis.analyze`.

Simulated policy (matching the analysis model):

* Fixed-priority preemptive scheduling, task list order = priority order.
* Jobs are released at ``k * T`` (jitter is realised as 0, one legal
  realisation of the jitter model).
* Shared resources follow the priority ceiling protocol: a job may enter a
  critical section only if its priority is strictly higher than the current
  system ceiling (the highest ceiling among locked resources).  Critical
  sections do not nest.  A job's critical sections occupy, in declaration
  order, the first units of its execution — a deterministic, worst-case
  friendly placement.

Time is discrete; each simulation step executes one time unit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import (
    Task,
    SupplyWindow,
    ValidationError,
    _as_int,
    _resource_ceilings,
    _validate_supply_window,
    _validate_task_set,
    minimum_supply,
)

__all__ = ["JobRecord", "SimResult", "simulate"]


@dataclass(frozen=True)
class JobRecord:
    """One completed job: release, completion and response time."""

    task: str
    release: int
    completion: int
    response: int


@dataclass(frozen=True)
class SimResult:
    """Outcome of a simulation run over ``horizon`` time units."""

    horizon: int
    jobs: tuple[JobRecord, ...]
    incomplete: tuple[tuple[str, int], ...]  # (task name, release) pairs

    def max_response(self) -> dict[str, int]:
        """Largest observed response time per task name."""
        worst: dict[str, int] = {}
        for job in self.jobs:
            worst[job.task] = max(worst.get(job.task, 0), job.response)
        return worst


def _slot_available(time: int, supply: SupplyWindow | None) -> bool:
    """Whether slot [time, time+1) lies in the last ``Q`` units of a period."""
    if supply is None:
        return True
    return time % supply.period >= supply.period - supply.window


def simulate(tasks, horizon, *, supply=None, phases=(0,)) -> SimResult:
    """Simulate the task set for ``horizon`` discrete time units.

    ``supply`` is an optional :class:`~rt_analysis.core.SupplyWindow`:
    outside the open slots no job executes.  ``phases`` enumerates integer
    arrival phases for the task set: for each phase ``phi`` every task ``i``
    releases jobs at ``phi + k * T_i``, so a caller can check the actual
    schedule under every worst-case-arrival-phase candidate (the phases are
    relative to the fixed supply window, which always opens its first window
    at ``P - Q``).  The returned :class:`SimResult` aggregates the jobs of
    every phase; duplicates (e.g. the default single phase 0) collapse.

    Raises :class:`ValidationError` on invalid input (same contract as
    :func:`rt_analysis.analyze`).
    """
    horizon = _as_int(horizon, field="horizon", index=None, task_name=None)
    if horizon < 1:
        raise ValidationError(f"'horizon' must be >= 1, got {horizon}")
    normalized = _validate_task_set(tasks)
    supply = _validate_supply_window(supply)
    ceilings = _resource_ceilings(normalized)

    if isinstance(phases, (str, bytes)) or not hasattr(phases, "__iter__"):
        raise ValidationError(
            f"'phases' must be an iterable of non-negative integers, "
            f"got {type(phases).__name__}"
        )
    phase_values: list[int] = []
    for k, phi in enumerate(phases):
        value = _as_int(phi, field=f"phases[{k}]", index=None, task_name=None)
        if value < 0:
            raise ValidationError(f"phases[{k}] must be >= 0, got {value}")
        if value not in phase_values:
            phase_values.append(value)
    if not phase_values:
        raise ValidationError("'phases' must contain at least one arrival phase")

    # Critical-section spans as execution-progress intervals [start, end).
    cs_spans: list[list[tuple[int, int, str]]] = []
    for task in normalized:
        spans = []
        offset = 0
        for cs in task.critical_sections:
            spans.append((offset, offset + cs.length, cs.resource))
            offset += cs.length
        cs_spans.append(spans)

    records: list[JobRecord] = []
    incomplete: list[tuple[str, int]] = []
    for phi in phase_values:
        jobs: list[dict] = []
        for i, task in enumerate(normalized):
            k = 0
            while phi + k * task.period < horizon:
                jobs.append(
                    {"task": i, "release": phi + k * task.period,
                     "progress": 0, "done": False}
                )
                k += 1

        locked: dict[str, int] = {}  # resource -> holding job index
        for time in range(horizon):
            if not _slot_available(time, supply):
                continue
            system_ceiling = None
            for resource in locked:
                ceiling = ceilings[resource]
                if system_ceiling is None or ceiling < system_ceiling:
                    system_ceiling = ceiling

            # Highest-priority ready job not blocked by the PCP rule.
            chosen = None
            for idx in sorted(range(len(jobs)), key=lambda kk: jobs[kk]["task"]):
                job = jobs[idx]
                if job["done"] or job["release"] > time:
                    continue
                ti = job["task"]
                entering = None
                for start, _end, resource in cs_spans[ti]:
                    if start == job["progress"]:
                        entering = resource
                        break
                if entering is not None:
                    holder = locked.get(entering)
                    if holder is not None and holder != idx:
                        continue  # resource busy: wait for the holder
                    if (
                        holder is None
                        and system_ceiling is not None
                        and ti >= system_ceiling
                    ):
                        continue  # blocked: priority not above system ceiling
                chosen = idx
                break
            if chosen is None:
                continue

            job = jobs[chosen]
            ti = job["task"]
            for start, _end, resource in cs_spans[ti]:
                if start == job["progress"] and resource not in locked:
                    locked[resource] = chosen
            job["progress"] += 1
            for _start, end, resource in cs_spans[ti]:
                if end == job["progress"] and locked.get(resource) == chosen:
                    del locked[resource]
            if job["progress"] == normalized[ti].wcet:
                job["done"] = True
                records.append(
                    JobRecord(
                        task=normalized[ti].name,
                        release=job["release"],
                        completion=time + 1,
                        response=time + 1 - job["release"],
                    )
                )

        for job in jobs:
            if not job["done"]:
                incomplete.append(
                    (normalized[job["task"]].name, job["release"])
                )

    return SimResult(
        horizon=horizon,
        jobs=tuple(records),
        incomplete=tuple(incomplete),
    )
