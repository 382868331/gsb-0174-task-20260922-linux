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
    ValidationError,
    _as_int,
    _resource_ceilings,
    _validate_task_set,
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


def simulate(tasks, horizon) -> SimResult:
    """Simulate the task set for ``horizon`` discrete time units.

    Raises :class:`ValidationError` on invalid input (same contract as
    :func:`rt_analysis.analyze`).
    """
    horizon = _as_int(horizon, field="horizon", index=None, task_name=None)
    if horizon < 1:
        raise ValidationError(f"'horizon' must be >= 1, got {horizon}")
    normalized = _validate_task_set(tasks)
    ceilings = _resource_ceilings(normalized)

    # Critical-section spans as execution-progress intervals [start, end).
    cs_spans: list[list[tuple[int, int, str]]] = []
    for task in normalized:
        spans = []
        offset = 0
        for cs in task.critical_sections:
            spans.append((offset, offset + cs.length, cs.resource))
            offset += cs.length
        cs_spans.append(spans)

    jobs: list[dict] = []
    for i, task in enumerate(normalized):
        release = 0
        while release < horizon:
            jobs.append(
                {"task": i, "release": release, "progress": 0, "done": False}
            )
            release += task.period

    locked: dict[str, int] = {}  # resource -> holding job index
    records: list[JobRecord] = []

    for time in range(horizon):
        system_ceiling = None
        for resource in locked:
            ceiling = ceilings[resource]
            if system_ceiling is None or ceiling < system_ceiling:
                system_ceiling = ceiling

        # Highest-priority ready job that is not blocked by the PCP rule.
        chosen = None
        for idx in sorted(range(len(jobs)), key=lambda k: jobs[k]["task"]):
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
                    continue  # blocked: priority not above the system ceiling
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

    incomplete = tuple(
        (normalized[job["task"]].name, job["release"])
        for job in jobs
        if not job["done"]
    )
    return SimResult(horizon=horizon, jobs=tuple(records), incomplete=incomplete)


# ---------------------------------------------------------------------------
# EDF validation simulator (independent of the dbf analysis)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdfJobRecord:
    """One completed EDF job: release, absolute deadline, completion."""

    task: str
    release: int
    deadline: int
    completion: int


@dataclass(frozen=True)
class EdfSimResult:
    """Outcome of an EDF simulation over 'horizon' discrete time units."""

    horizon: int
    jobs: tuple[EdfJobRecord, ...]
    misses: tuple[tuple[str, int, int], ...]  # (task, release, deadline)


def simulate_edf(tasks, horizon, *, arrivals=None) -> EdfSimResult:
    """Simulate preemptive EDF on a discrete time axis.

    'tasks' is a sequence of EdfTask with integral wcet (the discrete
    simulator executes one time unit per step; rational execution times are
    rejected).  'arrivals' optionally maps a task name to a sorted tuple of
    explicit release instants honouring the minimum inter-arrival time
    (sporadic model); by default jobs are released at k * period.  Ties on
    absolute deadline are broken by task order.  This is a validation aid
    for cross-checking analyze_edf, not part of the proof.
    """
    from .edf import EdfTask, _validate_edf_task_set

    horizon = _as_int(horizon, field="horizon", index=None, task_name=None)
    if horizon < 1:
        raise ValidationError(f"'horizon' must be >= 1, got {horizon}")
    normalized = _validate_edf_task_set(tasks)
    for name, c, _t, _d in normalized:
        if c.denominator != 1:
            raise ValidationError(
                f"task {name!r}: simulate_edf requires integral wcet, got {c}"
            )

    if arrivals is None:
        release_map = {}
    else:
        if not hasattr(arrivals, "items"):
            raise ValidationError(
                f"'arrivals' must be a mapping name -> releases, "
                f"got {type(arrivals).__name__}"
            )
        release_map = dict(arrivals)

    jobs: list[dict] = []
    for i, (name, c, period, deadline) in enumerate(normalized):
        if name in release_map:
            releases = list(release_map[name])
            prev = None
            for r in releases:
                r = _as_int(r, field=f"arrivals[{name!r}]", index=None, task_name=None)
                if r < 0 or r >= horizon:
                    raise ValidationError(
                        f"task {name!r}: release {r} outside [0, horizon={horizon})"
                    )
                if prev is not None and r - prev < period:
                    raise ValidationError(
                        f"task {name!r}: releases {prev} and {r} violate the "
                        f"minimum inter-arrival time T={period}"
                    )
                prev = r
        else:
            releases = list(range(0, horizon, period))
        for r in releases:
            jobs.append(
                {
                    "task": i,
                    "release": r,
                    "deadline": r + deadline,
                    "remaining": int(c),
                    "missed": False,
                }
            )

    records: list[EdfJobRecord] = []
    misses: list[tuple[str, int, int]] = []
    for time in range(horizon):
        for job in jobs:
            if (
                not job["missed"]
                and job["remaining"] > 0
                and job["deadline"] <= time
            ):
                job["missed"] = True
                misses.append(
                    (normalized[job["task"]][0], job["release"], job["deadline"])
                )
        ready = [
            job
            for job in jobs
            if job["release"] <= time and job["remaining"] > 0
        ]
        if not ready:
            continue
        chosen = min(ready, key=lambda j: (j["deadline"], j["task"]))
        chosen["remaining"] -= 1
        if chosen["remaining"] == 0:
            records.append(
                EdfJobRecord(
                    task=normalized[chosen["task"]][0],
                    release=chosen["release"],
                    deadline=chosen["deadline"],
                    completion=time + 1,
                )
            )
    return EdfSimResult(horizon=horizon, jobs=tuple(records), misses=tuple(misses))

