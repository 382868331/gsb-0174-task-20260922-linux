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
    CRITICALITY_HI,
    Task,
    ValidationError,
    _as_int,
    _resource_ceilings,
    _validate_mc_task_set,
    _validate_task_set,
)

__all__ = ["JobRecord", "SimResult", "simulate", "MCJobRecord", "MCSimResult", "simulate_mc"]


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
# Mixed-criticality mode-switch simulator (LO/HI, no shared resources)
#
# Concrete realization used to sanity-check analyze_mc's bounds:
#
# * Jobs release at k*T.  Fixed-priority preemptive scheduling, list order =
#   priority order, no resources/locks.
# * ``switch_time`` is the explicit LO -> HI switch instant (None = no
#   switch, a pure LO run).  Before it every job executes against its LO
#   budget; at the switch instant unfinished HI jobs are upgraded to their
#   HI budget (the overrun that drives the switch), and LO jobs that have
#   not completed are dropped.  No LO job is ever released at or after the
#   switch ("LO tasks start no new jobs").  HI jobs released at/after the
#   switch execute their HI budget from release.
#
# This generates one legal schedule instance; like simulate() it is a
# validation aid, not part of the guarantee and not a worst-case search.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCJobRecord:
    """One completed mixed-criticality job.

    ``switched`` is true when the job was released in HI mode or was still
    unfinished when the switch occurred (i.e. it executed under the HI
    budget); false for jobs completed entirely in LO mode.
    """

    task: str
    criticality: str
    release: int
    completion: int
    response: int
    switched: bool


@dataclass(frozen=True)
class MCSimResult:
    """Outcome of a mixed-criticality simulation over ``horizon`` units."""

    horizon: int
    switch_time: int | None
    jobs: tuple[MCJobRecord, ...]
    dropped: tuple[tuple[str, int], ...]      # LO jobs abandoned at the switch
    incomplete: tuple[tuple[str, int], ...]   # jobs unfinished by the horizon

    def max_response(self) -> dict[str, int]:
        """Largest observed response time per task name (completed jobs)."""
        worst: dict[str, int] = {}
        for job in self.jobs:
            worst[job.task] = max(worst.get(job.task, 0), job.response)
        return worst


def simulate_mc(tasks, horizon, *, switch_time=None) -> MCSimResult:
    """Simulate a dual-criticality task set with an explicit mode switch.

    ``switch_time`` is ``None`` (pure LO run) or an integer in
    ``1 .. horizon``: at that tick HI jobs are upgraded to their HI budget
    and unfinished LO jobs are dropped; no LO job releases at or after it.
    Raises :class:`ValidationError` on invalid input, including resource
    use (out of scope) or a LO-only task set (nothing could switch).
    """
    horizon = _as_int(horizon, field="horizon", index=None, task_name=None)
    if horizon < 1:
        raise ValidationError(f"'horizon' must be >= 1, got {horizon}")
    if switch_time is not None:
        switch_time = _as_int(
            switch_time, field="switch_time", index=None, task_name=None
        )
        if switch_time < 1:
            raise ValidationError(
                f"'switch_time' must be >= 1 or None, got {switch_time}"
            )
        if switch_time > horizon:
            raise ValidationError(
                f"'switch_time' ({switch_time}) must be <= 'horizon' ({horizon})"
            )
    normalized = _validate_mc_task_set(tasks)

    jobs: list[dict] = []
    for i, task in enumerate(normalized):
        release = 0
        while release < horizon:
            # No new LO jobs at or after the switch.
            if (
                task.criticality != CRITICALITY_HI
                and switch_time is not None
                and release >= switch_time
            ):
                break
            budget = task.wcet
            if (
                task.criticality == CRITICALITY_HI
                and switch_time is not None
                and release >= switch_time
            ):
                budget = task.wcet_hi
            jobs.append(
                {
                    "task": i,
                    "release": release,
                    "required": budget,
                    "progress": 0,
                    "done": False,
                    "dropped": False,
                    "upgraded": False,
                }
            )
            release += task.period

    records: list[MCJobRecord] = []

    for time in range(horizon):
        if time == switch_time:
            for job in jobs:
                if job["done"] or job["release"] > time:
                    continue
                ti = normalized[job["task"]]
                if ti.criticality == CRITICALITY_HI:
                    # An unfinished HI job overruns its LO budget: upgrade.
                    if not job["upgraded"]:
                        job["required"] = ti.wcet_hi
                        job["upgraded"] = True
                else:
                    # LO jobs do not run past the switch: drop them.
                    job["dropped"] = True

        chosen = None
        for idx in sorted(range(len(jobs)), key=lambda k: jobs[k]["task"]):
            job = jobs[idx]
            if job["done"] or job["dropped"] or job["release"] > time:
                continue
            chosen = idx
            break
        if chosen is None:
            continue

        job = jobs[chosen]
        ti = normalized[job["task"]]
        job["progress"] += 1
        if job["progress"] == job["required"]:
            job["done"] = True
            switched = (
                switch_time is not None
                and (job["release"] >= switch_time or job["upgraded"])
            )
            records.append(
                MCJobRecord(
                    task=ti.name,
                    criticality=ti.criticality,
                    release=job["release"],
                    completion=time + 1,
                    response=time + 1 - job["release"],
                    switched=switched,
                )
            )

    dropped = tuple(
        (normalized[job["task"]].name, job["release"])
        for job in jobs
        if job["dropped"]
    )
    incomplete = tuple(
        (normalized[job["task"]].name, job["release"])
        for job in jobs
        if not job["done"] and not job["dropped"]
    )
    return MCSimResult(
        horizon=horizon,
        switch_time=switch_time,
        jobs=tuple(records),
        dropped=dropped,
        incomplete=incomplete,
    )
