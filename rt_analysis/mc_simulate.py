"""Discrete-time validation simulator for the mixed-criticality model.

Validation aid only, like :mod:`rt_analysis.simulate`; it is not part of the
guarantee.  It generates one concrete schedule per deterministic scenario so
tests can check observed response times against the hand-derived LO and
AMC-rtb HI bounds.

Simulated policy (matching :mod:`rt_analysis.mixed`):

* Fixed-priority preemptive scheduling, list order = priority order, one CPU,
  no shared resources.  Jobs are released at ``k * T`` (jitter is realised as
  0, one legal realisation of the jitter model).
* Initially the system is in LO mode and every job executes within its LO
  budget.  A HI job named in ``overruns`` is given execution requirement
  ``wcet_hi``; the first time such a job needs execution past ``wcet_lo`` the
  system switches to HI mode.  From the switch instant:

    - no new low-criticality job is started (releases at/after the switch are
      ignored), and low jobs already released but unfinished are abandoned;
    - high-criticality jobs keep their fixed priorities and run to
      completion (overrun jobs up to ``wcet_hi``).

``overruns`` is either the string ``"all"`` (every released HI job whose
``wcet_hi > wcet_lo`` overruns) or an iterable of explicit ``(task name,
release time)`` pairs.  Time is discrete; each step executes one time unit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mixed import (
    CRITICALITY_HI,
    CRITICALITY_LO,
    MCTask,
    ValidationError,
    _validate_mc_set,
)
from .core import _as_int

__all__ = ["MCJobRecord", "MCSimResult", "simulate_mc", "OVERRUN_ALL"]

#: Selector: every HI job with headroom (wcet_hi > wcet_lo) overruns.
OVERRUN_ALL = "all"


@dataclass(frozen=True)
class MCJobRecord:
    """One completed job in the simulated schedule."""

    task: str
    criticality: str
    release: int
    completion: int
    response: int
    overrun: bool


@dataclass(frozen=True)
class MCSimResult:
    """Outcome of a mixed-criticality simulation over ``horizon`` time units.

    ``mode_switch`` is the integer time at which the HI mode started
    (exclusive lower bound of HI-mode ticks), or ``None`` when no overrun
    happened.  ``abandoned`` lists low-criticality jobs that were released
    but never started / finished because of the switch, as
    ``(task name, release)`` pairs.
    """

    horizon: int
    mode_switch: int | None
    jobs: tuple[MCJobRecord, ...]
    abandoned: tuple[tuple[str, int], ...]
    incomplete: tuple[tuple[str, int], ...]

    def max_response(self) -> dict[str, int]:
        """Largest observed response time per task name."""
        worst: dict[str, int] = {}
        for job in self.jobs:
            worst[job.task] = max(worst.get(job.task, 0), job.response)
        return worst


def _normalize_overruns(overruns, tasks: tuple[MCTask, ...], horizon: int):
    """Return a set of (task index, release) pairs naming overrun jobs."""
    by_name = {task.name: (i, task) for i, task in enumerate(tasks)}
    if overruns == OVERRUN_ALL:
        selected = set()
        for i, task in enumerate(tasks):
            if task.is_high and task.wcet_hi > task.wcet_lo:
                release = 0
                while release < horizon:
                    selected.add((i, release))
                    release += task.period
        return selected
    if isinstance(overruns, (str, bytes)) or not hasattr(overruns, "__iter__"):
        raise ValidationError(
            f"'overruns' must be 'all' or an iterable of (task name, release) "
            f"pairs, got {type(overruns).__name__} {overruns!r}"
        )
    selected = set()
    for k, pair in enumerate(overruns):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValidationError(
                f"overruns[{k}] must be a (task name, release) pair, got {pair!r}"
            )
        name, release = pair
        if not isinstance(name, str) or name not in by_name:
            raise ValidationError(
                f"overruns[{k}]: unknown task name {name!r}"
            )
        i, task = by_name[name]
        if not task.is_high:
            raise ValidationError(
                f"overruns[{k}]: task {name!r} is low-criticality; only a "
                f"high-criticality job can exceed its LO budget"
            )
        release = _as_int(
            release, field=f"overruns[{k}].release", index=None, task_name=name
        )
        if release < 0 or release >= horizon:
            raise ValidationError(
                f"overruns[{k}]: release {release} of task {name!r} is outside "
                f"the simulation horizon [0, {horizon})"
            )
        if release % task.period != 0:
            raise ValidationError(
                f"overruns[{k}]: task {name!r} releases at multiples of its "
                f"period {task.period}, so release {release} never happens"
            )
        if task.wcet_hi <= task.wcet_lo:
            raise ValidationError(
                f"overruns[{k}]: task {name!r} has wcet_hi == wcet_lo "
                f"({task.wcet_hi}); it cannot exceed its LO budget"
            )
        selected.add((i, release))
    return selected


def simulate_mc(tasks, horizon, *, overruns=()) -> MCSimResult:
    """Simulate the mixed-criticality task set for ``horizon`` time units.

    Raises :class:`ValidationError` on invalid input (same contract spirit as
    :func:`rt_analysis.analyze_mc`).
    """
    horizon = _as_int(horizon, field="horizon", index=None, task_name=None)
    if horizon < 1:
        raise ValidationError(f"'horizon' must be >= 1, got {horizon}")
    normalized = _validate_mc_set(tasks)
    overrun_set = _normalize_overruns(overruns, normalized, horizon)

    # Pre-create every LO-mode release (HI jobs releasing after the switch are
    # added on the fly; in practice pre-creating them is equivalent, but
    # keeping LO releases switch-aware makes the abandonment boundary exact).
    jobs: list[dict] = []

    def release_job(ti: int, release: int) -> None:
        task = normalized[ti]
        is_overrun = (ti, release) in overrun_set
        requirement = task.wcet_hi if is_overrun else task.wcet_lo
        jobs.append(
            {
                "task": ti,
                "release": release,
                "progress": 0,
                "requirement": requirement,
                "overrun": is_overrun,
                "done": False,
            }
        )

    for i, task in enumerate(normalized):
        release = 0
        while release < horizon:
            release_job(i, release)
            release += task.period

    records: list[MCJobRecord] = []
    abandoned: set[tuple[str, int]] = set()
    mode = CRITICALITY_LO
    switch_time: int | None = None

    for time in range(horizon):
        # Highest-priority eligible job.  In HI mode low-criticality jobs are
        # neither started nor continued: jobs not yet finished are abandoned.
        chosen = None
        for idx in sorted(range(len(jobs)), key=lambda k: jobs[k]["task"]):
            job = jobs[idx]
            if job["done"] or job["release"] > time:
                continue
            ti = job["task"]
            if mode == CRITICALITY_HI and not normalized[ti].is_high:
                continue
            chosen = idx
            break
        if chosen is None:
            continue

        job = jobs[chosen]
        ti = job["task"]
        task = normalized[ti]
        job["progress"] += 1

        if job["progress"] == job["requirement"]:
            job["done"] = True
            records.append(
                MCJobRecord(
                    task=task.name,
                    criticality=task.criticality,
                    release=job["release"],
                    completion=time + 1,
                    response=time + 1 - job["release"],
                    overrun=job["overrun"],
                )
            )
        elif (
            mode == CRITICALITY_LO
            and task.is_high
            and job["overrun"]
            and job["progress"] == task.wcet_lo
        ):
            # This job needs execution past its LO budget: switch at the end
            # of this tick.  Low jobs released before the switch but unfinished
            # (whether started or not) are abandoned; no new low job starts.
            mode = CRITICALITY_HI
            switch_time = time + 1
            for other in jobs:
                oj = other["task"]
                if (
                    not other["done"]
                    and not normalized[oj].is_high
                    and other["release"] < horizon
                ):
                    abandoned.add((normalized[oj].name, other["release"]))

    # All low jobs (including future releases, which are pre-created) that
    # never finish are abandoned by the switch; the set was populated at the
    # switch instant above.  Everything else unfinished only ran out of
    # horizon.
    incomplete = tuple(
        sorted(
            (normalized[job["task"]].name, job["release"])
            for job in jobs
            if not job["done"]
            and (normalized[job["task"]].name, job["release"]) not in abandoned
        )
    )
    return MCSimResult(
        horizon=horizon,
        mode_switch=switch_time,
        jobs=tuple(records),
        abandoned=tuple(sorted(abandoned)),
        incomplete=incomplete,
    )
