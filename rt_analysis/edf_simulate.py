"""Event-driven preemptive EDF simulation (validation aid, independent reference).

Unlike the fixed-priority simulator in simulate.py, this is a
pure event-driven simulator written directly from the scheduling rules:
* jobs are released at k*T for k*T < horizon (synchronous arrival
  pattern — one legal, often worst, realisation of the sporadic model);
* the ready job with the earliest absolute deadline runs (ties broken by
  earlier release, then task index); it is preempted as soon as a job
  with an earlier deadline is released;
* time and execution are exact Fraction arithmetic, so rational WCETs
  such as Fraction(1, 2) are handled exactly.

This simulator is an independent cross-check for tests/demos; it is not
part of the guarantee delivered by analyze_edf.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .core import ValidationError, _as_int
from .edf import _validate_edf_tasks

__all__ = ["EdfJobRecord", "EdfSimResult", "simulate_edf"]


@dataclass(frozen=True)
class EdfJobRecord:
    """One EDF job outcome (times are exact Fractions)."""

    task: str
    release: Fraction
    deadline: Fraction
    completion: Fraction
    response: Fraction
    missed: bool


@dataclass(frozen=True)
class EdfSimResult:
    """Outcome of an EDF simulation over 'horizon' time units."""

    horizon: Fraction
    jobs: tuple[EdfJobRecord, ...]
    incomplete: tuple[tuple[str, Fraction], ...]  # (task name, release) pairs

    def any_miss(self) -> bool:
        return any(job.missed for job in self.jobs)


def simulate_edf(tasks, horizon) -> EdfSimResult:
    """Simulate synchronous arrivals under preemptive EDF up to 'horizon'.

    Only jobs released strictly before 'horizon' are generated.  Jobs
    unfinished at 'horizon' are reported in 'incomplete'.  Raises
    ValidationError on invalid input.
    """
    horizon_int = _as_int(horizon, field="horizon", index=None, task_name=None)
    if horizon_int < 1:
        raise ValidationError(f"'horizon' must be >= 1, got {horizon_int}")
    normalized = _validate_edf_tasks(tasks)

    # Each job: dict(task index, release, abs deadline, remaining, done)
    jobs: list[dict] = []
    for i, (_name, c, period, deadline) in enumerate(normalized):
        k = 0
        while k * period < horizon_int:
            release = Fraction(k * period)
            jobs.append(
                {
                    "task": i,
                    "release": release,
                    "deadline": release + deadline,
                    "remaining": Fraction(c),
                    "done": False,
                }
            )
            k += 1

    jobs.sort(key=lambda j: (j["release"], j["task"]))
    records: list[EdfJobRecord] = []
    time = Fraction(0)
    cursor = 0  # next job index not yet released
    total = len(jobs)
    horizon_frac = Fraction(horizon_int)

    def release_due():
        nonlocal cursor, time
        while cursor < total and jobs[cursor]["release"] <= time:
            job = jobs[cursor]
            if job["remaining"] == 0:
                job["done"] = True
                name = normalized[job["task"]][0]
                records.append(
                    EdfJobRecord(
                        task=name,
                        release=job["release"],
                        deadline=job["deadline"],
                        completion=job["release"],
                        response=Fraction(0),
                        missed=False,
                    )
                )
            cursor += 1

    while True:
        release_due()
        ready = [j for j in jobs if not j["done"] and j["release"] <= time]
        if not ready:
            if cursor >= total:
                break
            time = jobs[cursor]["release"]
            continue
        current = min(ready, key=lambda j: (j["deadline"], j["release"], j["task"]))
        next_release = jobs[cursor]["release"] if cursor < total else None
        run_for = current["remaining"]
        if next_release is not None:
            run_for = min(run_for, next_release - time)
        # Do not run past the horizon: unfinished jobs become 'incomplete'.
        run_for = min(run_for, horizon_frac - time)
        time += run_for
        current["remaining"] -= run_for
        if current["remaining"] == 0:
            current["done"] = True
            name = normalized[current["task"]][0]
            records.append(
                EdfJobRecord(
                    task=name,
                    release=current["release"],
                    deadline=current["deadline"],
                    completion=time,
                    response=time - current["release"],
                    missed=time > current["deadline"],
                )
            )
        if time >= horizon_frac:
            break

    records.sort(key=lambda r: (r.release, r.task))
    incomplete = tuple(
        (normalized[j["task"]][0], j["release"])
        for j in jobs
        if not j["done"]
    )
    return EdfSimResult(horizon=horizon_frac, jobs=tuple(records), incomplete=incomplete)
