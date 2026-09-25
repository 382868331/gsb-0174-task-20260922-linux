"""Priority-assignment search: find an order that ``analyze`` can guarantee.

The task parameters and the blocking model are exactly those of
:mod:`rt_analysis.core`; this module only searches for a *priority order*
under which the existing analysis proves every task schedulable.

Search strategy (deterministic):

1. **Greedy pass, lowest priority level upward** (Audsley-style).  At each
   level the unassigned candidates are tried in task-ID (name) order.  A
   candidate is tested by building the tentative order

       <unassigned others, ID order> + [candidate] + <already placed below>

   and running the full :func:`rt_analysis.analyze` on it: resource ceilings
   and blocking terms are recomputed for *this* tentative order, never reused
   from the input order.  The candidate sits exactly at the level being
   filled, with every other unassigned task above it — which is precisely the
   situation any completion of the search would produce, so the per-candidate
   verdict is exact for the candidate's own task.  The first candidate (in ID
   order) that converges within ``D - J`` is placed, and the search moves one
   level up.

2. **Budgeted backtracking.**  For this model the per-task verdict depends
   only on the *sets* of higher- and lower-priority tasks (interference sums
   over the higher-priority set; blocking depends on the lower-priority set
   and on whether any user of a resource sits at or above the task — both
   set-determined), so the exchange conditions behind the greedy pass hold.
   Instead of relying on that argument alone, a greedy failure triggers a
   depth-first backtracking search over the same placement test, bounded by
   ``node_budget`` candidate evaluations.  Three outcomes:

   * a full order is found  -> ``assigned`` (re-verified by ``analyze``);
   * the whole tree is explored without success -> ``no_feasible_order``
     (proven *within this analysis model* — not a claim that no real-world
     schedule could exist);
   * the budget runs out first -> ``incomplete`` (nothing proven either way).

The new entry point accepts at most :data:`MAX_ASSIGN_TASKS` (8) tasks; the
``analyze`` limit of 20 tasks is unchanged.  With 8 tasks the full tree needs
at most ~110k candidate evaluations, so the default budget always suffices
for an exhaustive verdict at the interface limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .core import (
    DEFAULT_MAX_ITERATIONS,
    Report,
    STATUS_SCHEDULABLE,
    Task,
    ValidationError,
    _as_int,
    _validate_task_set,
    analyze,
)

__all__ = [
    "MAX_ASSIGN_TASKS",
    "DEFAULT_NODE_BUDGET",
    "STATUS_ASSIGNED",
    "STATUS_NO_FEASIBLE_ORDER",
    "STATUS_INCOMPLETE",
    "PlacementRecord",
    "AssignmentReport",
    "assign_priorities",
]

#: Maximum number of tasks accepted by :func:`assign_priorities`.
MAX_ASSIGN_TASKS = 8

#: Default budget of candidate evaluations for the backtracking pass.  The
#: full search tree for 8 tasks holds fewer than 110_000 candidate tests
#: (sum over levels of n!/(n-level-1)! placements), so this budget always
#: allows an exhaustive verdict at the interface limit.
DEFAULT_NODE_BUDGET = 200_000

#: Outcomes of :func:`assign_priorities`.
STATUS_ASSIGNED = "assigned"
STATUS_NO_FEASIBLE_ORDER = "no_feasible_order"
STATUS_INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class PlacementRecord:
    """One candidate evaluation during the search.

    ``level`` is the priority index being filled (0 = highest priority),
    ``phase`` is ``"greedy"`` or ``"backtrack"``, and ``outcome`` is
    ``"placed"`` or ``"rejected"`` with a human-readable reason.
    """

    phase: str
    level: int
    candidate: str
    outcome: str
    detail: str


@dataclass(frozen=True)
class AssignmentReport:
    """Result of :func:`assign_priorities`.

    ``order`` lists task names from highest to lowest priority and is
    ``None`` unless ``status == "assigned"``.  ``report`` is the full
    :func:`rt_analysis.analyze` evidence for that order (per-task blocking
    terms and per-round response-time iterations); it is the guarantee, the
    search merely located the order.  ``nodes_explored`` counts candidate
    evaluations across both phases.
    """

    status: str
    order: tuple[str, ...] | None
    report: Report | None
    placements: tuple[PlacementRecord, ...]
    nodes_explored: int
    node_budget: int
    summary: str


def assign_priorities(
    tasks: Sequence[Task],
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> AssignmentReport:
    """Search for a priority order that ``analyze`` proves schedulable.

    ``tasks`` is an unordered set of at most :data:`MAX_ASSIGN_TASKS` tasks
    (the input order carries no priority meaning here).  Candidates are
    tried in task-ID (name) order at every level, so the result is
    deterministic.  Raises :class:`ValidationError` on any contract
    violation; the function is pure.
    """
    node_budget = _as_int(node_budget, field="node_budget", index=None, task_name=None)
    if node_budget < 1:
        raise ValidationError(f"'node_budget' must be >= 1, got {node_budget}")
    normalized = _validate_task_set(tasks)
    if len(normalized) > MAX_ASSIGN_TASKS:
        raise ValidationError(
            f"priority assignment accepts at most {MAX_ASSIGN_TASKS} tasks, "
            f"got {len(normalized)} (analyze() itself still allows up to 20)"
        )
    n = len(normalized)
    by_id = tuple(sorted(normalized, key=lambda t: t.name))  # task ID order

    utilization = Fraction(0)
    for task in by_id:
        utilization += Fraction(task.wcet, task.period)

    placements: list[PlacementRecord] = []
    nodes = 0

    def evaluate(candidate: Task, rest: tuple[Task, ...], below: tuple[Task, ...]):
        """Test ``candidate`` at the current level: all of ``rest`` above it.

        The tentative order recomputes ceilings, blocking and response times
        from scratch via ``analyze``; nothing is inherited from the input
        order.  Returns (schedulable, per-task report for the candidate).
        """
        order = rest + (candidate,) + below
        report = analyze(order, max_iterations=max_iterations)
        task_report = report.tasks[len(rest)]
        return task_report.status == STATUS_SCHEDULABLE, task_report

    def record(phase, level, candidate, outcome, detail):
        placements.append(
            PlacementRecord(
                phase=phase,
                level=level,
                candidate=candidate,
                outcome=outcome,
                detail=detail,
            )
        )

    def finish(status, order, report, summary):
        return AssignmentReport(
            status=status,
            order=order,
            report=report,
            placements=tuple(placements),
            nodes_explored=nodes,
            node_budget=node_budget,
            summary=summary,
        )

    # A utilization above 1 cannot be guaranteed on one CPU under any order:
    # this is an exact, order-independent verdict, not a heuristic.
    if utilization > 1:
        return finish(
            STATUS_NO_FEASIBLE_ORDER,
            None,
            None,
            f"utilization {utilization} ({float(utilization):.4f}) > 1: no priority "
            f"order can be guaranteed on one CPU under the model",
        )

    # -- Phase 1: greedy, lowest level upward, candidates in task-ID order --
    below: tuple[Task, ...] = ()  # placed tasks, below[0] = highest of them
    unassigned: tuple[Task, ...] = by_id
    greedy_ok = True
    for level in range(n - 1, -1, -1):
        placed = None
        for candidate in unassigned:
            rest = tuple(t for t in unassigned if t is not candidate)
            ok, task_report = evaluate(candidate, rest, below)
            nodes += 1
            if ok:
                record(
                    "greedy", level, candidate.name, "placed", task_report.detail
                )
                placed = candidate
                break
            record("greedy", level, candidate.name, "rejected", task_report.detail)
        if placed is None:
            greedy_ok = False
            break
        unassigned = tuple(t for t in unassigned if t is not placed)
        below = (placed,) + below

    if greedy_ok:
        order = below  # highest priority first
        evidence = analyze(order, max_iterations=max_iterations)
        if not evidence.guaranteed:  # unreachable: each placement was exact
            raise AssertionError(
                "internal error: greedy order failed final verification"
            )
        names = tuple(t.name for t in order)
        return finish(
            STATUS_ASSIGNED,
            names,
            evidence,
            f"greedy placement found a guaranteeable order {names} after "
            f"{nodes} candidate evaluation(s)",
        )

    # -- Phase 2: budgeted backtracking over the same placement test --------
    found: tuple[Task, ...] | None = None
    exhausted = False  # set when the budget stops the search early

    def dfs(remaining: tuple[Task, ...], below: tuple[Task, ...]) -> None:
        nonlocal nodes, found, exhausted
        if found is not None or exhausted:
            return
        if not remaining:
            found = below
            return
        level = n - len(remaining)
        for candidate in remaining:
            if nodes >= node_budget:
                exhausted = True
                return
            rest = tuple(t for t in remaining if t is not candidate)
            ok, task_report = evaluate(candidate, rest, below)
            nodes += 1
            if ok:
                record(
                    "backtrack", level, candidate.name, "placed", task_report.detail
                )
                dfs(rest, (candidate,) + below)
                if found is not None or exhausted:
                    return
            else:
                record(
                    "backtrack", level, candidate.name, "rejected",
                    task_report.detail,
                )

    dfs(by_id, ())

    if found is not None:
        evidence = analyze(found, max_iterations=max_iterations)
        if not evidence.guaranteed:  # unreachable: each placement was exact
            raise AssertionError(
                "internal error: backtracked order failed final verification"
            )
        names = tuple(t.name for t in found)
        return finish(
            STATUS_ASSIGNED,
            names,
            evidence,
            f"backtracking found a guaranteeable order {names} after "
            f"{nodes} candidate evaluation(s)",
        )
    if exhausted:
        return finish(
            STATUS_INCOMPLETE,
            None,
            None,
            f"node budget of {node_budget} candidate evaluation(s) exhausted "
            f"after {nodes} nodes: no guaranteeable order found so far, but "
            f"the search was not exhaustive (inconclusive)",
        )
    return finish(
        STATUS_NO_FEASIBLE_ORDER,
        None,
        None,
        f"exhaustive search ({nodes} candidate evaluation(s)) found no "
        f"guaranteeable priority order under the analysis model; this does "
        f"not prove that no real-world schedule exists",
    )
