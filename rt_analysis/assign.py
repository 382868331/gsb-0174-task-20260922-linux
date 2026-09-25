"""Priority-order search: find a task ordering the analysis can guarantee.

This module adds a *priority assignment* entry point on top of the existing
response-time model (:mod:`rt_analysis.core`).  The task parameters, the
blocking model and the response-time recurrence are exactly those of
:func:`rt_analysis.analyze`; only the priority order is treated as unknown.

Search strategy (deterministic, lowest-priority-first):

* Slots are filled from the lowest priority upward.  At each slot the
  unassigned candidates are tried in task-ID order (task ID = index of the
  task in the input sequence).
* A candidate is tested under the assumption that every other unassigned
  task sits above it.  Resource ceilings and the blocking term are
  recomputed for this hypothetical order — never reused from the input
  order — because which critical sections can block a task depends on the
  candidate ordering.
* The candidate's response-time bound is then computed with the usual
  fixed-point recurrence.  This test is exact for the final order: the
  candidate's higher-priority *set* and lower-priority *set* are already
  fixed at test time, the interference sum does not depend on the order
  inside the higher-priority set, and the blocking condition
  ``ceiling <= candidate_index`` is invariant under permutations of the
  tasks above (a resource used by any task above has its ceiling above the
  candidate either way).
* The first passing candidate is placed (greedy step).  The model's
  exchange property makes this choice complete: moving a task upward
  replaces one critical section of blocking (length <= its own C) by at
  least one job of interference (>= its own C), so a task that passes the
  slot test with the full unassigned set above it cannot stop passing when
  some of those tasks move below it.  Hence if any completion is feasible,
  the first passing candidate admits one.  The DFS still tries the
  remaining candidates (backtracking) to finish the exhaustive proof when
  no order exists, and as a defensive mechanism should the model be
  extended; it is bounded by a fixed node budget
  (:data:`DEFAULT_NODE_BUDGET`).

Outcomes are reported as one of three statuses:

* ``assigned`` — an order was found and re-verified end-to-end with
  :func:`rt_analysis.analyze`; the full per-round evidence is attached.
* ``no_feasible_order`` — the search space was exhausted: no priority
  order is guaranteeable *under this analysis model*.  This is a statement
  about the model (a conservative sufficient test), not a proof that the
  task set is unschedulable in reality.
* ``search_budget_exhausted`` — a budget ran out first (the node budget,
  or the per-task iteration budget on some candidate); the result is
  inconclusive.  With the default budgets this cannot happen within the
  8-task interface limit (the exhaustive 8-task space is ~110k candidate
  tests, below the default node budget); it is reachable only with
  caller-reduced budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .core import (
    DEFAULT_MAX_ITERATIONS,
    Report,
    STATUS_BUDGET_EXHAUSTED as _ITERATION_BUDGET,
    STATUS_SCHEDULABLE,
    Task,
    ValidationError,
    _as_int,
    _blocking,
    _resource_ceilings,
    _response_time,
    _validate_task_set,
    analyze,
)

__all__ = [
    "MAX_ASSIGN_TASKS",
    "DEFAULT_NODE_BUDGET",
    "AssignmentReport",
    "assign_priorities",
]

#: Maximum number of tasks accepted by :func:`assign_priorities`.  This is
#: deliberately smaller than the analysis limit (20): the search is
#: worst-case factorial, and 8 tasks keeps the exhaustive space (~110k
#: candidate tests) inside the default node budget.
MAX_ASSIGN_TASKS = 8

#: Default upper bound on candidate tests (search nodes).  Exhausting it is
#: reported as ``search_budget_exhausted`` (inconclusive), distinct from a
#: proven ``no_feasible_order``.
DEFAULT_NODE_BUDGET = 200_000

#: Search outcomes.
STATUS_ASSIGNED = "assigned"
STATUS_NO_FEASIBLE = "no_feasible_order"
STATUS_BUDGET_EXHAUSTED = "search_budget_exhausted"


@dataclass(frozen=True)
class AssignmentReport:
    """Result of :func:`assign_priorities`.

    ``order`` is the guaranteeable priority order (index 0 = highest
    priority) when ``status == "assigned"`` and ``None`` otherwise.
    ``report`` is the :class:`rt_analysis.Report` produced by re-analyzing
    ``order`` end-to-end — the per-round response-time evidence — and is
    ``None`` unless ``status == "assigned"``.  ``candidates_tested`` counts
    the candidate tests (search nodes) performed.
    """

    status: str
    order: tuple[Task, ...] | None
    report: Report | None
    candidates_tested: int
    node_budget: int
    summary: str


def _validate_budget(value, *, field: str) -> int:
    budget = _as_int(value, field=field, index=None, task_name=None)
    if budget < 1:
        raise ValidationError(f"{field!r} must be >= 1, got {budget}")
    return budget


def assign_priorities(
    tasks: Sequence[Task],
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> AssignmentReport:
    """Search for a priority order that :func:`analyze` can guarantee.

    ``tasks`` may be given in any order; task ID = input index, and IDs
    define the deterministic candidate order of the search.  At most
    :data:`MAX_ASSIGN_TASKS` (8) tasks are accepted; the analysis-only
    entry point keeps its own 20-task limit.  Raises
    :class:`ValidationError` (locatable) on any contract violation; the
    function is pure.
    """
    max_iterations = _validate_budget(max_iterations, field="max_iterations")
    node_budget = _validate_budget(node_budget, field="node_budget")
    normalized = _validate_task_set(tasks)
    if len(normalized) > MAX_ASSIGN_TASKS:
        raise ValidationError(
            f"assign_priorities() accepts at most {MAX_ASSIGN_TASKS} tasks, "
            f"got {len(normalized)} (analyze() still accepts up to 20)"
        )
    n = len(normalized)

    utilization = Fraction(0)
    for task in normalized:
        utilization += Fraction(task.wcet, task.period)
    if utilization > 1:
        # Utilization is order-invariant: no permutation can be guaranteed.
        return AssignmentReport(
            status=STATUS_NO_FEASIBLE,
            order=None,
            report=None,
            candidates_tested=0,
            node_budget=node_budget,
            summary=(
                f"utilization {utilization} ({float(utilization):.4f}) > 1 for "
                f"every order: no feasible priority order under the analysis model"
            ),
        )

    assigned: list[int | None] = [None] * n  # slot -> task ID; slot n-1 = lowest
    used = [False] * n
    tested = 0
    node_budget_hit = False
    iteration_inconclusive = False

    def candidate_passes(slot: int, cand: int) -> bool | None:
        """Test ``cand`` at ``slot`` assuming all other unassigned tasks above.

        Ceilings and blocking are recomputed for the hypothetical order
        (unassigned others in ID order, then the candidate, then the tasks
        already assigned below); nothing is reused from the input order.
        Returns True/False for a proven pass/miss, or ``None`` when the
        per-task iteration budget is exhausted (inconclusive).
        """
        others = [i for i in range(n) if not used[i] and i != cand]
        below = [assigned[s] for s in range(slot + 1, n)]
        ordering = tuple(normalized[i] for i in (*others, cand, *below))
        ceilings = _resource_ceilings(ordering)
        pos = len(others)
        blocking = _blocking(pos, ordering, ceilings)
        report = _response_time(ordering[pos], ordering[:pos], blocking, max_iterations)
        if report.status == STATUS_SCHEDULABLE:
            return True
        if report.status == _ITERATION_BUDGET:
            return None
        return False

    def dfs(slot: int) -> bool:
        nonlocal tested, node_budget_hit, iteration_inconclusive
        if slot < 0:
            return True
        for cand in range(n):
            if used[cand]:
                continue
            if tested >= node_budget:
                node_budget_hit = True
                return False
            tested += 1
            outcome = candidate_passes(slot, cand)
            if outcome is None:
                iteration_inconclusive = True
                return False
            if outcome:
                used[cand] = True
                assigned[slot] = cand
                if dfs(slot - 1):
                    return True
                used[cand] = False
                assigned[slot] = None
        return False

    found = dfs(n - 1)
    if found:
        order = tuple(normalized[i] for i in assigned)
        # End-to-end evidence: re-analyze the found order from scratch.
        evidence = analyze(order, max_iterations=max_iterations)
        if not evidence.guaranteed:  # unreachable: per-slot tests are exact
            raise AssertionError(
                "internal error: searched order failed re-analysis: "
                + evidence.summary
            )
        listing = ", ".join(t.name for t in order)
        return AssignmentReport(
            status=STATUS_ASSIGNED,
            order=order,
            report=evidence,
            candidates_tested=tested,
            node_budget=node_budget,
            summary=(
                f"found guaranteeable order (highest priority first): {listing} "
                f"after {tested} candidate test(s); re-analysis confirms: "
                f"{evidence.summary}"
            ),
        )

    if node_budget_hit or iteration_inconclusive:
        why = (
            f"search node budget of {node_budget} candidate test(s) exhausted "
            f"after {tested} test(s)"
            if node_budget_hit
            else f"a candidate test hit the per-task iteration budget of "
                 f"{max_iterations} round(s) after {tested} candidate test(s)"
        )
        return AssignmentReport(
            status=STATUS_BUDGET_EXHAUSTED,
            order=None,
            report=None,
            candidates_tested=tested,
            node_budget=node_budget,
            summary=(
                f"{why}: inconclusive — neither a guaranteeable order nor "
                f"exhaustive infeasibility"
            ),
        )
    return AssignmentReport(
        status=STATUS_NO_FEASIBLE,
        order=None,
        report=None,
        candidates_tested=tested,
        node_budget=node_budget,
        summary=(
            f"exhaustive search ({tested} candidate test(s)): no priority "
            f"order is guaranteeable under this analysis model; this is a "
            f"model-scoped statement, not a proof of real-world unschedulability"
        ),
    )
