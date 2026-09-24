"""Reusable worst-case response-time analysis for fixed-priority task sets."""

from .core import (
    DEFAULT_MAX_ITERATIONS,
    MAX_TASKS,
    BlockingInfo,
    CriticalSection,
    IterationRecord,
    Report,
    Task,
    TaskReport,
    ValidationError,
    analyze,
)
from .simulate import JobRecord, SimResult, simulate
from .mixed import (
    CRITICALITY_HI,
    CRITICALITY_LO,
    MCIterationRecord,
    MCReport,
    MCTask,
    MCTaskReport,
    ModeReport,
    analyze_mc,
)
from .mc_simulate import (
    OVERRUN_ALL,
    MCJobRecord,
    MCSimResult,
    simulate_mc,
)

__all__ = [
    "MAX_TASKS",
    "DEFAULT_MAX_ITERATIONS",
    "ValidationError",
    "CriticalSection",
    "Task",
    "BlockingInfo",
    "IterationRecord",
    "TaskReport",
    "Report",
    "analyze",
    "JobRecord",
    "SimResult",
    "simulate",
    # Mixed-criticality iteration (LO/HI modes, two budgets).
    "CRITICALITY_LO",
    "CRITICALITY_HI",
    "MCTask",
    "MCIterationRecord",
    "ModeReport",
    "MCTaskReport",
    "MCReport",
    "analyze_mc",
    "MCJobRecord",
    "MCSimResult",
    "OVERRUN_ALL",
    "simulate_mc",
]
