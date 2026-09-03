from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agent import DebugAgent
from .dataset import DebugTask


@dataclass(frozen=True)
class EvaluationSummary:
    task_count: int
    baseline_success_count: int
    final_success_count: int
    repaired_count: int
    average_turns: float
    by_error_type: dict[str, dict[str, float | int]]

    @property
    def baseline_accuracy(self) -> float:
        return self.baseline_success_count / self.task_count if self.task_count else 0.0

    @property
    def final_accuracy(self) -> float:
        return self.final_success_count / self.task_count if self.task_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["baseline_accuracy"] = self.baseline_accuracy
        result["final_accuracy"] = self.final_accuracy
        return result


def evaluate(agent: DebugAgent, tasks: list[DebugTask]) -> tuple[EvaluationSummary, list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    baseline_success = 0
    final_success = 0
    repaired = 0
    total_turns = 0
    category_counts: dict[str, dict[str, int]] = {}

    for task in tasks:
        report = agent.run(task.question, task.initial_sql, task.reference_sql)
        first_passed = report.steps[0].passed
        baseline_success += int(first_passed)
        final_success += int(report.success)
        repaired += int(report.repaired)
        total_turns += len(report.steps)
        category = category_counts.setdefault(
            task.error_type,
            {"task_count": 0, "baseline_success_count": 0, "final_success_count": 0},
        )
        category["task_count"] += 1
        category["baseline_success_count"] += int(first_passed)
        category["final_success_count"] += int(report.success)
        reports.append(
            {
                "task": asdict(task),
                "baseline_passed": first_passed,
                "report": report.to_dict(),
            }
        )

    by_error_type: dict[str, dict[str, float | int]] = {}
    for error_type, counts in sorted(category_counts.items()):
        count = counts["task_count"]
        by_error_type[error_type] = {
            **counts,
            "baseline_accuracy": counts["baseline_success_count"] / count,
            "final_accuracy": counts["final_success_count"] / count,
        }

    summary = EvaluationSummary(
        task_count=len(tasks),
        baseline_success_count=baseline_success,
        final_success_count=final_success,
        repaired_count=repaired,
        average_turns=total_turns / len(tasks) if tasks else 0.0,
        by_error_type=by_error_type,
    )
    return summary, reports
