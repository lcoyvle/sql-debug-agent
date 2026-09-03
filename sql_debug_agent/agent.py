from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .repair import Repairer
from .verifier import SQLVerifier, Verification


@dataclass(frozen=True)
class DebugStep:
    turn: int
    sql: str
    passed: bool
    feedback: str
    reward: float


@dataclass(frozen=True)
class DebugReport:
    question: str
    initial_sql: str
    final_sql: str
    success: bool
    steps: tuple[DebugStep, ...]

    @property
    def repaired(self) -> bool:
        return self.success and self.final_sql.strip() != self.initial_sql.strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DebugAgent:
    def __init__(self, verifier: SQLVerifier, repairer: Repairer, max_turns: int = 3) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 必须大于等于 1")
        self.verifier = verifier
        self.repairer = repairer
        self.max_turns = max_turns

    def run(
        self, question: str, initial_sql: str, reference_sql: str | None = None
    ) -> DebugReport:
        current_sql = initial_sql.strip()
        history: list[DebugStep] = []

        for turn in range(1, self.max_turns + 1):
            verification: Verification = self.verifier.verify(current_sql, reference_sql)
            history.append(
                DebugStep(
                    turn=turn,
                    sql=current_sql,
                    passed=verification.passed,
                    feedback=verification.feedback,
                    reward=verification.total_reward,
                )
            )
            if verification.passed:
                break
            if turn == self.max_turns:
                break

            revised_sql = self.repairer.repair(question, current_sql, verification.feedback)
            if revised_sql.strip() == current_sql.strip():
                break
            current_sql = revised_sql.strip()

        return DebugReport(
            question=question,
            initial_sql=initial_sql,
            final_sql=current_sql,
            success=history[-1].passed,
            steps=tuple(history),
        )
