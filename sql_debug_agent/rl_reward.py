from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .safety import UnsafeQueryError, validate_read_only_sql
from .verifier import SQLVerifier


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    safe: bool
    executable_on_all_databases: bool
    matches_all_databases: bool
    execution_reward: float
    answer_reward: float
    repair_bonus: float
    repeat_penalty: float
    terminal_failure_penalty: float
    database_matches: tuple[bool, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RobustSQLReward:
    """Execution reward guarded by multiple databases with the same schema."""

    def __init__(self, database_paths: list[Path]) -> None:
        if len(database_paths) < 2:
            raise ValueError("稳健奖励至少需要两套同 Schema、不同数据分布的数据库")
        self.verifiers = [SQLVerifier(path) for path in database_paths]

    def score(
        self,
        candidate_sql: str,
        reference_sql: str,
        previous_sql: str | None = None,
        final_turn: bool = False,
    ) -> RewardBreakdown:
        try:
            validate_read_only_sql(candidate_sql)
        except UnsafeQueryError:
            return RewardBreakdown(
                total=-1.0,
                safe=False,
                executable_on_all_databases=False,
                matches_all_databases=False,
                execution_reward=0.0,
                answer_reward=0.0,
                repair_bonus=0.0,
                repeat_penalty=0.0,
                terminal_failure_penalty=0.0,
                database_matches=tuple(False for _ in self.verifiers),
            )

        executable_all = True
        database_matches: list[bool] = []
        for verifier in self.verifiers:
            reference = verifier.execute(reference_sql)
            if not reference.executable:
                raise ValueError(f"参考 SQL 无法执行：{reference.error}")
            candidate = verifier.execute(candidate_sql)
            executable_all = executable_all and candidate.executable
            database_matches.append(
                candidate.executable and candidate.rows == reference.rows
            )

        matches_all = executable_all and all(database_matches)
        execution_reward = 0.2 if executable_all else -0.2
        answer_reward = 1.0 if matches_all else 0.0
        repeated = previous_sql is not None and _normalize_sql(candidate_sql) == _normalize_sql(previous_sql)
        repeat_penalty = -0.2 if repeated else 0.0
        previous_failed = False
        if previous_sql is not None:
            previous_failed = not self.score(previous_sql, reference_sql).matches_all_databases
        repair_bonus = 0.3 if matches_all and previous_failed else 0.0
        terminal_penalty = -0.2 if final_turn and not matches_all else 0.0
        total = execution_reward + answer_reward + repair_bonus + repeat_penalty + terminal_penalty
        return RewardBreakdown(
            total=round(total, 6),
            safe=True,
            executable_on_all_databases=executable_all,
            matches_all_databases=matches_all,
            execution_reward=execution_reward,
            answer_reward=answer_reward,
            repair_bonus=repair_bonus,
            repeat_penalty=repeat_penalty,
            terminal_failure_penalty=terminal_penalty,
            database_matches=tuple(database_matches),
        )


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()
