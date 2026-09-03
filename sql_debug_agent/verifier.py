from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .safety import UnsafeQueryError, validate_read_only_sql


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    error: str | None = None

    @property
    def executable(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Verification:
    passed: bool
    feedback: str
    candidate: QueryResult
    reference: QueryResult | None
    execution_reward: float
    answer_reward: float

    @property
    def total_reward(self) -> float:
        return self.execution_reward + self.answer_reward


class SQLVerifier:
    def __init__(self, database_path: Path, max_rows: int = 1_000) -> None:
        self.database_path = database_path.resolve()
        self.max_rows = max_rows

    def execute(self, sql: str) -> QueryResult:
        try:
            safe_sql = validate_read_only_sql(sql)
        except UnsafeQueryError as exc:
            return QueryResult(error=f"安全检查失败：{exc}")

        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro", uri=True, timeout=2.0
        )
        connection.execute("PRAGMA query_only = ON")
        steps = 0

        def stop_long_query() -> int:
            nonlocal steps
            steps += 1
            return 1 if steps > 20_000 else 0

        connection.set_progress_handler(stop_long_query, 1_000)
        try:
            cursor = connection.execute(safe_sql)
            rows = tuple(tuple(row) for row in cursor.fetchmany(self.max_rows + 1))
            if len(rows) > self.max_rows:
                return QueryResult(error=f"结果超过 {self.max_rows} 行限制")
            columns = tuple(item[0] for item in (cursor.description or ()))
            return QueryResult(columns=columns, rows=rows)
        except sqlite3.Error as exc:
            return QueryResult(error=f"SQLite 执行错误：{exc}")
        finally:
            connection.close()

    def verify(self, candidate_sql: str, reference_sql: str | None = None) -> Verification:
        candidate = self.execute(candidate_sql)
        if not candidate.executable:
            return Verification(
                passed=False,
                feedback=candidate.error or "SQL 执行失败",
                candidate=candidate,
                reference=None,
                execution_reward=-0.2,
                answer_reward=0.0,
            )

        if reference_sql is None:
            return Verification(
                passed=True,
                feedback="SQL 可安全执行；未提供参考查询，因此没有验证语义正确性。",
                candidate=candidate,
                reference=None,
                execution_reward=0.2,
                answer_reward=0.0,
            )

        reference = self.execute(reference_sql)
        if not reference.executable:
            raise ValueError(f"参考 SQL 无法执行：{reference.error}")

        same_rows = candidate.rows == reference.rows
        # SQL expressions and aliases can produce different cursor labels while
        # returning the same answer. Execution-match accuracy evaluates values,
        # not whether the model copied the reference query's spelling.
        if same_rows:
            label_note = (
                " 返回列名与参考查询不同，但列值和顺序一致。"
                if candidate.columns != reference.columns
                else ""
            )
            return Verification(
                passed=True,
                feedback="SQL 可执行，且执行结果与参考答案一致。" + label_note,
                candidate=candidate,
                reference=reference,
                execution_reward=0.2,
                answer_reward=1.0,
            )

        details: list[str] = ["SQL 可以执行，但结果不正确。"]
        details.append(
            f"候选结果有 {len(candidate.rows)} 行，期望结果有 {len(reference.rows)} 行。"
        )
        return Verification(
            passed=False,
            feedback=" ".join(details),
            candidate=candidate,
            reference=reference,
            execution_reward=0.2,
            answer_reward=0.0,
        )
