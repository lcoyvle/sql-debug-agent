from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DebugTask:
    task_id: str
    question: str
    initial_sql: str
    reference_sql: str
    error_type: str
    template_id: str | None = None


def load_tasks(path: Path) -> list[DebugTask]:
    tasks: list[DebugTask] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                tasks.append(DebugTask(**item))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number} 数据格式错误：{exc}") from exc
    return tasks


def write_tasks(path: Path, tasks: list[DebugTask]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            item = {key: value for key, value in asdict(task).items() if value is not None}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path
