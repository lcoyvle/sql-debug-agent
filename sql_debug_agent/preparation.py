from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .database import get_schema
from .dataset import DebugTask
from .verifier import SQLVerifier


SYSTEM_PROMPT = """你是一个只读 SQLite 调试助手。根据用户问题、数据库 Schema、错误 SQL 和执行反馈，只返回一条修正后的 SQLite 查询。
规则：只做解决已知错误所需的最小修改；保留原 SQL 中题目要求的正确条件，不得凭空增加筛选条件；只能使用 Schema 中的字段；日期是 ISO 文本，直接比较 YYYY-MM-DD，不得使用 TO_DATE；需要包含零记录对象时，右表筛选应放在 JOIN ON 或 CASE 中，不得用 WHERE 破坏 LEFT JOIN；不得执行写操作或返回解释。"""


def build_sft_user_content(
    question: str, schema: str, sql: str, feedback: str
) -> str:
    return (
        f"用户问题：{question}\n\n"
        f"数据库 Schema：\n{schema}\n\n"
        f"错误 SQL：\n{sql}\n\n"
        f"验证反馈：\n{feedback}"
    )


def task_to_sft_record(
    task: DebugTask, verifier: SQLVerifier, schema: str
) -> dict[str, Any]:
    """Convert one raw bad case into a supervised correction trajectory."""
    baseline = verifier.verify(task.initial_sql, task.reference_sql)
    user_content = build_sft_user_content(
        task.question, schema, task.initial_sql, baseline.feedback
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": task.reference_sql},
        ],
        "metadata": {
            "task_id": task.task_id,
            "template_id": task.template_id,
            "error_type": task.error_type,
            "baseline_executable": baseline.candidate.executable,
            "baseline_reward": baseline.total_reward,
        },
    }


def _stable_group_order(group_key: str) -> str:
    return hashlib.sha256(group_key.encode("utf-8")).hexdigest()


def _choose_eval_groups(
    groups: dict[str, list[dict[str, Any]]], eval_ratio: float
) -> set[str]:
    """Choose whole template groups whose total size is closest to the target."""
    if len(groups) < 2:
        return set()

    ordered = sorted(groups, key=_stable_group_order)
    total_count = sum(len(records) for records in groups.values())
    target_count = total_count * eval_ratio

    # Dynamic programming keeps one deterministic subset for each achievable size.
    subsets: dict[int, tuple[str, ...]] = {0: ()}
    for key in ordered:
        group_size = len(groups[key])
        additions = {
            size + group_size: chosen + (key,)
            for size, chosen in list(subsets.items())
        }
        for size, chosen in additions.items():
            subsets.setdefault(size, chosen)

    candidates = [
        (size, chosen)
        for size, chosen in subsets.items()
        if 0 < len(chosen) < len(groups)
    ]
    _, chosen = min(
        candidates,
        key=lambda item: (abs(item[0] - target_count), item[0] > target_count, item[1]),
    )
    return set(chosen)


def prepare_sft_data(
    tasks: list[DebugTask],
    verifier: SQLVerifier,
    database_path: Path,
    output_dir: Path,
    eval_ratio: float = 0.2,
) -> tuple[Path, Path, int, int]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio 必须在 0 和 1 之间")

    schema = get_schema(database_path)
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task in tasks:
        record = task_to_sft_record(task, verifier, schema)
        split_key = task.template_id or task.task_id
        grouped[task.error_type][split_key].append(record)

    for error_type in sorted(grouped):
        error_groups = grouped[error_type]
        eval_keys = _choose_eval_groups(error_groups, eval_ratio)
        for split_key, records in error_groups.items():
            target = eval_records if split_key in eval_keys else train_records
            target.extend(records)

    # Tiny datasets can hash into an empty split. Move a complete template group so
    # paraphrases from the same generator never appear in both train and eval.
    if len(tasks) >= 2 and not eval_records:
        move_task = tasks[-1]
        moved = [
            record
            for record in train_records
            if _belongs_to_task_group(record, move_task)
        ]
        train_records = [record for record in train_records if record not in moved]
        eval_records.extend(moved)
    if len(tasks) >= 2 and not train_records:
        move_task = tasks[0]
        moved = [
            record
            for record in eval_records
            if _belongs_to_task_group(record, move_task)
        ]
        eval_records = [record for record in eval_records if record not in moved]
        train_records.extend(moved)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "sft_train.jsonl"
    eval_path = output_dir / "sft_eval.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(eval_path, eval_records)
    return train_path, eval_path, len(train_records), len(eval_records)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _belongs_to_task_group(record: dict[str, Any], task: DebugTask) -> bool:
    metadata = record["metadata"]
    if metadata["error_type"] != task.error_type:
        return False
    if task.template_id is None:
        return metadata["task_id"] == task.task_id
    return metadata["template_id"] == task.template_id
