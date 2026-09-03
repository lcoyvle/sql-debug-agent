from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .database import get_schema
from .dataset import DebugTask, load_tasks
from .preparation import SYSTEM_PROMPT, build_sft_user_content
from .rl_reward import RobustSQLReward
from .verifier import SQLVerifier


def build_rl_dataset(
    development_tasks_path: Path,
    v3_tasks_path: Path,
    demo_database_path: Path,
    training_database_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build offline preference pairs and GRPO prompts without final holdouts."""
    sources = [
        ("development", load_tasks(development_tasks_path), demo_database_path),
        ("v3_increment", load_tasks(v3_tasks_path), training_database_path),
    ]
    reward = RobustSQLReward([demo_database_path, training_database_path])
    schema = get_schema(demo_database_path)
    prompts: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    skipped_pairs = 0

    for source_name, tasks, source_database in sources:
        source_verifier = SQLVerifier(source_database)
        for task in tasks:
            feedback = source_verifier.verify(task.initial_sql, task.reference_sql).feedback
            user_content = build_sft_user_content(
                task.question, schema, task.initial_sql, feedback
            )
            prompts.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "reference_sql": task.reference_sql,
                    "previous_sql": task.initial_sql,
                    "metadata": {
                        "task_id": task.task_id,
                        "error_type": task.error_type,
                        "source": source_name,
                    },
                }
            )
            chosen_reward = reward.score(
                task.reference_sql,
                task.reference_sql,
                previous_sql=task.initial_sql,
            )
            negatives = {
                "unchanged": task.initial_sql,
                "empty_result_shortcut": _empty_result_shortcut(task.reference_sql),
                "unsafe_write": "DELETE FROM transactions",
            }
            for negative_type, rejected_sql in negatives.items():
                rejected_reward = reward.score(
                    rejected_sql,
                    task.reference_sql,
                    previous_sql=task.initial_sql,
                    final_turn=True,
                )
                if chosen_reward.total <= rejected_reward.total:
                    skipped_pairs += 1
                    continue
                pairs.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "chosen_sql": task.reference_sql,
                        "rejected_sql": rejected_sql,
                        "chosen_reward": chosen_reward.to_dict(),
                        "rejected_reward": rejected_reward.to_dict(),
                        "metadata": {
                            "task_id": task.task_id,
                            "error_type": task.error_type,
                            "source": source_name,
                            "negative_type": negative_type,
                        },
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "grpo_prompts.jsonl"
    pair_path = output_dir / "preference_pairs.jsonl"
    _write_jsonl(prompt_path, prompts)
    _write_jsonl(pair_path, pairs)
    margins = [
        pair["chosen_reward"]["total"] - pair["rejected_reward"]["total"]
        for pair in pairs
    ]
    manifest = {
        "purpose": "SFT V3 后的离线奖励审计、偏好优化和未来 GRPO prompts",
        "training_ready": False,
        "reason_not_training": "SFT V3 仍有回退；先验证奖励和数据，不启动 GRPO",
        "source_task_count": len(prompts),
        "preference_pair_count": len(pairs),
        "skipped_non_preferred_pairs": skipped_pairs,
        "source_distribution": dict(sorted(Counter(p["metadata"]["source"] for p in prompts).items())),
        "error_distribution": dict(sorted(Counter(p["metadata"]["error_type"] for p in prompts).items())),
        "negative_distribution": dict(sorted(Counter(p["metadata"]["negative_type"] for p in pairs).items())),
        "reward_margin": {
            "minimum": min(margins) if margins else None,
            "maximum": max(margins) if margins else None,
            "average": sum(margins) / len(margins) if margins else None,
        },
        "reward_databases": [str(demo_database_path.resolve()), str(training_database_path.resolve())],
        "excluded_final_holdouts": ["data/final_holdout.jsonl", "data/final_holdout_v3.jsonl"],
        "files": {
            "grpo_prompts": str(prompt_path.resolve()),
            "preference_pairs": str(pair_path.resolve()),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path = output_dir / "reward_audit.md"
    audit_path.write_text(_audit_markdown(manifest), encoding="utf-8")
    manifest["files"]["manifest"] = str(manifest_path.resolve())
    manifest["files"]["reward_audit"] = str(audit_path.resolve())
    return manifest


def _empty_result_shortcut(reference_sql: str) -> str:
    inner = reference_sql.strip().rstrip(";")
    return f"SELECT * FROM ({inner}) AS expected_shape WHERE 1=0"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _audit_markdown(manifest: dict[str, Any]) -> str:
    margin = manifest["reward_margin"]
    return "\n".join(
        [
            "# RL V1 数据与奖励审计",
            "",
            "## 数据范围",
            "",
            f"- Prompt 数：{manifest['source_task_count']}",
            f"- 偏好对数：{manifest['preference_pair_count']}",
            "- 数据来源：30 题开发集与 60 条 V3 增量数据。",
            "- 两套最终留出集均未进入 RL 数据。",
            "",
            "## 奖励设计",
            "",
            "| 行为 | 奖励 |",
            "| --- | ---: |",
            "| 两套数据库均安全可执行 | +0.2 |",
            "| 两套数据库结果均与参考一致 | +1.0 |",
            "| 从错误 SQL 成功修复 | +0.3 |",
            "| 重复提交相同 SQL | -0.2 |",
            "| 最后一轮仍失败 | -0.2 |",
            "| 写操作或越权 SQL | -1.0 |",
            "",
            "## 防奖励作弊",
            "",
            "- 空结果捷径必须在两套数据库都与参考结果一致才可能得答案奖励。",
            "- 删列或改变列顺序会改变结果元组，不能得答案奖励。",
            "- 只在单个小数据库上碰巧正确的过滤条件，在第二套数据库上会被识别。",
            "- SQL 列别名不同但结果值一致不会被误罚。",
            "- 危险 SQL 在执行前直接得到 -1.0。",
            "",
            "## 偏好间隔",
            "",
            f"- 最小：{margin['minimum']:.2f}",
            f"- 平均：{margin['average']:.2f}",
            f"- 最大：{margin['maximum']:.2f}",
            "",
            "## 阶段结论",
            "",
            "数据和奖励已可用于离线审计，但 `training_ready=false`。SFT V3 仍有回退，",
            "现阶段不启动 GRPO；先用这批偏好对验证奖励排序和错误覆盖。",
        ]
    ) + "\n"
