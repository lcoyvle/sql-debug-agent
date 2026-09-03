from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def analyze_badcases(
    report_path: Path,
    json_output: Path,
    markdown_output: Path,
    examples_per_type: int = 3,
) -> dict[str, Any]:
    """Turn one baseline report into machine- and human-readable failure analysis."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    failed_reports = [
        item for item in payload.get("reports", []) if not item["report"]["success"]
    ]

    behaviors: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in failed_reports:
        task = item["task"]
        report = item["report"]
        final_sql = report.get("final_sql", task["initial_sql"])
        steps = report.get("steps", [])
        feedback = steps[-1].get("feedback", "") if steps else ""

        if _normalize_sql(final_sql) == _normalize_sql(task["initial_sql"]):
            behaviors["unchanged_sql"] += 1
        else:
            behaviors["changed_but_still_wrong"] += 1
        if "执行错误" in feedback or "安全检查失败" in feedback:
            behaviors["execution_error"] += 1
        else:
            behaviors["wrong_result"] += 1

        error_type = task["error_type"]
        if len(examples[error_type]) < examples_per_type:
            examples[error_type].append(
                {
                    "task_id": task["task_id"],
                    "question": task["question"],
                    "initial_sql": task["initial_sql"],
                    "final_sql": final_sql,
                    "reference_sql": task["reference_sql"],
                    "feedback": feedback,
                }
            )

    priorities: list[dict[str, Any]] = []
    for error_type, metrics in summary.get("by_error_type", {}).items():
        failure_count = metrics["task_count"] - metrics["final_success_count"]
        priorities.append(
            {
                "error_type": error_type,
                "task_count": metrics["task_count"],
                "success_count": metrics["final_success_count"],
                "failure_count": failure_count,
                "accuracy": metrics["final_accuracy"],
            }
        )
    priorities.sort(key=lambda row: (-row["failure_count"], row["accuracy"], row["error_type"]))

    analysis: dict[str, Any] = {
        "source_report": str(report_path.resolve()),
        "run_config": payload.get("run_config", {}),
        "task_count": summary["task_count"],
        "success_count": summary["final_success_count"],
        "failure_count": len(failed_reports),
        "repair_accuracy": summary["final_accuracy"],
        "failure_behaviors": dict(sorted(behaviors.items())),
        "priorities": priorities,
        "failure_examples": dict(sorted(examples.items())),
        "decision": {
            "next_stage": "SFT",
            "reason": (
                "失败以基础纠错模式未学会或未改写 SQL 为主，适合先用监督数据学习稳定的"
                "问题、Schema、执行反馈到正确 SQL 的映射；暂不直接进入 RL。"
            ),
            "rl_entry_condition": (
                "SFT 后独立测试集准确率进入平台期，且剩余失败主要是可执行但语义错误、"
                "需要多步工具反馈或候选答案排序时，再构建 RL 数据。"
            ),
        },
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_output.write_text(_to_markdown(analysis), encoding="utf-8")
    return analysis


def _to_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Bad Case 分析",
        "",
        "## 总体结果",
        "",
        f"- 任务数：{analysis['task_count']}",
        f"- 修复成功：{analysis['success_count']}",
        f"- 修复失败：{analysis['failure_count']}",
        f"- 修复准确率：{analysis['repair_accuracy']:.1%}",
        "",
        "## 优先改进方向",
        "",
        "| 优先级 | 错误类型 | 成功/总数 | 失败数 | 准确率 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(analysis["priorities"], start=1):
        lines.append(
            f"| {rank} | {row['error_type']} | {row['success_count']}/{row['task_count']} "
            f"| {row['failure_count']} | {row['accuracy']:.1%} |"
        )

    behavior_labels = {
        "unchanged_sql": "模型未改写错误 SQL",
        "changed_but_still_wrong": "已改写但仍错误",
        "execution_error": "最终仍有执行错误",
        "wrong_result": "可以执行但结果错误",
    }
    lines.extend(["", "## 失败行为", ""])
    for behavior, count in analysis["failure_behaviors"].items():
        lines.append(f"- {behavior_labels.get(behavior, behavior)}：{count}")

    lines.extend(
        [
            "",
            "## 阶段决策",
            "",
            f"下一阶段：**{analysis['decision']['next_stage']}**。",
            "",
            analysis["decision"]["reason"],
            "",
            f"进入 RL 的条件：{analysis['decision']['rl_entry_condition']}",
            "",
            "## 失败样例",
        ]
    )
    for error_type, examples in analysis["failure_examples"].items():
        lines.extend(["", f"### {error_type}", ""])
        for example in examples:
            lines.extend(
                [
                    f"- `{example['task_id']}`：{example['question']}",
                    f"  - 反馈：{example['feedback']}",
                    f"  - 最终 SQL：`{example['final_sql']}`",
                    f"  - 参考 SQL：`{example['reference_sql']}`",
                ]
            )
    return "\n".join(lines) + "\n"
