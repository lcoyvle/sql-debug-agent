from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_evaluation_reports(
    base_report_path: Path,
    candidate_report_path: Path,
    json_output: Path,
    markdown_output: Path,
    base_label: str = "Base",
    candidate_label: str = "Candidate",
) -> dict[str, Any]:
    base = json.loads(base_report_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    base_tasks = _task_success_map(base)
    candidate_tasks = _task_success_map(candidate)
    if set(base_tasks) != set(candidate_tasks):
        raise ValueError("两个评测报告包含的 task_id 不一致，不能直接比较")

    fixed = sorted(
        task_id
        for task_id in base_tasks
        if not base_tasks[task_id] and candidate_tasks[task_id]
    )
    regressed = sorted(
        task_id
        for task_id in base_tasks
        if base_tasks[task_id] and not candidate_tasks[task_id]
    )
    remaining = sorted(
        task_id
        for task_id in base_tasks
        if not candidate_tasks[task_id]
    )

    base_by_type = base["summary"].get("by_error_type", {})
    candidate_by_type = candidate["summary"].get("by_error_type", {})
    category_comparison = []
    for error_type in sorted(set(base_by_type) | set(candidate_by_type)):
        base_metrics = base_by_type[error_type]
        candidate_metrics = candidate_by_type[error_type]
        category_comparison.append(
            {
                "error_type": error_type,
                "base_accuracy": base_metrics["final_accuracy"],
                "candidate_accuracy": candidate_metrics["final_accuracy"],
                "delta": (
                    candidate_metrics["final_accuracy"]
                    - base_metrics["final_accuracy"]
                ),
            }
        )

    base_accuracy = base["summary"]["final_accuracy"]
    candidate_accuracy = candidate["summary"]["final_accuracy"]
    delta = candidate_accuracy - base_accuracy
    if delta <= 0:
        reason = (
            f"{candidate_label} 没有在同一评测集上超过 {base_label}；"
            "应先分析数据分布和能力回退，暂不进入 RL。"
        )
    elif regressed:
        reason = (
            f"{candidate_label} 虽有净提升，但仍有 {len(regressed)} 个回退任务；"
            "应先用监督数据消除不稳定性，暂不进入 RL。"
        )
    else:
        reason = (
            f"{candidate_label} 已超过 {base_label} 且没有任务回退；"
            "仍需结合多轮策略错误和后续迭代是否稳定，再决定是否进入 RL。"
        )
    comparison = {
        "base_label": base_label,
        "candidate_label": candidate_label,
        "base_report": str(base_report_path.resolve()),
        "candidate_report": str(candidate_report_path.resolve()),
        "task_count": base["summary"]["task_count"],
        "base_accuracy": base_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": delta,
        "fixed_tasks": fixed,
        "regressed_tasks": regressed,
        "remaining_failed_tasks": remaining,
        "category_comparison": category_comparison,
        "decision": {
            "next_stage": "SFT_data_quality_and_generalization",
            "start_rl": False,
            "reason": reason,
        },
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_output.write_text(_to_markdown(comparison), encoding="utf-8")
    return comparison


def _task_success_map(report: dict[str, Any]) -> dict[str, bool]:
    return {
        item["task"]["task_id"]: bool(item["report"]["success"])
        for item in report["reports"]
    }


def _to_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        f"# {comparison['base_label']} 与 {comparison['candidate_label']} 对比",
        "",
        "## 总体结果",
        "",
        "| 模型 | 正确率 |",
        "| --- | ---: |",
        f"| {comparison['base_label']} | {comparison['base_accuracy']:.1%} |",
        f"| {comparison['candidate_label']} | {comparison['candidate_accuracy']:.1%} |",
        f"| 变化 | {comparison['accuracy_delta']:+.1%} |",
        "",
        "## 分类变化",
        "",
        f"| 错误类型 | {comparison['base_label']} | {comparison['candidate_label']} | 变化 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparison["category_comparison"]:
        lines.append(
            f"| {row['error_type']} | {row['base_accuracy']:.1%} | "
            f"{row['candidate_accuracy']:.1%} | {row['delta']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "## 任务变化",
            "",
            f"- 被 {comparison['candidate_label']} 修复："
            + (", ".join(f"`{task}`" for task in comparison["fixed_tasks"]) or "无"),
            f"- {comparison['candidate_label']} 后回退："
            + (
                ", ".join(f"`{task}`" for task in comparison["regressed_tasks"])
                or "无"
            ),
            "- 当前仍失败："
            + (
                ", ".join(
                    f"`{task}`" for task in comparison["remaining_failed_tasks"]
                )
                or "无"
            ),
            "",
            "## 阶段决策",
            "",
            f"暂不进入 RL：{comparison['decision']['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"
