from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_sft_experiment(
    base_report_path: Path,
    v1_report_path: Path,
    v2_report_path: Path,
    json_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    """Summarize three frozen-holdout runs and make the RL stage decision."""
    reports = {
        "MLX Base": _load_report(base_report_path),
        "SFT V1": _load_report(v1_report_path),
        "SFT V2": _load_report(v2_report_path),
    }
    task_sets = {
        label: set(_task_success_map(report)) for label, report in reports.items()
    }
    if len({frozenset(task_ids) for task_ids in task_sets.values()}) != 1:
        raise ValueError("三个评测报告包含的 task_id 不一致，不能直接比较")

    run_rows = []
    for label, report in reports.items():
        summary = report["summary"]
        run_rows.append(
            {
                "label": label,
                "success_count": summary["final_success_count"],
                "task_count": summary["task_count"],
                "accuracy": summary["final_accuracy"],
            }
        )

    categories = []
    error_types = sorted(
        set().union(
            *(report["summary"].get("by_error_type", {}) for report in reports.values())
        )
    )
    for error_type in error_types:
        categories.append(
            {
                "error_type": error_type,
                "base_accuracy": _category_accuracy(reports["MLX Base"], error_type),
                "v1_accuracy": _category_accuracy(reports["SFT V1"], error_type),
                "v2_accuracy": _category_accuracy(reports["SFT V2"], error_type),
            }
        )

    base_map = _task_success_map(reports["MLX Base"])
    v1_map = _task_success_map(reports["SFT V1"])
    v2_map = _task_success_map(reports["SFT V2"])
    transitions = {
        "base_to_v1": _transition(base_map, v1_map),
        "v1_to_v2": _transition(v1_map, v2_map),
        "base_to_v2": _transition(base_map, v2_map),
    }

    base_accuracy = reports["MLX Base"]["summary"]["final_accuracy"]
    v1_accuracy = reports["SFT V1"]["summary"]["final_accuracy"]
    v2_accuracy = reports["SFT V2"]["summary"]["final_accuracy"]
    best_sft_label, best_sft_accuracy = max(
        (("SFT V1", v1_accuracy), ("SFT V2", v2_accuracy)),
        key=lambda item: item[1],
    )
    best_sft_map = v1_map if best_sft_label == "SFT V1" else v2_map
    best_transition = _transition(base_map, best_sft_map)

    checks = {
        "best_sft_beats_base": best_sft_accuracy > base_accuracy,
        "best_sft_has_no_regression": not best_transition["regressed_tasks"],
        "second_sft_iteration_is_stable": v2_accuracy >= v1_accuracy,
    }
    start_rl = all(checks.values())
    if start_rl:
        next_stage = "RL_reward_and_data_design"
        reason = (
            "SFT 已在冻结留出集上超过 Base、没有任务回退，且第二轮没有退化；"
            "可以开始设计只针对多轮策略问题的 RL 奖励与数据。"
        )
    else:
        next_stage = "SFT_data_quality_and_generalization"
        reason = (
            "SFT 尚未稳定超过 Base，且存在任务回退或第二轮退化。此时做 RL 会强化"
            "尚未解决的数据分布和监督学习问题，应先改进 SFT 数据质量与泛化。"
        )

    winner = max(run_rows, key=lambda row: row["accuracy"])["label"]
    payload = {
        "reports": {
            "base": str(base_report_path.resolve()),
            "sft_v1": str(v1_report_path.resolve()),
            "sft_v2": str(v2_report_path.resolve()),
        },
        "evaluation_protocol": {
            "split": "frozen_final_holdout",
            "task_count": len(base_map),
            "holdout_used_for_training": False,
        },
        "runs": run_rows,
        "winner": winner,
        "category_comparison": categories,
        "transitions": transitions,
        "stage_gate": {
            "checks": checks,
            "start_rl": start_rl,
            "next_stage": next_stage,
            "reason": reason,
        },
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_output.write_text(_to_markdown(payload), encoding="utf-8")
    return payload


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_success_map(report: dict[str, Any]) -> dict[str, bool]:
    return {
        item["task"]["task_id"]: bool(item["report"]["success"])
        for item in report["reports"]
    }


def _category_accuracy(report: dict[str, Any], error_type: str) -> float:
    return report["summary"]["by_error_type"][error_type]["final_accuracy"]


def _transition(
    before: dict[str, bool], after: dict[str, bool]
) -> dict[str, list[str]]:
    return {
        "fixed_tasks": sorted(
            task_id for task_id in before if not before[task_id] and after[task_id]
        ),
        "regressed_tasks": sorted(
            task_id for task_id in before if before[task_id] and not after[task_id]
        ),
        "remaining_failed_tasks": sorted(
            task_id for task_id in after if not after[task_id]
        ),
    }


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# SFT V1/V2 最终留出集实验总结",
        "",
        "## 评测协议",
        "",
        "35 道最终留出题在 V2 数据构建前冻结，不参与 Base、SFT V1 或 SFT V2 的梯度更新。",
        "三组模型使用相同数据库、Prompt、最大轮数和执行结果匹配指标。",
        "",
        "## 总体结果",
        "",
        "| 模型 | 正确题数 | 准确率 |",
        "| --- | ---: | ---: |",
    ]
    for row in summary["runs"]:
        lines.append(
            f"| {row['label']} | {row['success_count']}/{row['task_count']} | "
            f"{row['accuracy']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"当前最优：**{summary['winner']}**。",
            "",
            "## 分类结果",
            "",
            "| 错误类型 | MLX Base | SFT V1 | SFT V2 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["category_comparison"]:
        lines.append(
            f"| {row['error_type']} | {row['base_accuracy']:.1%} | "
            f"{row['v1_accuracy']:.1%} | {row['v2_accuracy']:.1%} |"
        )

    lines.extend(["", "## 修复与回退", ""])
    transition_labels = {
        "base_to_v1": "Base → V1",
        "v1_to_v2": "V1 → V2",
        "base_to_v2": "Base → V2",
    }
    for key, label in transition_labels.items():
        transition = summary["transitions"][key]
        fixed = ", ".join(f"`{task}`" for task in transition["fixed_tasks"]) or "无"
        regressed = (
            ", ".join(f"`{task}`" for task in transition["regressed_tasks"])
            or "无"
        )
        lines.extend([f"### {label}", "", f"- 修复：{fixed}", f"- 回退：{regressed}", ""])

    gate = summary["stage_gate"]
    checks = gate["checks"]
    lines.extend(
        [
            "## RL 阶段门槛",
            "",
            "| 检查项 | 结果 |",
            "| --- | --- |",
            f"| 最优 SFT 在冻结留出集超过 Base | {'通过' if checks['best_sft_beats_base'] else '未通过'} |",
            f"| 最优 SFT 没有任务回退 | {'通过' if checks['best_sft_has_no_regression'] else '未通过'} |",
            f"| 第二轮 SFT 不低于第一轮 | {'通过' if checks['second_sft_iteration_is_stable'] else '未通过'} |",
            "",
            f"**结论：{'进入 RL' if gate['start_rl'] else '暂不进入 RL'}。** {gate['reason']}",
            "",
            "## Bad Case 结论",
            "",
            "- V1 学会了部分聚合和 JOIN 模式，但过滤条件泛化明显回退。",
            "- V2 内部验证损失很低，冻结留出集却继续下降，说明训练分布与真实评测分布不一致。",
            "- V2 典型错误包括 SQLite 中使用 `TO_DATE`、擅自增加业务过滤条件，以及保留会破坏 LEFT JOIN 的 WHERE 条件。",
            "- 下一轮应增加同问题多种值和 SQL 方言约束，并混入防遗忘的通用样本；不能把这 35 道留出题直接加入训练。",
        ]
    )
    return "\n".join(lines) + "\n"
