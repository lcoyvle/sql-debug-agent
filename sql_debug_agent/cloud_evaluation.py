from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_checkpoint_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_count = len(rows)
    correct_count = sum(bool(row["correct"]) for row in rows)
    totals = Counter(row["error_type"] for row in rows)
    correct = Counter(
        row["error_type"] for row in rows if bool(row["correct"])
    )
    by_error_type = {
        error_type: {
            "task_count": totals[error_type],
            "correct_count": correct[error_type],
            "accuracy": correct[error_type] / totals[error_type],
        }
        for error_type in sorted(totals)
    }
    return {
        "task_count": task_count,
        "correct_count": correct_count,
        "accuracy": correct_count / task_count if task_count else 0.0,
        "by_error_type": by_error_type,
    }


def compare_checkpoint_rows(
    base_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = {row["task_id"]: bool(row["correct"]) for row in base_rows}
    candidate = {row["task_id"]: bool(row["correct"]) for row in candidate_rows}
    if set(base) != set(candidate):
        raise ValueError("模型评测的 task_id 不一致，不能直接比较")
    fixed = sorted(
        task_id for task_id in base if not base[task_id] and candidate[task_id]
    )
    regressed = sorted(
        task_id for task_id in base if base[task_id] and not candidate[task_id]
    )
    base_accuracy = sum(base.values()) / len(base) if base else 0.0
    candidate_accuracy = sum(candidate.values()) / len(candidate) if candidate else 0.0
    return {
        "base_accuracy": base_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": candidate_accuracy - base_accuracy,
        "fixed_tasks": fixed,
        "regressed_tasks": regressed,
        "net_fixed_count": len(fixed) - len(regressed),
    }


def build_final_comparison(
    results: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    required = {"base", "sft", "grpo"}
    if set(results) != required:
        raise ValueError("最终评测必须同时包含 base、sft、grpo")
    summaries = {
        name: summarize_checkpoint_rows(rows) for name, rows in results.items()
    }
    base_to_sft = compare_checkpoint_rows(results["base"], results["sft"])
    sft_to_grpo = compare_checkpoint_rows(results["sft"], results["grpo"])
    grpo_delta = sft_to_grpo["accuracy_delta"]
    regressions = sft_to_grpo["regressed_tasks"]
    if grpo_delta > 0 and not regressions:
        verdict = "accept_grpo"
        reason = "GRPO 在冻结测试集上超过 SFT，且没有任务回退。"
    elif grpo_delta > 0:
        verdict = "grpo_promising_with_regressions"
        reason = "GRPO 总体准确率提升，但存在任务回退，需要分析 Bad Case。"
    elif grpo_delta == 0:
        verdict = "no_measurable_grpo_gain"
        reason = "GRPO 没有在冻结测试集上带来可测量的净提升。"
    else:
        verdict = "reject_grpo_regression"
        reason = "GRPO 在冻结测试集上低于 SFT，应保留 SFT 作为当前最佳版本。"
    return {
        "protocol": protocol,
        "summaries": summaries,
        "comparisons": {
            "base_to_sft": base_to_sft,
            "sft_to_grpo": sft_to_grpo,
        },
        "decision": {"verdict": verdict, "reason": reason},
        "results": results,
    }


def write_final_comparison(
    comparison: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_comparison.json"
    markdown_path = output_dir / "final_comparison.md"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(_to_markdown(comparison), encoding="utf-8")
    return json_path, markdown_path


def _to_markdown(comparison: dict[str, Any]) -> str:
    summaries = comparison["summaries"]
    change = comparison["comparisons"]["sft_to_grpo"]
    lines = [
        "# SQL Debug Agent 最终冻结评测",
        "",
        "## 总体结果",
        "",
        "| 模型 | 正确数 | 正确率 |",
        "| --- | ---: | ---: |",
    ]
    for name, label in (("base", "Base"), ("sft", "SFT"), ("grpo", "SFT + GRPO")):
        row = summaries[name]
        lines.append(
            f"| {label} | {row['correct_count']}/{row['task_count']} | "
            f"{row['accuracy']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## SFT → GRPO 变化",
            "",
            "- 修复任务：" + (", ".join(change["fixed_tasks"]) or "无"),
            "- 回退任务：" + (", ".join(change["regressed_tasks"]) or "无"),
            f"- 净变化：{change['accuracy_delta']:+.1%}",
            "",
            "## 结论",
            "",
            comparison["decision"]["reason"],
            "",
            "## 分类准确率",
            "",
            "| 错误类型 | Base | SFT | SFT + GRPO |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    error_types = sorted(
        set(summaries["base"]["by_error_type"])
        | set(summaries["sft"]["by_error_type"])
        | set(summaries["grpo"]["by_error_type"])
    )
    for error_type in error_types:
        values = [
            summaries[name]["by_error_type"][error_type]["accuracy"]
            for name in ("base", "sft", "grpo")
        ]
        lines.append(
            f"| {error_type} | {values[0]:.1%} | {values[1]:.1%} | {values[2]:.1%} |"
        )
    return "\n".join(lines) + "\n"
