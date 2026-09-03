from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .mlx_repair import MLXGeneratorClient
from .rl_reward import RobustSQLReward


def replay_model_rollouts(
    prompts_path: Path,
    demo_database_path: Path,
    training_database_path: Path,
    adapter_path: Path,
    output_dir: Path,
    model_name: str,
    cache_dir: Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    prompts = _read_jsonl(prompts_path)
    reward = RobustSQLReward([demo_database_path, training_database_path])
    client = MLXGeneratorClient(
        model_name=model_name,
        adapter_path=adapter_path,
        cache_dir=cache_dir,
    )
    rollouts: list[dict[str, Any]] = []
    for index, item in enumerate(prompts, start=1):
        candidate_sql = client.create_sql_repair(item["messages"])
        candidate_reward = reward.score(
            candidate_sql,
            item["reference_sql"],
            previous_sql=item["previous_sql"],
            final_turn=True,
        )
        baseline_reward = reward.score(
            item["previous_sql"],
            item["reference_sql"],
            previous_sql=item["previous_sql"],
            final_turn=True,
        )
        rollouts.append(
            {
                "metadata": item["metadata"],
                "candidate_sql": candidate_sql,
                "reference_sql": item["reference_sql"],
                "previous_sql": item["previous_sql"],
                "candidate_reward": candidate_reward.to_dict(),
                "baseline_reward": baseline_reward.to_dict(),
                "reward_delta": round(candidate_reward.total - baseline_reward.total, 6),
            }
        )
        if progress is not None and (index % 10 == 0 or index == len(prompts)):
            progress(index, len(prompts))

    summary = summarize_rollouts(rollouts)
    summary.update(
        {
            "model": model_name,
            "adapter_path": str(adapter_path.resolve()),
            "prompts_path": str(prompts_path.resolve()),
            "reward_databases": [
                str(demo_database_path.resolve()),
                str(training_database_path.resolve()),
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = output_dir / "rollouts.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_jsonl(rollouts_path, rollouts)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(_to_markdown(summary, rollouts), encoding="utf-8")
    summary["files"] = {
        "rollouts": str(rollouts_path.resolve()),
        "summary": str(summary_path.resolve()),
        "report": str(report_path.resolve()),
    }
    return summary


def summarize_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rollouts)
    rewards = [float(item["candidate_reward"]["total"]) for item in rollouts]
    safe_count = sum(bool(item["candidate_reward"]["safe"]) for item in rollouts)
    executable_count = sum(
        bool(item["candidate_reward"]["executable_on_all_databases"])
        for item in rollouts
    )
    correct_count = sum(
        bool(item["candidate_reward"]["matches_all_databases"])
        for item in rollouts
    )
    repeated_count = sum(
        float(item["candidate_reward"]["repeat_penalty"]) < 0 for item in rollouts
    )
    movement = Counter(
        "improved" if item["reward_delta"] > 0 else "regressed" if item["reward_delta"] < 0 else "tied"
        for item in rollouts
    )
    database_disagreement_count = sum(
        len(set(item["candidate_reward"]["database_matches"])) > 1
        for item in rollouts
    )
    by_type_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rollouts:
        by_type_raw[item["metadata"]["error_type"]].append(item)
    by_error_type = {}
    for error_type, items in sorted(by_type_raw.items()):
        successes = sum(
            bool(item["candidate_reward"]["matches_all_databases"])
            for item in items
        )
        by_error_type[error_type] = {
            "count": len(items),
            "success_count": successes,
            "accuracy": successes / len(items),
            "average_reward": sum(float(item["candidate_reward"]["total"]) for item in items) / len(items),
        }
    unique_rewards = sorted(set(rewards))
    mean = sum(rewards) / count if count else 0.0
    variance = sum((value - mean) ** 2 for value in rewards) / count if count else 0.0
    reward_checks = {
        "all_outputs_safe": safe_count == count,
        "reward_is_non_degenerate": len(unique_rewards) >= 2,
        "counterfactual_database_check_triggered": database_disagreement_count > 0,
        "preference_signal_present": movement["improved"] > 0 and movement["regressed"] + movement["tied"] > 0,
    }
    # SFT V3 still has a known holdout regression. Reward readiness is separate
    # from permission to start training, which remains blocked by the SFT gate.
    reward_ready = all(reward_checks.values())
    return {
        "rollout_count": count,
        "safe_count": safe_count,
        "safety_rate": safe_count / count if count else 0.0,
        "executable_on_all_count": executable_count,
        "executable_on_all_rate": executable_count / count if count else 0.0,
        "correct_on_all_count": correct_count,
        "correct_on_all_rate": correct_count / count if count else 0.0,
        "repeated_sql_count": repeated_count,
        "repeated_sql_rate": repeated_count / count if count else 0.0,
        "database_disagreement_count": database_disagreement_count,
        "movement_vs_initial": dict(sorted(movement.items())),
        "reward_distribution": {
            "minimum": min(rewards) if rewards else None,
            "maximum": max(rewards) if rewards else None,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "unique_values": unique_rewards,
        },
        "by_error_type": by_error_type,
        "reward_gate": {
            "checks": reward_checks,
            "reward_ready": reward_ready,
            "training_ready": False,
            "reason": (
                "奖励可用于真实 rollout 排序，但 SFT V3 在冻结留出集仍有回退；"
                "在回退消除前不启动 GRPO。"
                if reward_ready
                else "真实 rollout 未通过奖励有效性检查，必须先修正奖励。"
            ),
        },
    }


def export_hard_preferences(
    prompts_path: Path, rollouts_path: Path, output_path: Path
) -> dict[str, Any]:
    prompts = _read_jsonl(prompts_path)
    rollouts = _read_jsonl(rollouts_path)
    prompt_by_task = {
        item["metadata"]["task_id"]: item for item in prompts
    }
    pairs = []
    for rollout in rollouts:
        if rollout["candidate_reward"]["matches_all_databases"]:
            continue
        task_id = rollout["metadata"]["task_id"]
        prompt = prompt_by_task[task_id]
        pairs.append(
            {
                "prompt": prompt["messages"],
                "chosen_sql": rollout["reference_sql"],
                "rejected_sql": rollout["candidate_sql"],
                "chosen_reward": 1.5,
                "rejected_reward": rollout["candidate_reward"]["total"],
                "reward_margin": round(
                    1.5 - float(rollout["candidate_reward"]["total"]), 6
                ),
                "metadata": {
                    **rollout["metadata"],
                    "negative_type": "real_model_failure",
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, pairs)
    distribution = Counter(pair["metadata"]["error_type"] for pair in pairs)
    return {
        "pair_count": len(pairs),
        "error_distribution": dict(sorted(distribution.items())),
        "minimum_margin": min((pair["reward_margin"] for pair in pairs), default=None),
        "output": str(output_path.resolve()),
    }


def compare_replay_runs(
    base_summary_path: Path,
    candidate_summary_path: Path,
    base_rollouts_path: Path,
    candidate_rollouts_path: Path,
    json_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    base = json.loads(base_summary_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    base_rollouts = _read_jsonl(base_rollouts_path)
    candidate_rollouts = _read_jsonl(candidate_rollouts_path)
    base_success = {
        item["metadata"]["task_id"]: bool(item["candidate_reward"]["matches_all_databases"])
        for item in base_rollouts
    }
    candidate_success = {
        item["metadata"]["task_id"]: bool(item["candidate_reward"]["matches_all_databases"])
        for item in candidate_rollouts
    }
    if set(base_success) != set(candidate_success):
        raise ValueError("两次回放的 task_id 不一致")
    fixed = sorted(task for task in base_success if not base_success[task] and candidate_success[task])
    regressed = sorted(task for task in base_success if base_success[task] and not candidate_success[task])
    correct_delta = candidate["correct_on_all_count"] - base["correct_on_all_count"]
    repeat_delta = candidate["repeated_sql_count"] - base["repeated_sql_count"]
    accept = correct_delta > 0 and not regressed and repeat_delta <= 0
    payload = {
        "base_correct": base["correct_on_all_count"],
        "candidate_correct": candidate["correct_on_all_count"],
        "correct_delta": correct_delta,
        "base_repeated": base["repeated_sql_count"],
        "candidate_repeated": candidate["repeated_sql_count"],
        "repeat_delta": repeat_delta,
        "base_mean_reward": base["reward_distribution"]["mean"],
        "candidate_mean_reward": candidate["reward_distribution"]["mean"],
        "fixed_tasks": fixed,
        "regressed_tasks": regressed,
        "decision": {
            "accept_candidate": accept,
            "selected_adapter": "preference_v1_10" if accept else "sft_v3_30",
            "reason": (
                "候选提高双库正确数、没有任务回退且未增加重复提交。"
                if accept
                else "候选没有带来任务级净提升，拒绝替换当前 V3-30 Adapter。"
            ),
        },
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_output.write_text(_comparison_markdown(payload), encoding="utf-8")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _to_markdown(summary: dict[str, Any], rollouts: list[dict[str, Any]]) -> str:
    distribution = summary["reward_distribution"]
    lines = [
        "# SFT V3 真实 Rollout 奖励回放",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| Rollout 数 | {summary['rollout_count']} |",
        f"| 安全率 | {summary['safety_rate']:.1%} |",
        f"| 两库均可执行 | {summary['executable_on_all_rate']:.1%} |",
        f"| 两库结果均正确 | {summary['correct_on_all_rate']:.1%} |",
        f"| 重复原 SQL | {summary['repeated_sql_rate']:.1%} |",
        f"| 两库判定不一致 | {summary['database_disagreement_count']} |",
        "",
        "## 相对原错误 SQL",
        "",
    ]
    for key, label in (("improved", "改善"), ("tied", "持平"), ("regressed", "变差")):
        lines.append(f"- {label}：{summary['movement_vs_initial'].get(key, 0)}")
    lines.extend(
        [
            "",
            "## 奖励分布",
            "",
            f"- 最小：{distribution['minimum']:.2f}",
            f"- 平均：{distribution['mean']:.2f}",
            f"- 最大：{distribution['maximum']:.2f}",
            f"- 标准差：{distribution['standard_deviation']:.2f}",
            "",
            "## 分类结果",
            "",
            "| 错误类型 | 正确/总数 | 正确率 | 平均奖励 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for error_type, row in summary["by_error_type"].items():
        lines.append(
            f"| {error_type} | {row['success_count']}/{row['count']} | "
            f"{row['accuracy']:.1%} | {row['average_reward']:.2f} |"
        )
    failures = [
        item
        for item in rollouts
        if not item["candidate_reward"]["matches_all_databases"]
    ][:10]
    lines.extend(["", "## 失败样例（最多 10 条）", ""])
    for item in failures:
        lines.extend(
            [
                f"- `{item['metadata']['task_id']}`（{item['metadata']['error_type']}），奖励 {item['candidate_reward']['total']:.2f}",
                f"  - 模型 SQL：`{item['candidate_sql']}`",
                f"  - 参考 SQL：`{item['reference_sql']}`",
            ]
        )
    gate = summary["reward_gate"]
    lines.extend(
        [
            "",
            "## 阶段决策",
            "",
            f"奖励层：**{'通过' if gate['reward_ready'] else '未通过'}**。",
            f"GRPO 训练：**{'允许' if gate['training_ready'] else '暂不允许'}**。",
            "",
            gate["reason"],
        ]
    )
    return "\n".join(lines) + "\n"


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Corrective Replay 候选对比",
            "",
            "| 指标 | V3-30 | Corrective Replay-10 | 变化 |",
            "| --- | ---: | ---: | ---: |",
            f"| 双库正确数 | {comparison['base_correct']}/90 | {comparison['candidate_correct']}/90 | {comparison['correct_delta']:+d} |",
            f"| 重复原 SQL | {comparison['base_repeated']} | {comparison['candidate_repeated']} | {comparison['repeat_delta']:+d} |",
            f"| 平均奖励 | {comparison['base_mean_reward']:.2f} | {comparison['candidate_mean_reward']:.2f} | {comparison['candidate_mean_reward'] - comparison['base_mean_reward']:+.2f} |",
            "",
            f"- 新修复任务：{', '.join(comparison['fixed_tasks']) or '无'}",
            f"- 回退任务：{', '.join(comparison['regressed_tasks']) or '无'}",
            "",
            f"**是否接受候选：{'是' if comparison['decision']['accept_candidate'] else '否'}。**",
            "",
            comparison["decision"]["reason"],
            f"当前选择：`{comparison['decision']['selected_adapter']}`。",
        ]
    ) + "\n"
