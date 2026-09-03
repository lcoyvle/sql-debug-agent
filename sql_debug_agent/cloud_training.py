from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .mlx_repair import extract_sql
from .rl_reward import RobustSQLReward


def prepare_cloud_training_data(
    sft_train_path: Path,
    sft_eval_path: Path,
    grpo_prompts_path: Path,
    demo_database_path: Path,
    training_database_path: Path,
    output_dir: Path,
    hard_preferences_path: Path | None = None,
) -> dict[str, Any]:
    """Export only train/dev inputs needed by TRL; final holdouts stay excluded."""
    sft_train = [_to_prompt_completion(item) for item in _read_jsonl(sft_train_path)]
    sft_eval = [_to_prompt_completion(item) for item in _read_jsonl(sft_eval_path)]
    grpo_records = [_to_grpo_record(item) for item in _read_jsonl(grpo_prompts_path)]

    _validate_unique_task_ids(grpo_records)
    hard_grpo_records = _select_hard_grpo_records(
        grpo_records, hard_preferences_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_train_output = output_dir / "sft_train.jsonl"
    sft_eval_output = output_dir / "sft_eval.jsonl"
    grpo_output = output_dir / "grpo_train.jsonl"
    hard_grpo_output = output_dir / "grpo_hard_train.jsonl"
    demo_output = output_dir / "finance_demo.db"
    training_output = output_dir / "train_finance.db"
    _write_jsonl(sft_train_output, sft_train)
    _write_jsonl(sft_eval_output, sft_eval)
    _write_jsonl(grpo_output, grpo_records)
    _write_jsonl(hard_grpo_output, hard_grpo_records)
    shutil.copy2(demo_database_path, demo_output)
    shutil.copy2(training_database_path, training_output)

    manifest = {
        "format": "TRL conversational prompt-completion + GRPO prompt JSONL",
        "sft_train_count": len(sft_train),
        "sft_eval_count": len(sft_eval),
        "grpo_prompt_count": len(grpo_records),
        "grpo_hard_prompt_count": len(hard_grpo_records),
        "grpo_hard_source": "V3-30 真实 rollout 失败案例",
        "grpo_hard_error_distribution": dict(
            sorted(Counter(item["error_type"] for item in hard_grpo_records).items())
        ),
        "grpo_hard_source_distribution": dict(
            sorted(Counter(item["source"] for item in hard_grpo_records).items())
        ),
        "grpo_error_distribution": dict(
            sorted(Counter(item["error_type"] for item in grpo_records).items())
        ),
        "reward_database_count": 2,
        "final_holdouts_included": False,
        "excluded_files": ["data/final_holdout.jsonl", "data/final_holdout_v3.jsonl"],
        "mlx_adapter_included": False,
        "mlx_adapter_note": (
            "MLX Adapter 与 PyTorch/TRL Adapter 格式不同；云端先重建标准 PEFT SFT Adapter。"
        ),
        "files": {
            "sft_train": sft_train_output.name,
            "sft_eval": sft_eval_output.name,
            "grpo_train": grpo_output.name,
            "grpo_hard_train": hard_grpo_output.name,
            "demo_database": demo_output.name,
            "training_database": training_output.name,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def summarize_grpo_signal(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether sampled completion groups produced usable advantages."""
    reward_logs = [entry for entry in log_history if "reward_std" in entry]
    reward_stds = [_as_float(entry.get("reward_std")) for entry in reward_logs]
    rewards = [_as_float(entry.get("reward")) for entry in reward_logs]
    zero_std = [
        _as_float(entry.get("frac_reward_zero_std")) for entry in reward_logs
    ]
    varied_steps = sum(value > 1e-8 for value in reward_stds)
    logged_steps = len(reward_logs)
    return {
        "logged_steps": logged_steps,
        "steps_with_reward_variance": varied_steps,
        "reward_variance_step_rate": varied_steps / logged_steps if logged_steps else 0.0,
        "mean_reward": _mean(rewards),
        "mean_reward_std": _mean(reward_stds),
        "mean_frac_reward_zero_std": _mean(zero_std),
        "effective_learning_signal": varied_steps > 0,
        "interpretation": (
            "至少一个采样组的候选 SQL 奖励不同，GRPO 产生了非零优势信号。"
            if varied_steps > 0
            else "所有采样组的奖励都相同，本次 GRPO 没有产生有效学习信号。"
        ),
    }


def build_grpo_v2_patch_archive(project_root: Path, archive_path: Path) -> Path:
    """Create a tiny Colab patch that preserves an existing SFT adapter."""
    sources = [
        project_root / "cloud_grpo" / "train_grpo.py",
        project_root / "cloud_grpo" / "data" / "grpo_hard_train.jsonl",
        project_root / "sql_debug_agent" / "cloud_training.py",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"GRPO V2 补丁缺少文件：{', '.join(missing)}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(project_root))
    return archive_path


def make_sql_execution_reward(
    database_paths: list[Path],
) -> Callable[..., list[float]]:
    """Build a TRL reward callback backed by the project's dual-DB verifier."""
    scorer = RobustSQLReward(database_paths)

    def sql_execution_reward(
        completions: list[Any],
        reference_sql: list[str],
        previous_sql: list[str],
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion, reference, previous in zip(
            completions, reference_sql, previous_sql, strict=True
        ):
            candidate = extract_completion_sql(completion)
            rewards.append(
                scorer.score(
                    candidate,
                    reference,
                    previous_sql=previous,
                    final_turn=True,
                ).total
            )
        return rewards

    return sql_execution_reward


def extract_completion_sql(completion: Any) -> str:
    """Accept both standard and conversational completion shapes used by TRL."""
    if isinstance(completion, str):
        text = completion
    elif isinstance(completion, list) and completion:
        last = completion[-1]
        text = last.get("content", "") if isinstance(last, dict) else str(last)
    elif isinstance(completion, dict):
        text = completion.get("content", "")
    else:
        text = ""
    return extract_sql(str(text))


def build_colab_archive(project_root: Path, archive_path: Path) -> Path:
    """Create a small upload archive without model caches, adapters, or holdouts."""
    required = [
        project_root / "pyproject.toml",
        project_root / "cloud_grpo",
        project_root / "sql_debug_agent",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"云端包缺少文件：{', '.join(missing)}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = Path("sql-debug-agent-colab")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in required:
            if source.is_file():
                archive.write(source, prefix / source.name)
                continue
            for path in sorted(source.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                relative = path.relative_to(project_root)
                if _is_post_training_evaluation_file(relative):
                    continue
                archive.write(path, prefix / relative)
    return archive_path


def _is_post_training_evaluation_file(relative: Path) -> bool:
    """Keep frozen holdouts and their evaluator out of every training archive."""
    if relative.parts[:2] == ("cloud_grpo", "eval_data"):
        return True
    return relative.as_posix() in {
        "cloud_grpo/evaluate_checkpoints.py",
        "cloud_grpo/prepare_final_eval.py",
        "sql_debug_agent/cloud_evaluation.py",
    }


def _to_prompt_completion(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("SFT 记录缺少 messages")
    assistant = messages[-1]
    if assistant.get("role") != "assistant" or not assistant.get("content"):
        raise ValueError("SFT 记录最后一条必须是非空 assistant 答案")
    return {
        "prompt": messages[:-1],
        "completion": [assistant],
        "metadata": record.get("metadata", {}),
    }


def _to_grpo_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    metadata = record.get("metadata", {})
    required = ("reference_sql", "previous_sql")
    if not isinstance(messages, list) or not messages:
        raise ValueError("GRPO 记录缺少 messages")
    if any(not record.get(key) for key in required):
        raise ValueError("GRPO 记录缺少 reference_sql 或 previous_sql")
    if not metadata.get("task_id") or not metadata.get("error_type"):
        raise ValueError("GRPO 记录缺少 task_id 或 error_type")
    return {
        "prompt": messages,
        "reference_sql": record["reference_sql"],
        "previous_sql": record["previous_sql"],
        "task_id": metadata["task_id"],
        "error_type": metadata["error_type"],
        "source": metadata.get("source", "unknown"),
    }


def _validate_unique_task_ids(records: list[dict[str, Any]]) -> None:
    ids = [record["task_id"] for record in records]
    duplicates = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"GRPO task_id 重复：{', '.join(duplicates)}")


def _select_hard_grpo_records(
    grpo_records: list[dict[str, Any]], hard_preferences_path: Path | None
) -> list[dict[str, Any]]:
    if hard_preferences_path is None:
        return []
    preferences = _read_jsonl(hard_preferences_path)
    hard_ids = [item.get("metadata", {}).get("task_id") for item in preferences]
    if any(not task_id for task_id in hard_ids):
        raise ValueError("hard preference 缺少 metadata.task_id")
    duplicates = sorted(
        task_id for task_id, count in Counter(hard_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"hard preference task_id 重复：{', '.join(duplicates)}")
    by_id = {record["task_id"]: record for record in grpo_records}
    missing = sorted(set(hard_ids) - set(by_id))
    if missing:
        raise ValueError(f"hard preference 在 GRPO prompts 中不存在：{', '.join(missing)}")
    return [by_id[task_id] for task_id in hard_ids]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
