from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTECTION_TYPES = ("syntax_error", "schema_linking", "filter")


def build_corrective_replay_dataset(
    hard_preferences_path: Path,
    prompts_path: Path,
    rollouts_path: Path,
    validation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    hard_pairs = _read_jsonl(hard_preferences_path)
    prompts = _read_jsonl(prompts_path)
    rollouts = _read_jsonl(rollouts_path)
    prompt_by_task = {item["metadata"]["task_id"]: item for item in prompts}

    hard_records = [
        {
            "messages": pair["prompt"]
            + [{"role": "assistant", "content": pair["chosen_sql"]}],
            "metadata": {
                **pair["metadata"],
                "replay_role": "hard_correction",
            },
        }
        for pair in hard_pairs
    ]
    protection_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        error_type = rollout["metadata"]["error_type"]
        if (
            error_type in PROTECTION_TYPES
            and rollout["candidate_reward"]["matches_all_databases"]
        ):
            protection_candidates[error_type].append(rollout)

    protection_records: list[dict[str, Any]] = []
    target = len(hard_records)
    positions = {error_type: 0 for error_type in PROTECTION_TYPES}
    while len(protection_records) < target:
        added = False
        for error_type in PROTECTION_TYPES:
            items = protection_candidates[error_type]
            position = positions[error_type]
            if position >= len(items) or len(protection_records) >= target:
                continue
            rollout = items[position]
            positions[error_type] += 1
            prompt = prompt_by_task[rollout["metadata"]["task_id"]]
            protection_records.append(
                {
                    "messages": prompt["messages"]
                    + [{"role": "assistant", "content": rollout["reference_sql"]}],
                    "metadata": {
                        **rollout["metadata"],
                        "replay_role": "capability_protection",
                    },
                }
            )
            added = True
        if not added:
            break
    if len(protection_records) != target:
        raise RuntimeError(
            f"强项保护样本不足：需要 {target} 条，实际 {len(protection_records)} 条"
        )

    train_records = hard_records + protection_records
    validation_records = _read_jsonl(validation_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "sft_train.jsonl"
    valid_path = output_dir / "sft_eval.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(valid_path, validation_records)
    manifest = {
        "method": "corrective_replay_sft",
        "is_dpo": False,
        "reason": "本机 MLX-LM 0.31.3 不提供 DPO/GRPO trainer，使用可复现的短步数 Adapter 继续训练",
        "hard_correction_count": len(hard_records),
        "capability_protection_count": len(protection_records),
        "train_count": len(train_records),
        "valid_count": len(validation_records),
        "hard_error_distribution": dict(
            sorted(Counter(r["metadata"]["error_type"] for r in hard_records).items())
        ),
        "protection_distribution": dict(
            sorted(Counter(r["metadata"]["error_type"] for r in protection_records).items())
        ),
        "base_adapter": "artifacts/adapters/sft_v3_30/adapters.safetensors",
        "recommended_learning_rate": 5e-6,
        "candidate_steps": [5, 10],
        "files": {
            "train": str(train_path.resolve()),
            "valid": str(valid_path.resolve()),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["files"]["manifest"] = str(manifest_path.resolve())
    return manifest


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
