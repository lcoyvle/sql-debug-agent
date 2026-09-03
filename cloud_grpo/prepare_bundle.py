from __future__ import annotations

import argparse
from pathlib import Path

from sql_debug_agent.cloud_training import (
    build_colab_archive,
    build_grpo_v2_patch_archive,
    prepare_cloud_training_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 SQL Debug Agent 的 Colab 训练包")
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sql-debug-agent-colab.zip",
    )
    args = parser.parse_args()
    data_dir = PROJECT_ROOT / "cloud_grpo" / "data"
    manifest = prepare_cloud_training_data(
        PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_train.jsonl",
        PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_eval.jsonl",
        PROJECT_ROOT / "artifacts" / "rl_v1" / "grpo_prompts.jsonl",
        PROJECT_ROOT / "data" / "finance_demo.db",
        PROJECT_ROOT / "data" / "train_finance.db",
        data_dir,
        PROJECT_ROOT / "artifacts" / "rl_v1" / "hard_preference_pairs.jsonl",
    )
    archive = build_colab_archive(PROJECT_ROOT, args.archive)
    patch_archive = build_grpo_v2_patch_archive(
        PROJECT_ROOT,
        PROJECT_ROOT / "artifacts" / "sql-debug-agent-grpo-v2-patch.zip",
    )
    print(f"SFT 训练：{manifest['sft_train_count']} 条")
    print(f"SFT 验证：{manifest['sft_eval_count']} 条")
    print(f"GRPO prompts：{manifest['grpo_prompt_count']} 条")
    print(f"GRPO 真实失败难例：{manifest['grpo_hard_prompt_count']} 条")
    print("最终 holdout：未打包")
    print(f"Colab 上传包：{archive.resolve()}")
    print(f"GRPO V2 小补丁：{patch_archive.resolve()}")


if __name__ == "__main__":
    main()
