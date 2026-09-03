from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sql_debug_agent.cloud_evaluation import (
    build_final_comparison,
    write_final_comparison,
)
from sql_debug_agent.database import get_schema
from sql_debug_agent.dataset import DebugTask, load_tasks
from sql_debug_agent.mlx_repair import extract_sql
from sql_debug_agent.preparation import SYSTEM_PROMPT, build_sft_user_content
from sql_debug_agent.rl_reward import RobustSQLReward
from sql_debug_agent.verifier import SQLVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description="统一评测 Base、SFT 与 GRPO")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=PROJECT_ROOT / "cloud_grpo" / "eval_data" / "final_holdout_v3.jsonl",
    )
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        default=PROJECT_ROOT / "cloud_outputs" / "sft_adapter",
    )
    parser.add_argument(
        "--grpo-adapter",
        type=Path,
        default=PROJECT_ROOT / "cloud_outputs" / "grpo_v2_40",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "cloud_outputs" / "final_evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    _validate_paths(args)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tasks = load_tasks(args.tasks)
    data_dir = PROJECT_ROOT / "cloud_grpo" / "data"
    database_paths = [
        data_dir / "finance_demo.db",
        data_dir / "train_finance.db",
    ]
    prompts = _build_prompts(tasks, database_paths[0])
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=False,
    )
    base_model.eval()
    scorer = RobustSQLReward(database_paths)
    results: dict[str, list[dict[str, Any]]] = {}

    print("\n[1/3] 正在评测 Base...")
    base_outputs = _generate(
        base_model, tokenizer, prompts, args.batch_size, args.max_new_tokens
    )
    results["base"] = _score_outputs(tasks, base_outputs, scorer, "Base")

    print("\n[2/3] 正在评测 SFT...")
    model = PeftModel.from_pretrained(
        base_model, str(args.sft_adapter), adapter_name="sft", is_trainable=False
    )
    model.set_adapter("sft")
    model.eval()
    sft_outputs = _generate(
        model, tokenizer, prompts, args.batch_size, args.max_new_tokens
    )
    results["sft"] = _score_outputs(tasks, sft_outputs, scorer, "SFT")

    print("\n[3/3] 正在评测 SFT + GRPO...")
    model.load_adapter(str(args.grpo_adapter), adapter_name="grpo", is_trainable=False)
    model.set_adapter("grpo")
    model.eval()
    grpo_outputs = _generate(
        model, tokenizer, prompts, args.batch_size, args.max_new_tokens
    )
    results["grpo"] = _score_outputs(tasks, grpo_outputs, scorer, "GRPO")

    protocol = {
        "dataset": args.tasks.name,
        "dataset_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "task_count": len(tasks),
        "model": args.model,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "reward_databases": [path.name for path in database_paths],
        "single_repair_turn": True,
        "trained_after_holdout_reveal": False,
    }
    comparison = build_final_comparison(results, protocol)
    json_path, markdown_path = write_final_comparison(comparison, args.output_dir)
    _print_summary(comparison)
    print(f"\nJSON 报告：{json_path.resolve()}")
    print(f"Markdown 报告：{markdown_path.resolve()}")


def _validate_paths(args: argparse.Namespace) -> None:
    required = [
        args.tasks,
        args.sft_adapter / "adapter_config.json",
        args.sft_adapter / "adapter_model.safetensors",
        args.grpo_adapter / "adapter_config.json",
        args.grpo_adapter / "adapter_model.safetensors",
        PROJECT_ROOT / "cloud_grpo" / "data" / "finance_demo.db",
        PROJECT_ROOT / "cloud_grpo" / "data" / "train_finance.db",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("最终评测缺少文件：\n- " + "\n- ".join(missing))
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size 和 max-new-tokens 必须大于 0")


def _build_prompts(tasks: list[DebugTask], database_path: Path) -> list[list[dict[str, str]]]:
    verifier = SQLVerifier(database_path)
    schema = get_schema(database_path)
    prompts: list[list[dict[str, str]]] = []
    for task in tasks:
        baseline = verifier.verify(task.initial_sql, task.reference_sql)
        prompts.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_sft_user_content(
                        task.question,
                        schema,
                        task.initial_sql,
                        baseline.feedback,
                    ),
                },
            ]
        )
    return prompts


def _generate(
    model: Any,
    tokenizer: Any,
    prompts: list[list[dict[str, str]]],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    import torch

    outputs: list[str] = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        for sequence in generated:
            outputs.append(
                extract_sql(
                    tokenizer.decode(
                        sequence[prompt_length:], skip_special_tokens=True
                    )
                )
            )
        print(f"  已完成 {min(start + len(batch), len(prompts))}/{len(prompts)}")
    return outputs


def _score_outputs(
    tasks: list[DebugTask],
    outputs: list[str],
    scorer: RobustSQLReward,
    label: str,
) -> list[dict[str, Any]]:
    if len(tasks) != len(outputs):
        raise ValueError("模型输出数量与任务数量不一致")
    rows: list[dict[str, Any]] = []
    for task, candidate_sql in zip(tasks, outputs, strict=True):
        reward = scorer.score(
            candidate_sql,
            task.reference_sql,
            previous_sql=task.initial_sql,
            final_turn=True,
        )
        row = {
            "task_id": task.task_id,
            "error_type": task.error_type,
            "question": task.question,
            "initial_sql": task.initial_sql,
            "reference_sql": task.reference_sql,
            "candidate_sql": candidate_sql,
            "correct": reward.matches_all_databases,
            "reward": reward.to_dict(),
        }
        rows.append(row)
        mark = "✅" if row["correct"] else "❌"
        print(f"  {mark} {label}: {task.task_id}")
    return rows


def _print_summary(comparison: dict[str, Any]) -> None:
    print("\n===== 最终冻结评测 =====")
    for name, label in (("base", "Base"), ("sft", "SFT"), ("grpo", "SFT + GRPO")):
        summary = comparison["summaries"][name]
        print(
            f"{label}: {summary['correct_count']}/{summary['task_count']} "
            f"({summary['accuracy']:.1%})"
        )
    change = comparison["comparisons"]["sft_to_grpo"]
    print(f"SFT → GRPO 修复：{len(change['fixed_tasks'])} 题")
    print(f"SFT → GRPO 回退：{len(change['regressed_tasks'])} 题")
    print(f"结论：{comparison['decision']['reason']}")


if __name__ == "__main__":
    main()
