from __future__ import annotations

import argparse
import json
from pathlib import Path

from sql_debug_agent.cloud_training import make_sql_execution_reward, summarize_grpo_signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="在 SFT Adapter 上运行双数据库 GRPO")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--sft-adapter", type=Path, default=PROJECT_ROOT / "cloud_outputs" / "sft_adapter"
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--dataset",
        choices=("hard", "all"),
        default="hard",
        help="hard 使用 26 条真实失败难例；all 使用全部 90 条 prompts",
    )
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "cloud_outputs" / "grpo_adapter"
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("max-steps 必须大于 0")
    if args.num_generations < 2:
        raise ValueError("num-generations 必须至少为 2")
    if args.temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    if not (args.sft_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"没有找到 SFT Adapter：{args.sft_adapter}。请先运行 train_sft.py。"
        )

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    data_dir = PROJECT_ROOT / "cloud_grpo" / "data"
    dataset_name = "grpo_hard_train.jsonl" if args.dataset == "hard" else "grpo_train.jsonl"
    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"没有找到 GRPO 数据：{dataset_path}")
    dataset = Dataset.from_list(read_jsonl(dataset_path))
    print(
        f"GRPO 数据：{dataset_name}，{len(dataset)} 条；"
        f"每题采样 {args.num_generations} 个候选，temperature={args.temperature}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
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
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )
    model = PeftModel.from_pretrained(
        base_model, str(args.sft_adapter), is_trainable=True
    )

    reward = make_sql_execution_reward(
        [data_dir / "finance_demo.db", data_dir / "train_finance.db"]
    )
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=1e-5,
        beta=0.01,
        temperature=args.temperature,
        top_p=0.95,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.max_steps),
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        log_completions=True,
        seed=42,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # GRPOTrainer may add or reload PEFT adapters while it initializes its
    # reference-policy handling. Cast after trainer construction so every
    # trainable adapter parameter is FP32 before T4 AMP gradient unscaling.
    for parameter in trainer.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()
    signal_summary = summarize_grpo_signal(trainer.state.log_history)
    summary_path = args.output_dir / "grpo_signal_summary.json"
    summary_path.write_text(
        json.dumps(signal_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "GRPO 学习信号："
        f"{signal_summary['steps_with_reward_variance']}/"
        f"{signal_summary['logged_steps']} 个步骤有奖励差异"
    )
    print(f"信号审计报告：{summary_path.resolve()}")
    print(f"GRPO Adapter 已保存：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
