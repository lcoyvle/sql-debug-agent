from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="在 Colab T4 上重建标准 PEFT SFT Adapter")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "cloud_outputs" / "sft_adapter"
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("max-steps 必须大于 0")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    data_dir = PROJECT_ROOT / "cloud_grpo" / "data"
    train_dataset = Dataset.from_list(read_jsonl(data_dir / "sft_train.jsonl"))
    eval_dataset = Dataset.from_list(read_jsonl(data_dir / "sft_eval.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    config = SFTConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=max(1, args.max_steps),
        save_strategy="steps",
        save_steps=max(1, args.max_steps),
        max_length=1024,
        completion_only_loss=True,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        quantization_config=quantization,
        peft_config=lora,
    )
    # T4 has no native BF16 AMP path. Some Transformers/PEFT combinations
    # initialize LoRA parameters as BF16 even when fp16=True, which makes
    # GradScaler fail while unscaling gradients. PEFT recommends keeping all
    # trainable adapter parameters in FP32 under mixed-precision training.
    for parameter in trainer.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"SFT Adapter 已保存：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
