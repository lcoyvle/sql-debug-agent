from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .database import get_schema
from .dataset import load_tasks
from .model_errors import ModelConfigurationError
from .preparation import task_to_sft_record
from .verifier import SQLVerifier


DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"


def prepare_mlx_dataset(
    sft_train_path: Path,
    sft_eval_path: Path,
    test_tasks_path: Path,
    test_database_path: Path,
    output_dir: Path,
    sql_terminator: bool = False,
) -> dict[str, Any]:
    """Create the exact train/valid/test filenames and chat format MLX-LM expects."""
    train_records = _read_jsonl(sft_train_path)
    valid_records = _read_jsonl(sft_eval_path)
    test_tasks = load_tasks(test_tasks_path)
    test_verifier = SQLVerifier(test_database_path)
    test_schema = get_schema(test_database_path)
    test_records = [
        task_to_sft_record(task, test_verifier, test_schema) for task in test_tasks
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.jsonl",
        "valid": output_dir / "valid.jsonl",
        "test": output_dir / "test.jsonl",
    }
    _write_chat_jsonl(paths["train"], train_records, sql_terminator)
    _write_chat_jsonl(paths["valid"], valid_records, sql_terminator)
    _write_chat_jsonl(paths["test"], test_records, sql_terminator)

    manifest = {
        "format": "mlx-lm chat JSONL",
        "train_count": len(train_records),
        "valid_count": len(valid_records),
        "test_count": len(test_records),
        "test_excluded_from_gradient_updates": True,
        "sql_terminator": sql_terminator,
        "train_distribution": _distribution(train_records),
        "valid_distribution": _distribution(valid_records),
        "files": {name: str(path.resolve()) for name, path in paths.items()},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def build_mlx_lora_command(
    python_executable: str,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    iters: int,
    smoke: bool,
    learning_rate: float = 1e-5,
    resume_adapter_file: Path | None = None,
) -> list[str]:
    if iters < 1:
        raise ValueError("训练步数必须大于等于 1")
    command = [
        python_executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir.resolve()),
        "--adapter-path",
        str(adapter_path.resolve()),
        "--iters",
        str(iters),
        "--batch-size",
        "1",
        "--num-layers",
        "4",
        "--max-seq-length",
        "1024",
        "--learning-rate",
        str(learning_rate),
        "--grad-accumulation-steps",
        "4",
        "--steps-per-report",
        "1" if smoke else "10",
        "--steps-per-eval",
        str(max(1, min(iters, 50))),
        "--val-batches",
        "2" if smoke else "10",
        "--save-every",
        str(min(50, iters)),
        "--mask-prompt",
        "--grad-checkpoint",
        "--clear-cache-threshold",
        "1GB",
    ]
    if resume_adapter_file is not None:
        command.extend(
            ["--resume-adapter-file", str(resume_adapter_file.resolve())]
        )
    return command


def run_mlx_lora_training(
    model: str,
    data_dir: Path,
    adapter_path: Path,
    cache_dir: Path,
    iters: int,
    smoke: bool,
    learning_rate: float = 1e-5,
    resume_adapter_file: Path | None = None,
) -> Path:
    if importlib.util.find_spec("mlx_lm") is None:
        raise ModelConfigurationError(
            "当前 Python 没有安装 MLX-LM。请使用项目 .venv 中的 Python，或先运行："
            "python3 -m pip install -e '.[train]'"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    adapter_path.mkdir(parents=True, exist_ok=True)
    command = build_mlx_lora_command(
        sys.executable,
        model,
        data_dir,
        adapter_path,
        iters,
        smoke,
        learning_rate=learning_rate,
        resume_adapter_file=resume_adapter_file,
    )
    environment = os.environ.copy()
    environment["HF_HOME"] = str(cache_dir.resolve())
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log_parts: list[str] = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_parts.append(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise

    training_log = "".join(log_parts)
    log_path = adapter_path / "training.log"
    log_path.write_text(training_log, encoding="utf-8")
    if return_code != 0:
        raise ModelConfigurationError(
            f"MLX-LM 训练失败（退出码 {return_code}），请查看 {log_path.resolve()}。"
        )

    run_manifest = {
        "model": model,
        "adapter_path": str(adapter_path.resolve()),
        "data_dir": str(data_dir.resolve()),
        "iters": iters,
        "smoke": smoke,
        "learning_rate": learning_rate,
        "resume_adapter_file": (
            str(resume_adapter_file.resolve()) if resume_adapter_file else None
        ),
        "hardware_profile": {
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_layers": 4,
            "max_seq_length": 1024,
            "gradient_checkpointing": True,
        },
        "command": command,
        "metrics": _parse_training_metrics(training_log),
        "training_log": str(log_path.resolve()),
    }
    manifest_path = adapter_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_chat_jsonl(
    path: Path, records: list[dict[str, Any]], sql_terminator: bool
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            messages = [dict(message) for message in record["messages"]]
            if sql_terminator and messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] = messages[-1]["content"].rstrip("; ") + ";"
            handle.write(
                json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
            )


def _distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        record.get("metadata", {}).get("error_type", "unknown")
        for record in records
    )
    return dict(sorted(counts.items()))


def _parse_training_metrics(log: str) -> dict[str, Any]:
    validation = [
        {"iteration": int(step), "loss": float(loss)}
        for step, loss in re.findall(r"Iter (\d+): Val loss ([0-9.]+)", log)
    ]
    training = [
        {"iteration": int(step), "loss": float(loss)}
        for step, loss in re.findall(r"Iter (\d+): Train loss ([0-9.]+)", log)
    ]
    peak_values = [
        float(value) for value in re.findall(r"Peak mem ([0-9.]+) GB", log)
    ]
    return {
        "validation_loss": validation,
        "training_loss": training,
        "peak_memory_gb": max(peak_values) if peak_values else None,
    }
