from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = PROJECT_ROOT / "data" / "final_holdout_v3.jsonl"
    eval_dir = PROJECT_ROOT / "cloud_grpo" / "eval_data"
    eval_dir.mkdir(parents=True, exist_ok=True)
    target = eval_dir / source.name
    shutil.copy2(source, target)
    task_count = sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line)
    manifest = {
        "purpose": "post-training frozen evaluation only",
        "dataset": target.name,
        "task_count": task_count,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "included_in_sft_or_grpo_training": False,
        "warning": "上传后不得再使用该测试集继续训练或选择超参数。",
    }
    manifest_path = eval_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive_path = PROJECT_ROOT / "artifacts" / "sql-debug-agent-final-eval.zip"
    sources = [
        PROJECT_ROOT / "cloud_grpo" / "evaluate_checkpoints.py",
        target,
        manifest_path,
        PROJECT_ROOT / "sql_debug_agent" / "cloud_evaluation.py",
    ]
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(PROJECT_ROOT))
    print(f"冻结测试题：{task_count} 条")
    print(f"SHA256：{manifest['sha256']}")
    print(f"最终评测包：{archive_path.resolve()}")


if __name__ == "__main__":
    main()
