from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from .data_generation import MONTHS, REGIONS, RISKS, TXN_TYPES, task_fingerprint
from .database import get_schema
from .dataset import DebugTask, write_tasks
from .preparation import task_to_sft_record
from .verifier import SQLVerifier


V2_TARGET_DISTRIBUTION = {
    "aggregation": 12,
    "date": 12,
    "filter": 12,
    "join_type": 12,
    "schema_linking": 12,
}


def build_v2_dataset(
    database_path: Path,
    v1_train_path: Path,
    v1_eval_path: Path,
    raw_output_path: Path,
    output_dir: Path,
    forbidden_tasks: list[DebugTask],
) -> dict[str, Any]:
    verifier = SQLVerifier(database_path)
    forbidden = {task_fingerprint(task) for task in forbidden_tasks}
    additions = generate_v2_tasks(database_path, forbidden)
    write_tasks(raw_output_path, additions)

    v1_train = _read_jsonl(v1_train_path)
    v1_eval = _read_jsonl(v1_eval_path)
    schema = get_schema(database_path)
    addition_records = [
        task_to_sft_record(task, verifier, schema) for task in additions
    ]
    train_records = v1_train + addition_records

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "sft_train.jsonl"
    eval_path = output_dir / "sft_eval.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(eval_path, v1_eval)
    manifest = {
        "strategy": "V1 训练集 + 针对开发集剩余失败模式的新模板；V1 模板隔离集继续作内部验证",
        "v1_train_count": len(v1_train),
        "increment_count": len(additions),
        "train_count": len(train_records),
        "valid_count": len(v1_eval),
        "increment_distribution": dict(
            sorted(Counter(task.error_type for task in additions).items())
        ),
        "forbidden_exact_overlap_count": 0,
        "train_path": str(train_path.resolve()),
        "valid_path": str(eval_path.resolve()),
        "raw_increment_path": str(raw_output_path.resolve()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def generate_v2_tasks(
    database_path: Path, forbidden_fingerprints: set[str]
) -> list[DebugTask]:
    verifier = SQLVerifier(database_path)
    tasks: list[DebugTask] = []
    seen = set(forbidden_fingerprints)
    for error_type, target in V2_TARGET_DISTRIBUTION.items():
        accepted = 0
        for task in _candidates(error_type):
            fingerprint = task_fingerprint(task)
            if fingerprint in seen:
                continue
            if verifier.verify(task.initial_sql, task.reference_sql).passed:
                continue
            if not verifier.execute(task.reference_sql).executable:
                continue
            tasks.append(task)
            seen.add(fingerprint)
            accepted += 1
            if accepted == target:
                break
        if accepted != target:
            raise RuntimeError(f"V2 {error_type} 只生成 {accepted}/{target} 条")
    return tasks


def _task(error_type: str, index: int, template: str, question: str, initial: str, reference: str) -> DebugTask:
    return DebugTask(
        task_id=f"v2_{error_type}_{index:03d}",
        question=question,
        initial_sql=initial,
        reference_sql=reference,
        error_type=error_type,
        template_id=template,
    )


def _candidates(error_type: str) -> Iterable[DebugTask]:
    yield from {
        "aggregation": _aggregation_candidates,
        "date": _date_candidates,
        "filter": _filter_candidates,
        "join_type": _join_candidates,
        "schema_linking": _schema_candidates,
    }[error_type]()


def _aggregation_candidates() -> Iterable[DebugTask]:
    index = 0
    for txn_type, (month, start, end) in product(TXN_TYPES, MONTHS):
        index += 1
        reference = (
            "SELECT c.id, COUNT(t.id) AS txn_count FROM customers c "
            "LEFT JOIN accounts a ON a.customer_id=c.id "
            f"LEFT JOIN transactions t ON t.account_id=a.id AND t.txn_type='{txn_type}' "
            f"AND t.txn_date>='{start}' AND t.txn_date<'{end}' "
            "GROUP BY c.id ORDER BY c.id"
        )
        initial = reference.replace("COUNT(t.id)", "COUNT(*)")
        yield _task("aggregation", index, "v2_count_nullable_child", f"统计每位客户 {month} 的 {txn_type} 交易数，包括零次客户。", initial, reference)


def _date_candidates() -> Iterable[DebugTask]:
    index = 0
    boundaries = ("2025-01-08", "2025-01-20", "2025-02-01", "2025-02-15", "2025-03-05", "2025-04-10")
    for txn_type, boundary in product(TXN_TYPES, boundaries):
        index += 1
        reference = f"SELECT COUNT(*) AS txn_count FROM transactions WHERE txn_type='{txn_type}' AND txn_date>='{boundary}'"
        initial = reference.replace(boundary, "2024-12-15")
        yield _task("date", index, "v2_exact_boundary", f"统计 {boundary} 及之后的 {txn_type} 交易数量，必须使用给定日期。", initial, reference)


def _filter_candidates() -> Iterable[DebugTask]:
    index = 0
    for expected, wrong in product(RISKS, RISKS):
        if expected == wrong:
            continue
        index += 1
        reference = (
            "SELECT AVG(a.balance) AS avg_balance FROM customers c JOIN accounts a ON a.customer_id=c.id "
            f"WHERE c.risk_level='{expected}'"
        )
        initial = reference.replace(f"'{expected}'", f"'{wrong}'")
        yield _task("filter", index, "v2_risk_value_contrast", f"计算 {expected} 风险客户的平均账户余额，保留题目中的风险值。", initial, reference)
    for expected, wrong in product(REGIONS, REGIONS):
        if expected == wrong:
            continue
        index += 1
        reference = f"SELECT COUNT(*) AS customer_count FROM customers WHERE region='{expected}'"
        initial = reference.replace(f"'{expected}'", f"'{wrong}'")
        yield _task("filter", index, "v2_region_value_contrast", f"统计{expected}客户数，筛选值必须与问题一致。", initial, reference)


def _join_candidates() -> Iterable[DebugTask]:
    index = 0
    for txn_type, (month, start, end) in product(TXN_TYPES, MONTHS):
        index += 1
        reference = (
            "SELECT c.id, COALESCE(SUM(CASE WHEN "
            f"t.txn_type='{txn_type}' AND t.txn_date>='{start}' AND t.txn_date<'{end}' "
            "THEN t.amount ELSE 0 END),0) AS total_amount FROM customers c "
            "LEFT JOIN accounts a ON a.customer_id=c.id LEFT JOIN transactions t ON t.account_id=a.id "
            "GROUP BY c.id ORDER BY c.id"
        )
        initial = (
            "SELECT c.id, COALESCE(SUM(t.amount),0) AS total_amount FROM customers c "
            "LEFT JOIN accounts a ON a.customer_id=c.id LEFT JOIN transactions t ON t.account_id=a.id "
            f"WHERE t.txn_type='{txn_type}' AND t.txn_date>='{start}' AND t.txn_date<'{end}' "
            "GROUP BY c.id ORDER BY c.id"
        )
        yield _task("join_type", index, "v2_left_join_condition_position", f"统计每位客户 {month} 的 {txn_type} 总额，包括零金额客户。", initial, reference)


def _schema_candidates() -> Iterable[DebugTask]:
    index = 0
    mappings = (
        ("交易日期", "transactions t", "t.txn_date", "t.transaction_date"),
        ("交易金额", "transactions t", "t.amount", "t.transaction_amount"),
        ("账户类型", "accounts a", "a.account_type", "a.kind"),
        ("账户余额", "accounts a", "a.balance", "a.current_balance"),
    )
    for label, table, correct, wrong in mappings:
        for threshold in ("'2025-01-01'", "'2025-02-01'", "'2025-03-01'") if "日期" in label else ("0", "500", "1000"):
            index += 1
            operator = ">=" if "日期" in label else ">"
            reference = f"SELECT COUNT(*) AS row_count FROM {table} WHERE {correct} {operator} {threshold}"
            initial = reference.replace(correct, wrong)
            yield _task("schema_linking", index, "v2_schema_predicate", f"统计{label}{operator}{threshold}的记录数，使用 Schema 中真实字段。", initial, reference)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
