from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .data_generation import task_fingerprint
from .database import get_schema
from .dataset import DebugTask, load_tasks, write_tasks
from .preparation import SYSTEM_PROMPT, task_to_sft_record
from .verifier import SQLVerifier


V3_TARGET_DISTRIBUTION = {
    "aggregation": 8,
    "date": 10,
    "duplicate_counting": 6,
    "filter": 12,
    "join_type": 10,
    "schema_linking": 8,
    "syntax_error": 6,
}


def build_v3_holdout(database_path: Path, output_path: Path) -> tuple[Path, list[DebugTask]]:
    rows = [
        ("syntax_error", "计算全部账户余额总和。", "SELECT SUM(balance) FORM accounts", "SELECT SUM(balance) FROM accounts"),
        ("syntax_error", "列出客户编号和姓名。", "SELECT id name, FROM customers ORDER BY id", "SELECT id, name FROM customers ORDER BY id"),
        ("syntax_error", "统计交易记录数。", "SELECT COUNT(*) transactions", "SELECT COUNT(*) FROM transactions"),
        ("syntax_error", "查询最大交易金额。", "SELECT MAX(amount FROM transactions", "SELECT MAX(amount) FROM transactions"),
        ("schema_linking", "列出所有客户姓名。", "SELECT customer_name FROM customers ORDER BY id", "SELECT name FROM customers ORDER BY id"),
        ("schema_linking", "查询余额超过 9000 的账户数。", "SELECT COUNT(*) FROM accounts WHERE account_balance>9000", "SELECT COUNT(*) FROM accounts WHERE balance>9000"),
        ("schema_linking", "统计 debit 类型交易数。", "SELECT COUNT(*) FROM transactions WHERE type='debit'", "SELECT COUNT(*) FROM transactions WHERE txn_type='debit'"),
        ("schema_linking", "统计 2025-01-12 之后的交易数。", "SELECT COUNT(*) FROM transactions WHERE date>'2025-01-12'", "SELECT COUNT(*) FROM transactions WHERE txn_date>'2025-01-12'"),
        ("aggregation", "查询每种账户类型的最大余额。", "SELECT account_type,MIN(balance) FROM accounts GROUP BY account_type ORDER BY account_type", "SELECT account_type,MAX(balance) FROM accounts GROUP BY account_type ORDER BY account_type"),
        ("aggregation", "按交易类型统计交易数。", "SELECT txn_type,COUNT(*) FROM transactions ORDER BY txn_type", "SELECT txn_type,COUNT(*) FROM transactions GROUP BY txn_type ORDER BY txn_type"),
        ("aggregation", "统计每位客户的账户数，包括零账户客户。", "SELECT c.id,COUNT(*) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id GROUP BY c.id ORDER BY c.id", "SELECT c.id,COUNT(a.id) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id GROUP BY c.id ORDER BY c.id"),
        ("aggregation", "按月份统计交易总额。", "SELECT substr(txn_date,1,4),SUM(amount) FROM transactions GROUP BY substr(txn_date,1,4) ORDER BY 1", "SELECT substr(txn_date,1,7),SUM(amount) FROM transactions GROUP BY substr(txn_date,1,7) ORDER BY 1"),
        ("join_type", "列出账户编号及客户风险等级。", "SELECT a.id,c.risk_level FROM accounts a JOIN customers c ON a.id=c.id ORDER BY a.id", "SELECT a.id,c.risk_level FROM accounts a JOIN customers c ON a.customer_id=c.id ORDER BY a.id"),
        ("join_type", "统计每位客户的账户余额总和，包括零账户客户。", "SELECT c.id,COALESCE(SUM(a.balance),0) FROM customers c JOIN accounts a ON a.customer_id=c.id GROUP BY c.id ORDER BY c.id", "SELECT c.id,COALESCE(SUM(a.balance),0) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id GROUP BY c.id ORDER BY c.id"),
        ("join_type", "列出从未发生 debit 交易的账户。", "SELECT DISTINCT a.id FROM accounts a JOIN transactions t ON t.account_id=a.id WHERE t.txn_type!='debit' ORDER BY a.id", "SELECT a.id FROM accounts a WHERE NOT EXISTS (SELECT 1 FROM transactions t WHERE t.account_id=a.id AND t.txn_type='debit') ORDER BY a.id"),
        ("join_type", "统计每个账户的 credit 总额，包括零次账户。", "SELECT a.id,COALESCE(SUM(t.amount),0) FROM accounts a LEFT JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='credit' GROUP BY a.id ORDER BY a.id", "SELECT a.id,COALESCE(SUM(t.amount),0) FROM accounts a LEFT JOIN transactions t ON t.account_id=a.id AND t.txn_type='credit' GROUP BY a.id ORDER BY a.id"),
        ("filter", "列出华北地区客户姓名。", "SELECT name FROM customers WHERE region='西南' ORDER BY name", "SELECT name FROM customers WHERE region='华北' ORDER BY name"),
        ("filter", "统计 low 风险客户数。", "SELECT COUNT(*) FROM customers WHERE risk_level='high'", "SELECT COUNT(*) FROM customers WHERE risk_level='low'"),
        ("filter", "计算 debit 交易总额。", "SELECT SUM(amount) FROM transactions WHERE txn_type='credit'", "SELECT SUM(amount) FROM transactions WHERE txn_type='debit'"),
        ("filter", "统计华东客户的 checking 账户数。", "SELECT COUNT(a.id) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE c.region='华东'", "SELECT COUNT(a.id) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE c.region='华东' AND a.account_type='checking'"),
        ("date", "统计 2025-01-12 及之后的交易数。", "SELECT COUNT(*) FROM transactions WHERE txn_date>='2025-01-03'", "SELECT COUNT(*) FROM transactions WHERE txn_date>='2025-01-12'"),
        ("date", "统计 2025-01-20 之前的交易数。", "SELECT COUNT(*) FROM transactions WHERE txn_date<='2025-01-20'", "SELECT COUNT(*) FROM transactions WHERE txn_date<'2025-01-20'"),
        ("date", "计算 2025 年 2 月交易总额。", "SELECT SUM(amount) FROM transactions WHERE txn_date>='2025-01-01' AND txn_date<'2025-03-01'", "SELECT SUM(amount) FROM transactions WHERE txn_date>='2025-02-01' AND txn_date<'2025-03-01'"),
        ("date", "统计 2025-01-08 到 2025-01-20 之前的交易数。", "SELECT COUNT(*) FROM transactions WHERE txn_date BETWEEN '2025-01-08' AND '2025-01-20'", "SELECT COUNT(*) FROM transactions WHERE txn_date>='2025-01-08' AND txn_date<'2025-01-20'"),
        ("duplicate_counting", "统计拥有账户的客户数。", "SELECT COUNT(c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id", "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id"),
        ("duplicate_counting", "统计发生过 debit 交易的客户数。", "SELECT COUNT(t.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='debit'", "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='debit'"),
        ("duplicate_counting", "统计有交易的账户数。", "SELECT COUNT(t.id) FROM accounts a JOIN transactions t ON t.account_id=a.id", "SELECT COUNT(DISTINCT a.id) FROM accounts a JOIN transactions t ON t.account_id=a.id"),
        ("duplicate_counting", "统计发生过金额不少于 700 交易的客户数。", "SELECT COUNT(t.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.amount>=700", "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.amount>=700"),
    ]
    tasks = _rows_to_tasks(rows, "v3_holdout")
    _validate_tasks(tasks, SQLVerifier(database_path))
    return write_tasks(output_path, tasks), tasks


def build_v3_dataset(
    database_path: Path,
    v1_train_path: Path,
    v1_eval_path: Path,
    raw_output_path: Path,
    output_dir: Path,
    forbidden_tasks: list[DebugTask],
) -> dict[str, Any]:
    verifier = SQLVerifier(database_path)
    forbidden = {task_fingerprint(task) for task in forbidden_tasks}
    additions = _generate_v3_tasks(verifier, forbidden)
    write_tasks(raw_output_path, additions)
    schema = get_schema(database_path)
    v1_train = _read_jsonl(v1_train_path)
    v1_eval = _read_jsonl(v1_eval_path)
    for record in v1_train + v1_eval:
        record["messages"][0]["content"] = SYSTEM_PROMPT
    addition_records = [task_to_sft_record(task, verifier, schema) for task in additions]
    train_records = v1_train + addition_records
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "sft_train.jsonl"
    eval_path = output_dir / "sft_eval.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(eval_path, v1_eval)
    manifest = {
        "strategy": "V1 回放 + 多模板最小修改样本 + SQLite 方言约束；不沿用导致回退的 V2 增量",
        "prompt_profile": "sqlite_minimal_edit_v3",
        "v1_replay_count": len(v1_train),
        "increment_count": len(additions),
        "train_count": len(train_records),
        "valid_count": len(v1_eval),
        "increment_distribution": dict(sorted(Counter(t.error_type for t in additions).items())),
        "forbidden_exact_overlap_count": 0,
        "train_path": str(train_path.resolve()),
        "valid_path": str(eval_path.resolve()),
        "raw_increment_path": str(raw_output_path.resolve()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def _generate_v3_tasks(verifier: SQLVerifier, forbidden: set[str]) -> list[DebugTask]:
    accepted: list[DebugTask] = []
    seen = set(forbidden)
    candidates = list(_v3_candidates())
    for error_type, target in V3_TARGET_DISTRIBUTION.items():
        selected = 0
        for task in candidates:
            if task.error_type != error_type or task_fingerprint(task) in seen:
                continue
            if verifier.execute(task.reference_sql).executable and not verifier.verify(task.initial_sql, task.reference_sql).passed:
                accepted.append(task)
                seen.add(task_fingerprint(task))
                selected += 1
                if selected == target:
                    break
        if selected != target:
            raise RuntimeError(f"V3 {error_type} 只生成 {selected}/{target} 条")
    return accepted


def _v3_candidates() -> Iterable[DebugTask]:
    rows: list[tuple[str, str, str, str]] = []
    # Syntax-only repairs teach the model to preserve already-correct identifiers.
    for column, table in (("id", "customers"), ("balance", "accounts"), ("amount", "transactions"), ("risk_level", "customers"), ("account_type", "accounts"), ("txn_date", "transactions")):
        rows.append(("syntax_error", f"查询 {table} 的 {column}。", f"SELEC {column} FROM {table}", f"SELECT {column} FROM {table}"))
    mappings = (("name", "customer_name", "customers"), ("region", "area", "customers"), ("risk_level", "risk", "customers"), ("balance", "current_balance", "accounts"), ("account_type", "kind", "accounts"), ("txn_date", "posted_date", "transactions"), ("txn_type", "type", "transactions"), ("amount", "value", "transactions"))
    for correct, wrong, table in mappings:
        rows.append(("schema_linking", f"统计 {correct} 非空的记录数，只修正字段名。", f"SELECT COUNT(*) FROM {table} WHERE {wrong} IS NOT NULL", f"SELECT COUNT(*) FROM {table} WHERE {correct} IS NOT NULL"))
    for group, table, value in (("region", "customers", "id"), ("risk_level", "customers", "id"), ("account_type", "accounts", "id"), ("txn_type", "transactions", "id")):
        rows.append(("aggregation", f"按 {group} 统计记录数。", f"SELECT {group},COUNT({value}) FROM {table}", f"SELECT {group},COUNT({value}) FROM {table} GROUP BY {group}"))
    for agg_wrong, agg_right in (("MIN", "MAX"), ("SUM", "AVG"), ("AVG", "SUM"), ("MAX", "MIN")):
        rows.append(("aggregation", f"按账户类型计算 {agg_right} 余额。", f"SELECT account_type,{agg_wrong}(balance) FROM accounts GROUP BY account_type", f"SELECT account_type,{agg_right}(balance) FROM accounts GROUP BY account_type"))
    for txn_type in ("debit", "credit"):
        for threshold in (100, 500, 900):
            rows.append(("duplicate_counting", f"统计发生过金额大于 {threshold} 的 {txn_type} 交易的客户数。", f"SELECT COUNT(c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='{txn_type}' AND t.amount>{threshold}", f"SELECT COUNT(DISTINCT c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='{txn_type}' AND t.amount>{threshold}"))
    for expected, wrong in (("low", "medium"), ("medium", "high"), ("high", "low")):
        rows.append(("filter", f"统计 {expected} 风险客户数，只修正错误的风险值。", f"SELECT COUNT(*) FROM customers WHERE risk_level='{wrong}'", f"SELECT COUNT(*) FROM customers WHERE risk_level='{expected}'"))
    for expected, wrong in (("华东", "华南"), ("华南", "华北"), ("华北", "西南"), ("西南", "华东")):
        rows.append(("filter", f"统计{expected}客户数，不增加其他条件。", f"SELECT COUNT(*) FROM customers WHERE region='{wrong}'", f"SELECT COUNT(*) FROM customers WHERE region='{expected}'"))
    for txn_type, wrong in (("debit", "credit"), ("credit", "debit")):
        for threshold in (300, 800, 1500):
            rows.append(("filter", f"统计金额大于 {threshold} 的 {txn_type} 交易数，保留金额条件。", f"SELECT COUNT(*) FROM transactions WHERE amount>{threshold} AND txn_type='{wrong}'", f"SELECT COUNT(*) FROM transactions WHERE amount>{threshold} AND txn_type='{txn_type}'"))
    for expected, wrong in (("low", "high"), ("medium", "low"), ("high", "medium")):
        rows.append(("filter", f"列出 {expected} 风险客户姓名，只替换风险值。", f"SELECT name FROM customers WHERE risk_level='{wrong}' ORDER BY name", f"SELECT name FROM customers WHERE risk_level='{expected}' ORDER BY name"))
    for expected, wrong in (("华东", "华北"), ("华南", "西南"), ("东北", "华东")):
        rows.append(("filter", f"列出{expected}客户编号，不添加账户条件。", f"SELECT id FROM customers WHERE region='{wrong}' ORDER BY id", f"SELECT id FROM customers WHERE region='{expected}' ORDER BY id"))
    for boundary, wrong in (("2025-01-08", "2025-01-01"), ("2025-01-20", "2025-01-08"), ("2025-02-01", "2025-01-01"), ("2025-03-05", "2025-02-01"), ("2025-04-01", "2025-03-01")):
        rows.append(("date", f"统计 {boundary} 及之后的交易数；SQLite 日期文本直接比较。", f"SELECT COUNT(*) FROM transactions WHERE txn_date>='{wrong}'", f"SELECT COUNT(*) FROM transactions WHERE txn_date>='{boundary}'"))
        rows.append(("date", f"统计 {boundary} 之前的交易数；SQLite 日期文本直接比较。", f"SELECT COUNT(*) FROM transactions WHERE txn_date<'{wrong}'", f"SELECT COUNT(*) FROM transactions WHERE txn_date<'{boundary}'"))
    for txn_type in ("debit", "credit"):
        for account_type in ("checking", "saving"):
            rows.append(("join_type", f"统计每位客户的 {account_type} 账户数，包括零账户客户。", f"SELECT c.id,COUNT(a.id) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE a.account_type='{account_type}' GROUP BY c.id", f"SELECT c.id,COUNT(a.id) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id AND a.account_type='{account_type}' GROUP BY c.id"))
            rows.append(("join_type", f"统计每个账户的 {txn_type} 交易数，包括零次账户。", f"SELECT a.id,COUNT(t.id) FROM accounts a LEFT JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='{txn_type}' GROUP BY a.id", f"SELECT a.id,COUNT(t.id) FROM accounts a LEFT JOIN transactions t ON t.account_id=a.id AND t.txn_type='{txn_type}' GROUP BY a.id"))
    rows.extend([
        ("join_type", "列出没有 debit 交易的客户。", "SELECT DISTINCT c.id FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.txn_type!='debit'", "SELECT c.id FROM customers c WHERE NOT EXISTS (SELECT 1 FROM accounts a JOIN transactions t ON t.account_id=a.id WHERE a.customer_id=c.id AND t.txn_type='debit')"),
        ("join_type", "列出没有 credit 交易的账户。", "SELECT DISTINCT a.id FROM accounts a JOIN transactions t ON t.account_id=a.id WHERE t.txn_type!='credit'", "SELECT a.id FROM accounts a WHERE NOT EXISTS (SELECT 1 FROM transactions t WHERE t.account_id=a.id AND t.txn_type='credit')"),
        ("join_type", "列出账户编号和客户地区，只修正连接键。", "SELECT a.id,c.region FROM accounts a JOIN customers c ON a.id=c.id ORDER BY a.id", "SELECT a.id,c.region FROM accounts a JOIN customers c ON a.customer_id=c.id ORDER BY a.id"),
        ("join_type", "列出交易编号和账户类型，只修正连接键。", "SELECT t.id,a.account_type FROM transactions t JOIN accounts a ON t.id=a.id ORDER BY t.id", "SELECT t.id,a.account_type FROM transactions t JOIN accounts a ON t.account_id=a.id ORDER BY t.id"),
        ("join_type", "统计每位客户的账户余额，包括零账户客户。", "SELECT c.id,COALESCE(SUM(a.balance),0) FROM customers c JOIN accounts a ON a.customer_id=c.id GROUP BY c.id ORDER BY c.id", "SELECT c.id,COALESCE(SUM(a.balance),0) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id GROUP BY c.id ORDER BY c.id"),
        ("join_type", "统计每个账户的交易金额，包括零交易账户。", "SELECT a.id,COALESCE(SUM(t.amount),0) FROM accounts a JOIN transactions t ON t.account_id=a.id GROUP BY a.id ORDER BY a.id", "SELECT a.id,COALESCE(SUM(t.amount),0) FROM accounts a LEFT JOIN transactions t ON t.account_id=a.id GROUP BY a.id ORDER BY a.id"),
        ("join_type", "统计每个账户的 debit 总额，包括零次账户。", "SELECT a.id,COALESCE(SUM(t.amount),0) FROM accounts a LEFT JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='debit' GROUP BY a.id ORDER BY a.id", "SELECT a.id,COALESCE(SUM(CASE WHEN t.txn_type='debit' THEN t.amount ELSE 0 END),0) FROM accounts a LEFT JOIN transactions t ON t.account_id=a.id GROUP BY a.id ORDER BY a.id"),
        ("join_type", "统计每位客户的 credit 总额，包括零金额客户。", "SELECT c.id,COALESCE(SUM(t.amount),0) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id LEFT JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='credit' GROUP BY c.id ORDER BY c.id", "SELECT c.id,COALESCE(SUM(CASE WHEN t.txn_type='credit' THEN t.amount ELSE 0 END),0) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id LEFT JOIN transactions t ON t.account_id=a.id GROUP BY c.id ORDER BY c.id"),
    ])
    yield from _rows_to_tasks(rows, "v3_train")


def _rows_to_tasks(rows: list[tuple[str, str, str, str]], prefix: str) -> list[DebugTask]:
    counters: dict[str, int] = {}
    tasks = []
    for error_type, question, initial, reference in rows:
        counters[error_type] = counters.get(error_type, 0) + 1
        tasks.append(DebugTask(f"{prefix}_{error_type}_{counters[error_type]:02d}", question, initial, reference, error_type, f"{prefix}_{error_type}_{counters[error_type]:02d}"))
    return tasks


def _validate_tasks(tasks: list[DebugTask], verifier: SQLVerifier) -> None:
    for task in tasks:
        result = verifier.execute(task.reference_sql)
        if not result.executable:
            raise RuntimeError(f"{task.task_id} 参考 SQL 不可执行：{result.error}")
        if verifier.verify(task.initial_sql, task.reference_sql).passed:
            raise RuntimeError(f"{task.task_id} 不是有效 Bad Case")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
