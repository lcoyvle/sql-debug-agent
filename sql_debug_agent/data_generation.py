from __future__ import annotations

import hashlib
import re
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Iterable

from .dataset import DebugTask, write_tasks
from .verifier import SQLVerifier


TARGET_DISTRIBUTION = {
    "join_type": 60,
    "aggregation": 30,
    "duplicate_counting": 15,
    "filter": 35,
    "date": 30,
    "schema_linking": 20,
    "syntax_error": 10,
}

REGIONS = ("华东", "华南", "华北", "西南", "东北")
RISKS = ("low", "medium", "high")
ACCOUNT_TYPES = ("checking", "saving")
TXN_TYPES = ("debit", "credit")
MONTHS = (
    ("2025-01", "2025-01-01", "2025-02-01"),
    ("2025-02", "2025-02-01", "2025-03-01"),
    ("2025-03", "2025-03-01", "2025-04-01"),
    ("2025-04", "2025-04-01", "2025-05-01"),
    ("2025-05", "2025-05-01", "2025-06-01"),
    ("2025-06", "2025-06-01", "2025-07-01"),
)


def task_fingerprint(task: DebugTask) -> str:
    normalized = "\n".join(
        re.sub(r"\s+", " ", value.strip().lower())
        for value in (task.question, task.initial_sql, task.reference_sql)
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def generate_training_tasks(
    database_path: Path, excluded_fingerprints: set[str] | None = None
) -> list[DebugTask]:
    verifier = SQLVerifier(database_path)
    tasks: list[DebugTask] = []
    fingerprints: set[str] = set()
    excluded_fingerprints = excluded_fingerprints or set()

    for error_type, target_count in TARGET_DISTRIBUTION.items():
        accepted = 0
        for candidate in _candidate_tasks(error_type):
            fingerprint = task_fingerprint(candidate)
            if fingerprint in fingerprints or fingerprint in excluded_fingerprints:
                continue
            verification = verifier.verify(
                candidate.initial_sql, candidate.reference_sql
            )
            if verification.passed:
                continue
            tasks.append(candidate)
            fingerprints.add(fingerprint)
            accepted += 1
            if accepted == target_count:
                break
        if accepted != target_count:
            raise RuntimeError(
                f"{error_type} 只生成了 {accepted}/{target_count} 条有效 Bad Case"
            )

    return tasks


def build_training_dataset(
    database_path: Path,
    output_path: Path,
    excluded_fingerprints: set[str] | None = None,
) -> tuple[Path, list[DebugTask]]:
    tasks = generate_training_tasks(database_path, excluded_fingerprints)
    return write_tasks(output_path, tasks), tasks


def summarize_distribution(tasks: list[DebugTask]) -> dict[str, int]:
    return dict(sorted(Counter(task.error_type for task in tasks).items()))


def _candidate_tasks(error_type: str) -> Iterable[DebugTask]:
    factories = {
        "join_type": _join_candidates,
        "aggregation": _aggregation_candidates,
        "duplicate_counting": _duplicate_counting_candidates,
        "filter": _filter_candidates,
        "date": _date_candidates,
        "schema_linking": _schema_candidates,
        "syntax_error": _syntax_candidates,
    }
    yield from factories[error_type]()


def _make_task(
    error_type: str,
    sequence: int,
    template_id: str,
    question: str,
    initial_sql: str,
    reference_sql: str,
) -> DebugTask:
    return DebugTask(
        task_id=f"train_{error_type}_{sequence:03d}",
        question=question,
        initial_sql=initial_sql,
        reference_sql=reference_sql,
        error_type=error_type,
        template_id=template_id,
    )


def _join_candidates() -> Iterable[DebugTask]:
    sequence = 0
    for txn_type, (month, start, end) in product(TXN_TYPES, MONTHS):
        sequence += 1
        question = f"统计每位客户 {month} 的 {txn_type} 交易次数，包括零次客户，按客户编号排序。"
        reference = (
            "SELECT c.id, COUNT(t.id) AS txn_count FROM customers c "
            "LEFT JOIN accounts a ON a.customer_id = c.id "
            f"LEFT JOIN transactions t ON t.account_id = a.id AND t.txn_type = '{txn_type}' "
            f"AND t.txn_date >= '{start}' AND t.txn_date < '{end}' "
            "GROUP BY c.id ORDER BY c.id"
        )
        initial = reference.replace("LEFT JOIN", "JOIN").replace(
            f"JOIN transactions t ON t.account_id = a.id AND t.txn_type = '{txn_type}' "
            f"AND t.txn_date >= '{start}' AND t.txn_date < '{end}' ",
            f"JOIN transactions t ON t.account_id = a.id WHERE t.txn_type = '{txn_type}' "
            f"AND t.txn_date >= '{start}' AND t.txn_date < '{end}' ",
        )
        yield _make_task("join_type", sequence, "join_zero_count", question, initial, reference)

    for txn_type, (month, start, end) in product(TXN_TYPES, MONTHS):
        sequence += 1
        question = f"统计每位客户 {month} 的 {txn_type} 总金额，包括零金额客户，按客户编号排序。"
        reference = (
            "SELECT c.id, COALESCE(SUM(CASE WHEN t.txn_type = "
            f"'{txn_type}' AND t.txn_date >= '{start}' AND t.txn_date < '{end}' "
            "THEN t.amount ELSE 0 END), 0) AS total_amount FROM customers c "
            "LEFT JOIN accounts a ON a.customer_id = c.id "
            "LEFT JOIN transactions t ON t.account_id = a.id "
            "GROUP BY c.id ORDER BY c.id"
        )
        initial = (
            "SELECT c.id, COALESCE(SUM(t.amount), 0) AS total_amount FROM customers c "
            "LEFT JOIN accounts a ON a.customer_id = c.id "
            "LEFT JOIN transactions t ON t.account_id = a.id "
            f"WHERE t.txn_type = '{txn_type}' AND t.txn_date >= '{start}' "
            f"AND t.txn_date < '{end}' GROUP BY c.id ORDER BY c.id"
        )
        yield _make_task("join_type", sequence, "join_zero_sum", question, initial, reference)

    for region, account_type in product(REGIONS, ACCOUNT_TYPES):
        sequence += 1
        question = f"统计{region}每位客户的 {account_type} 账户数，包括零账户客户，按客户编号排序。"
        reference = (
            "SELECT c.id, COUNT(a.id) AS account_count FROM customers c "
            f"LEFT JOIN accounts a ON a.customer_id = c.id AND a.account_type = '{account_type}' "
            f"WHERE c.region = '{region}' GROUP BY c.id ORDER BY c.id"
        )
        initial = (
            "SELECT c.id, COUNT(a.id) AS account_count FROM customers c "
            f"JOIN accounts a ON a.customer_id = c.id WHERE c.region = '{region}' "
            f"AND a.account_type = '{account_type}' GROUP BY c.id ORDER BY c.id"
        )
        yield _make_task("join_type", sequence, "join_zero_accounts", question, initial, reference)

    for txn_type, (month, start, end) in product(TXN_TYPES, MONTHS):
        sequence += 1
        question = f"列出 {month} 从未发生 {txn_type} 交易的客户编号，按编号排序。"
        reference = (
            "SELECT c.id FROM customers c WHERE NOT EXISTS ("
            "SELECT 1 FROM accounts a JOIN transactions t ON t.account_id = a.id "
            f"WHERE a.customer_id = c.id AND t.txn_type = '{txn_type}' "
            f"AND t.txn_date >= '{start}' AND t.txn_date < '{end}') ORDER BY c.id"
        )
        initial = (
            "SELECT DISTINCT c.id FROM customers c JOIN accounts a ON a.customer_id = c.id "
            "JOIN transactions t ON t.account_id = a.id "
            f"WHERE t.txn_type != '{txn_type}' AND t.txn_date >= '{start}' "
            f"AND t.txn_date < '{end}' ORDER BY c.id"
        )
        yield _make_task("join_type", sequence, "join_not_exists", question, initial, reference)

    for account_type, threshold in product(ACCOUNT_TYPES, (0, 5000, 10000, 15000, 20000)):
        sequence += 1
        question = f"列出余额不低于 {threshold} 的 {account_type} 账户编号和客户姓名，按账户编号排序。"
        reference = (
            "SELECT a.id, c.name FROM accounts a "
            "JOIN customers c ON a.customer_id = c.id "
            f"WHERE a.account_type = '{account_type}' AND a.balance >= {threshold} ORDER BY a.id"
        )
        initial = reference.replace("a.customer_id = c.id", "a.id = c.id")
        yield _make_task("join_type", sequence, "join_wrong_key", question, initial, reference)

    for account_type, region in product(ACCOUNT_TYPES, REGIONS):
        sequence += 1
        question = f"列出{region}没有 {account_type} 账户的客户编号，按编号排序。"
        reference = (
            "SELECT c.id FROM customers c WHERE c.region = "
            f"'{region}' AND NOT EXISTS (SELECT 1 FROM accounts a WHERE a.customer_id = c.id "
            f"AND a.account_type = '{account_type}') ORDER BY c.id"
        )
        initial = (
            "SELECT DISTINCT c.id FROM customers c JOIN accounts a ON a.customer_id = c.id "
            f"WHERE c.region = '{region}' AND a.account_type != '{account_type}' ORDER BY c.id"
        )
        yield _make_task("join_type", sequence, "join_missing_relation", question, initial, reference)


def _aggregation_candidates() -> Iterable[DebugTask]:
    sequence = 0
    dimensions = (
        ("地区", "c.region", "region"),
        ("风险等级", "c.risk_level", "risk_level"),
        ("账户类型", "a.account_type", "account_type"),
    )
    filters = (("", ""), ("，仅计算余额大于5000的账户", " WHERE a.balance > 5000"),
               ("，仅计算余额大于10000的账户", " WHERE a.balance > 10000"),
               ("，仅计算余额大于15000的账户", " WHERE a.balance > 15000"))
    for (label, expression, alias), (question_filter, sql_filter) in product(dimensions, filters):
        sequence += 1
        question = f"按{label}统计账户余额总和{question_filter}，按{label}升序。"
        base = (
            f"SELECT {expression} AS {alias}, SUM(a.balance) AS total_balance "
            "FROM customers c JOIN accounts a ON a.customer_id = c.id"
        )
        reference = f"{base}{sql_filter} GROUP BY {expression} ORDER BY {alias}"
        initial = f"{base}{sql_filter} ORDER BY {alias}"
        yield _make_task("aggregation", sequence, "aggregation_missing_group", question, initial, reference)

    for txn_type, threshold in product(TXN_TYPES, (0, 100, 300, 600, 1000, 1500)):
        sequence += 1
        question = f"按月份统计金额大于 {threshold} 的 {txn_type} 交易总额，月份为 YYYY-MM，按月份升序。"
        reference = (
            "SELECT substr(txn_date, 1, 7) AS month, SUM(amount) AS total_amount "
            f"FROM transactions WHERE txn_type = '{txn_type}' AND amount > {threshold} "
            "GROUP BY substr(txn_date, 1, 7) ORDER BY month"
        )
        initial = reference.replace("substr(txn_date, 1, 7)", "substr(txn_date, 1, 4)")
        yield _make_task("aggregation", sequence, "aggregation_wrong_grain", question, initial, reference)

    for txn_type, (month, start, end) in product(TXN_TYPES, MONTHS):
        sequence += 1
        question = f"计算 {month} 每种账户类型的 {txn_type} 平均交易金额，按账户类型升序。"
        reference = (
            "SELECT a.account_type, AVG(t.amount) AS avg_amount FROM accounts a "
            "JOIN transactions t ON t.account_id = a.id "
            f"WHERE t.txn_type = '{txn_type}' AND t.txn_date >= '{start}' "
            f"AND t.txn_date < '{end}' GROUP BY a.account_type ORDER BY a.account_type"
        )
        initial = reference.replace("AVG(t.amount)", "SUM(t.amount)")
        yield _make_task("aggregation", sequence, "aggregation_wrong_function", question, initial, reference)


def _duplicate_counting_candidates() -> Iterable[DebugTask]:
    sequence = 0
    for txn_type, start_date in product(TXN_TYPES, ("2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01")):
        sequence += 1
        question = f"统计 {start_date} 及之后发生过 {txn_type} 交易的客户数量。"
        joins = (
            "FROM customers c JOIN accounts a ON a.customer_id = c.id "
            "JOIN transactions t ON t.account_id = a.id "
            f"WHERE t.txn_type = '{txn_type}' AND t.txn_date >= '{start_date}'"
        )
        reference = f"SELECT COUNT(DISTINCT c.id) AS customer_count {joins}"
        initial = f"SELECT COUNT(*) AS customer_count {joins}"
        yield _make_task("duplicate_counting", sequence, "distinct_customer_date", question, initial, reference)

    for region, txn_type in product(REGIONS, TXN_TYPES):
        sequence += 1
        question = f"统计{region}发生过 {txn_type} 交易的客户数量。"
        joins = (
            "FROM customers c JOIN accounts a ON a.customer_id = c.id "
            "JOIN transactions t ON t.account_id = a.id "
            f"WHERE c.region = '{region}' AND t.txn_type = '{txn_type}'"
        )
        reference = f"SELECT COUNT(DISTINCT c.id) AS customer_count {joins}"
        initial = f"SELECT COUNT(t.id) AS customer_count {joins}"
        yield _make_task("duplicate_counting", sequence, "distinct_customer_region", question, initial, reference)


def _filter_candidates() -> Iterable[DebugTask]:
    sequence = 0
    for expected, wrong in product(RISKS, RISKS):
        if expected == wrong:
            continue
        sequence += 1
        question = f"统计 {expected} 风险客户的账户余额总和。"
        reference = (
            "SELECT SUM(a.balance) AS total_balance FROM customers c "
            f"JOIN accounts a ON a.customer_id = c.id WHERE c.risk_level = '{expected}'"
        )
        initial = reference.replace(f"'{expected}'", f"'{wrong}'")
        yield _make_task("filter", sequence, "filter_wrong_risk", question, initial, reference)

    for expected, wrong in product(REGIONS, REGIONS):
        if expected == wrong:
            continue
        sequence += 1
        question = f"统计{expected}客户的账户数量。"
        reference = (
            "SELECT COUNT(a.id) AS account_count FROM customers c "
            f"JOIN accounts a ON a.customer_id = c.id WHERE c.region = '{expected}'"
        )
        initial = reference.replace(f"'{expected}'", f"'{wrong}'")
        yield _make_task("filter", sequence, "filter_wrong_region", question, initial, reference)

    for expected, wrong in (("checking", "saving"), ("saving", "checking")):
        sequence += 1
        question = f"统计 {expected} 账户的余额总和。"
        reference = f"SELECT SUM(balance) AS total_balance FROM accounts WHERE account_type = '{expected}'"
        initial = reference.replace(f"'{expected}'", f"'{wrong}'")
        yield _make_task("filter", sequence, "filter_wrong_account_type", question, initial, reference)

    for expected, wrong in (("debit", "credit"), ("credit", "debit")):
        for threshold in (0, 300, 800, 1500):
            sequence += 1
            question = f"统计金额大于 {threshold} 的 {expected} 交易总额。"
            reference = (
                f"SELECT SUM(amount) AS total_amount FROM transactions WHERE txn_type = '{expected}' "
                f"AND amount > {threshold}"
            )
            initial = reference.replace(f"'{expected}'", f"'{wrong}'")
            yield _make_task("filter", sequence, "filter_wrong_txn_type", question, initial, reference)

    for threshold in (100, 300, 500, 800, 1000, 1500, 2000, 3000):
        sequence += 1
        question = f"统计金额不低于 {threshold} 的交易数量。"
        reference = f"SELECT COUNT(*) AS txn_count FROM transactions WHERE amount >= {threshold}"
        initial = "SELECT COUNT(*) AS txn_count FROM transactions"
        yield _make_task("filter", sequence, "filter_missing_threshold", question, initial, reference)

    for region in REGIONS:
        sequence += 1
        question = f"统计{region}客户数量。"
        reference = f"SELECT COUNT(*) AS customer_count FROM customers WHERE region = '{region}'"
        initial = "SELECT COUNT(*) AS customer_count FROM customers"
        yield _make_task("filter", sequence, "filter_missing_region", question, initial, reference)

        sequence += 1
        question = f"计算{region}客户的平均账户余额。"
        reference = (
            "SELECT AVG(a.balance) AS avg_balance FROM customers c "
            f"JOIN accounts a ON a.customer_id = c.id WHERE c.region = '{region}'"
        )
        initial = (
            "SELECT AVG(a.balance) AS avg_balance FROM customers c "
            "JOIN accounts a ON a.customer_id = c.id"
        )
        yield _make_task("filter", sequence, "filter_missing_region_balance", question, initial, reference)

    for risk in RISKS:
        sequence += 1
        question = f"统计 {risk} 风险客户数量。"
        reference = f"SELECT COUNT(*) AS customer_count FROM customers WHERE risk_level = '{risk}'"
        initial = "SELECT COUNT(*) AS customer_count FROM customers"
        yield _make_task("filter", sequence, "filter_missing_risk", question, initial, reference)

        sequence += 1
        question = f"统计 {risk} 风险客户的账户余额总和。"
        reference = (
            "SELECT SUM(a.balance) AS total_balance FROM customers c "
            f"JOIN accounts a ON a.customer_id = c.id WHERE c.risk_level = '{risk}'"
        )
        initial = (
            "SELECT SUM(a.balance) AS total_balance FROM customers c "
            "JOIN accounts a ON a.customer_id = c.id"
        )
        yield _make_task("filter", sequence, "filter_missing_risk_balance", question, initial, reference)


def _date_candidates() -> Iterable[DebugTask]:
    sequence = 0
    for txn_type, (month, start, end) in product(TXN_TYPES, MONTHS):
        sequence += 1
        wrong_end = "2025-07-01" if end != "2025-07-01" else "2025-06-01"
        question = f"统计 {month} 的 {txn_type} 交易数量。"
        reference = (
            "SELECT COUNT(*) AS txn_count FROM transactions "
            f"WHERE txn_type = '{txn_type}' AND txn_date >= '{start}' AND txn_date < '{end}'"
        )
        initial = reference.replace(f"txn_date < '{end}'", f"txn_date < '{wrong_end}'")
        yield _make_task("date", sequence, "date_wrong_month_end", question, initial, reference)

    dates = ("2025-01-01", "2025-01-20", "2025-02-15", "2025-03-10", "2025-04-01", "2025-05-15")
    for txn_type, boundary in product(TXN_TYPES, dates):
        sequence += 1
        question = f"统计 {boundary} 及之后的 {txn_type} 交易数量。"
        reference = (
            "SELECT COUNT(*) AS txn_count FROM transactions "
            f"WHERE txn_type = '{txn_type}' AND txn_date >= '{boundary}'"
        )
        initial = reference.replace(f"txn_date >= '{boundary}'", "txn_date >= '2024-12-01'")
        yield _make_task("date", sequence, "date_wrong_start", question, initial, reference)

    for txn_type, threshold in product(TXN_TYPES, (0, 300, 800, 1500)):
        sequence += 1
        question = f"按月份统计金额大于 {threshold} 的 {txn_type} 交易数，月份为 YYYY-MM。"
        reference = (
            "SELECT substr(txn_date, 1, 7) AS month, COUNT(*) AS txn_count "
            f"FROM transactions WHERE txn_type = '{txn_type}' AND amount > {threshold} "
            "GROUP BY substr(txn_date, 1, 7) ORDER BY month"
        )
        initial = reference.replace("substr(txn_date, 1, 7)", "substr(txn_date, 1, 4)")
        yield _make_task("date", sequence, "date_month_granularity", question, initial, reference)


def _schema_candidates() -> Iterable[DebugTask]:
    sequence = 0
    mappings = (
        ("客户姓名", "customers c", "c.name", "c.customer_name"),
        ("客户地区", "customers c", "c.region", "c.area"),
        ("客户风险等级", "customers c", "c.risk_level", "c.risk"),
        ("账户余额", "accounts a", "a.balance", "a.amount"),
        ("账户类型", "accounts a", "a.account_type", "a.type"),
        ("交易日期", "transactions t", "t.txn_date", "t.transaction_date"),
        ("交易金额", "transactions t", "t.amount", "t.amounts"),
        ("交易类型", "transactions t", "t.txn_type", "t.type"),
    )
    for label, table, correct, wrong in mappings:
        sequence += 1
        question = f"列出所有{label}，按该字段升序。"
        reference = f"SELECT {correct} FROM {table} ORDER BY {correct}"
        initial = reference.replace(correct, wrong)
        yield _make_task("schema_linking", sequence, "schema_select", question, initial, reference)

        sequence += 1
        question = f"统计不同{label}的数量。"
        reference = f"SELECT COUNT(DISTINCT {correct}) AS value_count FROM {table}"
        initial = reference.replace(correct, wrong)
        yield _make_task("schema_linking", sequence, "schema_aggregate", question, initial, reference)

        sequence += 1
        question = f"查询{label}不为空的记录数量。"
        reference = f"SELECT COUNT(*) AS row_count FROM {table} WHERE {correct} IS NOT NULL"
        initial = reference.replace(correct, wrong)
        yield _make_task("schema_linking", sequence, "schema_filter", question, initial, reference)


def _syntax_candidates() -> Iterable[DebugTask]:
    cases = (
        ("统计客户数量。", "SELEC COUNT(*) AS customer_count FROM customers", "SELECT COUNT(*) AS customer_count FROM customers"),
        ("统计账户余额总和。", "SELECT SUM(balance) AS total_balance FORM accounts", "SELECT SUM(balance) AS total_balance FROM accounts"),
        ("统计 debit 交易数量。", "SELECT COUNT(*) AS txn_count FROM transactions WHER txn_type = 'debit'", "SELECT COUNT(*) AS txn_count FROM transactions WHERE txn_type = 'debit'"),
        ("按地区统计客户数量。", "SELECT region COUNT(*) AS customer_count FROM customers GROUP BY region", "SELECT region, COUNT(*) AS customer_count FROM customers GROUP BY region"),
        ("计算账户平均余额。", "SELECT AVG(balance AS avg_balance FROM accounts", "SELECT AVG(balance) AS avg_balance FROM accounts"),
        ("列出所有账户编号。", "SELECT id FRM accounts ORDER BY id", "SELECT id FROM accounts ORDER BY id"),
        ("统计 high 风险客户数量。", "SELECT COUNT(*) FROM customers WERE risk_level = 'high'", "SELECT COUNT(*) FROM customers WHERE risk_level = 'high'"),
        ("查询最大交易金额。", "SELECT MAX(amount)) AS max_amount FROM transactions", "SELECT MAX(amount) AS max_amount FROM transactions"),
        ("按类型统计账户数。", "SELECT account_type, COUNT(*) FROM accounts GROUP account_type", "SELECT account_type, COUNT(*) FROM accounts GROUP BY account_type"),
        ("列出客户姓名并按编号排序。", "SELECT name FROM customers ODER BY id", "SELECT name FROM customers ORDER BY id"),
    )
    for sequence, (question, initial, reference) in enumerate(cases, start=1):
        yield _make_task(
            "syntax_error", sequence, f"syntax_{sequence:02d}", question, initial, reference
        )
