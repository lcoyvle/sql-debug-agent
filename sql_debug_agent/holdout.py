from __future__ import annotations

from pathlib import Path

from .dataset import DebugTask, write_tasks
from .verifier import SQLVerifier


def build_final_holdout(database_path: Path, output_path: Path) -> tuple[Path, list[DebugTask]]:
    """Build a frozen final set that is not used for V2 data design or training."""
    tasks = _holdout_tasks()
    verifier = SQLVerifier(database_path)
    for task in tasks:
        reference = verifier.execute(task.reference_sql)
        if not reference.executable:
            raise RuntimeError(f"{task.task_id} 参考 SQL 不可执行：{reference.error}")
        if verifier.verify(task.initial_sql, task.reference_sql).passed:
            raise RuntimeError(f"{task.task_id} 不是有效 Bad Case")
    return write_tasks(output_path, tasks), tasks


def _holdout_tasks() -> list[DebugTask]:
    rows = [
        # Syntax: new typo shapes, not copied from the development set.
        ("syntax_error", "统计客户的平均编号。", "SELCT AVG(id) FROM customers", "SELECT AVG(id) FROM customers"),
        ("syntax_error", "查询最小账户余额。", "SELECT MIN(balance) FRO accounts", "SELECT MIN(balance) FROM accounts"),
        ("syntax_error", "按风险等级统计客户数。", "SELECT risk_level COUNT(*) FROM customers GROUP BY risk_level", "SELECT risk_level, COUNT(*) FROM customers GROUP BY risk_level"),
        ("syntax_error", "列出交易编号并排序。", "SELECT id FROM transactions ORDER id", "SELECT id FROM transactions ORDER BY id"),
        ("syntax_error", "计算 credit 交易平均金额。", "SELECT AVG(amount) FROM transactions WHERE txn_type == 'credit", "SELECT AVG(amount) FROM transactions WHERE txn_type = 'credit'"),
        # Schema linking.
        ("schema_linking", "按客户编号列出客户姓名。", "SELECT c.full_name FROM customers c ORDER BY c.id", "SELECT c.name FROM customers c ORDER BY c.id"),
        ("schema_linking", "查询账户的当前余额并按账户编号排序。", "SELECT a.current_balance FROM accounts a ORDER BY a.id", "SELECT a.balance FROM accounts a ORDER BY a.id"),
        ("schema_linking", "统计交易金额大于 500 的记录数。", "SELECT COUNT(*) FROM transactions t WHERE t.value > 500", "SELECT COUNT(*) FROM transactions t WHERE t.amount > 500"),
        ("schema_linking", "统计 2025-01-08 及之后的交易数。", "SELECT COUNT(*) FROM transactions t WHERE t.posted_date >= '2025-01-08'", "SELECT COUNT(*) FROM transactions t WHERE t.txn_date >= '2025-01-08'"),
        ("schema_linking", "统计 saving 类型账户数。", "SELECT COUNT(*) FROM accounts a WHERE a.kind = 'saving'", "SELECT COUNT(*) FROM accounts a WHERE a.account_type = 'saving'"),
        # Aggregation and counting grain.
        ("aggregation", "按风险等级统计客户数量并排序。", "SELECT risk_level, COUNT(*) FROM customers ORDER BY risk_level", "SELECT risk_level, COUNT(*) FROM customers GROUP BY risk_level ORDER BY risk_level"),
        ("aggregation", "按地区计算客户账户的平均余额。", "SELECT c.region, SUM(a.balance) FROM customers c JOIN accounts a ON a.customer_id = c.id GROUP BY c.region ORDER BY c.region", "SELECT c.region, AVG(a.balance) FROM customers c JOIN accounts a ON a.customer_id = c.id GROUP BY c.region ORDER BY c.region"),
        ("aggregation", "统计每个账户的交易数，包括零交易账户。", "SELECT a.id, COUNT(*) FROM accounts a LEFT JOIN transactions t ON t.account_id = a.id GROUP BY a.id ORDER BY a.id", "SELECT a.id, COUNT(t.id) FROM accounts a LEFT JOIN transactions t ON t.account_id = a.id GROUP BY a.id ORDER BY a.id"),
        ("aggregation", "按月份统计交易平均金额。", "SELECT substr(txn_date,1,4), AVG(amount) FROM transactions GROUP BY substr(txn_date,1,4) ORDER BY 1", "SELECT substr(txn_date,1,7), AVG(amount) FROM transactions GROUP BY substr(txn_date,1,7) ORDER BY 1"),
        ("aggregation", "统计发生过交易的账户数量。", "SELECT COUNT(t.id) FROM accounts a JOIN transactions t ON t.account_id = a.id", "SELECT COUNT(DISTINCT a.id) FROM accounts a JOIN transactions t ON t.account_id = a.id"),
        # Join semantics.
        ("join_type", "统计每位客户的 credit 总额，包括零金额客户。", "SELECT c.id, COALESCE(SUM(t.amount),0) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id LEFT JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='credit' GROUP BY c.id ORDER BY c.id", "SELECT c.id, COALESCE(SUM(CASE WHEN t.txn_type='credit' THEN t.amount ELSE 0 END),0) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id LEFT JOIN transactions t ON t.account_id=a.id GROUP BY c.id ORDER BY c.id"),
        ("join_type", "统计每个账户的 debit 交易数，包括零次账户。", "SELECT a.id, COUNT(t.id) FROM accounts a JOIN transactions t ON t.account_id=a.id AND t.txn_type='debit' GROUP BY a.id ORDER BY a.id", "SELECT a.id, COUNT(t.id) FROM accounts a LEFT JOIN transactions t ON t.account_id=a.id AND t.txn_type='debit' GROUP BY a.id ORDER BY a.id"),
        ("join_type", "列出从未发生 credit 交易的客户。", "SELECT DISTINCT c.id FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.txn_type!='credit' ORDER BY c.id", "SELECT c.id FROM customers c WHERE NOT EXISTS (SELECT 1 FROM accounts a JOIN transactions t ON t.account_id=a.id WHERE a.customer_id=c.id AND t.txn_type='credit') ORDER BY c.id"),
        ("join_type", "列出账户编号及所属客户姓名。", "SELECT a.id,c.name FROM accounts a JOIN customers c ON a.id=c.id ORDER BY a.id", "SELECT a.id,c.name FROM accounts a JOIN customers c ON a.customer_id=c.id ORDER BY a.id"),
        ("join_type", "统计每位客户的 saving 账户数，包括零账户客户。", "SELECT c.id,COUNT(a.id) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE a.account_type='saving' GROUP BY c.id ORDER BY c.id", "SELECT c.id,COUNT(a.id) FROM customers c LEFT JOIN accounts a ON a.customer_id=c.id AND a.account_type='saving' GROUP BY c.id ORDER BY c.id"),
        # Filters.
        ("filter", "计算 high 风险客户的平均账户余额。", "SELECT AVG(a.balance) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE c.risk_level='medium'", "SELECT AVG(a.balance) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE c.risk_level='high'"),
        ("filter", "统计金额超过 400 的 credit 交易数。", "SELECT COUNT(*) FROM transactions WHERE amount>400", "SELECT COUNT(*) FROM transactions WHERE amount>400 AND txn_type='credit'"),
        ("filter", "统计华东客户的 saving 账户数。", "SELECT COUNT(a.id) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE c.region='华东'", "SELECT COUNT(a.id) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE c.region='华东' AND a.account_type='saving'"),
        ("filter", "统计金额不超过 700 的交易数。", "SELECT COUNT(*) FROM transactions WHERE amount<700", "SELECT COUNT(*) FROM transactions WHERE amount<=700"),
        ("filter", "统计 medium 风险客户的账户数。", "SELECT COUNT(a.id) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE c.risk_level='low'", "SELECT COUNT(a.id) FROM customers c JOIN accounts a ON a.customer_id=c.id WHERE c.risk_level='medium'"),
        # Date boundaries.
        ("date", "统计 2025-01-08 到 2025-02-01 之前的交易数。", "SELECT COUNT(*) FROM transactions WHERE txn_date>='2025-01-01' AND txn_date<'2025-02-01'", "SELECT COUNT(*) FROM transactions WHERE txn_date>='2025-01-08' AND txn_date<'2025-02-01'"),
        ("date", "统计 2025-02-01 及之后的 debit 交易数。", "SELECT COUNT(*) FROM transactions WHERE txn_type='debit' AND txn_date>='2025-01-01'", "SELECT COUNT(*) FROM transactions WHERE txn_type='debit' AND txn_date>='2025-02-01'"),
        ("date", "统计 2025-01-31 之前的交易数。", "SELECT COUNT(*) FROM transactions WHERE txn_date<='2025-01-31'", "SELECT COUNT(*) FROM transactions WHERE txn_date<'2025-01-31'"),
        ("date", "统计 2025 年 3 月 credit 交易总额。", "SELECT SUM(amount) FROM transactions WHERE txn_type='credit' AND txn_date>='2025-01-01' AND txn_date<'2025-04-01'", "SELECT SUM(amount) FROM transactions WHERE txn_type='credit' AND txn_date>='2025-03-01' AND txn_date<'2025-04-01'"),
        ("date", "统计 2025-01-20 当天的交易数。", "SELECT COUNT(*) FROM transactions WHERE txn_date>='2025-01-20'", "SELECT COUNT(*) FROM transactions WHERE txn_date='2025-01-20'"),
        # Duplicate counting.
        ("duplicate_counting", "统计发生过交易的客户数。", "SELECT COUNT(t.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id", "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id"),
        ("duplicate_counting", "统计发生过 debit 交易的账户数。", "SELECT COUNT(t.id) FROM accounts a JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='debit'", "SELECT COUNT(DISTINCT a.id) FROM accounts a JOIN transactions t ON t.account_id=a.id WHERE t.txn_type='debit'"),
        ("duplicate_counting", "统计 2025 年 1 月有交易的客户数。", "SELECT COUNT(*) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.txn_date>='2025-01-01' AND t.txn_date<'2025-02-01'", "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.txn_date>='2025-01-01' AND t.txn_date<'2025-02-01'"),
        ("duplicate_counting", "统计有 checking 账户且发生过交易的客户数。", "SELECT COUNT(c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE a.account_type='checking'", "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE a.account_type='checking'"),
        ("duplicate_counting", "统计发生过金额大于 500 交易的客户数。", "SELECT COUNT(t.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.amount>500", "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN accounts a ON a.customer_id=c.id JOIN transactions t ON t.account_id=a.id WHERE t.amount>500"),
    ]
    counters: dict[str, int] = {}
    tasks: list[DebugTask] = []
    for error_type, question, initial_sql, reference_sql in rows:
        counters[error_type] = counters.get(error_type, 0) + 1
        tasks.append(
            DebugTask(
                task_id=f"holdout_{error_type}_{counters[error_type]:02d}",
                question=question,
                initial_sql=initial_sql,
                reference_sql=reference_sql,
                error_type=error_type,
            )
        )
    return tasks
