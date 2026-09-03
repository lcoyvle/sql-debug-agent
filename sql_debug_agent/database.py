from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path


DEMO_DB_VERSION = 2
TRAIN_DB_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    risk_level TEXT NOT NULL
);

CREATE TABLE accounts (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    account_type TEXT NOT NULL,
    balance REAL NOT NULL
);

CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    txn_date TEXT NOT NULL,
    txn_type TEXT NOT NULL CHECK (txn_type IN ('debit', 'credit')),
    amount REAL NOT NULL
);
"""

SEED_SQL = """
INSERT INTO customers VALUES
    (1, '张伟', '华东', 'low'),
    (2, '李娜', '华南', 'medium'),
    (3, '王强', '华东', 'high'),
    (4, '赵敏', '华北', 'low'),
    (5, '陈晨', '西南', 'medium');

INSERT INTO accounts VALUES
    (101, 1, 'checking', 12000.0),
    (102, 1, 'saving', 30000.0),
    (201, 2, 'checking', 8000.0),
    (301, 3, 'checking', 5000.0),
    (401, 4, 'saving', 18000.0);

INSERT INTO transactions VALUES
    (1, 101, '2025-01-03', 'debit', 300.0),
    (2, 101, '2025-01-08', 'credit', 2000.0),
    (3, 102, '2025-02-01', 'debit', 700.0),
    (4, 201, '2025-01-12', 'debit', 1200.0),
    (5, 201, '2025-03-05', 'credit', 500.0),
    (6, 301, '2025-01-20', 'debit', 2000.0),
    (7, 301, '2025-01-21', 'debit', 1000.0),
    (8, 201, '2025-01-31', 'debit', 400.0);
"""


def create_demo_database(path: Path) -> Path:
    """Create a deterministic synthetic finance database."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA_SQL)
        connection.executescript(SEED_SQL)
        connection.execute(f"PRAGMA user_version = {DEMO_DB_VERSION}")
        connection.commit()
    finally:
        connection.close()
    return path


def create_training_database(path: Path) -> Path:
    """Create a larger deterministic database used only for data construction."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    regions = ("华东", "华南", "华北", "西南", "东北")
    risks = ("low", "medium", "high")
    customers = [
        (customer_id, f"训练客户{customer_id:02d}", regions[(customer_id - 1) % 5], risks[(customer_id - 1) % 3])
        for customer_id in range(1, 19)
    ]

    accounts: list[tuple[int, int, str, float]] = []
    for customer_id in range(1, 16):
        account_count = 2 if customer_id % 3 == 0 else 1
        for position in range(account_count):
            account_id = customer_id * 100 + position + 1
            account_type = "checking" if (customer_id + position) % 2 else "saving"
            balance = float(1_500 + customer_id * 1_375 + position * 4_250)
            accounts.append((account_id, customer_id, account_type, balance))

    transactions: list[tuple[int, int, str, str, float]] = []
    transaction_id = 1
    start = date(2024, 12, 15)
    for account_index, (account_id, _, _, _) in enumerate(accounts):
        # Several accounts intentionally have no transactions so LEFT JOIN cases
        # have a different answer from INNER JOIN cases.
        if account_index % 7 == 0:
            continue
        transaction_count = 3 + account_index % 5
        for position in range(transaction_count):
            txn_date = start + timedelta(days=(account_index * 11 + position * 17) % 200)
            txn_type = "debit" if (account_index + position) % 3 else "credit"
            amount = float(75 + ((account_index + 2) * (position + 3) * 47) % 4_800)
            transactions.append(
                (transaction_id, account_id, txn_date.isoformat(), txn_type, amount)
            )
            transaction_id += 1

    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA_SQL)
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)
        connection.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?)", accounts)
        connection.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?)", transactions)
        connection.execute(f"PRAGMA user_version = {TRAIN_DB_VERSION}")
        connection.commit()
    finally:
        connection.close()
    return path


def get_database_version(path: Path) -> int:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


def get_schema(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    finally:
        connection.close()
    return "\n\n".join(sql for _, sql in rows if sql)


def get_columns(path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        table_names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if not row[0].startswith("sqlite_")
        ]
        columns: list[str] = []
        for table_name in table_names:
            safe_name = table_name.replace('"', '""')
            for row in connection.execute(f'PRAGMA table_info("{safe_name}")'):
                columns.append(str(row[1]))
        return sorted(set(columns))
    finally:
        connection.close()
