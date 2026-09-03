from __future__ import annotations

import re


class UnsafeQueryError(ValueError):
    """Raised when a query is outside the read-only project scope."""


FORBIDDEN_KEYWORDS = {
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "UPDATE",
    "VACUUM",
}


def validate_read_only_sql(sql: str) -> str:
    cleaned = sql.strip()
    if not cleaned:
        raise UnsafeQueryError("SQL 不能为空")

    without_last_semicolon = cleaned[:-1] if cleaned.endswith(";") else cleaned
    if ";" in without_last_semicolon:
        raise UnsafeQueryError("只允许执行一条 SQL")

    first_word_match = re.match(r"^[\s(]*([A-Za-z]+)", without_last_semicolon)
    first_word = first_word_match.group(1).upper() if first_word_match else ""
    if first_word not in {"SELECT", "WITH"}:
        raise UnsafeQueryError("只允许 SELECT 或 WITH 查询")

    tokens = set(re.findall(r"\b[A-Za-z]+\b", without_last_semicolon.upper()))
    forbidden = sorted(tokens & FORBIDDEN_KEYWORDS)
    if forbidden:
        raise UnsafeQueryError(f"检测到禁止关键字：{', '.join(forbidden)}")
    return without_last_semicolon
