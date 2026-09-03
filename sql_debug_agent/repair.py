from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .database import get_columns


class Repairer(Protocol):
    def repair(self, question: str, sql: str, feedback: str) -> str:
        """Return one revised SQL candidate."""


@dataclass
class RuleBasedRepairer:
    """A transparent offline baseline, not a substitute for a trained model."""

    database_path: Path

    def repair(self, question: str, sql: str, feedback: str) -> str:
        revised = sql.strip()

        if re.search(r"\bFORM\b", revised, flags=re.IGNORECASE):
            return re.sub(r"\bFORM\b", "FROM", revised, flags=re.IGNORECASE)

        missing_column = re.search(r"no such column:\s*([\w.]+)", feedback, re.I)
        if missing_column:
            qualified_name = missing_column.group(1)
            prefix, _, bare_name = qualified_name.rpartition(".")
            matches = difflib.get_close_matches(
                bare_name, get_columns(self.database_path), n=1, cutoff=0.6
            )
            if matches:
                replacement = f"{prefix}.{matches[0]}" if prefix else matches[0]
                return re.sub(
                    rf"\b{re.escape(qualified_name)}\b",
                    replacement,
                    revised,
                    flags=re.I,
                )

        if "客户数量" in question and re.search(r"COUNT\s*\(\s*\*\s*\)", revised, re.I):
            return re.sub(
                r"COUNT\s*\(\s*\*\s*\)",
                "COUNT(DISTINCT c.id)",
                revised,
                count=1,
                flags=re.I,
            )

        if "包括没有交易" in question and "LEFT JOIN" not in revised.upper():
            return re.sub(r"\bJOIN\b", "LEFT JOIN", revised, flags=re.I)

        per_group = any(term in question for term in ("每个地区", "每位客户", "每个风险等级"))
        has_aggregate = bool(re.search(r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(", revised, re.I))
        if per_group and has_aggregate and "GROUP BY" not in revised.upper():
            select_match = re.search(r"\bSELECT\s+(.+?)\s+FROM\b", revised, re.I | re.S)
            if select_match:
                group_expression = select_match.group(1).split(",", 1)[0].strip()
                insertion = re.search(r"\b(ORDER\s+BY|LIMIT)\b", revised, re.I)
                index = insertion.start() if insertion else len(revised.rstrip(";"))
                suffix = revised[index:].lstrip()
                prefix = revised[:index].rstrip().rstrip(";")
                return f"{prefix} GROUP BY {group_expression} {suffix}".strip()

        return revised
