from __future__ import annotations

from pathlib import Path
from typing import Any

from .database import get_schema


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "error_type": {
            "type": "string",
            "enum": [
                "syntax_error",
                "schema_linking",
                "aggregation",
                "duplicate_counting",
                "join_type",
                "filter",
                "date",
                "other",
            ],
        },
        "change_summary": {"type": "string"},
        "corrected_sql": {"type": "string"},
    },
    "required": ["error_type", "change_summary", "corrected_sql"],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTIONS = """你是一个金融分析场景的只读 SQL 调试助手。
你的唯一任务是根据用户问题、SQLite Schema、当前 SQL 和验证反馈，返回一条修正后的查询。
遵守以下边界：
1. 只允许 SELECT 或 WITH；不得生成写操作、DDL、PRAGMA、ATTACH 或多条语句。
2. 不要假设 Schema 中不存在的表或字段。
3. 保持用户要求的筛选条件、聚合粒度、排序和是否包含零记录对象。
4. 验证反馈可能只说明结果不一致；此时从问题语义检查 JOIN、COUNT DISTINCT、GROUP BY、日期边界和过滤条件。
5. corrected_sql 只能包含 SQL，不要使用 Markdown 代码块。
"""


def build_sql_repair_prompt(
    database_path: Path,
    question: str,
    sql: str,
    feedback: str,
) -> str:
    return (
        "<user_question>\n"
        f"{question}\n"
        "</user_question>\n\n"
        "<database_schema>\n"
        f"{get_schema(database_path)}\n"
        "</database_schema>\n\n"
        "<current_sql>\n"
        f"{sql}\n"
        "</current_sql>\n\n"
        "<verification_feedback>\n"
        f"{feedback}\n"
        "</verification_feedback>"
    )
