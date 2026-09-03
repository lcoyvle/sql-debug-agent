from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .model_errors import ModelAPIError, ModelConfigurationError
from .model_prompt import RESPONSE_SCHEMA, SYSTEM_INSTRUCTIONS, build_sql_repair_prompt


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.6-luna"

Transport = Callable[[urllib.request.Request, float], bytes]


def _default_transport(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


@dataclass
class OpenAIResponsesClient:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 60.0
    transport: Transport = _default_transport

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
    ) -> "OpenAIResponsesClient":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ModelConfigurationError(
                "没有找到 OPENAI_API_KEY。请先在终端设置环境变量；不要把密钥写进代码或提交到 Git。"
            )
        return cls(
            api_key=api_key,
            model=model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        )

    def create_sql_repair(self, prompt: str) -> dict[str, str]:
        payload = {
            "model": self.model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "sql_repair",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                }
            },
            "reasoning": {"effort": "low"},
            "max_output_tokens": 600,
            "store": False,
        }
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            raw_response = self.transport(request, self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelAPIError(f"模型 API 返回 HTTP {exc.code}：{body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise ModelAPIError(f"无法连接模型 API：{exc.reason}") from exc
        except TimeoutError as exc:
            raise ModelAPIError("模型 API 请求超时") from exc

        try:
            response_payload = json.loads(raw_response.decode("utf-8"))
            output_text = _extract_output_text(response_payload)
            decision = json.loads(output_text)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelAPIError("模型返回内容无法解析为 SQL 修复结果") from exc

        corrected_sql = decision.get("corrected_sql")
        if not isinstance(corrected_sql, str) or not corrected_sql.strip():
            raise ModelAPIError("模型返回结果缺少 corrected_sql")
        return {
            "error_type": str(decision.get("error_type", "other")),
            "change_summary": str(decision.get("change_summary", "")),
            "corrected_sql": corrected_sql.strip(),
        }


def _extract_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text:
                    return text
    raise KeyError("output_text")


@dataclass
class OpenAIRepairer:
    database_path: Path
    client: OpenAIResponsesClient

    def repair(self, question: str, sql: str, feedback: str) -> str:
        prompt = build_sql_repair_prompt(
            self.database_path, question, sql, feedback
        )
        return self.client.create_sql_repair(prompt)["corrected_sql"]
