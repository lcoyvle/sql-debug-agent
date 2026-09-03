from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .model_errors import ModelAPIError
from .model_prompt import RESPONSE_SCHEMA, SYSTEM_INSTRUCTIONS, build_sql_repair_prompt


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:1.5b-instruct"

Transport = Callable[[urllib.request.Request, float], bytes]


def _default_transport(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


@dataclass
class OllamaChatClient:
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout: float = 120.0
    transport: Transport = _default_transport

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
    ) -> "OllamaChatClient":
        return cls(
            model=model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=base_url
            or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        )

    def create_sql_repair(self, prompt: str) -> dict[str, str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw_response = self.transport(request, self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                raise ModelAPIError(
                    f"Ollama 找不到模型 {self.model!r}。请先运行：ollama pull {self.model}"
                ) from exc
            raise ModelAPIError(f"Ollama 返回 HTTP {exc.code}：{body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise ModelAPIError(
                "无法连接本地 Ollama。请确认 Ollama 已安装并正在运行；"
                f"当前地址：{self.base_url}"
            ) from exc
        except TimeoutError as exc:
            raise ModelAPIError("Ollama 请求超时，模型首次加载可能需要更长时间") from exc

        try:
            response_payload = json.loads(raw_response.decode("utf-8"))
            if response_payload.get("error"):
                raise ModelAPIError(f"Ollama 调用失败：{response_payload['error']}")
            message = response_payload["message"]
            content = message["content"]
            decision = content if isinstance(content, dict) else json.loads(content)
        except ModelAPIError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelAPIError("Ollama 返回内容无法解析为 SQL 修复结果") from exc

        corrected_sql = decision.get("corrected_sql")
        if not isinstance(corrected_sql, str) or not corrected_sql.strip():
            raise ModelAPIError("Ollama 返回结果缺少 corrected_sql")
        return {
            "error_type": str(decision.get("error_type", "other")),
            "change_summary": str(decision.get("change_summary", "")),
            "corrected_sql": corrected_sql.strip(),
        }


@dataclass
class OllamaRepairer:
    database_path: Path
    client: OllamaChatClient

    def repair(self, question: str, sql: str, feedback: str) -> str:
        prompt = build_sql_repair_prompt(
            self.database_path, question, sql, feedback
        )
        return self.client.create_sql_repair(prompt)["corrected_sql"]
