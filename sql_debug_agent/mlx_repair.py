from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .database import get_schema
from .model_errors import ModelConfigurationError
from .mlx_training import DEFAULT_MLX_MODEL
from .preparation import SYSTEM_PROMPT, build_sft_user_content


Loader = Callable[..., tuple[Any, Any]]
Generator = Callable[..., Any]


@dataclass
class MLXGeneratorClient:
    model_name: str = DEFAULT_MLX_MODEL
    adapter_path: Path | None = None
    max_tokens: int = 256
    cache_dir: Path | None = None
    loader: Loader | None = None
    generator: Generator | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def create_sql_repair(self, messages: list[dict[str, str]]) -> str:
        self._ensure_loaded()
        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        result = self.generator(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
        )
        if isinstance(result, str):
            text = result
        elif hasattr(result, "__iter__"):
            chunks: list[str] = []
            for response in result:
                chunks.append(getattr(response, "text", str(response)))
                partial = "".join(chunks)
                sql_start = re.search(r"\b(SELECT|WITH)\b", partial, re.I)
                sql_text = partial[sql_start.start() :] if sql_start else ""
                if sql_start and (";" in sql_text or "\n" in sql_text):
                    break
            text = "".join(chunks)
        else:
            text = getattr(result, "text", str(result))
        return extract_sql(text)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(self.cache_dir.resolve())
        if self.loader is None or self.generator is None:
            try:
                from mlx_lm import load, stream_generate
            except ImportError as exc:
                raise ModelConfigurationError(
                    "当前 Python 没有安装 MLX-LM，请使用项目 .venv 运行。"
                ) from exc
            self.loader = load
            self.generator = stream_generate
        adapter = str(self.adapter_path.resolve()) if self.adapter_path else None
        self._model, self._tokenizer = self.loader(
            self.model_name, adapter_path=adapter
        )


@dataclass
class MLXRepairer:
    database_path: Path
    client: MLXGeneratorClient

    def repair(self, question: str, sql: str, feedback: str) -> str:
        user_content = build_sft_user_content(
            question, get_schema(self.database_path), sql, feedback
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.client.create_sql_repair(messages)


def extract_sql(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.I | re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    match = re.search(r"\b(SELECT|WITH)\b", cleaned, re.I)
    if match:
        cleaned = cleaned[match.start() :]
    cleaned = re.sub(r"<\|[^>]+\|>", "", cleaned)
    if ";" in cleaned:
        cleaned = cleaned.split(";", 1)[0] + ";"
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0]
    cleaned = re.sub(r"(?:\s*[!！]\s*)+$", "", cleaned)
    return cleaned.strip()
