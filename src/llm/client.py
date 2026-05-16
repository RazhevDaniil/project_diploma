"""LLM client for Cloud.ru Foundation Models via OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from src.runtime_config import RuntimeSettings, build_runtime_settings

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


def _get_openai_client(settings: RuntimeSettings) -> "OpenAI":
    """Return a Foundation Models client configured by env/runtime settings."""
    from openai import OpenAI

    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
    )


def _call_foundation_models(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
    settings: RuntimeSettings,
) -> str:
    """Call Cloud.ru Foundation Models via the official OpenAI client."""
    client = _get_openai_client(settings)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content if response.choices else ""
    if content is None:
        return ""
    return content


def _llm_cache_key(
    system_prompt: Optional[str],
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update((system_prompt or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update(prompt.encode("utf-8"))
    h.update(b"\x1f")
    h.update(model.encode("utf-8"))
    h.update(b"\x1f")
    h.update(f"{temperature:.4f}".encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(max_tokens).encode("utf-8"))
    return h.hexdigest()


def _llm_cache_dir() -> "Path":
    import os
    from pathlib import Path
    base = os.getenv("LLM_CACHE_DIR")
    if base:
        return Path(base)
    # default: <project_root>/llm_cache, with project_root = .../project_diploma
    here = Path(__file__).resolve().parents[2]
    return here / "llm_cache"


def _llm_cache_enabled() -> bool:
    import os
    return os.getenv("LLM_CACHE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def _llm_cache_get(key: str) -> Optional[str]:
    if not _llm_cache_enabled():
        return None
    try:
        p = _llm_cache_dir() / f"{key}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception as exc:
        logger.debug("LLM cache get failed: %s", exc)
    return None


def _llm_cache_put(key: str, value: str) -> None:
    if not _llm_cache_enabled():
        return
    try:
        d = _llm_cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.txt").write_text(value, encoding="utf-8")
    except Exception as exc:
        logger.debug("LLM cache put failed: %s", exc)


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float | None = None,
    max_tokens: int = 4096,
    max_retries: int = 3,
    settings: RuntimeSettings | dict | None = None,
) -> str:
    """Simple helper: send a prompt, get a string back. Retries on timeout.

    Опциональный файловый кеш ответов: включается LLM_CACHE_ENABLED=true.
    Назначение — детерминированные итерации в разработке/eval: ключ
    хеширует (system_prompt, prompt, model, temperature, max_tokens), так
    что повторный запуск с тем же кодом и теми же требованиями даёт
    идентичный verdict. В production кеш по умолчанию выключен."""
    import time

    runtime_settings = build_runtime_settings(settings)
    effective_temperature = runtime_settings.openai_temperature if temperature is None else temperature

    cache_key = _llm_cache_key(
        system_prompt=system_prompt,
        prompt=prompt,
        model=runtime_settings.openai_model,
        temperature=effective_temperature,
        max_tokens=max_tokens,
    )
    cached = _llm_cache_get(cache_key)
    if cached is not None:
        return cached

    for attempt in range(1, max_retries + 1):
        try:
            response = _call_foundation_models(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                settings=runtime_settings,
            )
            if runtime_settings.llm_request_delay > 0:
                time.sleep(runtime_settings.llm_request_delay)
            _llm_cache_put(cache_key, response)
            return response
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 5 * attempt
            logger.warning("LLM call failed (attempt %d/%d), retrying in %ds: %s", attempt, max_retries, wait, e)
            time.sleep(wait)


def _extract_json(text: str) -> dict | list | None:
    """Try multiple strategies to extract JSON from LLM response."""
    import re

    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    cleaned = text.strip()
    if cleaned.startswith("[") and not cleaned.endswith("]"):
        cleaned += "]"
    elif cleaned.startswith("{") and not cleaned.endswith("}"):
        cleaned += "}"
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


def call_llm_json(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float | None = None,
    max_tokens: int = 4096,
    settings: RuntimeSettings | dict | None = None,
) -> dict | list:
    """Call LLM and parse the response as JSON."""
    runtime_settings = build_runtime_settings(settings)
    raw = call_llm(prompt, system_prompt, temperature, max_tokens, settings=runtime_settings)

    result = _extract_json(raw)
    if result is not None:
        return result

    repair_prompt = f"""Преобразуй следующий ответ модели в валидный JSON.

Правила:
- Верни только JSON без markdown и пояснений.
- Не добавляй новые факты.
- Сохрани все элементы и поля, которые можно восстановить из ответа.
- Если ответ содержит список объектов, верни JSON-массив.
- Если восстановить структуру невозможно, верни пустой JSON-массив [].

Исходный ответ:
---
{raw}
---
"""
    try:
        repaired_raw = call_llm(
            repair_prompt,
            system_prompt="Ты исправляешь поврежденный JSON. Отвечай только валидным JSON.",
            temperature=0,
            max_tokens=max_tokens,
            max_retries=2,
            settings=runtime_settings,
        )
        repaired = _extract_json(repaired_raw)
        if repaired is not None:
            logger.info("Successfully repaired LLM JSON response")
            return repaired
    except Exception as exc:
        logger.warning("Failed to repair LLM JSON response: %s", exc)

    logger.warning("Failed to parse LLM JSON response, returning raw text")
    logger.debug("Raw LLM response: %s", raw[:500])
    return {"raw": raw}
