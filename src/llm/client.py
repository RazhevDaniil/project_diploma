"""LLM client for Cloud.ru Foundation Models via OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from typing import Optional

from openai import OpenAI

import config as cfg

logger = logging.getLogger(__name__)


def _get_openai_client() -> OpenAI:
    """Return a Foundation Models client configured by env/runtime settings."""
    return OpenAI(
        api_key=cfg.OPENAI_API_KEY,
        base_url=cfg.OPENAI_API_BASE,
    )


def _call_foundation_models(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Cloud.ru Foundation Models via the official OpenAI client."""
    client = _get_openai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=cfg.OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content if response.choices else ""
    if content is None:
        return ""
    return content


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.05,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> str:
    """Simple helper: send a prompt, get a string back. Retries on timeout."""
    import time

    for attempt in range(1, max_retries + 1):
        try:
            response = _call_foundation_models(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            time.sleep(1)
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
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> dict | list:
    """Call LLM and parse the response as JSON."""
    raw = call_llm(prompt, system_prompt, temperature, max_tokens)

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
