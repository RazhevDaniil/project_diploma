"""LLM Client — abstracts GigaChat and OpenAI-compatible APIs."""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI

import config as cfg

logger = logging.getLogger(__name__)


def _get_gigachat(temperature: float, max_tokens: int) -> BaseChatModel:
    from langchain_gigachat import GigaChat

    return GigaChat(
        credentials=cfg.GIGACHAT_CREDENTIALS,
        model=cfg.GIGACHAT_MODEL,
        scope=cfg.GIGACHAT_SCOPE,
        temperature=temperature,
        max_tokens=max_tokens,
        verify_ssl_certs=False,
        timeout=600,
    )


def _get_openai_client() -> OpenAI:
    """Return an OpenAI-compatible client configured by env/runtime settings."""
    return OpenAI(
        api_key=cfg.OPENAI_API_KEY,
        base_url=cfg.OPENAI_API_BASE,
    )


def _call_gigachat(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call GigaChat through LangChain and return plain text."""
    llm = _get_gigachat(temperature=temperature, max_tokens=max_tokens)
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    return response.content


def _call_openai_compatible(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call an OpenAI-compatible API via the official OpenAI client."""
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
            if cfg.LLM_PROVIDER == "gigachat":
                response = _call_gigachat(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = _call_openai_compatible(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            # Small delay between successful calls to avoid rate limiting
            time.sleep(1)
            return response
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 5 * attempt  # 5s, 10s, 15s
            logger.warning("LLM call failed (attempt %d/%d), retrying in %ds: %s", attempt, max_retries, wait, e)
            time.sleep(wait)


def _extract_json(text: str) -> dict | list | None:
    """Try multiple strategies to extract JSON from LLM response."""
    import re

    text = text.strip()

    # Strategy 1: markdown code block ```json ... ```
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    # Strategy 2: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first [ ... ] or { ... } in the text
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    # Strategy 4: try to fix common issues — trailing commas, missing brackets
    cleaned = text.strip()
    if cleaned.startswith("[") and not cleaned.endswith("]"):
        cleaned += "]"
    elif cleaned.startswith("{") and not cleaned.endswith("}"):
        cleaned += "}"
    # Remove trailing commas before ] or }
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
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

    logger.warning("Failed to parse LLM JSON response, returning raw text")
    logger.debug("Raw LLM response: %s", raw[:500])
    return {"raw": raw}
