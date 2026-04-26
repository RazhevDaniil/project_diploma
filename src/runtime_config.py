"""Helpers for applying runtime LLM settings from the UI/API."""

from __future__ import annotations

import logging

import config as cfg

logger = logging.getLogger(__name__)


def apply_runtime_settings(settings: dict | None) -> None:
    """Apply runtime settings sent by the UI to the process-wide config."""

    if not settings:
        return

    field_map = {
        "openai_api_base": "OPENAI_API_BASE",
        "openai_api_key": "OPENAI_API_KEY",
        "openai_model": "OPENAI_MODEL",
        "managed_rag_url": "MANAGED_RAG_URL",
        "managed_rag_kb_version": "MANAGED_RAG_KB_VERSION",
        "managed_rag_api_key": "MANAGED_RAG_API_KEY",
        "managed_rag_results": "MANAGED_RAG_RESULTS",
        "managed_rag_context_chunks": "MANAGED_RAG_CONTEXT_CHUNKS",
        "managed_rag_max_tokens": "MANAGED_RAG_MAX_TOKENS",
        "managed_rag_temperature": "MANAGED_RAG_TEMPERATURE",
    }

    for incoming_key, cfg_key in field_map.items():
        value = settings.get(incoming_key)
        if value in (None, ""):
            continue
        current = getattr(cfg, cfg_key)
        if isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        current = getattr(cfg, cfg_key)
        if current != value:
            setattr(cfg, cfg_key, value)
            logger.info("Runtime setting updated: %s", cfg_key)
