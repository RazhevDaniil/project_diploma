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
        "provider": "LLM_PROVIDER",
        "gigachat_credentials": "GIGACHAT_CREDENTIALS",
        "gigachat_model": "GIGACHAT_MODEL",
        "gigachat_scope": "GIGACHAT_SCOPE",
        "gigachat_embedding_model": "GIGACHAT_EMBEDDING_MODEL",
        "openai_api_base": "OPENAI_API_BASE",
        "openai_api_key": "OPENAI_API_KEY",
        "openai_model": "OPENAI_MODEL",
    }
    embedding_fields = {
        "gigachat_credentials",
        "gigachat_scope",
        "gigachat_embedding_model",
    }

    refresh_vectorstore = False

    for incoming_key, cfg_key in field_map.items():
        value = settings.get(incoming_key)
        if value in (None, ""):
            continue
        current = getattr(cfg, cfg_key)
        if current != value:
            setattr(cfg, cfg_key, value)
            logger.info("Runtime setting updated: %s", cfg_key)
            if incoming_key in embedding_fields:
                refresh_vectorstore = True

    if refresh_vectorstore:
        from src.knowledge_base.store import invalidate_cached_runtime

        invalidate_cached_runtime()
