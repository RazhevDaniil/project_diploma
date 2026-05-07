"""Runtime settings helpers.

The backend may process several runs at the same time, so request-specific
settings must travel with the run instead of mutating the process-wide config.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any

import config as cfg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSettings:
    """Immutable settings snapshot for one extraction/analysis request."""

    openai_api_base: str
    openai_api_key: str
    openai_model: str
    openai_temperature: float
    llm_request_delay: float
    parser_mode: str
    parser_chunk_size: int
    parser_concurrency: int
    parser_fast_min_requirements: int
    parser_fast_max_requirements: int
    max_requirements_per_batch: int
    analysis_rag_mode: str
    analysis_batch_concurrency: int
    managed_rag_url: str
    managed_rag_kb_version: str
    managed_rag_api_key: str
    managed_rag_results: int
    managed_rag_context_chunks: int
    managed_rag_max_tokens: int
    managed_rag_temperature: float
    managed_rag_retrieval_type: str
    managed_rag_concurrency: int
    managed_rag_cache_enabled: bool


FIELD_MAP = {
    "openai_api_base": ("OPENAI_API_BASE", str),
    "openai_api_key": ("OPENAI_API_KEY", str),
    "openai_model": ("OPENAI_MODEL", str),
    "openai_temperature": ("OPENAI_TEMPERATURE", float),
    "llm_request_delay": ("LLM_REQUEST_DELAY", float),
    "parser_mode": ("PARSER_MODE", lambda value: str(value).lower()),
    "parser_chunk_size": ("PARSER_CHUNK_SIZE", int),
    "parser_concurrency": ("PARSER_CONCURRENCY", int),
    "parser_fast_min_requirements": ("PARSER_FAST_MIN_REQUIREMENTS", int),
    "parser_fast_max_requirements": ("PARSER_FAST_MAX_REQUIREMENTS", int),
    "max_requirements_per_batch": ("MAX_REQUIREMENTS_PER_BATCH", int),
    "analysis_rag_mode": ("ANALYSIS_RAG_MODE", lambda value: str(value).lower()),
    "analysis_batch_concurrency": ("ANALYSIS_BATCH_CONCURRENCY", int),
    "managed_rag_url": ("MANAGED_RAG_URL", str),
    "managed_rag_kb_version": ("MANAGED_RAG_KB_VERSION", str),
    "managed_rag_api_key": ("MANAGED_RAG_API_KEY", str),
    "managed_rag_results": ("MANAGED_RAG_RESULTS", int),
    "managed_rag_context_chunks": ("MANAGED_RAG_CONTEXT_CHUNKS", int),
    "managed_rag_max_tokens": ("MANAGED_RAG_MAX_TOKENS", int),
    "managed_rag_temperature": ("MANAGED_RAG_TEMPERATURE", float),
    "managed_rag_retrieval_type": ("MANAGED_RAG_RETRIEVAL_TYPE", str),
    "managed_rag_concurrency": ("MANAGED_RAG_CONCURRENCY", int),
    "managed_rag_cache_enabled": ("MANAGED_RAG_CACHE_ENABLED", None),
}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def default_runtime_settings() -> RuntimeSettings:
    """Build settings from environment-backed config defaults."""

    return RuntimeSettings(
        openai_api_base=cfg.OPENAI_API_BASE,
        openai_api_key=cfg.OPENAI_API_KEY,
        openai_model=cfg.OPENAI_MODEL,
        openai_temperature=cfg.OPENAI_TEMPERATURE,
        llm_request_delay=cfg.LLM_REQUEST_DELAY,
        parser_mode=cfg.PARSER_MODE,
        parser_chunk_size=cfg.PARSER_CHUNK_SIZE,
        parser_concurrency=cfg.PARSER_CONCURRENCY,
        parser_fast_min_requirements=cfg.PARSER_FAST_MIN_REQUIREMENTS,
        parser_fast_max_requirements=cfg.PARSER_FAST_MAX_REQUIREMENTS,
        max_requirements_per_batch=cfg.MAX_REQUIREMENTS_PER_BATCH,
        analysis_rag_mode=cfg.ANALYSIS_RAG_MODE,
        analysis_batch_concurrency=cfg.ANALYSIS_BATCH_CONCURRENCY,
        managed_rag_url=cfg.MANAGED_RAG_URL,
        managed_rag_kb_version=cfg.MANAGED_RAG_KB_VERSION,
        managed_rag_api_key=cfg.MANAGED_RAG_API_KEY,
        managed_rag_results=cfg.MANAGED_RAG_RESULTS,
        managed_rag_context_chunks=cfg.MANAGED_RAG_CONTEXT_CHUNKS,
        managed_rag_max_tokens=cfg.MANAGED_RAG_MAX_TOKENS,
        managed_rag_temperature=cfg.MANAGED_RAG_TEMPERATURE,
        managed_rag_retrieval_type=cfg.MANAGED_RAG_RETRIEVAL_TYPE,
        managed_rag_concurrency=cfg.MANAGED_RAG_CONCURRENCY,
        managed_rag_cache_enabled=cfg.MANAGED_RAG_CACHE_ENABLED,
    )


def build_runtime_settings(settings: RuntimeSettings | dict | None = None) -> RuntimeSettings:
    """Return an immutable runtime settings snapshot.

    ``settings`` may be a payload from the UI/API or an existing
    ``RuntimeSettings`` instance.
    """

    if isinstance(settings, RuntimeSettings):
        return settings

    runtime = default_runtime_settings()
    if not settings:
        return runtime
    if not isinstance(settings, dict):
        logger.warning("Ignoring unsupported runtime settings payload: %s", type(settings).__name__)
        return runtime

    updates = {}
    for incoming_key, (_cfg_key, caster) in FIELD_MAP.items():
        value = settings.get(incoming_key)
        if value in (None, ""):
            continue
        try:
            if caster is None:
                updates[incoming_key] = _coerce_bool(value)
            else:
                updates[incoming_key] = caster(value)
        except (TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid runtime setting %s=%r: %s", incoming_key, value, exc)

    if not updates:
        return runtime
    return replace(runtime, **updates)


def apply_runtime_settings(settings: dict | None) -> RuntimeSettings:
    """Compatibility wrapper.

    Older call sites used this function to mutate ``config``. It now returns a
    per-request snapshot instead; callers should pass it into parser/analyzer
    functions.
    """

    return build_runtime_settings(settings)
