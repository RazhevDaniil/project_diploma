"""Persistent UI/runtime settings storage."""

from __future__ import annotations

from datetime import datetime
import json
import threading

import config as cfg

_lock = threading.Lock()

SENSITIVE_KEYS = {"openai_api_key", "managed_rag_api_key"}

ALLOWED_SETTINGS_KEYS = {
    "openai_api_base",
    "openai_model",
    "openai_temperature",
    "llm_request_delay",
    "parser_mode",
    "parser_chunk_size",
    "parser_concurrency",
    "parser_fast_min_requirements",
    "parser_fast_max_requirements",
    "max_requirements_per_batch",
    "analysis_rag_mode",
    "analysis_batch_concurrency",
    "managed_rag_url",
    "managed_rag_kb_version",
    "managed_rag_results",
    "managed_rag_context_chunks",
    "managed_rag_max_tokens",
    "managed_rag_temperature",
    "managed_rag_concurrency",
    "managed_rag_cache_enabled",
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _coerce_value(key: str, value):
    int_keys = {
        "parser_chunk_size",
        "parser_concurrency",
        "parser_fast_min_requirements",
        "parser_fast_max_requirements",
        "max_requirements_per_batch",
        "analysis_batch_concurrency",
        "managed_rag_results",
        "managed_rag_context_chunks",
        "managed_rag_max_tokens",
        "managed_rag_concurrency",
    }
    float_keys = {"openai_temperature", "llm_request_delay", "managed_rag_temperature"}
    bool_keys = {"managed_rag_cache_enabled"}

    if key in int_keys:
        return int(value)
    if key in float_keys:
        return float(value)
    if key in bool_keys:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return str(value)


def sanitize_settings(settings: dict | None, include_secrets: bool = False) -> dict:
    if not isinstance(settings, dict):
        return {}

    result = {}
    allowed = set(ALLOWED_SETTINGS_KEYS)
    if include_secrets:
        allowed.update(SENSITIVE_KEYS)

    for key, value in settings.items():
        if key not in allowed or value in (None, ""):
            continue
        result[key] = _coerce_value(key, value) if key in ALLOWED_SETTINGS_KEYS else str(value)
    return result


def load_ui_settings() -> dict:
    if not cfg.UI_SETTINGS_PATH.exists():
        return {"settings": {}, "updated_at": ""}
    try:
        payload = json.loads(cfg.UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"settings": {}, "updated_at": ""}

    return {
        "settings": sanitize_settings(payload.get("settings", {})),
        "updated_at": str(payload.get("updated_at", "") or ""),
    }


def save_ui_settings(settings: dict) -> dict:
    clean = sanitize_settings(settings)
    payload = {"settings": clean, "updated_at": _now()}
    with _lock:
        cfg.SETTINGS_DIR.mkdir(exist_ok=True)
        tmp = cfg.UI_SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cfg.UI_SETTINGS_PATH)
    return payload
