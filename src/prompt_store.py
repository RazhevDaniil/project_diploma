"""Versioned prompt storage for parser and analyzer prompts."""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

import config as cfg
from src.analysis import prompts as analysis_prompts
from src.parser import requirement_extractor


PROMPT_DEFINITIONS = {
    "parser_system": {
        "label": "Парсер: system prompt",
        "default_content": requirement_extractor.EXTRACTION_SYSTEM_PROMPT,
    },
    "parser_user_template": {
        "label": "Парсер: user template",
        "default_content": requirement_extractor.EXTRACTION_PROMPT_TEMPLATE,
    },
    "analysis_system": {
        "label": "Анализатор: system prompt",
        "default_content": analysis_prompts.ANALYSIS_SYSTEM,
    },
    "analysis_user_template": {
        "label": "Анализатор: user template",
        "default_content": analysis_prompts.ANALYSIS_PROMPT_TEMPLATE,
    },
    "summary_system": {
        "label": "Summary: system prompt",
        "default_content": analysis_prompts.SUMMARY_SYSTEM,
    },
    "summary_user_template": {
        "label": "Summary: user template",
        "default_content": analysis_prompts.SUMMARY_PROMPT_TEMPLATE,
    },
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_store() -> dict:
    prompts = {}
    for key, definition in PROMPT_DEFINITIONS.items():
        version_id = "default"
        prompts[key] = {
            "label": definition["label"],
            "active_version": version_id,
            "versions": [
                {
                    "id": version_id,
                    "label": "Базовая версия",
                    "created_at": _now(),
                    "content": definition["default_content"],
                }
            ],
        }
    return {"prompts": prompts}


def _read_store() -> dict:
    if not cfg.PROMPT_STORE_PATH.exists():
        store = _default_store()
        _write_store(store)
        return store

    try:
        store = json.loads(cfg.PROMPT_STORE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        store = _default_store()
        _write_store(store)
        return store

    changed = False
    prompts = store.setdefault("prompts", {})
    for key, definition in PROMPT_DEFINITIONS.items():
        if key not in prompts:
            prompts[key] = _default_store()["prompts"][key]
            changed = True
        else:
            prompts[key].setdefault("label", definition["label"])
            prompts[key].setdefault("active_version", "default")
            prompts[key].setdefault("versions", [])
            if not prompts[key]["versions"]:
                prompts[key]["versions"].append(_default_store()["prompts"][key]["versions"][0])
                changed = True

    if changed:
        _write_store(store)
    return store


def _write_store(store: dict) -> None:
    cfg.PROMPT_VERSIONS_DIR.mkdir(exist_ok=True)
    cfg.PROMPT_STORE_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_prompts() -> dict:
    return _read_store()


def get_prompt(prompt_key: str) -> str:
    store = _read_store()
    prompt = store["prompts"].get(prompt_key)
    if not prompt:
        raise KeyError(f"Unknown prompt key: {prompt_key}")
    active_version = prompt.get("active_version")
    for version in prompt.get("versions", []):
        if version.get("id") == active_version:
            return str(version.get("content", ""))
    return str(prompt["versions"][0].get("content", ""))


def create_prompt_version(prompt_key: str, content: str, label: str | None = None, activate: bool = True) -> dict:
    store = _read_store()
    prompt = store["prompts"].get(prompt_key)
    if not prompt:
        raise KeyError(f"Unknown prompt key: {prompt_key}")

    version = {
        "id": uuid4().hex,
        "label": label or f"Версия от {_now()}",
        "created_at": _now(),
        "content": content,
    }
    prompt["versions"].append(version)
    if activate:
        prompt["active_version"] = version["id"]
    _write_store(store)
    return version


def activate_prompt_version(prompt_key: str, version_id: str) -> dict:
    store = _read_store()
    prompt = store["prompts"].get(prompt_key)
    if not prompt:
        raise KeyError(f"Unknown prompt key: {prompt_key}")
    if not any(version.get("id") == version_id for version in prompt.get("versions", [])):
        raise KeyError(f"Unknown version id for {prompt_key}: {version_id}")
    prompt["active_version"] = version_id
    _write_store(store)
    return prompt
