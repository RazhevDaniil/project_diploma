"""Streamlit UI for the Cloud.ru TZ analyzer backend."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import unquote

import requests
import streamlit as st


DEFAULT_BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")
MODEL_OPTIONS = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "GLM-4.6": "zai-org/GLM-4.6",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
}
PREFERRED_PLATFORM_ORDER = ["ГосОблако", "Evolution", "Advanced", "Облако VMware"]
PERSISTED_SETTINGS_KEYS = [
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
]


def init_state():
    defaults = {
        "analysis_report": None,
        "requirements": None,
        "analysis_search_mode": "managed_rag",
        "parsed_files": [],
        "report_markdown": "",
        "downloads": {},
        "analysis_qa_history": [],
        "analysis_question_input": "",
        "pending_analysis_question": None,
        "selected_run_id": None,
        "active_run": None,
        "runs_list": [],
        "suppress_auto_load_latest": False,
        "upload_widget_version": 0,
        "auto_refresh_run": True,
        "last_backend_health": None,
        "last_prompts_payload": None,
        "settings_loaded_for_backend": "",
        "last_persisted_settings_payload": None,
        "last_settings_save_error": "",
        "last_settings_saved_at": "",
        "backend_api_url": DEFAULT_BACKEND_API_URL,
        "openai_api_base": os.getenv("OPENAI_API_BASE", "https://foundation-models.api.cloud.ru/v1"),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_model": os.getenv("OPENAI_MODEL", MODEL_OPTIONS["Qwen3-Next-80B-A3B-Instruct"]),
        "openai_temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.05")),
        "llm_request_delay": float(os.getenv("LLM_REQUEST_DELAY", "0")),
        "parser_mode": os.getenv("PARSER_MODE", "fast").lower(),
        "parser_chunk_size": int(os.getenv("PARSER_CHUNK_SIZE", "6000")),
        "parser_concurrency": int(os.getenv("PARSER_CONCURRENCY", "4")),
        "parser_fast_min_requirements": int(os.getenv("PARSER_FAST_MIN_REQUIREMENTS", "20")),
        "parser_fast_max_requirements": int(os.getenv("PARSER_FAST_MAX_REQUIREMENTS", "1000")),
        "max_requirements_per_batch": int(os.getenv("MAX_REQUIREMENTS_PER_BATCH", "8")),
        "analysis_rag_mode": os.getenv("ANALYSIS_RAG_MODE", "grouped").lower(),
        "analysis_batch_concurrency": int(os.getenv("ANALYSIS_BATCH_CONCURRENCY", "2")),
        "managed_rag_url": os.getenv(
            "MANAGED_RAG_URL",
            "https://e424a162-618c-4862-b789-b089abd81b46.managed-rag.inference.cloud.ru/api/v2/retrieve_generate",
        ),
        "managed_rag_kb_version": os.getenv("MANAGED_RAG_KB_VERSION", "eb73eb63-ec91-47c9-851e-1c14949b7a14"),
        "managed_rag_api_key": os.getenv("MANAGED_RAG_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        "managed_rag_results": int(os.getenv("MANAGED_RAG_RESULTS", "6")),
        "managed_rag_context_chunks": int(os.getenv("MANAGED_RAG_CONTEXT_CHUNKS", "6")),
        "managed_rag_max_tokens": int(os.getenv("MANAGED_RAG_MAX_TOKENS", "768")),
        "managed_rag_temperature": float(os.getenv("MANAGED_RAG_TEMPERATURE", "0.01")),
        "managed_rag_concurrency": int(os.getenv("MANAGED_RAG_CONCURRENCY", "4")),
        "managed_rag_cache_enabled": os.getenv("MANAGED_RAG_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
        "metrics_mode": "platform",  # «platform» или «portfolio» — Блок 9
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def llm_settings_payload() -> dict:
    return {
        "openai_api_base": st.session_state.openai_api_base,
        "openai_api_key": st.session_state.openai_api_key,
        "openai_model": st.session_state.openai_model,
        "openai_temperature": st.session_state.openai_temperature,
        "llm_request_delay": st.session_state.llm_request_delay,
        "parser_mode": st.session_state.parser_mode,
        "parser_chunk_size": st.session_state.parser_chunk_size,
        "parser_concurrency": st.session_state.parser_concurrency,
        "parser_fast_min_requirements": st.session_state.parser_fast_min_requirements,
        "parser_fast_max_requirements": st.session_state.parser_fast_max_requirements,
        "max_requirements_per_batch": st.session_state.max_requirements_per_batch,
        "analysis_rag_mode": st.session_state.analysis_rag_mode,
        "analysis_batch_concurrency": st.session_state.analysis_batch_concurrency,
        "managed_rag_url": st.session_state.managed_rag_url,
        "managed_rag_kb_version": st.session_state.managed_rag_kb_version,
        "managed_rag_api_key": st.session_state.managed_rag_api_key or st.session_state.openai_api_key,
        "managed_rag_results": st.session_state.managed_rag_results,
        "managed_rag_context_chunks": st.session_state.managed_rag_context_chunks,
        "managed_rag_max_tokens": st.session_state.managed_rag_max_tokens,
        "managed_rag_temperature": st.session_state.managed_rag_temperature,
        "managed_rag_concurrency": st.session_state.managed_rag_concurrency,
        "managed_rag_cache_enabled": st.session_state.managed_rag_cache_enabled,
    }


def persisted_settings_payload() -> dict:
    return {key: st.session_state.get(key) for key in PERSISTED_SETTINGS_KEYS}


def apply_settings_to_state(settings: dict):
    if not isinstance(settings, dict):
        return
    for key in PERSISTED_SETTINGS_KEYS:
        if key in settings and settings[key] not in (None, ""):
            st.session_state[key] = settings[key]


def load_persisted_settings_once():
    base_url = st.session_state.backend_api_url
    if st.session_state.settings_loaded_for_backend == base_url:
        return
    response = requests.get(f"{base_url}/settings", timeout=10)
    response.raise_for_status()
    payload = response.json()
    settings = payload.get("settings", {})
    apply_settings_to_state(settings)
    st.session_state.last_persisted_settings_payload = persisted_settings_payload()
    st.session_state.settings_loaded_for_backend = base_url
    st.session_state.last_settings_save_error = ""


def save_persisted_settings():
    payload = persisted_settings_payload()
    response = requests.post(
        f"{st.session_state.backend_api_url}/settings",
        json={"settings": payload},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    st.session_state.last_persisted_settings_payload = persisted_settings_payload()
    st.session_state.last_settings_saved_at = data.get("updated_at", "")
    st.session_state.last_settings_save_error = ""
    return data


def api_get(path: str, timeout: int = 10) -> dict:
    url = f"{st.session_state.backend_api_url}{path}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=15, show_spinner=False)
def cached_backend_health(base_url: str) -> dict:
    response = requests.get(f"{base_url}/health", timeout=15)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def cached_prompts(base_url: str) -> dict:
    response = requests.get(f"{base_url}/prompts", timeout=20)
    response.raise_for_status()
    return response.json()


def api_post_json(path: str, payload: dict, timeout: int = 120) -> dict:
    url = f"{st.session_state.backend_api_url}{path}"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return {"content": response.content, "filename": _response_filename(response)}


def api_post_files(path: str, files: list, data: dict, timeout: int = 1200) -> dict:
    url = f"{st.session_state.backend_api_url}{path}"
    response = requests.post(url, files=files, data=data, timeout=timeout)
    response.raise_for_status()
    return response.json()


def model_label(model_value: str) -> str:
    for label, value in MODEL_OPTIONS.items():
        if value == model_value:
            return label
    return model_value


def _response_filename(response: requests.Response, default_name: str = "download.bin") -> str:
    disposition = response.headers.get("content-disposition", "")
    if not disposition:
        return default_name

    parts = [part.strip() for part in disposition.split(";")]

    for part in parts:
        if part.lower().startswith("filename*="):
            value = part.split("=", 1)[1].strip()
            if "''" in value:
                _, encoded = value.split("''", 1)
                return unquote(encoded.strip('"')) or default_name
            return unquote(value.strip('"')) or default_name

    for part in parts:
        if part.lower().startswith("filename="):
            value = part.split("=", 1)[1].strip().strip('"')
            return value or default_name

    return default_name


def build_upload_files(uploaded_files) -> list:
    payload = []
    for upload in uploaded_files:
        payload.append(
            (
                "files",
                (
                    upload.name,
                    upload.getvalue(),
                    upload.type or "application/octet-stream",
                ),
            )
        )
    return payload


def fetch_runs(limit: int = 50) -> list[dict]:
    payload = api_get(f"/runs?limit={limit}", timeout=20)
    return payload.get("runs", [])


def fetch_run(run_id: str) -> dict:
    return api_get(f"/runs/{run_id}", timeout=20)


def fetch_run_status(run_id: str) -> dict:
    return api_get(f"/runs/{run_id}/status", timeout=5)


def load_run_into_state(run: dict):
    previous_run_id = st.session_state.get("selected_run_id")
    st.session_state.selected_run_id = run.get("id")
    st.session_state.active_run = run
    st.session_state.suppress_auto_load_latest = False
    st.session_state.requirements = run.get("requirements") or None
    st.session_state.parsed_files = run.get("parsed_files") or []
    st.session_state.analysis_report = run.get("report")
    st.session_state.analysis_search_mode = "managed_rag"
    if previous_run_id != run.get("id"):
        st.session_state.analysis_qa_history = []
        st.session_state.downloads = {}
        st.session_state.report_markdown = ""
    elif not run.get("report"):
        st.session_state.report_markdown = ""


def load_run_status_into_state(status_payload: dict):
    current = st.session_state.active_run or {}
    previous_status = current.get("status")
    merged = {**current, **status_payload}
    st.session_state.active_run = merged
    if status_payload.get("status") in {"extracted", "completed", "failed"} and previous_status != status_payload.get("status"):
        full_run = fetch_run(status_payload["id"])
        load_run_into_state(full_run)
        return True
    return False


def reset_current_outputs():
    st.session_state.analysis_report = None
    st.session_state.requirements = None
    st.session_state.parsed_files = []
    st.session_state.report_markdown = ""
    st.session_state.downloads = {}
    st.session_state.analysis_qa_history = []


def start_new_analysis():
    st.session_state.selected_run_id = None
    st.session_state.active_run = None
    st.session_state.suppress_auto_load_latest = True
    st.session_state.upload_widget_version += 1
    reset_current_outputs()


def status_label(status: str) -> str:
    return {
        "created": "Создан",
        "queued": "В очереди",
        "extracting": "Извлечение требований",
        "extracted": "Требования извлечены",
        "analyzing": "Анализ требований",
        "completed": "Готово",
        "failed": "Ошибка",
    }.get(status or "", status or "Неизвестно")


def platform_cell(assessment: dict) -> str:
    symbol = {
        "match": "+",
        "partial": "±",
        "mismatch": "-",
        "needs_clarification": "?",
    }.get(assessment.get("verdict"), "?")
    refs = assessment.get("_ui_refs") or assessment.get("evidence_refs") or []
    return f"{symbol} {', '.join(refs[:2])}".strip()


def canonical_platform_name(value: str) -> str:
    text = (value or "").strip()
    lowered = text.lower()
    if "гособлак" in lowered or "гос облак" in lowered or "goscloud" in lowered:
        return "ГосОблако"
    if "vmware" in lowered or "vcloud" in lowered or "облако vmware" in lowered:
        return "Облако VMware"
    if "advanced" in lowered:
        return "Advanced"
    if "evolution" in lowered:
        return "Evolution"
    return text


# ---------------------------------------------------------------------------
# UI helpers (Блок 1 редизайна) — небольшие утилиты, на которых строятся
# остальные блоки (метрики, матрица, drill-down, popover'ы для сносок).
# ---------------------------------------------------------------------------

_SECTION_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+){1,4})\b")
_TABLE_ROW_RE = re.compile(r"Табл(?:ица)?\s*№?\s*(\d+)[^,]*,?\s*стр[оа]?[а-я]*\s*(\d+)", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"пункт\s+(\d+)", re.IGNORECASE)


def short_section(section: str, max_len: int = 70) -> str:
    """Сжимает breadcrumb-section до читаемого ID для таблиц.

    Примеры (длинный текст → короткий):
        "Общие требования … > Таблица №1. Параметры …, строка 5"
            → "Таблица 1, стр. 5"
        "Требования к Услуге ВВИ > … / пункт 3"
            → "ВВИ / пункт 3"
        "7.2.4"
            → "7.2.4"
        "Требования к Личному кабинету … / пункт 1"
            → "ЛК / пункт 1"
    """
    if not section:
        return "—"
    section = str(section).strip()
    if len(section) <= max_len:
        return section

    # 1. Если это явный иерархический номер «8.1.3» — возвращаем как есть.
    pure_number_match = re.fullmatch(r"\d+(?:\.\d+){1,5}", section)
    if pure_number_match:
        return section

    # 2. Таблица + строка («Таблица №1, строка 5»)
    table_match = _TABLE_ROW_RE.search(section)
    if table_match:
        return f"Таблица {table_match.group(1)}, стр. {table_match.group(2)}"

    # 3. Если в section есть номер раздела типа «7.2.4» где-то внутри — берём его
    #    плюс возможный «/ пункт N».
    list_match = _LIST_ITEM_RE.search(section)
    if list_match:
        # Сначала пытаемся найти alias по ВСЕМУ section'у — типичные разделы
        # ТЗ для пресейла Cloud.ru сжимаются в короткие метки.
        alias_map = (
            ("виртуальная вычислительная инфраструктура", "ВВИ"),
            ("личному кабинету", "ЛК"),
            ("личного кабинета", "ЛК"),
            ("личный кабинет", "ЛК"),
            ("услуге интернет на гарантированной", "Интернет"),
            ("услуге размещения оборудования", "Колокация"),
            ("площадкам размещения инфраструктуры", "ЦОД"),
            ("качеству и безопасности", "Качество/ИБ"),
            ("гарантийному сроку", "Гарантия"),
            ("требования к услугам", "Услуги"),
        )
        section_lower = section.lower()
        short_head = ""
        for key, alias in alias_map:
            if key in section_lower:
                short_head = alias
                break
        if not short_head:
            # Fallback: берём первый сегмент breadcrumb'а и сжимаем.
            head = section.split(">", 1)[0].strip()
            words = head.split()
            short_head = " ".join(words[:3])
            if len(short_head) > 28:
                short_head = short_head[:28].rstrip() + "…"
        return f"{short_head} / пункт {list_match.group(1)}"

    # 4. Берём последние 2 сегмента breadcrumb'а и ужимаем.
    parts = [p.strip() for p in section.split(">") if p.strip()]
    if len(parts) >= 2:
        tail = " > ".join(parts[-2:])
    else:
        tail = parts[-1] if parts else section
    if len(tail) > max_len:
        tail = tail[: max_len - 1].rstrip() + "…"
    return tail


CATEGORY_LABELS = {
    "technical": "Техническое",
    "sla": "SLA",
    "security": "ИБ",
    "legal": "Юридическое",
    "commercial": "Коммерческое",
    "other": "Прочее",
}


def category_label(category: str) -> str:
    """Канонизированная метка категории требования для UI-таблиц."""
    return CATEGORY_LABELS.get((category or "").lower(), category or "—")


_VERDICT_EMOJI = {
    "match": "✅",
    "partial": "🟡",
    "mismatch": "❌",
    "needs_clarification": "❓",
}
_VERDICT_LABELS = {
    "match": "Соответствует",
    "partial": "Частично",
    "mismatch": "Не соответствует",
    "needs_clarification": "Уточнить",
}


def verdict_emoji_label(verdict: str, with_label: bool = False) -> str:
    """Возвращает эмодзи (или эмодзи+метку) для вердикта."""
    emoji = _VERDICT_EMOJI.get(verdict, "·")
    if not with_label:
        return emoji
    label = _VERDICT_LABELS.get(verdict, verdict or "—")
    return f"{emoji} {label}"


def recommended_metric(report: dict) -> dict:
    """Достаёт «главную метрику» отчёта — покрытие на рекомендуемой платформе.

    Возвращает dict с ключами:
        platform        — имя платформы (например, «Облако VMware»). Может быть
                          пустым, если платформа не определена.
        percent         — процент покрытия (0..100). Если recommended-метрика
                          отсутствует, используем общий compliance_percentage.
        is_fallback     — True, если показываем общий процент вместо платформенного
                          (для UI это сигнал оформить визуально иначе).
        general_percent — общий compliance_percentage отчёта (best-case).
        score, max_score — баллы (для подписи прогресс-бара).
    """
    if not isinstance(report, dict):
        return {
            "platform": "",
            "percent": 0.0,
            "is_fallback": True,
            "general_percent": 0.0,
            "score": 0,
            "max_score": 0,
        }
    general = float(report.get("compliance_percentage") or 0)
    platform = (report.get("recommended_platform") or "").strip()
    rec_pct = report.get("recommended_platform_compliance")
    try:
        rec_pct_value = float(rec_pct) if rec_pct is not None else None
    except (TypeError, ValueError):
        rec_pct_value = None
    if platform and rec_pct_value is not None and rec_pct_value > 0:
        return {
            "platform": platform,
            "percent": rec_pct_value,
            "is_fallback": False,
            "general_percent": general,
            "score": int(report.get("score") or 0),
            "max_score": int(report.get("max_score") or 0),
        }
    return {
        "platform": "",
        "percent": general,
        "is_fallback": True,
        "general_percent": general,
        "score": int(report.get("score") or 0),
        "max_score": int(report.get("max_score") or 0),
    }


def metric_color(percent: float) -> str:
    """Цвет (CSS) для процента покрытия — единая шкала во всём UI."""
    if percent >= 75:
        return "green"
    if percent >= 55:
        return "orange"
    return "red"


# --- Heuristic compact-label для ячейки «Требование» в матрице ---------------
# Полный текст требования живёт в drill-down карточке. В таблице нужен
# короткий понятный label, чтобы пресейл с одного взгляда понимал смысл строки.

_REQ_LABEL_RULES = [
    # SLA таблица 4 — узнаём по «Время решения» + приоритет/тип
    (re.compile(r"Время решения инцидентов первого приоритета.*?Целевое значение[^:]*:\s*(\d+)", re.IGNORECASE | re.DOTALL),
     lambda m: f"SLA: критичный приоритет ≤ {m.group(1)} ч"),
    (re.compile(r"Время решения инцидентов второго приоритета.*?Целевое значение[^:]*:\s*(\d+)", re.IGNORECASE | re.DOTALL),
     lambda m: f"SLA: высокий приоритет ≤ {m.group(1)} ч"),
    (re.compile(r"Время решения инцидентов третьего приоритета.*?Целевое значение[^:]*:\s*(\d+)", re.IGNORECASE | re.DOTALL),
     lambda m: f"SLA: средний приоритет ≤ {m.group(1)} раб. ч"),
    (re.compile(r"Время решения инцидентов четвертого приоритета.*?Целевое значение[^:]*:\s*(\d+)", re.IGNORECASE | re.DOTALL),
     lambda m: f"SLA: низкий приоритет ≤ {m.group(1)} раб. ч"),
    (re.compile(r"стандартных запросов на изменение.*?Целевое значение[^:]*:\s*(\d+)", re.IGNORECASE | re.DOTALL),
     lambda m: f"SLA: запрос на изменение ≤ {m.group(1)} раб. ч"),
    (re.compile(r"запросов об увеличении ресурсов.*?Целевое значение[^:]*:\s*(\d+)", re.IGNORECASE | re.DOTALL),
     lambda m: f"SLA: запрос ресурсов ≤ {m.group(1)} ч"),
    # Таблица 1 — узнаём по «Параметр услуги»
    (re.compile(r"Процессор[^;]*; Объем услуги:\s*(\d+)", re.IGNORECASE),
     lambda m: f"vCPU суммарно — {m.group(1)} шт."),
    (re.compile(r"Память[^;]*; Объем услуги:\s*(\d+)", re.IGNORECASE),
     lambda m: f"RAM суммарно — {m.group(1)} ГБ"),
    (re.compile(r"Дисковое пространство[^;]*SSD[^;]*; Объем услуги:\s*(\d+)", re.IGNORECASE),
     lambda m: f"Диск суммарно — {m.group(1)} ГБ SSD/SAS"),
    (re.compile(r"IP-адрес[^;]*; Объем услуги:\s*(\d+)", re.IGNORECASE),
     lambda m: f"Публичных IP — {m.group(1)} шт."),
    (re.compile(r"Astra Linux[^;]*; Объем услуги:\s*(\d+)", re.IGNORECASE),
     lambda m: f"Astra Linux / RED OS (ФСТЭК) — {m.group(1)} шт."),
    (re.compile(r"Интернет.*?Безлимитный доступ\s*([\d-]+)\s*Мбит", re.IGNORECASE | re.DOTALL),
     lambda m: f"Интернет — {m.group(1)} Мбит/с"),
    # ВВИ — характеристики ВМ
    (re.compile(r"Размер Памяти на 1 Виртуальную машину\s*не менее\s*(\d+)\s*ГБ", re.IGNORECASE),
     lambda m: f"RAM на 1 ВМ ≥ {m.group(1)} ГБ"),
    (re.compile(r"Количество Процессоров на 1 Виртуальную машину.*?(\d+)\s*шт", re.IGNORECASE | re.DOTALL),
     lambda m: f"vCPU на 1 ВМ — до {m.group(1)} шт."),
    (re.compile(r"Количество дисков на 1 Виртуальную машину.*?не менее\s*(\d+)", re.IGNORECASE | re.DOTALL),
     lambda m: f"Дисков на 1 ВМ ≥ {m.group(1)}"),
    (re.compile(r"белых IP-адресов на каждую Виртуальную машину.*?(\d+)", re.IGNORECASE | re.DOTALL),
     lambda m: f"IP на 1 ВМ — {m.group(1)}"),
    (re.compile(r"Размер каждого подключенного к Виртуальной машине диска.*?(\d+)\s*ТБ", re.IGNORECASE | re.DOTALL),
     lambda m: f"Размер диска ВМ — до {m.group(1)} ТБ"),
    (re.compile(r"CPU Ready time.*?(\d+)\s*%", re.IGNORECASE | re.DOTALL),
     lambda m: f"CPU Ready time — не менее {m.group(1)}% (формулировка спорная)"),
    (re.compile(r"BPS между Виртуальными машинами.*?(\d[\d ]*)\s*Мбит", re.IGNORECASE | re.DOTALL),
     lambda m: f"BPS между ВМ — до {m.group(1).strip()} Мбит/с"),
    (re.compile(r"SSD\s*[-—–]\s*до\s*([\d ]+)\s*IOPS\s*на\s*1\s*Тб", re.IGNORECASE),
     lambda m: f"SSD — до {m.group(1).strip()} IOPS на 1 ТБ"),
    (re.compile(r"SSD\s*[-—–]\s*до\s*(\d+)\s*мс", re.IGNORECASE),
     lambda m: f"SSD — задержка ≤ {m.group(1)} мс"),
    (re.compile(r"99,?\s*982\s*%", re.IGNORECASE),
     lambda m: "Доступность ≥ 99,982%"),
    # Личный кабинет
    (re.compile(r"Создания/изменения профиля организации", re.IGNORECASE),
     lambda m: "ЛК: профиль организации"),
    (re.compile(r"Приглашения новых пользователей.*?ролевой? моделью", re.IGNORECASE | re.DOTALL),
     lambda m: "ЛК: ролевая модель + приглашение пользователей"),
    (re.compile(r"Просмотра технической информации.*?потребляемых объемов", re.IGNORECASE | re.DOTALL),
     lambda m: "ЛК: мониторинг потребляемых ресурсов"),
    (re.compile(r"Логирования событий", re.IGNORECASE),
     lambda m: "ЛК: логирование событий"),
    (re.compile(r"Создания обращений в техническую поддержку", re.IGNORECASE),
     lambda m: "ЛК: тикеты в техподдержку"),
    (re.compile(r"двухфактор", re.IGNORECASE),
     lambda m: "ЛК: двухфакторная аутентификация (2FA)"),
    (re.compile(r"графический интерфейс в Личном кабинете", re.IGNORECASE),
     lambda m: "ЛК: графический интерфейс"),
    (re.compile(r"публичной документацией на русском", re.IGNORECASE),
     lambda m: "ЛК: документация и интерфейс на русском"),
    (re.compile(r"Личный кабинет должен быть доступен через современный веб-браузер", re.IGNORECASE),
     lambda m: "ЛК: доступ через веб-браузер"),
    # ВВИ — обязанности
    (re.compile(r"бесперебойное функционирование:\s*Пула ресурсов", re.IGNORECASE),
     lambda m: "ВВИ: пул ресурсов и доступ к нему"),
    (re.compile(r"бесперебойное функционирование:\s*Возможность создания Виртуальных машин", re.IGNORECASE),
     lambda m: "ВВИ: создание ВМ под управлением ОС"),
    (re.compile(r"бесперебойное функционирование:\s*Виртуальную сеть", re.IGNORECASE),
     lambda m: "ВВИ: виртуальная сеть + шлюз + IP"),
    (re.compile(r"бесперебойное функционирование:\s*Доступ\s*\(сетевое подключение\)", re.IGNORECASE),
     lambda m: "ВВИ: доступ через интернет/локальную сеть"),
    # Колокация (7.2.x)
    (re.compile(r"забронировано не менее\s*(\d+)U.*?(\d+)\s*Вт", re.IGNORECASE | re.DOTALL),
     lambda m: f"Колокация: ≥ {m.group(1)}U / {m.group(2)} Вт на юнит"),
    (re.compile(r"подключается к внутренней сети передачи данных ЦОД", re.IGNORECASE),
     lambda m: "Колокация: подключение к сети ЦОД"),
    (re.compile(r"Физический доступ сотрудников Заказчика", re.IGNORECASE),
     lambda m: "Колокация: физический доступ заказчика"),
    (re.compile(r"Администрирование оборудования и программного обеспечения", re.IGNORECASE),
     lambda m: "Колокация: администрирование самостоятельное"),
    (re.compile(r"Передача оборудования.*?Акт", re.IGNORECASE | re.DOTALL),
     lambda m: "Колокация: акт сдачи-приёмки"),
    (re.compile(r"сохранность оборудования", re.IGNORECASE),
     lambda m: "Колокация: сохранность оборудования"),
    (re.compile(r"размещения сетевого оборудования Заказчика в том же ЦОД", re.IGNORECASE),
     lambda m: "Колокация: размещение оборудования в ЦОД"),
    # Интернет / сеть
    (re.compile(r"телекоммуникационные услуги доступа в Интернет.*?(\d+)\s*Мбит", re.IGNORECASE | re.DOTALL),
     lambda m: f"Интернет: гарантированно ≥ {m.group(1)} Мбит/с"),
    (re.compile(r"Процент потерянных пакетов.*?(\d+[,.]?\d*)\s*%", re.IGNORECASE | re.DOTALL),
     lambda m: f"Сеть: потери ≤ {m.group(1).replace('.', ',')}%"),
    (re.compile(r"Средняя сетевая задержка.*?(\d+)\s*мс", re.IGNORECASE | re.DOTALL),
     lambda m: f"Сеть: задержка ≤ {m.group(1)} мс"),
    # ЦОД
    (re.compile(r"Помещения ЦОД.*?долгосрочной.*?аренде", re.IGNORECASE | re.DOTALL),
     lambda m: "ЦОД: собственный или долгосрочная аренда (≥ 3 лет)"),
    (re.compile(r"резервирования технических средств", re.IGNORECASE),
     lambda m: "ЦОД: резервирование инфраструктуры"),
    (re.compile(r"автоматического и ручного резервного копирования", re.IGNORECASE),
     lambda m: "ЦОД: бэкап (auto + manual)"),
    (re.compile(r"автоматическое выявление недоступности", re.IGNORECASE),
     lambda m: "ЦОД: автомониторинг доступности"),
    (re.compile(r"автоматическое оповещение", re.IGNORECASE),
     lambda m: "ЦОД: автооповещения о сбоях"),
    (re.compile(r"плановое и превентивное обслуживание", re.IGNORECASE),
     lambda m: "ЦОД: плановое ТО"),
    (re.compile(r"ремонт и замена компонентов", re.IGNORECASE),
     lambda m: "ЦОД: ремонт/замена компонентов"),
    (re.compile(r"добавление или удаление элементов", re.IGNORECASE),
     lambda m: "ЦОД: добавление/удаление элементов"),
    (re.compile(r"тестирование компонентов и систем в целом", re.IGNORECASE),
     lambda m: "ЦОД: тестирование компонентов"),
    (re.compile(r"круглосуточное нахождение в здании ЦОД сотрудников", re.IGNORECASE),
     lambda m: "ЦОД: 24/7 сотрудники сервисной поддержки"),
    (re.compile(r"проведения плановых работ.*?без отключения", re.IGNORECASE | re.DOTALL),
     lambda m: "ЦОД: плановые работы без даунтайма"),
    # Compliance / лицензии
    (re.compile(r"К1.*?приказ\s*ФСТЭК\s*России\s*[№N]\s*17", re.IGNORECASE | re.DOTALL),
     lambda m: "ИБ: К1 (ФСТЭК № 17)"),
    (re.compile(r"первого уровня защищ.*?персональных данных", re.IGNORECASE | re.DOTALL),
     lambda m: "ИБ: УЗ-1 (ПП № 1119)"),
    (re.compile(r"ИСО/МЭК\s*27001", re.IGNORECASE),
     lambda m: "ИБ: ISO/IEC 27001:2021"),
    (re.compile(r"лицензию ФСТЭК России", re.IGNORECASE),
     lambda m: "Лицензия ФСТЭК ТЗКИ"),
    (re.compile(r"лицензию Роскомнадзора", re.IGNORECASE),
     lambda m: "Лицензия Роскомнадзор (каналы связи)"),
    (re.compile(r"аттестат соответствия платформы виртуализации", re.IGNORECASE),
     lambda m: "ИБ: аттестат платформы виртуализации"),
    (re.compile(r"выписку из модели угроз", re.IGNORECASE),
     lambda m: "ИБ: выписка из модели угроз"),
    # Гарантии
    (re.compile(r"Контроль основных параметров Услуги", re.IGNORECASE),
     lambda m: "Гарантия: контроль параметров"),
    (re.compile(r"Устранение сбоев в работе Услуги", re.IGNORECASE),
     lambda m: "Гарантия: устранение сбоев"),
    (re.compile(r"Профилактические работы\s*$", re.IGNORECASE),
     lambda m: "Гарантия: профилактика"),
    (re.compile(r"Консультации Заказчика", re.IGNORECASE),
     lambda m: "Гарантия: консультации"),
    (re.compile(r"круглосуточную техническую поддержку", re.IGNORECASE),
     lambda m: "Поддержка 24/7"),
    (re.compile(r"единой точке обращения", re.IGNORECASE),
     lambda m: "Единая точка обращений: ЛК + e-mail + телефон"),
    (re.compile(r"присвоением каждой заявке уникального", re.IGNORECASE),
     lambda m: "Регистрация заявок с уникальным ID"),
    (re.compile(r"профилактические работы.*?согласованию с Заказчиком.*?(\d+)\s*час", re.IGNORECASE | re.DOTALL),
     lambda m: f"Профилактика: согласование за ≥ {m.group(1)} ч"),
    (re.compile(r"устранить все недостатки", re.IGNORECASE),
     lambda m: "Устранение недостатков своими силами"),
]


def compact_requirement_label(text: str, max_len: int = 90) -> str:
    """Эвристическое сжатие текста требования в человекочитаемую короткую
    метку для отображения в матрице. Полный текст остаётся доступен в
    drill-down dialog'е.

    Применяется набор регулярных выражений (см. `_REQ_LABEL_RULES`) для
    типичных требований ТЗ Cloud.ru: SLA, vCPU/RAM/диски, ЛК, колокация,
    ИБ, гарантии. Если ни одно правило не срабатывает — fallback на
    первые 2 предложения с многоточием.
    """
    if not text:
        return "—"
    src = text.strip()

    for pattern, formatter in _REQ_LABEL_RULES:
        match = pattern.search(src)
        if match:
            try:
                label = formatter(match)
                if label:
                    return label
            except (IndexError, ValueError):
                continue

    # Fallback — первое осмысленное предложение, без хвостов с двоеточиями
    # и intro-конструкций.
    cleaned = re.sub(
        r"^(Исполнитель должен |Исполнитель обязан |Исполнитель обязуется |Заказчик обязан )",
        "", src, count=1, flags=re.IGNORECASE,
    )
    # Берём до первого «;» или конца предложения.
    head = re.split(r"[;.](?:\s|$)", cleaned, maxsplit=1)[0]
    head = head.strip()
    if len(head) > max_len:
        head = head[: max_len - 1].rstrip() + "…"
    # С большой буквы.
    if head:
        head = head[0].upper() + head[1:]
    return head or "—"


def compute_metrics_by_mode(report: dict, mode: str) -> dict:
    """Возвращает счётчики и процент в выбранном режиме.

    mode == "platform"  — считаем match/partial/mismatch ТОЛЬКО по
                          platform_assessments рекомендуемой платформы.
                          Те требования, у которых эта платформа не оценена,
                          выпадают из знаменателя (как делает
                          recommended_platform_compliance в models.py).
    mode == "portfolio" — старая логика: по overall_verdict каждого
                          требования (best-case по любой платформе).

    В UI это нужно для согласованности: процент в шапке и счётчики в сводке
    должны отражать одну и ту же логику. Раньше шапка показывала 77% «по
    VMware», а счётчики — best-case → возникала путаница.
    """
    if not isinstance(report, dict):
        report = {}
    if mode == "platform":
        platform = (report.get("recommended_platform") or "").strip()
        if platform:
            counts = {"match": 0, "partial": 0, "mismatch": 0, "needs_clarification": 0}
            score = 0
            for verdict in report.get("verdicts", []) or []:
                # Берём оценку именно этой платформы из platform_assessments.
                chosen = None
                for assessment in verdict.get("platform_assessments", []) or []:
                    a_platform = (assessment.get("platform_name") or "").strip()
                    if canonical_platform_name(a_platform) == platform:
                        chosen = assessment
                        break
                if not chosen:
                    continue
                v = chosen.get("verdict") or "needs_clarification"
                if v not in counts:
                    v = "needs_clarification"
                counts[v] += 1
                if v == "match":
                    score += 2
                elif v == "partial":
                    score += 1
            total = sum(counts.values())
            max_score = total * 2
            percent = round((score / max_score * 100) if max_score else 0.0, 1)
            return {
                "mode": "platform",
                "platform": platform,
                "total": total,
                "match_count": counts["match"],
                "partial_count": counts["partial"],
                "mismatch_count": counts["mismatch"],
                "clarification_count": counts["needs_clarification"],
                "score": score,
                "max_score": max_score,
                "percent": percent,
            }
    # Fallback: режим "portfolio" — старая логика.
    return {
        "mode": "portfolio",
        "platform": "",
        "total": int(report.get("total") or 0),
        "match_count": int(report.get("match_count") or 0),
        "partial_count": int(report.get("partial_count") or 0),
        "mismatch_count": int(report.get("mismatch_count") or 0),
        "clarification_count": int(report.get("clarification_count") or 0),
        "score": int(report.get("score") or 0),
        "max_score": int(report.get("max_score") or 0),
        "percent": float(report.get("compliance_percentage") or 0.0),
    }


def get_metrics_mode() -> str:
    """Текущий режим из session_state, по умолчанию — «по рекомендуемой платформе»."""
    return st.session_state.get("metrics_mode", "platform")


def is_matrix_platform_name(value: str) -> bool:
    text = canonical_platform_name(value)
    lowered = text.lower()
    if not text:
        return False
    if lowered.startswith("cloud.ru источник"):
        return False
    if "документация не найдена" in lowered or "платформа не определена" in lowered:
        return False
    return True


def _is_canonical_cloud_platform(name: str) -> bool:
    """True, если name — каноническая платформа Cloud.ru
    (ГосОблако / Облако VMware / Advanced / Evolution)."""
    return canonical_platform_name(name) in PREFERRED_PLATFORM_ORDER


def report_platform_names(report: dict) -> list[str]:
    """Список платформ для шапки матрицы.

    Фильтр source_type="external_service" срабатывает ТОЛЬКО если
    platform_name НЕ совпадает с канонической платформой Cloud.ru. Бывает,
    что LLM ошибочно ставит source_type=external_service для канонической
    платформы (например, ГосОблако с verdict=partial и source_type=external)
    — раньше такой assessment пропадал из матрицы, хотя в drill-down карточке
    был. Доверяем имени платформы как ground truth.
    """
    names = []
    for verdict in report.get("verdicts", []):
        for assessment in verdict.get("platform_assessments", []) or []:
            raw_name = assessment.get("platform_name") or ""
            platform_name = canonical_platform_name(raw_name)
            if assessment.get("source_type") == "external_service" and not _is_canonical_cloud_platform(raw_name):
                continue
            if is_matrix_platform_name(platform_name) and platform_name not in names:
                names.append(platform_name)

    preferred = [name for name in PREFERRED_PLATFORM_ORDER if name in names]
    other = sorted([name for name in names if name not in PREFERRED_PLATFORM_ORDER], key=str.casefold)
    return preferred + other


def report_reference_map(report: dict) -> dict[str, int]:
    refs = {}
    for verdict in report.get("verdicts", []):
        for assessment in verdict.get("platform_assessments", []) or []:
            source_urls = assessment.get("source_urls") or []
            source_titles = assessment.get("source_titles") or []
            key = (source_urls[0] if source_urls else None) or (source_titles[0] if source_titles else None)
            if key and key not in refs:
                refs[key] = len(refs) + 1
        for url in verdict.get("source_urls", []) or []:
            if url and url not in refs:
                refs[url] = len(refs) + 1
    return refs


def best_platform_assessment(items: list[dict], refs: dict[str, int]) -> dict | None:
    if not items:
        return None
    rank = {"match": 3, "partial": 2, "needs_clarification": 1, "mismatch": 0}
    best = sorted(
        items,
        key=lambda item: (rank.get(item.get("verdict"), -1), float(item.get("confidence") or 0)),
        reverse=True,
    )[0].copy()
    source_urls = best.get("source_urls") or []
    source_titles = best.get("source_titles") or []
    key = (source_urls[0] if source_urls else None) or (source_titles[0] if source_titles else None)
    best["_ui_refs"] = [f"[{refs[key]}]"] if key in refs else []
    return best


def report_platform_matrix(report: dict) -> list[dict]:
    """Строки матрицы платформ для UI.

    UX-Блок 6: дополнено колонкой «Категория» и сжатым полем «Пункт ТЗ»
    (через short_section). Полный section хранится в `_full_section` для
    drill-down (Блок 8). Полный текст требования — в `_full_text`.
    """
    refs = report_reference_map(report)
    platform_names = report_platform_names(report)
    if not platform_names:
        return []

    rows = []
    for verdict in report.get("verdicts", []):
        full_section = verdict.get("section") or f"#{verdict.get('requirement_id')}"
        full_text = verdict.get("requirement_text") or ""
        row = {
            "Пункт ТЗ": short_section(full_section),
            "Категория": category_label(verdict.get("category", "")),
            # Heuristic-сжатие: «Время решения инцидентов первого
            # приоритета… Целевое значение*: 2…» → «SLA: критичный
            # приоритет ≤ 2 ч». Полный текст — в drill-down карточке.
            "Требование": compact_requirement_label(full_text),
            "_full_section": full_section,
            "_full_text": full_text,
            "_requirement_id": verdict.get("requirement_id"),
        }
        by_platform = {platform_name: [] for platform_name in platform_names}
        for assessment in verdict.get("platform_assessments", []) or []:
            raw_name = assessment.get("platform_name") or ""
            # Фильтр external_service срабатывает ТОЛЬКО для НЕ-канонических
            # платформ Cloud.ru. Если LLM ошибочно пометил ГосОблако/VMware/
            # Advanced/Evolution как external — этот assessment всё равно
            # должен попасть в матрицу (доверяем platform_name).
            if assessment.get("source_type") == "external_service" and not _is_canonical_cloud_platform(raw_name):
                continue
            platform_name = canonical_platform_name(raw_name)
            if platform_name in by_platform:
                by_platform[platform_name].append(assessment)
        for platform_name in platform_names:
            assessment = best_platform_assessment(by_platform[platform_name], refs)
            # Пустая ячейка = «LLM не вернул оценку для этой платформы».
            # Честнее показать «?» (уточнить), чем «-» (несоответствие):
            # отсутствие оценки ≠ доказанное несоответствие. Backend-страховка
            # (_fill_missing_canonical_platforms) обычно гарантирует, что
            # ассессмент будет — но если что-то проскочило, рендерим «?».
            row[platform_name] = platform_cell(assessment) if assessment else "?"
        rows.append(row)
    return rows


def report_suspicious_items(report: dict) -> list[dict]:
    items = report.get("suspicious_items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]

    result = []
    for verdict in report.get("verdicts", []):
        reasons = list(verdict.get("evidence_contract_notes") or [])
        if verdict.get("verdict") == "needs_clarification":
            reasons.append("Требует ручного уточнения")
        if float(verdict.get("confidence") or 0) < 0.55:
            reasons.append("Низкая уверенность")
        if verdict.get("requires_external_service"):
            reasons.append("Нужна проработка внешних услуг / подрядчиков")
        if verdict.get("evidence_status") in {"missing", "weak", "downgraded"}:
            reasons.append(f"Статус доказательств: {verdict.get('evidence_status')}")
        if reasons:
            result.append(
                {
                    "section": verdict.get("section") or f"#{verdict.get('requirement_id')}",
                    "requirement_text": verdict.get("requirement_text", ""),
                    "verdict": verdict.get("verdict", ""),
                    "confidence": verdict.get("confidence", 0),
                    "reasons": list(dict.fromkeys(reasons)),
                }
            )
    return result


def render_run_status(run: dict):
    status = run.get("status", "unknown")
    stage = run.get("stage", "")
    done = int(run.get("progress_done") or 0)
    total = int(run.get("progress_total") or 0)

    if status == "failed":
        st.error(f"{status_label(status)}: {run.get('error') or 'без деталей'}")
    elif status in {"queued", "extracting", "analyzing"}:
        st.info(f"{status_label(status)} · {stage}")
    elif status == "completed":
        st.success("Анализ завершён")
    elif status == "extracted":
        st.success("Требования извлечены, можно запускать анализ")
    else:
        st.caption(f"{status_label(status)} · {stage}")

    if total > 0:
        st.progress(min(max(done / total, 0.0), 1.0), text=f"{done}/{total}")


def refresh_selected_run():
    run_id = st.session_state.get("selected_run_id")
    if not run_id:
        return None
    run = fetch_run(run_id)
    load_run_into_state(run)
    return run


def refresh_selected_run_status():
    run_id = st.session_state.get("selected_run_id")
    if not run_id:
        return None
    status_payload = fetch_run_status(run_id)
    full_refreshed = load_run_status_into_state(status_payload)
    return {"status": status_payload, "full_refreshed": full_refreshed}


def render_current_run_panel():
    active_run = st.session_state.active_run
    if not active_run:
        return

    st.subheader("Текущий запуск")
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"**{active_run.get('document_name', 'document')}**  \n"
            f"`{active_run.get('id')}`  \n"
            f"Обновлено: {active_run.get('updated_at', '')}"
        )
        render_run_status(active_run)
    with right:
        st.checkbox("Автообновлять", key="auto_refresh_run")
        if st.button("Обновить статус", use_container_width=True, key="refresh_run_status"):
            try:
                refresh_result = refresh_selected_run_status()
                if refresh_result and refresh_result.get("full_refreshed"):
                    st.rerun()
            except Exception as exc:
                show_request_error(exc)


def fetch_report_markdown():
    report = st.session_state.analysis_report
    if not report:
        return
    result = api_post_json("/reports/markdown", {"report": report}, timeout=120)
    st.session_state.report_markdown = result.get("markdown", "")


def apply_pending_analysis_question():
    pending_question = st.session_state.get("pending_analysis_question")
    if pending_question is None:
        return
    st.session_state.analysis_question_input = pending_question
    st.session_state.pending_analysis_question = None


def prepare_download(format_name: str):
    report = st.session_state.analysis_report
    if not report:
        return
    url = f"{st.session_state.backend_api_url}/reports/export/{format_name}"
    response = requests.post(url, json={"report": report}, timeout=300)
    response.raise_for_status()
    extension_map = {
        "md": "report.md",
        "docx": "report.docx",
        "pdf": "report.pdf",
        "xlsx": "report.xlsx",
    }
    st.session_state.downloads[format_name] = {
        "content": response.content,
        "filename": _response_filename(response, default_name=extension_map.get(format_name, "download.bin")),
    }


def render_download_controls():
    st.subheader("Скачать полный отчёт")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        if st.button("💾 Подготовить Markdown", key="prepare_md_top"):
            try:
                prepare_download("md")
            except Exception as exc:
                show_request_error(exc)
        if "md" in st.session_state.downloads:
            st.download_button(
                "📥 Скачать MD",
                data=st.session_state.downloads["md"]["content"],
                file_name=st.session_state.downloads["md"]["filename"],
                mime="text/markdown",
                key="download_md_top",
            )

    with col_b:
        if st.button("💾 Подготовить DOCX", key="prepare_docx_top"):
            try:
                prepare_download("docx")
            except Exception as exc:
                show_request_error(exc)
        if "docx" in st.session_state.downloads:
            st.download_button(
                "📥 Скачать DOCX",
                data=st.session_state.downloads["docx"]["content"],
                file_name=st.session_state.downloads["docx"]["filename"],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_docx_top",
            )

    with col_c:
        if st.button("💾 Подготовить PDF", key="prepare_pdf_top"):
            try:
                prepare_download("pdf")
            except Exception as exc:
                show_request_error(exc)
        if "pdf" in st.session_state.downloads:
            st.download_button(
                "📥 Скачать PDF",
                data=st.session_state.downloads["pdf"]["content"],
                file_name=st.session_state.downloads["pdf"]["filename"],
                mime="application/pdf",
                key="download_pdf_top",
            )

    with col_d:
        if st.button("💾 Подготовить Excel", key="prepare_xlsx_top"):
            try:
                prepare_download("xlsx")
            except Exception as exc:
                show_request_error(exc)
        if "xlsx" in st.session_state.downloads:
            st.download_button(
                "📥 Скачать XLSX",
                data=st.session_state.downloads["xlsx"]["content"],
                file_name=st.session_state.downloads["xlsx"]["filename"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_xlsx_top",
            )


def render_metrics_mode_selector(report: dict):
    """Переключатель: считать процент и счётчики по рекомендуемой платформе
    или по портфелю (best-case).

    Раньше шапка показывала «Покрытие 77% по VMware», а сводка ниже — общие
    счётчики по best-case (37/41/0/11). Это вводило в путаницу: процент
    говорил про платформу, а распределение — про другое. Теперь пользователь
    явно выбирает один из режимов, и весь отчёт согласован.
    """
    rec = recommended_metric(report)
    if rec["is_fallback"] or not rec["platform"]:
        # Если рекомендуемой платформы нет (старые отчёты или нулевой анализ),
        # переключатель не имеет смысла — всегда показываем общий процент.
        st.session_state["metrics_mode"] = "portfolio"
        return
    options = ["platform", "portfolio"]
    labels = {
        "platform": f"По рекомендуемой платформе ({rec['platform']})",
        "portfolio": "Best-case по портфелю Cloud.ru",
    }
    current = get_metrics_mode()
    if current not in options:
        current = "platform"
    selected = st.radio(
        "Режим расчёта процента и счётчиков",
        options=options,
        index=options.index(current),
        format_func=lambda value: labels.get(value, value),
        horizontal=True,
        key="metrics_mode_selector",
        help=(
            "**По рекомендуемой платформе** — считаем процент, как будто КП "
            "построено только на одной платформе Cloud.ru (например, на "
            "Облако VMware). Так считает заказчик при оценке тендера: «эта "
            "платформа закрывает 88% наших требований».\n\n"
            "**Best-case по портфелю Cloud.ru** — берём ЛУЧШИЙ вердикт по "
            "каждому требованию из всех 4 платформ. Так считаем мы, когда "
            "решаем «какой максимум Cloud.ru может закрыть в принципе» — "
            "без жёсткой привязки к одной платформе. Полезно для гибридных "
            "сценариев (ГосОблако + VMware + colocation)."
        ),
    )
    st.session_state["metrics_mode"] = selected


def render_report_header(report: dict):
    """Шапка отчёта: главная цифра — процент в выбранном режиме (платформа
    или портфель), под ней — альтернативная цифра меньшим шрифтом.

    Кнопка-переключатель режима показывается над шапкой через
    `render_metrics_mode_selector`.
    """
    rec = recommended_metric(report)
    if rec["is_fallback"]:
        # Нет рекомендуемой платформы — старое поведение, один общий процент.
        color = metric_color(rec["percent"])
        progress_value = min(max(rec["percent"] / 100.0, 0.0), 1.0)
        st.markdown(
            f"## Общий процент соответствия: "
            f"<span style='color:{color}'>{rec['percent']:.1f}%</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{rec['score']} из {rec['max_score']} баллов**")
        st.progress(progress_value)
        return

    mode = get_metrics_mode()
    metrics = compute_metrics_by_mode(report, mode)
    color = metric_color(metrics["percent"])
    progress_value = min(max(metrics["percent"] / 100.0, 0.0), 1.0)

    if mode == "platform":
        st.markdown(
            f"## Покрытие на рекомендуемой платформе "
            f"({metrics['platform']}): "
            f"<span style='color:{color}'>{metrics['percent']:.1f}%</span>",
            unsafe_allow_html=True,
            help=(
                f"Доля требований ТЗ, которые Cloud.ru закрывает, если "
                f"строить КП только на платформе **{metrics['platform']}** "
                f"(без миксования с другими). Считается как "
                f"`(match × 2 + partial × 1) / (всего × 2)`. "
                f"Так же считает заказчик при оценке тендера: «насколько "
                f"эта платформа подходит для наших задач»."
            ),
        )
        st.caption(
            f"Доля требований ТЗ, которые закрываются на платформе "
            f"{metrics['platform']}. Счётчики ниже — тоже только по этой платформе."
        )
    else:
        st.markdown(
            f"## Общий процент соответствия портфеля Cloud.ru (best-case): "
            f"<span style='color:{color}'>{metrics['percent']:.1f}%</span>",
            unsafe_allow_html=True,
            help=(
                "По каждому требованию берётся ЛУЧШИЙ вердикт из всех 4 "
                "платформ Cloud.ru. Это «теоретический максимум» — "
                "сколько Cloud.ru может закрыть **в принципе**, если "
                "разрешено мешать платформы (ГосОблако + VMware + Advanced). "
                "Реальное КП обычно строится на одной платформе → "
                "переключитесь в режим «По рекомендуемой платформе»."
            ),
        )
        st.caption(
            "Берётся лучший вердикт по каждому требованию из всех платформ "
            "Cloud.ru. Используйте, если рассматриваете гибридное предложение."
        )
    st.progress(progress_value)

    # Альтернативная цифра — мелким, для контекста.
    if mode == "platform":
        st.markdown(
            f"**Best-case по портфелю Cloud.ru:** "
            f"{rec['general_percent']:.1f}% · "
            f"{rec['score']} из {rec['max_score']} баллов"
        )
    else:
        st.markdown(
            f"**Покрытие на рекомендуемой платформе ({rec['platform']}):** "
            f"{rec['percent']:.1f}%"
        )


def render_summary_table(report: dict):
    """Сводная таблица: всего / match / partial / mismatch / NC.

    Счётчики берутся из выбранного режима (платформа или портфель), чтобы
    цифры в сводке согласовались с процентом в шапке.
    """
    mode = get_metrics_mode()
    metrics = compute_metrics_by_mode(report, mode)
    cols = st.columns(5)
    cols[0].metric(
        "Всего",
        metrics["total"],
        help="Сколько отдельных требований извлекли из ТЗ. Парсер бьёт ТЗ на атомарные пункты — один пункт списка = одно требование.",
    )
    cols[1].metric(
        "✅ Соответствует",
        metrics["match_count"],
        help="Cloud.ru закрывает требование без оговорок. Есть прямое подтверждение в документации (цитата, число, фраза).",
    )
    cols[2].metric(
        "🟡 Частично",
        metrics["partial_count"],
        help="Cloud.ru закрывает на части платформ или ниже целевого уровня. Нужно уточнение в КП или партнёрская услуга.",
    )
    cols[3].metric(
        "❌ Не соответствует",
        metrics["mismatch_count"],
        help="Явное противоречие: нашли значение в Cloud.ru — оно хуже, или Cloud.ru явно не предоставляет. Реальный блокер тендера.",
    )
    cols[4].metric(
        "❓ Уточнить",
        metrics["clarification_count"],
        help="Данных в RAG не хватает или требование сформулировано нечётко. Нужна проверка у клиентского менеджера Cloud.ru или у заказчика.",
    )

    rec = recommended_metric(report)
    if not rec["is_fallback"]:
        # Контекстный subline: показываем оба процента, чтобы пользователь
        # видел разницу.
        st.caption(
            f"📊 Покрытие на **{rec['platform']}**: {rec['percent']:.1f}% · "
            f"Best-case по портфелю: {rec['general_percent']:.1f}% · "
            f"Текущий режим: **{('платформа' if mode == 'platform' else 'портфель')}**"
        )

    # Процедурные пункты закупки (ОКПД, цена, обеспечение заявки и т.п.) —
    # исключены из знаменателя compliance%, но не пропали: показываем их
    # отдельной подсказкой и блоком render_procedural_block ниже.
    procedural_count = int(report.get("procedural_count") or 0)
    if procedural_count > 0:
        total_with_proc = int(report.get("total_with_procedural") or 0) or (metrics["total"] + procedural_count)
        st.caption(
            f"📎 Дополнительно извлечено **{procedural_count}** процедурных пунктов закупки "
            f"(ОКПД, цена, обеспечение заявки, антикоррупция и т.п.) — они вне технической "
            f"оценки Cloud.ru, поэтому не влияют на процент соответствия. "
            f"Всего в исходном ТЗ распарсено: {total_with_proc} пунктов."
        )


def render_suspicious_block(report: dict):
    """Сомнительные места: таблица + RAG-трассировка."""
    suspicious_items = report_suspicious_items(report)
    if not suspicious_items:
        return
    st.divider()
    st.subheader("Сомнительные места")
    st.caption(
        "Пункты, где мало доказательств, слабая релевантность RAG, "
        "низкая уверенность или нужна ручная проработка."
    )
    suspicious_rows = []
    for item in suspicious_items:
        section_short = short_section(item.get("section") or f"#{item.get('requirement_id')}")
        suspicious_rows.append(
            {
                "Пункт ТЗ": section_short,
                "Вердикт": verdict_emoji_label(item.get("verdict", ""), with_label=True),
                "Уверенность": f"{float(item.get('confidence') or 0):.0%}",
                "Причины": "; ".join(item.get("reasons", []) or []),
                "Требование": (item.get("requirement_text") or "")[:240],
            }
        )
    st.dataframe(suspicious_rows, use_container_width=True, hide_index=True)

    with st.expander("Трассировка RAG по сомнительным местам"):
        verdict_by_section = {
            (verdict.get("section") or f"#{verdict.get('requirement_id')}"): verdict
            for verdict in report.get("verdicts", [])
        }
        for item in suspicious_items[:30]:
            section = item.get("section") or f"#{item.get('requirement_id')}"
            verdict = verdict_by_section.get(section, {})
            trace = verdict.get("trace") or {}
            st.markdown(f"**{short_section(section)}**")
            profile = trace.get("profile") or {}
            if profile:
                st.caption(
                    f"Профиль: {profile.get('cluster', 'n/a')}; "
                    f"платформа: {profile.get('platform_hint') or 'не определена'}"
                )
            for source in (trace.get("selected_sources") or [])[:3]:
                title = source.get("title") or "Источник"
                score = source.get("score", 0)
                reasons = "; ".join(source.get("reasons") or [])
                st.markdown(f"- `{score}` **{title}** — {reasons}")


def render_platform_matrix_block(report: dict):
    """Матрица по платформам.

    Клик по строке матрицы → открывается drill-down dialog с полной карточкой
    требования (Streamlit ≥ 1.35 c `on_select="rerun"`). Если selection API
    недоступен — отдельный селектор в `render_drill_down_block` остаётся как
    fallback.

    Внутренние ключи (`_full_*`, `_requirement_id`) скрыты в `column_config` —
    они нужны для drill-down. Сноски RAG рендерятся popover'ами под матрицей.
    """
    platform_matrix = report_platform_matrix(report)
    if not platform_matrix:
        return
    st.divider()
    st.subheader("Матрица по платформам")
    st.caption(
        "`+` соответствует, `±` частично, `-` не подтверждено, `?` уточнить. "
        "**Клик по строке** — открыть полную карточку требования."
    )

    # --- A9: фильтр строк матрицы (вердикт × платформа) ---
    # Два управления над матрицей:
    #   1. Какие вердикты показывать (multiselect)
    #   2. По какой платформе смотреть (selectbox: любая / конкретная)
    # Дефолт: «все вердикты × любая платформа» — видно всю матрицу.
    # Полезные комбинации:
    #   • «❌ + ❓» × «любая» → всё, что не закрыто хотя бы на одной платформе.
    #   • «❌ + ❓» × «Облако VMware» → что именно НЕ закрыто на VMware — это
    #     ровно те пункты, которые блокируют КП на VMware.
    #   • «✅» × «Advanced» → что Advanced точно закрывает.
    VERDICT_FILTER_OPTIONS = {
        "✅ Соответствует (+)": "+",
        "🟡 Частично (±)": "±",
        "❌ Не соответствует (−)": "-",
        "❓ Уточнить (?)": "?",
    }
    platform_names_for_filter = report_platform_names(report)

    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        selected_verdict_filters = st.multiselect(
            "Показать строки с вердиктом:",
            options=list(VERDICT_FILTER_OPTIONS.keys()),
            default=list(VERDICT_FILTER_OPTIONS.keys()),
            key="matrix_verdict_filter",
            help=(
                "По умолчанию показываются все вердикты. Снимите галочки — "
                "оставлены только строки с нужными вердиктами. Например, "
                "оставьте только «❌» и «❓» — увидите всё, что не закрыто."
            ),
        )
    with filter_col2:
        ANY_PLATFORM_LABEL = "Любая платформа"
        platform_filter_options = [ANY_PLATFORM_LABEL] + platform_names_for_filter
        selected_platform_filter = st.selectbox(
            "По какой платформе:",
            options=platform_filter_options,
            index=0,
            key="matrix_platform_filter",
            help=(
                "**Любая платформа** — строка попадает в фильтр, если ВЫБРАННЫЙ "
                "вердикт встретился хотя бы у одной из 4 платформ Cloud.ru. "
                "Полезно когда хочется увидеть «общую картину»: что вообще "
                "не закрыто Cloud.ru.\n\n"
                "**Конкретная платформа** — строка попадает только если ИМЕННО "
                "у этой платформы выбранный вердикт. Полезно, когда КП "
                "строится на одной платформе и нужно увидеть её узкие места."
            ),
        )

    if not selected_verdict_filters:
        st.warning("Выберите хотя бы один вердикт, чтобы увидеть строки матрицы.")
        return

    is_filtered = (
        len(selected_verdict_filters) < len(VERDICT_FILTER_OPTIONS)
        or selected_platform_filter != ANY_PLATFORM_LABEL
    )
    if is_filtered:
        selected_symbols = {VERDICT_FILTER_OPTIONS[label] for label in selected_verdict_filters}

        if selected_platform_filter == ANY_PLATFORM_LABEL:
            # Хотя бы одна из платформ имеет выбранный вердикт.
            def _row_matches(row):
                for pname in platform_names_for_filter:
                    cell = (row.get(pname) or "").strip()
                    first_char = cell[0] if cell else ""
                    if first_char in selected_symbols:
                        return True
                return False
        else:
            # Конкретная платформа имеет выбранный вердикт.
            def _row_matches(row):
                cell = (row.get(selected_platform_filter) or "").strip()
                first_char = cell[0] if cell else ""
                return first_char in selected_symbols

        platform_matrix = [r for r in platform_matrix if _row_matches(r)]
        if not platform_matrix:
            st.info(
                "Нет требований с выбранными параметрами фильтра. "
                "Попробуйте расширить список вердиктов или выбрать другую платформу."
            )
            return

        scope_text = (
            "хотя бы по одной платформе"
            if selected_platform_filter == ANY_PLATFORM_LABEL
            else f"по платформе «{selected_platform_filter}»"
        )
        st.caption(
            f"Отфильтровано: **{len(platform_matrix)}** требований "
            f"({', '.join(selected_verdict_filters)} {scope_text})."
        )

    column_config = {
        "Пункт ТЗ": st.column_config.TextColumn(
            "Пункт ТЗ",
            width="small",
            help=(
                "Сокращённый идентификатор раздела ТЗ (например, «7.2.3», "
                "«ВВИ / пункт 4», «Таблица 1, стр. 3»). Полный путь в "
                "оригинальном ТЗ — кликни по строке."
            ),
        ),
        "Категория": st.column_config.TextColumn(
            "Категория",
            width="small",
            help=(
                "Тип требования: **Техническое** (vCPU, RAM, сеть), **SLA** "
                "(доступность, время реакции), **ИБ** (аттестаты, лицензии, "
                "защита данных), **Юридическое** (договорные нюансы), "
                "**Прочее**. Категорию определяет парсер по ключевым словам."
            ),
        ),
        "Требование": st.column_config.TextColumn(
            "Требование",
            width="large",
            help=(
                "Эвристически сжатая формулировка требования (например, "
                "«vCPU на 1 ВМ — до 24 шт.», «SLA: критичный приоритет ≤ 2 ч»). "
                "Полный текст ТЗ — кликни по строке."
            ),
        ),
        "_full_section": None,
        "_full_text": None,
        "_requirement_id": None,
    }
    platform_names = report_platform_names(report)
    platform_descriptions = {
        "ГосОблако": "Cloud.ru ГосОблако (ГИС ГТ) — аттестована под ГИС К1, УЗ-1, КИИ-1. Для госсектора.",
        "Облако VMware": "Cloud.ru Облако VMware — vCloud Director на VMware, аттестация К1/УЗ-1, SLA 99.982%.",
        "Advanced": "Cloud.ru Advanced — OpenStack-стэк (EVS/ECS), УЗ-1 без К1, высокие лимиты на 1 ВМ.",
        "Evolution": "Cloud.ru Evolution — массовая публичная платформа, УЗ-1 без К1, Compute SLA 99.9%.",
    }
    for name in platform_names:
        desc = platform_descriptions.get(name, name)
        column_config[name] = st.column_config.TextColumn(
            name,
            width="small",
            help=(
                f"{desc}\n\n"
                f"Вердикт: `+` соответствует, `±` частично (нужно уточнение/доработка), "
                f"`-` не соответствует (блокер), `?` уточнить у клиентского менеджера. "
                f"`[N]` — номер RAG-источника, по которому LLM сделала вывод."
            ),
        )

    height = min(700, max(280, 38 * (len(platform_matrix) + 1)))

    # Streamlit ≥ 1.35 поддерживает on_select на st.dataframe — это и есть
    # «клик по строке». Если фича недоступна — рендерим обычный dataframe
    # без selection, и пользователь использует селектор в render_drill_down_block.
    selection_supported = True
    selected_row_idx: int | None = None
    try:
        event = st.dataframe(
            platform_matrix,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=height,
            on_select="rerun",
            selection_mode="single-row",
            key="platform_matrix_dataframe",
        )
        if hasattr(event, "selection") and getattr(event.selection, "rows", None):
            rows = list(event.selection.rows)
            if rows:
                selected_row_idx = int(rows[0])
    except TypeError:
        # Старый Streamlit без on_select — обычный рендер.
        selection_supported = False
        st.dataframe(
            platform_matrix,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=height,
        )

    if not selection_supported or selected_row_idx is None:
        return

    # Защита от «диалог открывается на каждом rerun'е»: запоминаем последний
    # row_idx, по которому ОТКРЫВАЛИ диалог. Только если новый row_idx
    # отличается от запомненного — открываем дилог.
    last_key = "platform_matrix_last_dialog_row"
    last_handled = st.session_state.get(last_key)
    if selected_row_idx == last_handled:
        return

    if selected_row_idx >= len(platform_matrix):
        return
    row = platform_matrix[selected_row_idx]
    requirement_id = row.get("_requirement_id")
    verdicts = report.get("verdicts") or []
    selected_verdict = next(
        (v for v in verdicts if v.get("requirement_id") == requirement_id),
        None,
    )
    if not selected_verdict:
        return

    refs = report_reference_map(report)

    # Сохраняем индекс ДО открытия dialog'а, чтобы при следующем rerun
    # (если пользователь не снял selection) мы не открывали повторно.
    st.session_state[last_key] = selected_row_idx

    if hasattr(st, "dialog"):
        @st.dialog(
            f"Карточка требования: {short_section(selected_verdict.get('section') or '')}",
            width="large",
        )
        def _drill_dialog():
            _render_drill_down_body(selected_verdict, refs)
        try:
            _drill_dialog()
            return
        except Exception:
            pass

    # Fallback: показать в expander под матрицей.
    with st.expander("Карточка требования", expanded=True):
        _render_drill_down_body(selected_verdict, refs)


_HTML_ENTITY_QUOTE = "&#34;"


def _decode_excerpt(raw: str) -> tuple[str, str, str]:
    """Расшифровывает excerpt из RAG (часто это HTML-entity-encoded JSON-блоб
    типа `{"title": "...", "description": "...", "meta_product": "..."}`).

    Возвращает (clean_excerpt, fallback_title, fallback_url):
    - clean_excerpt — человекочитаемый текст (description / content), без
      JSON-обёртки. Если парсинг не удался — возвращаем raw как есть.
    - fallback_title — title или meta_product из JSON, если они есть.
    - fallback_url — link / url из JSON.
    """
    if not raw:
        return "", "", ""
    text = raw
    if _HTML_ENTITY_QUOTE in text:
        try:
            import html
            text = html.unescape(text)
        except Exception:
            pass
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return text, "", ""
    try:
        obj = json.loads(stripped)
    except Exception:
        return text, "", ""
    if not isinstance(obj, dict):
        return text, "", ""
    description = (
        obj.get("description")
        or obj.get("content")
        or obj.get("text")
        or ""
    )
    fallback_title = (
        obj.get("title")
        or obj.get("meta_document")
        or obj.get("meta_product")
        or ""
    )
    fallback_url = obj.get("link") or obj.get("url") or ""
    if isinstance(description, str) and description.strip():
        return description.strip(), str(fallback_title), str(fallback_url)
    # Если description пустой — собираем читаемое summary из метаданных.
    parts = []
    if obj.get("meta_product"):
        parts.append(f"Продукт: {obj['meta_product']}")
    if obj.get("meta_document"):
        parts.append(f"Документ: {obj['meta_document']}")
    if isinstance(obj.get("meta_platform"), list) and obj["meta_platform"]:
        parts.append("Платформа: " + ", ".join(obj["meta_platform"]))
    if obj.get("meta_tags"):
        parts.append(f"Теги: {obj['meta_tags']}")
    if parts:
        return " · ".join(parts), str(fallback_title), str(fallback_url)
    return text, str(fallback_title), str(fallback_url)


def collect_rag_sources(report: dict) -> list[dict]:
    """Возвращает список источников RAG, которые упоминаются как сноски в
    отчёте, с обогащением title/url/excerpt из платформенных assessment'ов
    и trace.selected_sources.

    Структура элемента: {index, key, title, url, excerpt, score}.
    """
    refs = report_reference_map(report)
    if not refs:
        return []
    enriched: dict[str, dict] = {}
    # 1. Сначала идём по trace.selected_sources всех verdict'ов — там есть
    #    title, url, excerpt, score (это самый богатый источник данных).
    for verdict in report.get("verdicts", []) or []:
        trace = verdict.get("trace") or {}
        for source in (trace.get("selected_sources") or []):
            if not isinstance(source, dict):
                continue
            url = source.get("url") or ""
            title = source.get("title") or ""
            key = url or title
            if not key:
                continue
            if key not in refs:
                continue
            existing = enriched.get(key) or {}
            existing.setdefault("title", title)
            existing.setdefault("url", url)
            existing.setdefault("excerpt", source.get("excerpt") or "")
            score = source.get("score")
            if score is not None and "score" not in existing:
                existing["score"] = score
            enriched[key] = existing
    # 2. Дополняем недостающие из platform_assessments — там есть title без excerpt.
    for verdict in report.get("verdicts", []) or []:
        for assessment in verdict.get("platform_assessments", []) or []:
            urls = assessment.get("source_urls") or []
            titles = assessment.get("source_titles") or []
            for u, t in zip(urls, titles + [""] * len(urls)):
                key = u or t
                if not key or key not in refs:
                    continue
                existing = enriched.get(key) or {}
                existing.setdefault("url", u or "")
                existing.setdefault("title", t or "")
                enriched[key] = existing
    # 3. Собираем итоговый список, отсортированный по индексу сноски.
    result = []
    for key, index in sorted(refs.items(), key=lambda item: item[1]):
        meta = enriched.get(key, {})
        url = meta.get("url") or (key if str(key).startswith("http") else "")
        title = meta.get("title") or (key if not str(key).startswith("http") else "")
        result.append({
            "index": index,
            "key": key,
            "url": url,
            "title": title,
            "excerpt": meta.get("excerpt") or "",
            "score": meta.get("score"),
        })
    return result


def render_rag_references_popovers(report: dict):
    """Рендерит сноски RAG как горизонтальную полосу popover'ов.

    Клик по `[N]` открывает мини-окно с заголовком, URL и (если есть)
    кратким excerpt'ом из retrieved chunk. Удобно: пользователь видит
    `[4]` в матрице → ищет глазами `[4]` под матрицей → клик = деталь.
    """
    sources = collect_rag_sources(report)
    if not sources:
        return
    st.markdown("##### 🔖 Сноски RAG")
    st.caption("Кликни по номеру, чтобы посмотреть документ Cloud.ru, на который опирался анализ.")
    # Все источники в одну строку с горизонтальным скроллом (CSS-класс
    # rag-references-host) — экономит вертикальное место и удобнее даже
    # при 20-25 источниках.
    st.markdown('<div class="rag-references-host">', unsafe_allow_html=True)
    cols = st.columns(max(len(sources), 1))
    for i, src in enumerate(sources):
        label = f"[{src['index']}]"
        with cols[i]:
            if hasattr(st, "popover"):
                with st.popover(label, use_container_width=True):
                    raw_excerpt = (src.get("excerpt") or "").strip()
                    clean_excerpt, fb_title, fb_url = _decode_excerpt(raw_excerpt)
                    title = (
                        src.get("title")
                        or fb_title
                        or src.get("key")
                        or "Источник"
                    )
                    st.markdown(f"**{title}**")
                    url = src.get("url") or fb_url
                    if url:
                        st.markdown(f"🔗 [{url}]({url})")
                    if src.get("score") is not None:
                        try:
                            score_val = float(src["score"])
                            st.caption(f"RAG score: {score_val:.2f}")
                        except (TypeError, ValueError):
                            pass
                    if clean_excerpt:
                        st.markdown("**Фрагмент:**")
                        st.markdown(f"> {clean_excerpt[:600]}")
            else:
                if src.get("url"):
                    st.markdown(f"{label} [{src.get('title') or src['url']}]({src['url']})")
                else:
                    st.markdown(f"{label} {src.get('title') or src.get('key') or ''}")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_drill_down_body(verdict: dict, refs: dict[str, int]):
    """Содержимое модалки drill-down по одному verdict'у. Используется и
    внутри st.dialog (новые Streamlit), и как inline-блок (fallback).
    """
    section = verdict.get("section") or f"#{verdict.get('requirement_id')}"
    overall = verdict.get("verdict", "")
    confidence = float(verdict.get("confidence") or 0)

    st.markdown(f"#### {short_section(section)}")
    st.caption(f"Полный путь в ТЗ: `{section}`")

    # Шапка — overall verdict + категория + уверенность.
    head_cols = st.columns([1, 1, 2])
    head_cols[0].markdown(
        f"**Вердикт:** {verdict_emoji_label(overall, with_label=True)}",
        help="Итоговый вердикт по требованию — best-case по 4 платформам Cloud.ru.",
    )
    head_cols[1].markdown(
        f"**Категория:** {category_label(verdict.get('category', ''))}",
        help="К какой группе ТЗ относится требование (определяет парсер).",
    )
    head_cols[2].markdown(
        f"**Уверенность:** {confidence:.0%}",
        help=(
            "Насколько LLM уверена в вердикте. **90-100%** — есть прямое "
            "подтверждение в RAG. **55-70%** — вердикт построен на косвенных "
            "признаках, есть смысл уточнить. **<50%** — рекомендую ручную "
            "проверку у клиентского менеджера."
        ),
    )

    st.markdown("**Текст требования:**")
    st.markdown(f"> {verdict.get('requirement_text') or '—'}")

    if verdict.get("reasoning"):
        st.markdown("**Обоснование:**")
        st.markdown(verdict["reasoning"])

    if verdict.get("evidence"):
        st.markdown("**Цитата из источников (evidence):**")
        st.markdown(f"> {verdict['evidence']}")

    if verdict.get("recommendation"):
        st.markdown("**Рекомендация:**")
        st.markdown(verdict["recommendation"])

    # Платформенные assessment'ы — каждая платформа со своим вердиктом.
    platform_assessments = verdict.get("platform_assessments") or []
    if platform_assessments:
        st.divider()
        st.markdown("**Оценки по платформам Cloud.ru:**")
        for assessment in platform_assessments:
            name = assessment.get("platform_name") or "—"
            v = assessment.get("verdict", "")
            conf = float(assessment.get("confidence") or 0)
            st.markdown(
                f"- **{name}** — {verdict_emoji_label(v, with_label=True)} "
                f"(уверенность {conf:.0%})"
            )
            reasoning = assessment.get("reasoning") or ""
            if reasoning:
                st.caption(reasoning)
            evidence_refs = assessment.get("evidence_refs") or []
            source_titles = assessment.get("source_titles") or []
            if evidence_refs or source_titles:
                pieces = []
                if evidence_refs:
                    pieces.append("Сноски: " + " ".join(evidence_refs))
                if source_titles:
                    pieces.append("Документы: " + ", ".join(source_titles[:3]))
                st.caption(" · ".join(pieces))

    # Если есть requires_external_service — показываем партнёрские услуги.
    if verdict.get("requires_external_service"):
        st.divider()
        st.markdown("**🤝 Требуется партнёрская услуга:**")
        st.markdown(verdict.get("external_service_notes") or "Без подробностей")

    # RAG-трассировка: title, url, score, excerpt каждого выбранного источника.
    trace = verdict.get("trace") or {}
    selected_sources = trace.get("selected_sources") or []
    if selected_sources:
        st.divider()
        st.markdown("**🔍 RAG-источники, на которых построен вердикт:**")
        for idx, source in enumerate(selected_sources[:6], start=1):
            if not isinstance(source, dict):
                continue
            title = source.get("title") or f"Источник {idx}"
            url = source.get("url") or ""
            score = source.get("score", 0)
            excerpt = (source.get("excerpt") or "").strip()
            ref_idx = refs.get(url) or refs.get(title) or idx
            st.markdown(f"- **[{ref_idx}] {title}** — score `{score}`")
            if url:
                st.markdown(f"  🔗 [{url}]({url})")
            if excerpt:
                st.markdown(f"  > {excerpt[:400]}")

    # Кнопка «Спросить чат об этом пункте» — pre-fills поле ввода.
    st.divider()
    if st.button(
        f"💬 Спросить чат об этом пункте",
        use_container_width=True,
        key=f"ask_chat_about_{verdict.get('requirement_id')}",
    ):
        st.session_state.pending_analysis_question = (
            f"Расскажи подробнее, почему пункт «{short_section(section)}» получил вердикт "
            f"«{verdict_emoji_label(overall, with_label=True)}». Какие источники RAG использовались "
            f"и что из этого следует для коммерческого предложения?"
        )
        st.rerun()


def render_drill_down_block(report: dict):
    """Селектор пункта ТЗ + модалка с полной детализацией (drill-down).

    UX-Блок 8: пресейл может выбрать любой пункт ТЗ из dropdown и увидеть
    полный verdict, все platform_assessments, источники, рекомендацию.
    Самое полное место отчёта — раньше это было «погребено» в скачанном
    Markdown.
    """
    verdicts = report.get("verdicts") or []
    if not verdicts:
        return
    refs = report_reference_map(report)
    st.divider()
    st.markdown("### 🔬 Углубиться в пункт ТЗ")
    st.caption("Выбери пункт — откроется полная карточка вердикта со всеми сноcками RAG.")

    # Опции для селектора: «{короткий ID} ({категория}) — {начало текста}».
    options = []
    by_id: dict[int, dict] = {}
    for v in verdicts:
        rid = v.get("requirement_id")
        if rid is None:
            continue
        section = v.get("section") or f"#{rid}"
        category = category_label(v.get("category", ""))
        head = (v.get("requirement_text") or "")[:80]
        ev_emoji = verdict_emoji_label(v.get("verdict", ""))
        options.append(rid)
        by_id[rid] = {
            "_label": f"{ev_emoji} {short_section(section)} · {category} · {head}",
            "verdict": v,
        }

    if not options:
        return
    selected = st.selectbox(
        "Пункт ТЗ",
        options,
        format_func=lambda rid: by_id[rid]["_label"] if rid in by_id else f"#{rid}",
        key="drill_down_selected_id",
    )
    if selected is None:
        return

    selected_verdict = by_id[selected]["verdict"]

    # st.dialog (≥ 1.36) — модалка. Открываем СТРОГО по клику на кнопку
    # (`st.button` возвращает True только в тот rerun, в который её
    # реально нажали). Никаких флагов в session_state — иначе при любом
    # последующем rerun'е (например, после suggested-вопроса в чате)
    # диалог открывался бы повторно, что и было багом UX.
    if hasattr(st, "dialog"):
        # Кнопка с ключом, привязанным к id требования: при смене требования
        # в селекторе старая кнопка демонтируется, новая поднимается.
        open_clicked = st.button(
            "Открыть полную карточку",
            use_container_width=True,
            key=f"open_drill_{selected}",
        )
        if open_clicked:
            @st.dialog(
                f"Карточка требования: {short_section(selected_verdict.get('section') or '')}",
                width="large",
            )
            def _drill_dialog():
                _render_drill_down_body(selected_verdict, refs)

            try:
                _drill_dialog()
            except Exception:
                # Streamlit < 1.36 или иной отказ dialog API — fallback
                # на inline-expander под кнопкой.
                with st.expander("Карточка требования", expanded=True):
                    _render_drill_down_body(selected_verdict, refs)
    else:
        # Fallback: показываем содержимое в expander под селектором.
        with st.expander("Карточка требования", expanded=True):
            _render_drill_down_body(selected_verdict, refs)


_EXTERNAL_SERVICE_KIND_RULES = [
    ("Colocation / физический ЦОД", re.compile(r"colocation|размещени.*оборудовани|физическ.*доступ.*ЦОД|стойк[аи]|юнит", re.IGNORECASE)),
    ("Лицензированный оператор связи", re.compile(r"оператор.*связ|лицензи.*связ|роскомнадзор|канал.*связ", re.IGNORECASE)),
    ("Защита информации / ИБ-подрядчик", re.compile(r"защит.*информаци|ИБ-подрядчик|серт.*ИБ|аттестац", re.IGNORECASE)),
    ("Поставщик ПО / лицензии", re.compile(r"лицензи.*ПО|поставк.*ПО|прав.*использован", re.IGNORECASE)),
]


def _classify_external_service(text: str) -> str:
    """Эвристически определяет тип партнёрской услуги. Используется для
    группировки строк блока «Внешние услуги / подрядчики», чтобы пресейл
    видел сразу 3-4 категории, а не плоский список из 40+ требований."""
    for label, pattern in _EXTERNAL_SERVICE_KIND_RULES:
        if pattern.search(text or ""):
            return label
    return "Прочее"


def render_external_services_block(report: dict):
    """Компактный блок «Внешние услуги / подрядчики».

    UX-Блок A8: вместо плоского списка из всех `requires_external_service`
    требований (часто 30-40+ строк) — группировка по типу услуги
    (colocation / телеком / ИБ-подрядчик). Внутри каждой группы — список
    section'ов в одну строку через запятую, без обрезанного текста
    требования (полный текст уже есть в drill-down карточке матрицы).
    """
    external_items = [
        verdict for verdict in report.get("verdicts", [])
        if verdict.get("requires_external_service")
    ]
    if not external_items:
        return

    # Группируем по типу услуги. Тип определяется по external_service_notes
    # с fallback'ом на requirement_text.
    groups: dict[str, list[dict]] = {}
    for verdict in external_items:
        notes = verdict.get("external_service_notes") or ""
        signal = notes if notes.strip() else (verdict.get("requirement_text") or "")
        kind = _classify_external_service(signal)
        groups.setdefault(kind, []).append(verdict)

    # Шапка — общий счётчик и разворачиваемый блок с группами.
    with st.expander(
        f"🤝 Внешние услуги / подрядчики ({len(external_items)} требований, "
        f"{len(groups)} категорий)"
    ):
        st.caption(
            "🤝 Это требования, которые сама платформа Cloud.ru напрямую не закрывает, "
            "но они решаются через **партнёрские услуги** (Сбер colocation, "
            "лицензированные операторы связи, ИБ-подрядчики). Включите их в КП "
            "как отдельные пункты предложения — без этого тендер не закроется. "
            "Полный текст каждого требования — кликом по строке в матрице платформ выше."
        )
        # Сортируем группы по убыванию числа требований.
        for kind in sorted(groups, key=lambda k: -len(groups[k])):
            items = groups[kind]
            # Уникальные external_service_notes внутри группы — короткое
            # перечисление, без дубликатов.
            unique_notes = sorted(
                {(v.get("external_service_notes") or "").strip() for v in items} - {""}
            )
            note_line = " · ".join(unique_notes) if unique_notes else "—"
            sections = [short_section(v.get("section") or f"#{v.get('requirement_id')}") for v in items]
            # Уникальные section'ы (бывает дубликаты после парсера),
            # сохраняем порядок появления.
            unique_sections = list(dict.fromkeys(sections))
            preview = ", ".join(unique_sections[:8])
            tail = "" if len(unique_sections) <= 8 else f" + ещё {len(unique_sections) - 8}"
            st.markdown(
                f"**{kind}** ({len(items)}) — "
                f"<span style='color:#666'>{note_line}</span><br>"
                f"<small>{preview}{tail}</small>",
                unsafe_allow_html=True,
            )


def render_procedural_block(report: dict):
    """Блок «Процедурные пункты закупки».

    Показывает пункты, помеченные парсером как category='procedural':
    ОКПД, цена, обеспечение заявки, антикоррупция, идентификация участников
    и т.п. Они исключены из знаменателя compliance%, но не пропали:
    tech-sales должен видеть их в отчёте, чтобы передать коммерческой/
    правовой команде Cloud.ru при подготовке КП.

    Пункты сворачиваются в expander (по умолчанию закрыт), чтобы не
    отвлекать от технической части. Группируем по разделу первого уровня
    (например, «1.», «2.»), чтобы пресейл сразу видел структуру.
    """
    procedural_items = [
        verdict for verdict in report.get("verdicts", []) or []
        if (verdict.get("category") or "").lower() == "procedural"
        or verdict.get("verdict") == "out_of_scope"
    ]
    if not procedural_items:
        return

    # Группируем по first-level section (до первой точки): «1», «3.7» → «3».
    groups: dict[str, list[dict]] = {}
    for verdict in procedural_items:
        section = (verdict.get("section") or "").strip()
        # Извлекаем самый верхний номер «1», «3», «4».
        match = re.match(r"^\s*(\d+)", section)
        top = match.group(1) if match else "—"
        groups.setdefault(top, []).append(verdict)

    with st.expander(
        f"📎 Процедурные пункты закупки ({len(procedural_items)} требований) — "
        f"вне технической оценки"
    ):
        st.caption(
            "📎 Эти пункты относятся к **коммерческо-правовой обвязке тендера** "
            "(ОКПД, начальная максимальная цена, обеспечение заявки, антикоррупция, "
            "идентификация участников закупки, реквизиты сторон). Они **не оценивают "
            "технические возможности Cloud.ru** и поэтому исключены из процента "
            "соответствия. Передайте этот список в коммерческую/правовую команду "
            "Cloud.ru при подготовке КП — там не должно быть пропусков."
        )
        rows = []
        for verdict in procedural_items:
            section = verdict.get("section") or f"#{verdict.get('requirement_id')}"
            text = (verdict.get("requirement_text") or "").strip()
            preview = text if len(text) <= 200 else text[:197].rstrip() + "…"
            rows.append({
                "Пункт ТЗ": short_section(section),
                "Требование": preview,
            })
        if rows:
            st.dataframe(
                rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Пункт ТЗ": st.column_config.TextColumn(width="small"),
                    "Требование": st.column_config.TextColumn(width="large"),
                },
            )


def render_methodology_expander(report: dict):
    """Свёрнутый блок «Методика оценки» — что значат баллы."""
    with st.expander("ℹ️ Методика оценки — как считается процент покрытия"):
        st.markdown(
            "**Принцип:** каждое требование даёт от 0 до 2 баллов.\n\n"
            "| Вердикт | Баллы | Что это значит для пресейла |\n"
            "|---|---|---|\n"
            "| ✅ Полное соответствие (match) | **2** | Можно смело брать в КП — Cloud.ru закрывает без оговорок. |\n"
            "| 🟡 Частичное (partial) | **1** | В КП с пометкой «доработка/уточнение» — Cloud.ru закрывает на части платформ или ниже целевого уровня. |\n"
            "| ❌ Не соответствует (mismatch) | **0** | Блокер тендера — нужно либо менять платформу, либо привлекать партнёра (colocation, оператор связи). |\n"
            "| ❓ Требует уточнения (NC) | **0** | Пресейлу нужно уточнить у клиентского менеджера Cloud.ru или у заказчика. |\n\n"
            f"**Максимальный балл** = {report.get('total', 0)} требований × 2 = "
            f"**{report.get('max_score', 0)}**. Набрано **{report.get('score', 0)}** баллов.\n\n"
            "Процент покрытия = `набрано / максимум × 100%`. "
            "Это упрощённая шкала: даже одно `❌` на критичное требование может "
            "сорвать тендер — смотрите блок **Ключевые блокеры** в Резюме."
        )


def render_full_report(report: dict):
    """Главный композит отчёта — собирает все блоки в правильном порядке.

    Используется на табе «Анализ» после завершения анализа. Блоки внутри
    самостоятельны: каждый можно вынести в фрагмент или drill-down dialog
    в последующих UX-блоках без изменений снаружи.
    """
    if report is None:
        st.info("Загрузите ТЗ и запустите анализ — отчёт появится здесь.")
        return
    render_metrics_mode_selector(report)
    render_report_header(report)
    render_methodology_expander(report)
    st.divider()
    render_summary_table(report)
    # Блок A7: переключаемое резюме — два варианта на бэкенде, фронт
    # выбирает нужный по текущему режиму шапки. Поле `summary` остаётся
    # для обратной совместимости (старые отчёты, где раздельных нет).
    mode = get_metrics_mode()
    summary_platform = (report.get("summary_platform") or "").strip()
    summary_portfolio = (report.get("summary_portfolio") or "").strip()
    summary_fallback = (report.get("summary") or "").strip()

    if mode == "platform":
        chosen = summary_platform or summary_portfolio or summary_fallback
        mode_label = "По рекомендуемой платформе"
    else:
        chosen = summary_portfolio or summary_fallback or summary_platform
        mode_label = "Best-case по портфелю Cloud.ru"

    if chosen:
        st.markdown("### Резюме")
        # Подсказка: каким режимом сейчас живёт резюме. И показ обоих —
        # маленькая плашка с альтернативным резюме (через expander).
        rec = recommended_metric(report)
        if mode == "platform" and not summary_platform and not rec["is_fallback"]:
            st.caption(
                "ℹ️ Резюме под платформенный режим ещё не сгенерировано "
                "(старый прогон до dual-summary). Показано best-case резюме."
            )
        else:
            st.caption(f"ℹ️ Режим: **{mode_label}**. Переключатель в шапке выше.")
        st.markdown(chosen)

        # Альтернативное резюме под спойлером — для пресейла, который
        # хочет сравнить «жёсткий» вариант с «гибридным» в одном экране.
        alt = summary_portfolio if mode == "platform" else summary_platform
        if alt and alt != chosen:
            alt_label = "Best-case по портфелю" if mode == "platform" else "По рекомендуемой платформе"
            with st.expander(f"Альтернативное резюме: {alt_label}"):
                st.markdown(alt)
    # render_suspicious_block и render_drill_down_block (отдельный
    # селектор) убраны — каждое требование доступно через клик по строке
    # матрицы (drill-down dialog), включая reasons / RAG-трассировку.
    # Сами функции оставлены в коде как fallback, на случай если понадобятся.
    render_platform_matrix_block(report)
    # Блок A6: RAG-сноски popover'ами убраны из основного отчёта —
    # дубликат с теми же ссылками в drill-down карточке требования. Если
    # пресейлу нужны источники, он открывает строку матрицы и видит
    # сноски прямо там, в контексте конкретного требования.
    render_external_services_block(report)
    render_procedural_block(report)
    st.divider()
    render_download_controls()


def _render_chat_history():
    """Рендер истории сообщений в стиле ChatGPT — без expander'ов, просто
    последовательные сообщения user/assistant. Самые свежие — снизу.
    """
    history = st.session_state.analysis_qa_history or []
    if not history:
        st.caption("История пока пуста. Задай вопрос — ответ и источники появятся здесь.")
        return
    # st.chat_message — нативный Streamlit-стиль для чатов, появляется в виде
    # карточек user/assistant. Доступен с Streamlit ≥ 1.24, гарантировано
    # в нашем образе.
    for item in history:
        with st.chat_message("user"):
            st.markdown(item.get("question", ""))
        with st.chat_message("assistant"):
            st.markdown(item.get("answer", ""))
            related = item.get("related_sections") or []
            if related:
                st.caption(
                    "Связанные пункты ТЗ: " + ", ".join(short_section(s) for s in related[:8])
                )
            urls = item.get("source_urls") or []
            if urls:
                st.markdown("**Источники:**")
                for url in urls[:8]:
                    if str(url).startswith("http"):
                        st.markdown(f"- [{url}]({url})")
                    else:
                        st.markdown(f"- {url}")


def _ask_question(report: dict, question: str):
    """Отправляет вопрос на /analysis/ask и добавляет в историю чата."""
    if not question.strip():
        return
    try:
        with st.spinner("Готовлю ответ..."):
            answer = api_post_json(
                "/analysis/ask",
                {
                    "question": question.strip(),
                    "report": report,
                    "requirements": st.session_state.requirements or [],
                    "search_mode": st.session_state.analysis_search_mode,
                    "history": st.session_state.analysis_qa_history,
                    "llm_settings": llm_settings_payload(),
                },
                timeout=3600,
            )
        st.session_state.analysis_qa_history.append(
            {
                "question": question.strip(),
                "answer": answer.get("answer", ""),
                "related_sections": answer.get("related_sections", []),
                "source_urls": answer.get("source_urls", []),
            }
        )
        # Сбрасываем поле ввода, чтобы не отправилось второй раз.
        st.session_state.pending_analysis_question = None
    except Exception as exc:
        show_request_error(exc)


def _chat_body(report: dict):
    """Тело чата — отделено от декоратора, чтобы fallback без st.fragment
    тоже работал.
    """
    st.caption("Спроси по конкретному пункту ТЗ, спорному вердикту или источникам RAG.")

    # История сообщений (как в ChatGPT).
    _render_chat_history()

    st.divider()

    # Suggested questions — вертикально, под узкую правую колонку.
    suggested_questions = [
        "Почему этот пункт признан несоответствием?",
        "Что перепроверить вручную?",
        "Какие 3 самых рискованных пункта?",
    ]
    for idx, question in enumerate(suggested_questions):
        if st.button(question, key=f"suggested_question_{idx}", use_container_width=True):
            _ask_question(report, question)
            st.rerun()

    # Pending-вопрос (если задан кнопкой «связать с пунктом» из drill-down).
    apply_pending_analysis_question()

    st.text_area(
        "Свой вопрос",
        key="analysis_question_input",
        height=110,
        placeholder=(
            "Например: «Почему 8.1.3 отмечен как partial?» или "
            "«Назови 3 главных открытых вопроса заказчику»"
        ),
    )
    ask_disabled = not (st.session_state.get("analysis_question_input") or "").strip()
    if st.button("Спросить", disabled=ask_disabled, use_container_width=True, key="chat_ask_btn"):
        current_question = st.session_state.analysis_question_input.strip()
        _ask_question(report, current_question)
        st.rerun()


def render_analysis_chat(report: dict):
    """Чат по анализу. Раньше оборачивался в `@st.fragment` (когда был в
    правой колонке отчёта), чтобы скролл основного отчёта не сбрасывался.
    После переезда в sidebar (Блок A6) фрагмент не нужен — sidebar и так
    отдельная зона, плюс `@st.fragment` внутри `with st.sidebar:` ломал
    обработчики кнопок (нажатие «Спросить» не запускало запрос). Теперь
    просто вызываем тело напрямую — обычный rerun отрабатывает нормально.
    """
    _chat_body(report)


def show_request_error(exc: Exception):
    detail = str(exc)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
            detail = payload.get("detail", detail)
        except ValueError:
            detail = response.text or detail
    st.error(detail)


st.set_page_config(
    page_title="Cloud.ru — Анализ ТЗ",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",  # Sidebar теперь хост чата (см. блок A6)
)
init_state()

# Блок A6: чат теперь живёт в sidebar (см. ниже `with st.sidebar`),
# где переключатель «Чат / Настройки» в шапке сайдбара выбирает режим.
# Sticky-CSS оставлен только для совместимости — фактически старая
# `.sticky-chat-host` больше не используется (правая колонка с чатом
# удалена из основного потока отчёта). Этот блок отвечает только за CSS
# RAG-сносок и вспомогательные стили.
st.markdown(
    """
    <style>
    /* RAG-сноски popover'ы — компактнее, в одну линию со скроллом */
    .rag-references-host [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    .rag-references-host [data-testid="stHorizontalBlock"] > div {
        flex: 0 0 auto;
        min-width: 80px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if hasattr(st, "fragment"):
    @st.fragment(run_every="3s")
    def render_live_run_panel():
        active_run = st.session_state.active_run or {}
        if (
            st.session_state.get("selected_run_id")
            and st.session_state.get("auto_refresh_run")
            and active_run.get("status") in {"queued", "extracting", "analyzing"}
        ):
            try:
                refresh_result = refresh_selected_run_status()
                if refresh_result and refresh_result.get("full_refreshed"):
                    st.rerun()
            except Exception:
                pass
        render_current_run_panel()
else:
    def render_live_run_panel():
        render_current_run_panel()

backend_health = None
backend_error = None
backend_health_stale = False
try:
    backend_health = cached_backend_health(st.session_state.backend_api_url)
    st.session_state.last_backend_health = backend_health
except Exception as exc:
    backend_error = str(exc)
    backend_health = st.session_state.last_backend_health
    backend_health_stale = backend_health is not None

if backend_health and not backend_health_stale:
    try:
        load_persisted_settings_once()
    except Exception as exc:
        st.session_state.last_settings_save_error = f"Не удалось загрузить сохранённые настройки: {exc}"

with st.sidebar:
    # Блок A6: переключатель «Чат / Настройки» в шапке сайдбара. По умолчанию
    # — Чат (если есть готовый отчёт), иначе — Настройки. Радио хранится в
    # session_state, чтобы переключение не сбрасывалось на rerun'ах.
    has_report = bool(st.session_state.get("analysis_report"))
    if "sidebar_view" not in st.session_state:
        st.session_state.sidebar_view = "💬 Чат" if has_report else "⚙️ Настройки"
    st.title("☁️ Cloud.ru TZ Analyzer")
    sidebar_view = st.radio(
        "Боковая панель",
        ["💬 Чат", "⚙️ Настройки"],
        horizontal=True,
        key="sidebar_view",
        label_visibility="collapsed",
    )
    st.divider()

if sidebar_view == "💬 Чат":
  with st.sidebar:
    if has_report:
        render_analysis_chat(st.session_state.analysis_report)
    else:
        st.info(
            "Загрузите ТЗ и запустите анализ — после этого здесь появится "
            "чат для уточняющих вопросов по отчёту."
        )
        st.caption(
            "Переключитесь на «Настройки», чтобы проверить параметры "
            "Backend / RAG / LLM перед прогоном."
        )

if sidebar_view == "⚙️ Настройки":
  with st.sidebar:
    st.markdown("UI работает через отдельный backend API")
    st.divider()

    st.subheader("Backend API")
    st.text_input("Backend URL", key="backend_api_url")

    if backend_health and not backend_health_stale:
        st.success(
            f"Backend доступен: {backend_health.get('status', 'ok')}, "
            f"RAG: {backend_health.get('rag_provider', 'managed_rag')}"
        )
    elif backend_health_stale:
        st.warning(
            "Backend отвечает медленно. Использую последний успешный статус; "
            "долгий анализ может продолжаться в фоне."
        )
    else:
        st.error(f"Backend недоступен: {backend_error}")

    st.divider()

    st.subheader("Foundation Models API")
    st.text_input("API Base URL", key="openai_api_base")
    st.text_input("API Key", type="password", key="openai_api_key")
    st.caption("Ключи используются для текущей обработки, но не сохраняются в настройках UI.")
    model_values = list(MODEL_OPTIONS.values())
    if st.session_state.openai_model not in model_values:
        st.session_state.openai_model = MODEL_OPTIONS["Qwen3-Next-80B-A3B-Instruct"]
    model_index = model_values.index(st.session_state.openai_model) if st.session_state.openai_model in model_values else 0
    st.selectbox("LLM Model", model_values, index=model_index, key="openai_model", format_func=model_label)
    st.slider(
        "Температура LLM",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="openai_temperature",
        help="Ниже — стабильнее и строже, выше — больше вариативности в формулировках.",
    )
    with st.expander("Скорость обработки"):
        parser_modes = ["fast", "hybrid", "llm"]
        if st.session_state.parser_mode not in parser_modes:
            st.session_state.parser_mode = "fast"
        st.selectbox(
            "Режим извлечения требований",
            parser_modes,
            key="parser_mode",
            format_func=lambda value: {
                "fast": "Быстрый локальный",
                "hybrid": "Локальный + fallback LLM",
                "llm": "Только LLM",
            }.get(value, value),
        )
        st.number_input("Размер чанка парсера", min_value=3000, max_value=20000, step=1000, key="parser_chunk_size")
        st.number_input("Параллельность парсера", min_value=1, max_value=10, key="parser_concurrency")
        st.number_input("Мин. требований для fast/hybrid", min_value=1, max_value=200, key="parser_fast_min_requirements")
        st.number_input("Макс. требований для анализа", min_value=20, max_value=1000, step=20, key="parser_fast_max_requirements")
        st.number_input("Требований в батче анализа", min_value=5, max_value=50, step=5, key="max_requirements_per_batch")
        rag_modes = ["grouped", "per_requirement"]
        if st.session_state.analysis_rag_mode not in rag_modes:
            st.session_state.analysis_rag_mode = "grouped"
        st.selectbox(
            "RAG для анализа",
            rag_modes,
            key="analysis_rag_mode",
            format_func=lambda value: "1 RAG-запрос на батч" if value == "grouped" else "RAG-запрос на каждое требование",
        )
        st.number_input("Параллельность батчей анализа", min_value=1, max_value=6, key="analysis_batch_concurrency")
        st.number_input("Пауза между LLM-запросами, сек", min_value=0.0, max_value=5.0, step=0.1, key="llm_request_delay")

    st.divider()

    st.subheader("Managed RAG")
    st.text_input("RAG URL", key="managed_rag_url")
    st.text_input("Knowledge Base Version", key="managed_rag_kb_version")
    st.text_input("RAG API Key", type="password", key="managed_rag_api_key")
    st.number_input("Кол-во результатов", min_value=1, max_value=10, key="managed_rag_results")
    st.number_input("Чанков в контексте", min_value=1, max_value=20, key="managed_rag_context_chunks")
    st.number_input("Макс. токенов RAG", min_value=128, max_value=4096, step=128, key="managed_rag_max_tokens")
    st.number_input("Температура RAG", min_value=0.0, max_value=1.0, step=0.01, key="managed_rag_temperature")
    st.number_input("Параллельность RAG", min_value=1, max_value=10, key="managed_rag_concurrency")
    st.checkbox("Кэшировать RAG-ответы", key="managed_rag_cache_enabled")

    st.divider()
    st.subheader("Настройки")
    st.caption("Версия базы знаний, параметры скорости и RAG сохраняются на backend и применяются к новым запускам.")
    if backend_health and not backend_health_stale:
        current_settings_payload = persisted_settings_payload()
        settings_loaded = st.session_state.settings_loaded_for_backend == st.session_state.backend_api_url
        if settings_loaded and current_settings_payload != st.session_state.last_persisted_settings_payload:
            try:
                save_persisted_settings()
            except Exception as exc:
                st.session_state.last_settings_save_error = str(exc)
        if st.button("Сохранить сейчас", use_container_width=True):
            try:
                save_persisted_settings()
                st.success("Настройки сохранены")
            except Exception as exc:
                show_request_error(exc)
        if st.session_state.last_settings_saved_at:
            st.caption(f"Последнее сохранение: {st.session_state.last_settings_saved_at}")
        if st.session_state.last_settings_save_error:
            st.warning(st.session_state.last_settings_save_error)
    else:
        st.info("Настройки сохранятся, когда backend будет доступен.")

if backend_health:
    try:
        if st.session_state.selected_run_id:
            refresh_selected_run_status()
        else:
            st.session_state.runs_list = fetch_runs()
        if (
            not st.session_state.selected_run_id
            and not st.session_state.suppress_auto_load_latest
            and st.session_state.runs_list
        ):
            latest_run = fetch_run(st.session_state.runs_list[0]["id"])
            load_run_into_state(latest_run)
    except Exception:
        pass

# UX-Блок 3: убрали отдельный таб «📊 Отчёт» — отчёт теперь на табе
# «📄 Анализ ТЗ», прямо под загрузкой/прогрессом. Чат — справа в правой
# колонке (Блок 4).
tab_analyze, tab_history, tab_prompts = st.tabs(["📄 Анализ ТЗ", "🕘 История", "✍️ Промпты"])

with tab_analyze:
    has_report = bool(st.session_state.analysis_report)
    active_run = st.session_state.active_run or {}
    current_status = active_run.get("status")
    is_busy = current_status in {"queued", "extracting", "analyzing"}

    # Шапка таба компактнее: при готовом отчёте показываем 1-строчную
    # плашку «✅ Готово · ТЗ.docx · N требований» + кнопку «Новый анализ».
    # Загрузка и кнопки 1/2 убираются в expander.
    if has_report and not is_busy:
        doc_name = active_run.get("document_name") or "—"
        total_reqs = len(st.session_state.requirements or [])
        updated = active_run.get("updated_at", "")
        col_status, col_new = st.columns([3, 1])
        with col_status:
            st.success(
                f"✅ Анализ готов · **{doc_name}** · {total_reqs} требований"
                + (f" · обновлено {updated}" if updated else "")
            )
        with col_new:
            if st.button("Новый анализ", width="stretch"):
                start_new_analysis()
                st.rerun()
        upload_expanded = False
    else:
        st.header("Загрузка и анализ ТЗ")
        st.caption("Проверка возможностей Cloud.ru выполняется через Managed RAG.")
        col_title, col_new = st.columns([3, 1])
        with col_new:
            if st.button("Новый анализ", width="stretch"):
                start_new_analysis()
                st.rerun()
        if st.session_state.active_run:
            render_live_run_panel()
            st.divider()
        upload_expanded = True

    upload_label = (
        "📂 Загрузить новое ТЗ / запустить шаги извлечения и анализа"
        if has_report and not is_busy
        else "📂 Документ и шаги обработки"
    )
    with st.expander(upload_label, expanded=upload_expanded):
        uploaded_files = st.file_uploader(
            "Загрузите ТЗ (PDF, DOCX, XLSX, TXT)",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt"],
            accept_multiple_files=True,
            key=f"uploaded_files_{st.session_state.upload_widget_version}",
        )

        if uploaded_files:
            for upload in uploaded_files:
                st.info(f"📎 {upload.name} ({upload.size / 1024:.0f} КБ)")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "1️⃣ Запустить извлечение",
                disabled=not uploaded_files or not backend_health,
                use_container_width=True,
            ):
                try:
                    with st.spinner("Создаю запуск на backend..."):
                        run = api_post_files(
                            "/runs/extract",
                            build_upload_files(uploaded_files),
                            {"llm_settings_json": json.dumps(llm_settings_payload(), ensure_ascii=False)},
                            timeout=120,
                        )
                        reset_current_outputs()
                        load_run_into_state(run)
                    st.success("Запуск создан. Извлечение продолжится на backend.")
                    st.rerun()
                except Exception as exc:
                    show_request_error(exc)

        with col2:
            can_analyze = bool(st.session_state.requirements and st.session_state.selected_run_id)
            if st.button(
                "2️⃣ Запустить анализ",
                disabled=not can_analyze or is_busy or not backend_health,
                use_container_width=True,
            ):
                try:
                    with st.spinner("Ставлю анализ в очередь на backend..."):
                        run = api_post_json(
                            f"/runs/{st.session_state.selected_run_id}/analysis",
                            {"llm_settings": llm_settings_payload()},
                            timeout=60,
                        )
                        load_run_into_state(run)
                        st.session_state.analysis_qa_history = []
                        st.session_state.downloads = {}
                    st.success("Анализ запущен. Можно обновлять страницу и возвращаться позже.")
                    st.rerun()
                except Exception as exc:
                    show_request_error(exc)

        # Парсенные файлы и live-прогресс — внутри того же expander'а,
        # чтобы не загромождать главный экран при готовом отчёте.
        if has_report and active_run:
            st.divider()
            render_run_status(active_run)

        if st.session_state.parsed_files:
            st.divider()
            st.markdown("**Обработанные файлы:**")
            for item in st.session_state.parsed_files:
                st.markdown(
                    f"- **{item['filename']}**: {item['text_chars']} символов, "
                    f"{item['table_count']} таблиц, {item['requirements_found']} требований"
                )

    if st.session_state.requirements and not has_report:
        st.caption(
            f"Извлечено требований: {len(st.session_state.requirements)}. "
            "После анализа полный список появится в матрице."
        )

    # UX-Блок A6: чат теперь в sidebar (см. блок выше), отчёт занимает
    # всю ширину рабочей области. Это снимает проблему «чат не двигается
    # при скролле» (sidebar и так sticky по умолчанию в Streamlit).
    if st.session_state.analysis_report:
        st.divider()
        render_full_report(st.session_state.analysis_report)

with tab_history:
    st.header("История запусков")
    st.caption("Здесь можно вернуться к прошлым обработкам даже после обновления или закрытия страницы.")

    if not backend_health:
        st.info("Backend недоступен")
    else:
        if st.button("Обновить историю"):
            try:
                st.session_state.runs_list = fetch_runs()
            except Exception as exc:
                show_request_error(exc)

        runs = st.session_state.runs_list or []
        if not runs:
            st.info("История пока пустая")
        else:
            for run_info in runs:
                cols = st.columns([3, 1.3, 1.2, 1])
                with cols[0]:
                    st.markdown(
                        f"**{run_info.get('document_name', 'document')}**  \n"
                        f"`{run_info.get('id')}`"
                    )
                    if run_info.get("error"):
                        st.caption(run_info["error"])
                with cols[1]:
                    st.markdown(status_label(run_info.get("status", "")))
                    st.caption(run_info.get("updated_at", ""))
                with cols[2]:
                    st.metric("Требований", run_info.get("total_requirements", 0))
                with cols[3]:
                    if st.button("Открыть", key=f"open_run_{run_info.get('id')}", use_container_width=True):
                        try:
                            run = fetch_run(run_info["id"])
                            load_run_into_state(run)
                            st.rerun()
                        except Exception as exc:
                            show_request_error(exc)
                st.divider()

with tab_prompts:
    st.header("Промпты")

    if not backend_health:
        st.info("Backend недоступен")
    else:
        try:
            prompts_payload = cached_prompts(st.session_state.backend_api_url)
            st.session_state.last_prompts_payload = prompts_payload
        except Exception as exc:
            prompts_payload = st.session_state.last_prompts_payload or {"prompts": {}}
            if not prompts_payload.get("prompts"):
                show_request_error(exc)
            else:
                st.warning("Промпты временно отвечают медленно. Показываю последнюю успешную версию.")

        prompts = prompts_payload.get("prompts", {})
        prompt_keys = list(prompts.keys())
        if not prompt_keys:
            st.warning("Промпты пока не инициализированы")
        else:
            left, main = st.columns([1, 3])
            with left:
                selected_key = st.selectbox(
                    "Промпт",
                    prompt_keys,
                    format_func=lambda key: prompts[key].get("label", key),
                    key="selected_prompt_key",
                )
                selected_prompt = prompts[selected_key]
                versions = selected_prompt.get("versions", [])
                active_version = selected_prompt.get("active_version")
                version_by_id = {version.get("id"): version for version in versions}
                version_ids = list(version_by_id.keys())
                active_index = version_ids.index(active_version) if active_version in version_ids else 0
                selected_version_id = st.selectbox(
                    "Версия",
                    version_ids,
                    index=active_index,
                    format_func=lambda version_id: (
                        f"{version_by_id[version_id].get('label', 'Версия')} · "
                        f"{version_by_id[version_id].get('created_at', '')}"
                    ),
                    key=f"selected_version_{selected_key}",
                )
                selected_version = version_by_id[selected_version_id]
                if selected_version_id != active_version:
                    if st.button("Сделать активной", use_container_width=True):
                        try:
                            api_post_json(
                                "/prompts/activate",
                                {"prompt_key": selected_key, "version_id": selected_version_id},
                                timeout=30,
                            )
                            cached_prompts.clear()
                            st.success("Активная версия обновлена")
                            st.rerun()
                        except Exception as exc:
                            show_request_error(exc)

            with main:
                st.subheader(selected_prompt.get("label", selected_key))
                st.caption(f"Активная версия: {active_version}")
                editor_key = f"prompt_editor_{selected_key}_{selected_version.get('id')}"
                edited_content = st.text_area(
                    "Текст промпта",
                    value=selected_version.get("content", ""),
                    height=420,
                    key=editor_key,
                )
                label_key = f"prompt_label_{selected_key}_{selected_version.get('id')}"
                version_label = st.text_input("Название новой версии", value="", key=label_key)
                col_save, col_hint = st.columns([1, 2])
                with col_save:
                    if st.button("Сохранить новую версию", use_container_width=True):
                        try:
                            api_post_json(
                                "/prompts/version",
                                {
                                    "prompt_key": selected_key,
                                    "content": edited_content,
                                    "label": version_label,
                                    "activate": True,
                                },
                                timeout=30,
                            )
                            cached_prompts.clear()
                            st.success("Новая версия сохранена и активирована")
                            st.rerun()
                        except Exception as exc:
                            show_request_error(exc)
                with col_hint:
                    if selected_key == "parser_user_template":
                        st.caption("Доступная переменная: {document_text}")
                    elif selected_key == "analysis_user_template":
                        st.caption("Доступные переменные: {requirements_block}, {context}")
                    elif selected_key == "summary_user_template":
                        st.caption("Доступные переменные: {doc_name}, {total}, {match_count}, {partial_count}, {mismatch_count}, {clarification_count}, {compliance_pct}, {top_mismatches}")

# UX-Блок 3: блок `with tab_report:` удалён. Содержимое отчёта (шапка,
# методика, сводка, резюме, сомнительные места, матрица платформ,
# сноски, внешние услуги, скачивания) собрано в `render_full_report` и
# вызывается на табе «📄 Анализ ТЗ» под загрузкой/прогрессом.
