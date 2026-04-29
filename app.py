"""Streamlit UI for the Cloud.ru TZ analyzer backend."""

from __future__ import annotations

import json
import os
import time
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
        "auto_refresh_run": True,
        "last_backend_health": None,
        "last_prompts_payload": None,
        "backend_api_url": DEFAULT_BACKEND_API_URL,
        "openai_api_base": os.getenv("OPENAI_API_BASE", "https://foundation-models.api.cloud.ru/v1"),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_model": os.getenv("OPENAI_MODEL", MODEL_OPTIONS["gpt-oss-120b"]),
        "openai_temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.05")),
        "managed_rag_url": os.getenv(
            "MANAGED_RAG_URL",
            "https://e424a162-618c-4862-b789-b089abd81b46.managed-rag.inference.cloud.ru/api/v2/retrieve_generate",
        ),
        "managed_rag_kb_version": os.getenv("MANAGED_RAG_KB_VERSION", "eb73eb63-ec91-47c9-851e-1c14949b7a14"),
        "managed_rag_api_key": os.getenv("MANAGED_RAG_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        "managed_rag_results": int(os.getenv("MANAGED_RAG_RESULTS", "2")),
        "managed_rag_context_chunks": int(os.getenv("MANAGED_RAG_CONTEXT_CHUNKS", "3")),
        "managed_rag_max_tokens": int(os.getenv("MANAGED_RAG_MAX_TOKENS", "256")),
        "managed_rag_temperature": float(os.getenv("MANAGED_RAG_TEMPERATURE", "0.01")),
        "managed_rag_concurrency": int(os.getenv("MANAGED_RAG_CONCURRENCY", "4")),
        "managed_rag_cache_enabled": os.getenv("MANAGED_RAG_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
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


def load_run_into_state(run: dict):
    previous_run_id = st.session_state.get("selected_run_id")
    st.session_state.selected_run_id = run.get("id")
    st.session_state.active_run = run
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


def reset_current_outputs():
    st.session_state.analysis_report = None
    st.session_state.requirements = None
    st.session_state.parsed_files = []
    st.session_state.report_markdown = ""
    st.session_state.downloads = {}
    st.session_state.analysis_qa_history = []


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
    refs = assessment.get("evidence_refs") or []
    return f"{symbol} {', '.join(refs[:2])}".strip()


def report_platform_matrix(report: dict) -> list[dict]:
    platform_names = []
    for verdict in report.get("verdicts", []):
        for assessment in verdict.get("platform_assessments", []) or []:
            name = assessment.get("platform_name") or "Не определено"
            if name not in platform_names:
                platform_names.append(name)

    rows = []
    for verdict in report.get("verdicts", []):
        row = {
            "Пункт ТЗ": verdict.get("section") or f"#{verdict.get('requirement_id')}",
            "Требование": (verdict.get("requirement_text") or "")[:140],
        }
        by_platform = {
            assessment.get("platform_name") or "Не определено": assessment
            for assessment in verdict.get("platform_assessments", []) or []
        }
        for platform_name in platform_names:
            row[platform_name] = platform_cell(by_platform[platform_name]) if platform_name in by_platform else "-"
        rows.append(row)
    return rows


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


st.set_page_config(page_title="Cloud.ru — Анализ ТЗ", page_icon="☁️", layout="wide")
init_state()

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

with st.sidebar:
    st.title("☁️ Cloud.ru TZ Analyzer")
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
    model_values = list(MODEL_OPTIONS.values())
    if st.session_state.openai_model not in model_values:
        st.session_state.openai_model = MODEL_OPTIONS["gpt-oss-120b"]
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

    st.divider()

    st.subheader("Managed RAG")
    st.text_input("RAG URL", key="managed_rag_url")
    st.text_input("Knowledge Base Version", key="managed_rag_kb_version")
    st.text_input("RAG API Key", type="password", key="managed_rag_api_key")
    st.number_input("Кол-во результатов", min_value=1, max_value=10, key="managed_rag_results")
    st.number_input("Чанков в контексте", min_value=1, max_value=20, key="managed_rag_context_chunks")
    st.number_input("Макс. токенов RAG", min_value=128, max_value=4096, step=128, key="managed_rag_max_tokens")
    st.number_input("Параллельность RAG", min_value=1, max_value=10, key="managed_rag_concurrency")
    st.checkbox("Кэшировать RAG-ответы", key="managed_rag_cache_enabled")

if backend_health:
    try:
        st.session_state.runs_list = fetch_runs()
        if st.session_state.selected_run_id:
            refresh_selected_run()
        elif st.session_state.runs_list:
            latest_run = fetch_run(st.session_state.runs_list[0]["id"])
            load_run_into_state(latest_run)
    except Exception:
        pass

tab_analyze, tab_history, tab_prompts, tab_report = st.tabs(["📄 Анализ ТЗ", "🕘 История", "✍️ Промпты", "📊 Отчёт"])

with tab_analyze:
    st.header("Загрузка и анализ ТЗ")
    st.caption("Проверка возможностей Cloud.ru выполняется через Managed RAG.")

    active_run = st.session_state.active_run
    if active_run:
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
            if st.button("Обновить статус", use_container_width=True):
                try:
                    refresh_selected_run()
                    st.rerun()
                except Exception as exc:
                    show_request_error(exc)

        st.divider()

    uploaded_files = st.file_uploader(
        "Загрузите ТЗ (PDF, DOCX, XLSX, TXT)",
        type=["pdf", "docx", "doc", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for upload in uploaded_files:
            st.info(f"📎 {upload.name} ({upload.size / 1024:.0f} КБ)")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("1️⃣ Запустить извлечение", disabled=not uploaded_files or not backend_health, use_container_width=True):
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
        current_status = (st.session_state.active_run or {}).get("status")
        is_busy = current_status in {"queued", "extracting", "analyzing"}
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

    if st.session_state.parsed_files:
        st.divider()
        st.subheader("Обработанные файлы")
        for item in st.session_state.parsed_files:
            st.markdown(
                f"- **{item['filename']}**: {item['text_chars']} символов, "
                f"{item['table_count']} таблиц, {item['requirements_found']} требований"
            )

    if st.session_state.requirements:
        st.divider()
        st.subheader(f"Извлечённые требования ({len(st.session_state.requirements)})")
        categories = sorted({req["category"] for req in st.session_state.requirements})
        selected_cats = st.multiselect("Фильтр по категории", categories, default=categories)
        filtered = [req for req in st.session_state.requirements if req["category"] in selected_cats]

        for req in filtered:
            cat_label = {
                "technical": "🔧 Техническое",
                "sla": "📊 SLA",
                "legal": "⚖️ Юридическое",
                "commercial": "💰 Коммерческое",
                "security": "🔒 ИБ",
                "other": "📌 Прочее",
            }.get(req["category"], req["category"])

            with st.expander(f"#{req['id']} [{req['section']}] {cat_label} — {req['text'][:80]}..."):
                st.markdown(f"**Категория:** {cat_label}")
                st.markdown(f"**Раздел:** {req['section']}")
                st.markdown(f"**Текст:** {req['text']}")
                if req.get("tables"):
                    st.markdown("**Таблица:**")
                    st.markdown(req["tables"])

    active_run = st.session_state.active_run
    if (
        active_run
        and st.session_state.auto_refresh_run
        and active_run.get("status") in {"queued", "extracting", "analyzing"}
    ):
        time.sleep(3)
        st.rerun()

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

with tab_report:
    st.header("Отчёт по анализу")

    report = st.session_state.analysis_report
    if report is None:
        st.info("Сначала выполните анализ ТЗ на первой вкладке")
    else:
        pct = report.get("compliance_percentage", 0)
        color = "green" if pct >= 80 else "orange" if pct >= 60 else "red"
        st.markdown(f"## Общий процент соответствия: <span style='color:{color}'>{pct}%</span>", unsafe_allow_html=True)
        st.markdown(f"**{report.get('score', 0)} из {report.get('max_score', 0)} баллов**")
        st.progress(min(max(float(pct) / 100, 0.0), 1.0))

        with st.expander("Методика оценки"):
            st.markdown(
                "| Вердикт | Баллы |\n|---|---|\n"
                "| Полное соответствие | 2 |\n"
                "| Частичное соответствие | 1 |\n"
                "| Несоответствие / Требует уточнения | 0 |\n\n"
                f"Максимальный балл = {report.get('total', 0)} × 2 = {report.get('max_score', 0)}. "
                f"Набрано **{report.get('score', 0)}** баллов."
            )

        st.divider()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Всего", report.get("total", 0))
        col2.metric("✅ Соответствует", report.get("match_count", 0))
        col3.metric("🟡 Частично", report.get("partial_count", 0))
        col4.metric("❌ Не соответствует", report.get("mismatch_count", 0))
        col5.metric("❓ Уточнить", report.get("clarification_count", 0))

        if report.get("summary"):
            st.markdown("### Резюме")
            st.markdown(report["summary"])

        platform_matrix = report_platform_matrix(report)
        if platform_matrix:
            st.divider()
            st.subheader("Матрица по платформам")
            st.caption("`+` — соответствует, `±` — частично, `-` — не подтверждено, `?` — нужно уточнить. Сноски раскрыты в скачиваемом отчёте.")
            st.dataframe(platform_matrix, use_container_width=True, hide_index=True)

        external_items = [
            verdict for verdict in report.get("verdicts", [])
            if verdict.get("requires_external_service")
        ]
        if external_items:
            with st.expander(f"Внешние услуги / подрядчики ({len(external_items)})"):
                for verdict in external_items:
                    st.markdown(
                        f"**{verdict.get('section') or '#' + str(verdict.get('requirement_id'))}** — "
                        f"{(verdict.get('requirement_text') or '')[:220]}"
                    )
                    if verdict.get("external_service_notes"):
                        st.caption(verdict["external_service_notes"])

        st.divider()

        st.subheader("Задать вопрос по анализу")
        st.caption("Можно переспросить по конкретному пункту ТЗ, по спорному вердикту или по источникам Managed RAG.")

        apply_pending_analysis_question()

        suggested_questions = [
            "Почему этот пункт признан несоответствием?",
            "Что именно нужно перепроверить вручную в этом ТЗ?",
            "Какие 3 самых рискованных пункта в этом анализе?",
        ]
        cols = st.columns(len(suggested_questions))
        for idx, question in enumerate(suggested_questions):
            if cols[idx].button(question, key=f"suggested_question_{idx}", use_container_width=True):
                st.session_state.pending_analysis_question = question
                st.rerun()

        st.text_area(
            "Вопрос по анализу",
            key="analysis_question_input",
            height=100,
            placeholder="Например: Почему пункт 7.2.4 отмечен как несоответствие и на какие источники вы опирались?",
        )

        if st.button("Спросить", disabled=not st.session_state.analysis_question_input.strip(), use_container_width=True):
            try:
                current_question = st.session_state.analysis_question_input.strip()
                with st.spinner("Готовлю ответ по контексту ТЗ, анализа и Managed RAG..."):
                    answer = api_post_json(
                        "/analysis/ask",
                        {
                            "question": current_question,
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
                        "question": current_question,
                        "answer": answer.get("answer", ""),
                        "related_sections": answer.get("related_sections", []),
                        "source_urls": answer.get("source_urls", []),
                    }
                )
                st.session_state.pending_analysis_question = ""
                st.rerun()
            except Exception as exc:
                show_request_error(exc)

        if st.session_state.analysis_qa_history:
            for idx, item in enumerate(reversed(st.session_state.analysis_qa_history), start=1):
                with st.expander(f"Вопрос {len(st.session_state.analysis_qa_history) - idx + 1}: {item['question']}", expanded=(idx == 1)):
                    st.markdown(item["answer"])
                    if item.get("related_sections"):
                        st.caption("Связанные пункты ТЗ: " + ", ".join(item["related_sections"]))
                    if item.get("source_urls"):
                        st.markdown("**Источники:**")
                        for url in item["source_urls"]:
                            st.markdown(f"- [{url}]({url})")

        st.divider()

        st.caption("Полная детализация скрыта в UI. Для проверки проблемных пунктов и полного реестра скачайте отчёт в нужном формате.")

        st.divider()
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            if st.button("💾 Подготовить Markdown"):
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
                )

        with col_b:
            if st.button("💾 Подготовить DOCX"):
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
                )

        with col_c:
            if st.button("💾 Подготовить PDF"):
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
                )

        with col_d:
            if st.button("💾 Подготовить Excel"):
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
                )
