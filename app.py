"""Streamlit UI for the Cloud.ru TZ analyzer backend."""

from __future__ import annotations

import json
import os
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
        "backend_api_url": DEFAULT_BACKEND_API_URL,
        "openai_api_base": os.getenv("OPENAI_API_BASE", "https://foundation-models.api.cloud.ru/v1"),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_model": os.getenv("OPENAI_MODEL", MODEL_OPTIONS["gpt-oss-120b"]),
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
try:
    backend_health = api_get("/health", timeout=5)
except Exception as exc:
    backend_error = str(exc)

with st.sidebar:
    st.title("☁️ Cloud.ru TZ Analyzer")
    st.markdown("UI работает через отдельный backend API")
    st.divider()

    st.subheader("Backend API")
    st.text_input("Backend URL", key="backend_api_url")

    if backend_health:
        st.success(
            f"Backend доступен: {backend_health.get('status', 'ok')}, "
            f"RAG: {backend_health.get('rag_provider', 'managed_rag')}"
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

tab_analyze, tab_prompts, tab_report = st.tabs(["📄 Анализ ТЗ", "✍️ Промпты", "📊 Отчёт"])

with tab_analyze:
    st.header("Загрузка и анализ ТЗ")
    st.caption("Проверка возможностей Cloud.ru выполняется через Managed RAG.")

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
        if st.button("1️⃣ Извлечь требования", disabled=not uploaded_files or not backend_health, use_container_width=True):
            try:
                with st.spinner("Извлечение требований через backend..."):
                    result = api_post_files(
                        "/requirements/extract",
                        build_upload_files(uploaded_files),
                        {"llm_settings_json": json.dumps(llm_settings_payload(), ensure_ascii=False)},
                        timeout=1800,
                    )
                st.session_state.requirements = result.get("requirements", [])
                st.session_state.parsed_files = result.get("files", [])
                st.session_state.analysis_report = None
                st.session_state.analysis_qa_history = []
                st.session_state.report_markdown = ""
                st.session_state.downloads = {}
                st.success(f"Найдено {result.get('total_requirements', 0)} требований")
            except Exception as exc:
                show_request_error(exc)

    with col2:
        if st.button(
            "2️⃣ Запустить анализ",
            disabled=not st.session_state.requirements or not backend_health,
            use_container_width=True,
        ):
            try:
                with st.spinner("Анализ требований..."):
                    report = api_post_json(
                        "/analysis/report",
                        {
                            "document_name": uploaded_files[0].name if uploaded_files else "document",
                            "search_mode": "managed_rag",
                            "requirements": st.session_state.requirements,
                            "llm_settings": llm_settings_payload(),
                        },
                        timeout=3600,
                    )
                    st.session_state.analysis_report = report
                    st.session_state.analysis_search_mode = "managed_rag"
                    st.session_state.analysis_qa_history = []
                    st.session_state.downloads = {}
                    fetch_report_markdown()
                st.success(
                    "Анализ завершён! "
                    f"Соответствие: {report.get('compliance_percentage', 0)}% "
                    f"({report.get('score', 0)}/{report.get('max_score', 0)} баллов)"
                )
                st.balloons()
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

with tab_prompts:
    st.header("Промпты")

    if not backend_health:
        st.info("Backend недоступен")
    else:
        try:
            prompts_payload = api_get("/prompts", timeout=20)
        except Exception as exc:
            show_request_error(exc)
            prompts_payload = {"prompts": {}}

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
