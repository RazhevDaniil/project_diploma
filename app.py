"""Streamlit UI for the Cloud.ru TZ analyzer backend."""

from __future__ import annotations

import json
import os

import requests
import streamlit as st


DEFAULT_BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")
MODEL_OPTIONS = ["GigaChat-2", "GigaChat-2-Pro", "GigaChat", "GigaChat-2-Max"]


def init_state():
    defaults = {
        "analysis_report": None,
        "requirements": None,
        "parsed_files": [],
        "report_markdown": "",
        "downloads": {},
        "backend_api_url": DEFAULT_BACKEND_API_URL,
        "llm_provider": os.getenv("LLM_PROVIDER", "gigachat"),
        "gigachat_credentials": os.getenv("GIGACHAT_CREDENTIALS", ""),
        "gigachat_model": os.getenv("GIGACHAT_MODEL", MODEL_OPTIONS[0]),
        "gigachat_scope": os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        "gigachat_embedding_model": os.getenv("GIGACHAT_EMBEDDING_MODEL", "Embeddings"),
        "openai_api_base": os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1"),
        "openai_api_key": os.getenv("OPENAI_API_KEY", "not-needed"),
        "openai_model": os.getenv("OPENAI_MODEL", "local-model"),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def llm_settings_payload() -> dict:
    return {
        "provider": st.session_state.llm_provider,
        "gigachat_credentials": st.session_state.gigachat_credentials,
        "gigachat_model": st.session_state.gigachat_model,
        "gigachat_scope": st.session_state.gigachat_scope,
        "gigachat_embedding_model": st.session_state.gigachat_embedding_model,
        "openai_api_base": st.session_state.openai_api_base,
        "openai_api_key": st.session_state.openai_api_key,
        "openai_model": st.session_state.openai_model,
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


def _response_filename(response: requests.Response) -> str:
    disposition = response.headers.get("content-disposition", "")
    if "filename=" not in disposition:
        return "download.bin"
    return disposition.split("filename=", 1)[1].strip().strip('"')


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


def prepare_download(format_name: str):
    report = st.session_state.analysis_report
    if not report:
        return
    url = f"{st.session_state.backend_api_url}/reports/export/{format_name}"
    response = requests.post(url, json={"report": report}, timeout=300)
    response.raise_for_status()
    st.session_state.downloads[format_name] = {
        "content": response.content,
        "filename": _response_filename(response),
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
            f"векторов: {backend_health.get('vector_count', 0)}"
        )
    else:
        st.error(f"Backend недоступен: {backend_error}")

    st.divider()

    st.subheader("Настройки LLM")
    st.selectbox("Провайдер", ["gigachat", "openai_compatible"], key="llm_provider")
    st.text_input("GigaChat Credentials", type="password", key="gigachat_credentials")
    st.text_input("GigaChat Scope", key="gigachat_scope")
    st.text_input("Embedding Model", key="gigachat_embedding_model")

    if st.session_state.llm_provider == "gigachat":
        model_index = MODEL_OPTIONS.index(st.session_state.gigachat_model) if st.session_state.gigachat_model in MODEL_OPTIONS else 0
        st.selectbox("Модель", MODEL_OPTIONS, index=model_index, key="gigachat_model")
    else:
        st.caption("Даже в этом режиме RAG и эмбеддинги используют GigaChat credentials выше.")
        st.text_input("API Base URL", key="openai_api_base")
        st.text_input("API Key", type="password", key="openai_api_key")
        st.text_input("Model Name", key="openai_model")

    st.divider()

    st.subheader("База знаний")
    if st.button("🗑️ Сбросить базу знаний", disabled=not backend_health):
        try:
            api_post_json("/kb/reset", {}, timeout=60)
            st.success("База знаний очищена")
            st.rerun()
        except Exception as exc:
            show_request_error(exc)

tab_analyze, tab_kb, tab_report = st.tabs(["📄 Анализ ТЗ", "📚 База знаний", "📊 Отчёт"])

with tab_analyze:
    st.header("Загрузка и анализ ТЗ")

    search_mode = st.radio(
        "Режим поиска информации",
        ["rag", "live"],
        format_func=lambda x: {
            "rag": "📦 RAG (по проиндексированной базе знаний)",
            "live": "🌐 Live Search (поиск в интернете и cloud.ru/docs)",
        }[x],
        horizontal=True,
    )

    if search_mode == "live":
        st.caption(
            "Технические/SLA/ИБ требования ищутся по cloud.ru/docs, "
            "юридические и коммерческие могут уходить в live search."
        )

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
                            "search_mode": search_mode,
                            "requirements": st.session_state.requirements,
                            "llm_settings": llm_settings_payload(),
                        },
                        timeout=3600,
                    )
                    st.session_state.analysis_report = report
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

with tab_kb:
    st.header("База знаний — документация Cloud.ru")
    st.markdown(
        """
        База знаний наполняется из официальной документации [cloud.ru/docs](https://cloud.ru/docs).
        Краулинг и индексация выполняются на backend-сервисе.
        """
    )

    kb_status = None
    if backend_health:
        try:
            kb_status = api_get("/kb/status", timeout=10)
        except Exception as exc:
            show_request_error(exc)

    if kb_status:
        st.info(f"Индекс: {kb_status.get('vector_count', 0)} векторов")

    col_crawl1, col_crawl2 = st.columns(2)
    with col_crawl1:
        max_pages = st.number_input("Макс. страниц (0 = все ~7000)", min_value=0, max_value=10000, value=0, step=100)
    with col_crawl2:
        concurrency = st.number_input("Параллельность запросов", min_value=1, max_value=50, value=10)

    if st.button("🌐 Запустить краулинг cloud.ru/docs", use_container_width=True, disabled=not backend_health):
        try:
            with st.spinner("Краулинг и индексация на backend..."):
                result = api_post_json(
                    "/kb/crawl",
                    {
                        "max_pages": int(max_pages),
                        "concurrency": int(concurrency),
                        "llm_settings": llm_settings_payload(),
                    },
                    timeout=7200,
                )
            st.success(
                f"Проиндексировано {result.get('indexed_pages', 0)} страниц. "
                f"Всего векторов: {result.get('vector_count', 0)}"
            )
            st.balloons()
        except Exception as exc:
            show_request_error(exc)

    st.divider()

    with st.expander("📁 Загрузить дополнительные файлы вручную"):
        kb_files = st.file_uploader(
            "Дополнительная документация (TXT, MD, HTML, PDF, DOCX)",
            type=["txt", "md", "html", "pdf", "docx"],
            accept_multiple_files=True,
            key="kb_uploader",
        )

        if st.button("📥 Индексировать файлы", disabled=not kb_files or not backend_health):
            try:
                with st.spinner("Индексация файлов на backend..."):
                    result = api_post_files(
                        "/kb/index-files",
                        build_upload_files(kb_files),
                        {"llm_settings_json": json.dumps(llm_settings_payload(), ensure_ascii=False)},
                        timeout=3600,
                    )
                st.success(f"Индексировано! Всего векторов в базе: {result.get('vector_count', 0)}")
                st.rerun()
            except Exception as exc:
                show_request_error(exc)

    st.divider()

    st.subheader("Тестовый поиск по базе")
    test_query = st.text_input("Поисковый запрос")
    if st.button("🔍 Искать", disabled=not backend_health) and test_query:
        try:
            result = api_post_json(
                "/kb/search",
                {
                    "query": test_query,
                    "k": 5,
                    "llm_settings": llm_settings_payload(),
                },
                timeout=120,
            )
            results = result.get("results", [])
            if results:
                for i, doc in enumerate(results):
                    label = f"{doc.get('title', '')} — {doc.get('source', 'unknown')}".strip(" —")
                    with st.expander(f"Результат {i + 1} — {label}"):
                        st.markdown(doc.get("content", "")[:500])
                        source = doc.get("source", "")
                        if source.startswith("http"):
                            st.markdown(f"[Открыть в документации]({source})")
            else:
                st.warning("Ничего не найдено")
        except Exception as exc:
            show_request_error(exc)

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

        if not st.session_state.report_markdown:
            try:
                fetch_report_markdown()
            except Exception as exc:
                show_request_error(exc)

        if st.session_state.report_markdown:
            st.markdown(st.session_state.report_markdown)

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
