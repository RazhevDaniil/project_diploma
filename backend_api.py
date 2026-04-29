"""FastAPI backend for the Cloud.ru TZ analyzer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from uuid import uuid4

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config as cfg
from src.analysis.analyzer import analyze_requirements
from src.managed_rag.client import retrieve_generate
from src.models import AnalysisReport, PlatformAssessment, Requirement, RequirementVerdict
from src.parser.document_parser import parse_document
from src.parser.requirement_extractor import extract_requirements
from src.prompt_store import activate_prompt_version, create_prompt_version, list_prompts
from src.report.generator import generate_markdown, save_docx, save_excel, save_markdown, save_pdf
from src.runtime_config import apply_runtime_settings
from src.run_store import create_run, get_run, list_runs, update_run
from src.llm.client import call_llm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Cloud.ru TZ Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_settings_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid llm_settings_json: {exc}") from exc


async def _save_upload(upload: UploadFile, target_dir: Path) -> Path:
    suffix = Path(upload.filename or "").suffix
    safe_name = Path(upload.filename or "upload").name
    destination = target_dir / f"{uuid4().hex}_{safe_name}"
    content = await upload.read()
    destination.write_bytes(content)
    if not suffix and upload.filename:
        destination = destination.with_suffix(Path(upload.filename).suffix)
    return destination


def _requirement_from_dict(item: dict) -> Requirement:
    return Requirement(
        id=int(item["id"]),
        section=str(item.get("section", "")),
        text=str(item.get("text", "")),
        category=str(item.get("category", "other")),
        tables=str(item.get("tables", "")),
    )


def _report_from_dict(item: dict) -> AnalysisReport:
    def _safe_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "да", "нужно", "required"}
        return bool(value)

    verdicts = []
    for verdict in item.get("verdicts", []):
        platform_assessments = []
        for platform_item in verdict.get("platform_assessments", []) or []:
            if not isinstance(platform_item, dict):
                continue
            platform_assessments.append(
                PlatformAssessment(
                    platform_name=str(platform_item.get("platform_name", "") or "Не определено"),
                    verdict=str(platform_item.get("verdict", "needs_clarification") or "needs_clarification"),
                    confidence=float(platform_item.get("confidence", 0.0) or 0.0),
                    reasoning=str(platform_item.get("reasoning", "") or ""),
                    evidence_refs=[str(ref) for ref in platform_item.get("evidence_refs", []) or []],
                    source_urls=[str(url) for url in platform_item.get("source_urls", []) or []],
                    source_titles=[str(title) for title in platform_item.get("source_titles", []) or []],
                    source_type=str(platform_item.get("source_type", "platform") or "platform"),
                    recommendation=str(platform_item.get("recommendation", "") or ""),
                )
            )
        verdicts.append(
            RequirementVerdict(
                requirement_id=int(verdict["requirement_id"]),
                section=str(verdict.get("section", "")),
                requirement_text=str(verdict.get("requirement_text", "")),
                category=str(verdict.get("category", "other")),
                verdict=str(verdict.get("verdict", "needs_clarification")),
                confidence=float(verdict.get("confidence", 0.0)),
                reasoning=str(verdict.get("reasoning", "")),
                evidence=str(verdict.get("evidence", "")),
                recommendation=str(verdict.get("recommendation", "")),
                source_urls=[str(url) for url in verdict.get("source_urls", [])],
                platform_assessments=platform_assessments,
                requires_external_service=_safe_bool(verdict.get("requires_external_service", False)),
                external_service_notes=str(verdict.get("external_service_notes", "") or ""),
            )
        )
    return AnalysisReport(
        document_name=str(item.get("document_name", "document")),
        verdicts=verdicts,
        summary=str(item.get("summary", "")),
    )


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9_-]{3,}", text.lower())}


def _verdict_relevance(question: str, verdict: RequirementVerdict) -> tuple[int, float]:
    q_tokens = _tokenize(question)
    haystack = " ".join(
        [
            verdict.section or "",
            verdict.requirement_text or "",
            verdict.reasoning or "",
            verdict.recommendation or "",
            verdict.evidence or "",
        ]
    ).lower()
    score = sum(1 for token in q_tokens if token in haystack)

    section_refs = re.findall(r"\b\d+(?:\.\d+)+\b", question)
    if section_refs and any(ref in (verdict.section or "") for ref in section_refs):
        score += 6

    verdict_rank = {
        "mismatch": 0,
        "needs_clarification": 1,
        "partial": 2,
        "match": 3,
    }.get(verdict.verdict, 4)
    return score, -verdict_rank + verdict.confidence


def _requirement_relevance(question: str, req: Requirement) -> int:
    q_tokens = _tokenize(question)
    haystack = " ".join([req.section or "", req.text or "", req.tables or ""]).lower()
    score = sum(1 for token in q_tokens if token in haystack)

    section_refs = re.findall(r"\b\d+(?:\.\d+)+\b", question)
    if section_refs and any(ref in (req.section or "") for ref in section_refs):
        score += 6
    return score


def _build_analysis_chat_context(
    question: str,
    report: AnalysisReport,
    requirements: list[Requirement],
    search_mode: str,
) -> tuple[str, list[str], list[str]]:
    sorted_verdicts = sorted(
        report.verdicts,
        key=lambda verdict: _verdict_relevance(question, verdict),
        reverse=True,
    )
    relevant_verdicts = [verdict for verdict in sorted_verdicts[:5] if _verdict_relevance(question, verdict)[0] > 0]
    if not relevant_verdicts:
        relevant_verdicts = sorted_verdicts[:3]

    sorted_requirements = sorted(
        requirements,
        key=lambda req: _requirement_relevance(question, req),
        reverse=True,
    )
    relevant_requirements = [req for req in sorted_requirements[:5] if _requirement_relevance(question, req) > 0]
    if not relevant_requirements:
        relevant_requirements = sorted_requirements[:3]

    related_sections = list(
        dict.fromkeys(
            [verdict.section for verdict in relevant_verdicts if verdict.section]
            + [req.section for req in relevant_requirements if req.section]
        )
    )

    source_urls: list[str] = []
    verdict_lines = []
    for verdict in relevant_verdicts:
        source_urls.extend(verdict.source_urls)
        platform_lines = []
        for assessment in verdict.platform_assessments:
            platform_lines.append(
                f"{assessment.platform_name} ({assessment.source_type}): "
                f"{assessment.verdict}, {assessment.reasoning}"
            )
        verdict_lines.append(
            "\n".join(
                [
                    f"Пункт ТЗ: {verdict.section or f'#{verdict.requirement_id}'}",
                    f"Категория: {verdict.category}",
                    f"Вердикт: {verdict.verdict}",
                    f"Требование: {verdict.requirement_text}",
                    f"Обоснование: {verdict.reasoning}",
                    f"Рекомендация: {verdict.recommendation}",
                    "Оценка по платформам: " + ("; ".join(platform_lines) if platform_lines else "нет"),
                    f"Внешние услуги/подрядчики: {verdict.external_service_notes if verdict.requires_external_service else 'нет'}",
                    f"Источники: {', '.join(verdict.source_urls[:5]) if verdict.source_urls else 'нет'}",
                ]
            )
        )

    requirement_lines = []
    for req in relevant_requirements:
        requirement_lines.append(
            "\n".join(
                [
                    f"Пункт ТЗ: {req.section or f'#{req.id}'}",
                    f"Категория: {req.category}",
                    f"Требование: {req.text}",
                    f"Таблицы: {req.tables or 'нет'}",
                ]
            )
        )

    managed_rag_lines = []
    try:
        rag_result = retrieve_generate(question, number_of_results=5)
        managed_rag_lines.append(rag_result.as_context())
    except Exception as exc:
        logger.warning("Managed RAG search for analysis chat failed: %s", exc)

    context_parts = [
        f"Документ: {report.document_name}",
        f"Сводка анализа: {report.summary}",
        "Релевантные выводы анализа:\n" + ("\n\n---\n\n".join(verdict_lines) if verdict_lines else "нет"),
        "Релевантные требования ТЗ:\n" + ("\n\n---\n\n".join(requirement_lines) if requirement_lines else "нет"),
        "Контекст Managed RAG:\n" + ("\n\n---\n\n".join(managed_rag_lines) if managed_rag_lines else "нет"),
    ]

    dedup_urls = list(dict.fromkeys([url for url in source_urls if url]))
    return "\n\n".join(context_parts), related_sections[:10], dedup_urls[:15]


ANALYSIS_CHAT_SYSTEM = """Ты — технический пресейл-ассистент Cloud.ru.
Отвечай на вопросы только на основании переданного контекста: ТЗ, уже проведённого анализа, базы знаний и найденных источников.

Правила ответа:
- Отвечай по-русски.
- Начинай с прямого ответа на вопрос.
- Если вопрос связан с конкретным пунктом ТЗ, обязательно укажи номер/раздел этого пункта.
- Если данных недостаточно, честно скажи, чего не хватает.
- Не выдумывай документы, цифры или источники.
- Если есть риск ошибки в анализе, прямо предложи, что именно перепроверить вручную.
"""


def _format_history(history: list[dict]) -> str:
    lines = []
    for item in history[-6:]:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        lines.append(f"Вопрос: {question}\nОтвет: {answer}")
    return "\n\n".join(lines) if lines else "нет"


def _run_extract_requirements(run_id: str, uploads: list[dict], settings: dict | None) -> None:
    try:
        apply_runtime_settings(settings)
        update_run(
            run_id,
            status="extracting",
            stage="extracting_requirements",
            progress_done=0,
            progress_total=len(uploads),
            error="",
        )

        all_requirements = []
        parsed_files = []
        for index, item in enumerate(uploads, start=1):
            path = Path(item["path"])
            parsed = parse_document(path)
            requirements = extract_requirements(parsed.full_text)
            all_requirements.extend(requirements)
            parsed_files.append(
                {
                    "filename": item.get("filename") or path.name,
                    "text_chars": len(parsed.text),
                    "table_count": len(parsed.tables),
                    "requirements_found": len(requirements),
                }
            )
            update_run(
                run_id,
                parsed_files=parsed_files,
                requirements=[req.to_dict() for req in all_requirements],
                progress_done=index,
            )

        update_run(
            run_id,
            status="extracted",
            stage="requirements_ready",
            parsed_files=parsed_files,
            requirements=[req.to_dict() for req in all_requirements],
            progress_done=len(uploads),
            progress_total=len(uploads),
            error="",
        )
    except Exception as exc:
        logger.exception("Run %s extraction failed", run_id)
        update_run(run_id, status="failed", stage="extract_failed", error=str(exc))


def _run_analyze_requirements(run_id: str, settings: dict | None) -> None:
    try:
        apply_runtime_settings(settings)
        run = get_run(run_id)
        if not run:
            return
        requirements = [_requirement_from_dict(item) for item in run.get("requirements", [])]
        update_run(
            run_id,
            status="analyzing",
            stage="analysis_running",
            progress_done=0,
            progress_total=len(requirements),
            error="",
        )

        def progress(done: int, total: int) -> None:
            update_run(run_id, progress_done=done, progress_total=total)

        report = analyze_requirements(
            requirements=requirements,
            document_name=str(run.get("document_name", "document")),
            search_mode="managed_rag",
            progress_callback=progress,
        )
        update_run(
            run_id,
            status="completed",
            stage="analysis_completed",
            report=report.to_dict(),
            progress_done=len(requirements),
            progress_total=len(requirements),
            error="",
        )
    except Exception as exc:
        logger.exception("Run %s analysis failed", run_id)
        update_run(run_id, status="failed", stage="analysis_failed", error=str(exc))


@app.get("/health")
async def healthcheck():
    return {
        "status": "ok",
        "provider": "foundation_models",
        "rag_provider": "managed_rag",
        "llm_model": cfg.OPENAI_MODEL,
        "llm_temperature": cfg.OPENAI_TEMPERATURE,
        "managed_rag_kb_version": cfg.MANAGED_RAG_KB_VERSION,
    }


@app.get("/prompts")
def prompts_list():
    return list_prompts()


@app.post("/prompts/version")
def prompts_create_version(payload: dict = Body(...)):
    prompt_key = str(payload.get("prompt_key", "")).strip()
    content = str(payload.get("content", "")).strip()
    if not prompt_key or not content:
        raise HTTPException(status_code=400, detail="prompt_key and content are required")
    try:
        version = create_prompt_version(
            prompt_key=prompt_key,
            content=content,
            label=str(payload.get("label", "")).strip() or None,
            activate=bool(payload.get("activate", True)),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "version": version, "prompts": list_prompts()}


@app.post("/prompts/activate")
def prompts_activate(payload: dict = Body(...)):
    prompt_key = str(payload.get("prompt_key", "")).strip()
    version_id = str(payload.get("version_id", "")).strip()
    if not prompt_key or not version_id:
        raise HTTPException(status_code=400, detail="prompt_key and version_id are required")
    try:
        prompt = activate_prompt_version(prompt_key, version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "prompt": prompt, "prompts": list_prompts()}


@app.get("/runs")
def runs_list(limit: int = 50):
    return {"runs": list_runs(limit=limit)}


@app.get("/runs/{run_id}")
def runs_get(run_id: str):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@app.post("/runs/extract")
async def runs_start_extract(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    llm_settings_json: str | None = Form(default=None),
):
    settings = _parse_settings_json(llm_settings_json) or {}
    saved_files = []
    for upload in files:
        tmp_path = await _save_upload(upload, cfg.UPLOAD_DIR)
        saved_files.append(
            {
                "filename": upload.filename or tmp_path.name,
                "path": str(tmp_path),
                "size": tmp_path.stat().st_size,
            }
        )

    document_name = saved_files[0]["filename"] if saved_files else "document"
    run = create_run(document_name=document_name, files=saved_files, settings=settings)
    update_run(
        run["id"],
        status="queued",
        stage="extract_queued",
        progress_done=0,
        progress_total=len(saved_files),
    )
    background_tasks.add_task(_run_extract_requirements, run["id"], saved_files, settings)
    return get_run(run["id"])


@app.post("/runs/{run_id}/analysis")
def runs_start_analysis(run_id: str, background_tasks: BackgroundTasks, payload: dict = Body(...)):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    if not run.get("requirements"):
        raise HTTPException(status_code=400, detail="Run has no extracted requirements")
    if run.get("status") in {"extracting", "analyzing", "queued"}:
        return run

    settings = payload.get("llm_settings") or run.get("settings") or {}
    update_run(
        run_id,
        status="queued",
        stage="analysis_queued",
        settings=settings,
        progress_done=0,
        progress_total=len(run.get("requirements", [])),
        error="",
    )
    background_tasks.add_task(_run_analyze_requirements, run_id, settings)
    return get_run(run_id)


@app.post("/requirements/extract")
async def extract_requirements_endpoint(
    files: list[UploadFile] = File(...),
    llm_settings_json: str | None = Form(default=None),
):
    apply_runtime_settings(_parse_settings_json(llm_settings_json))

    all_requirements = []
    parsed_files = []

    for upload in files:
        tmp_path = await _save_upload(upload, cfg.UPLOAD_DIR)
        parsed = parse_document(tmp_path)
        requirements = extract_requirements(parsed.full_text)
        all_requirements.extend(requirements)
        parsed_files.append(
            {
                "filename": upload.filename or tmp_path.name,
                "text_chars": len(parsed.text),
                "table_count": len(parsed.tables),
                "requirements_found": len(requirements),
            }
        )

    return {
        "files": parsed_files,
        "requirements": [req.to_dict() for req in all_requirements],
        "total_requirements": len(all_requirements),
    }


@app.post("/analysis/report")
def analysis_report(payload: dict = Body(...)):
    apply_runtime_settings(payload.get("llm_settings"))

    requirements = [_requirement_from_dict(item) for item in payload.get("requirements", [])]
    if not requirements:
        raise HTTPException(status_code=400, detail="At least one requirement is required")

    report = analyze_requirements(
        requirements=requirements,
        document_name=str(payload.get("document_name", "document")),
        search_mode=str(payload.get("search_mode", "managed_rag")),
    )
    return report.to_dict()


@app.post("/analysis/ask")
def analysis_ask(payload: dict = Body(...)):
    apply_runtime_settings(payload.get("llm_settings"))

    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    report_payload = payload.get("report")
    if not report_payload:
        raise HTTPException(status_code=400, detail="Report is required")

    report = _report_from_dict(report_payload)
    requirements = [_requirement_from_dict(item) for item in payload.get("requirements", [])]
    search_mode = str(payload.get("search_mode", "managed_rag"))
    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []

    context, related_sections, source_urls = _build_analysis_chat_context(
        question=question,
        report=report,
        requirements=requirements,
        search_mode=search_mode,
    )

    prompt = f"""Вопрос пользователя:
{question}

История предыдущих вопросов по этому анализу:
{_format_history(history)}

Контекст анализа:
{context}

Сформируй краткий, но содержательный ответ.
Если уместно, используй структуру:
1. Короткий вывод
2. Почему такой вывод
3. Что перепроверить вручную
4. На какие пункты ТЗ и источники опираться
"""

    answer = call_llm(
        prompt,
        system_prompt=ANALYSIS_CHAT_SYSTEM,
        max_tokens=1800,
    )

    return {
        "answer": answer,
        "related_sections": related_sections,
        "source_urls": source_urls,
    }


@app.post("/reports/markdown")
def render_markdown(payload: dict = Body(...)):
    report = _report_from_dict(payload.get("report", payload))
    return {"markdown": generate_markdown(report)}


@app.post("/reports/export/{format_name}")
def export_report(format_name: str, payload: dict = Body(...)):
    report = _report_from_dict(payload.get("report", payload))

    if format_name == "md":
        path = save_markdown(report)
        media_type = "text/markdown"
    elif format_name == "docx":
        path = save_docx(report)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif format_name == "pdf":
        path = save_pdf(report)
        media_type = "application/pdf"
    elif format_name == "xlsx":
        path = save_excel(report)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise HTTPException(status_code=404, detail=f"Unsupported report format: {format_name}")

    return FileResponse(path, filename=path.name, media_type=media_type)
