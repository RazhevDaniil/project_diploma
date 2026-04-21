"""FastAPI backend for the Cloud.ru TZ analyzer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config as cfg
from src.analysis.analyzer import analyze_requirements
from src.crawler.spider import crawl_docs_sync, fetch_sitemap_urls, filter_docs_urls, index_crawled_pages
from src.knowledge_base.indexer import index_raw_texts
from src.knowledge_base.store import get_persisted_vector_count, reset_vectorstore, search
from src.models import AnalysisReport, Requirement, RequirementVerdict
from src.parser.document_parser import parse_document
from src.parser.requirement_extractor import extract_requirements
from src.report.generator import generate_markdown, save_docx, save_excel, save_markdown, save_pdf
from src.runtime_config import apply_runtime_settings

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
    verdicts = []
    for verdict in item.get("verdicts", []):
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
            )
        )
    return AnalysisReport(
        document_name=str(item.get("document_name", "document")),
        verdicts=verdicts,
        summary=str(item.get("summary", "")),
    )


def _search_results_to_dict(results) -> list[dict]:
    payload = []
    for doc in results:
        payload.append(
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata),
                "source": doc.metadata.get("url", doc.metadata.get("source", "unknown")),
                "title": doc.metadata.get("title", ""),
            }
        )
    return payload


@app.get("/health")
def healthcheck():
    return {
        "status": "ok",
        "provider": "foundation_models",
        "vector_count": get_persisted_vector_count(),
        "llm_model": cfg.OPENAI_MODEL,
        "embedding_model": cfg.OPENAI_EMBEDDING_MODEL,
    }


@app.get("/kb/status")
def knowledge_base_status():
    count = get_persisted_vector_count()
    return {
        "index_exists": count > 0,
        "vector_count": count,
        "paths": {
            "faiss_index": str(cfg.FAISS_INDEX_DIR),
            "reports": str(cfg.REPORTS_DIR),
        },
    }


@app.post("/kb/reset")
def knowledge_base_reset():
    reset_vectorstore()
    return {"ok": True, "vector_count": 0}


@app.post("/kb/crawl")
def knowledge_base_crawl(payload: dict = Body(...)):
    apply_runtime_settings(payload.get("llm_settings"))

    max_pages = int(payload.get("max_pages", cfg.CRAWL_MAX_PAGES))
    concurrency = int(payload.get("concurrency", cfg.CRAWL_CONCURRENCY))

    all_urls = fetch_sitemap_urls()
    doc_urls = filter_docs_urls(all_urls)
    total_urls = len(doc_urls) if max_pages == 0 else min(max_pages, len(doc_urls))

    pages = crawl_docs_sync(
        urls=doc_urls,
        max_pages=max_pages,
        concurrency=concurrency,
    )
    total_vectors = index_crawled_pages(pages)

    return {
        "ok": True,
        "found_urls": len(doc_urls),
        "crawled_urls": total_urls,
        "indexed_pages": len(pages),
        "vector_count": total_vectors,
    }


@app.post("/kb/index-files")
async def knowledge_base_index_files(
    files: list[UploadFile] = File(...),
    llm_settings_json: str | None = Form(default=None),
):
    apply_runtime_settings(_parse_settings_json(llm_settings_json))

    texts = []
    file_summaries = []

    for upload in files:
        tmp_path = await _save_upload(upload, cfg.KNOWLEDGE_BASE_DIR)
        suffix = tmp_path.suffix.lower()
        if suffix in (".pdf", ".docx", ".doc"):
            parsed = parse_document(tmp_path)
            text = parsed.full_text
        else:
            text = tmp_path.read_text(encoding="utf-8", errors="replace")

        texts.append({"text": text, "source": upload.filename or tmp_path.name, "filename": upload.filename or tmp_path.name})
        file_summaries.append(
            {
                "filename": upload.filename or tmp_path.name,
                "chars": len(text),
            }
        )

    total = index_raw_texts(texts)
    return {"ok": True, "files": file_summaries, "vector_count": total}


@app.post("/kb/search")
def knowledge_base_search(payload: dict = Body(...)):
    apply_runtime_settings(payload.get("llm_settings"))

    query = str(payload.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    k = int(payload.get("k", cfg.TOP_K_RESULTS))
    results = search(query, k=k)
    return {"results": _search_results_to_dict(results)}


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
        search_mode=str(payload.get("search_mode", "rag")),
    )
    return report.to_dict()


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
