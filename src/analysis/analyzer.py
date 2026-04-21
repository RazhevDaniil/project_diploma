"""Compliance analyzer — the core RAG analysis engine."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from src.analysis.prompts import (
    ANALYSIS_PROMPT_TEMPLATE,
    ANALYSIS_SYSTEM,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM,
)
from src.models import AnalysisReport, Requirement, RequirementVerdict
from src.knowledge_base.store import search
from src.llm.client import call_llm, call_llm_json

import config as cfg

logger = logging.getLogger(__name__)

# Domains considered relevant for Cloud.ru analysis
_TRUSTED_DOMAINS = {"cloud.ru", "cloudru.tech", "sbercloud.ru",
                    "fstec.ru", "rkn.gov.ru", "consultant.ru", "garant.ru"}


def _filter_urls(urls: list) -> list[str]:
    """Filter out irrelevant/junk URLs, keep only trusted domains."""
    if not isinstance(urls, list):
        return []
    filtered = []
    for url in urls:
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        try:
            hostname = urlparse(url).hostname or ""
            for domain in _TRUSTED_DOMAINS:
                if hostname == domain or hostname.endswith("." + domain):
                    filtered.append(url)
                    break
        except Exception:
            continue
    return filtered


def analyze_requirements(
    requirements: list[Requirement],
    document_name: str,
    search_mode: str = "rag",  # "rag" or "live"
    batch_size: int = cfg.MAX_REQUIREMENTS_PER_BATCH,
    progress_callback=None,
) -> AnalysisReport:
    """Analyze all requirements against the Cloud.ru knowledge base.

    Args:
        search_mode: "rag" uses FAISS vector store, "live" searches web/docs per requirement.
        progress_callback: optional callable(done, total) for progress updates.
    """
    report = AnalysisReport(document_name=document_name)

    if search_mode == "live":
        # Live mode: analyze one requirement at a time
        total = len(requirements)
        for i, req in enumerate(requirements):
            logger.info("Live analyzing requirement %d/%d", i + 1, total)
            verdict = _analyze_single_live(req)
            report.verdicts.append(verdict)
            if progress_callback:
                progress_callback(i + 1, total)
    else:
        # RAG mode: analyze in batches
        total_batches = (len(requirements) + batch_size - 1) // batch_size
        for i in range(0, len(requirements), batch_size):
            batch = requirements[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info("Analyzing batch %d/%d (%d requirements)", batch_num, total_batches, len(batch))
            verdicts = _analyze_batch(batch)
            report.verdicts.extend(verdicts)
            if progress_callback:
                progress_callback(min(i + batch_size, len(requirements)), len(requirements))

    report.summary = _generate_summary(report)
    return report


def _analyze_single_live(req: Requirement) -> RequirementVerdict:
    """Analyze a single requirement using live web/docs search."""
    from src.search.live_search import search_for_requirement

    # Search for relevant content
    query = req.text
    if req.tables:
        query += "\n" + req.tables
    results = search_for_requirement(query, category=req.category, max_results=5)

    # Build context from search results
    source_urls: list[str] = []
    if results:
        context_parts = []
        for r in results:
            context_parts.append(f"[Источник: {r.url}]\n[Заголовок: {r.title}]\n{r.content[:2000]}")
            if r.url and r.url not in source_urls:
                source_urls.append(r.url)
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "Не найдено релевантной информации в документации и в интернете."

    # Format requirement
    req_line = f"ID={req.id} | Раздел: {req.section} | Категория: {req.category}\nТребование: {req.text}"
    if req.tables:
        req_line += f"\nТаблица:\n{req.tables}"

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        requirements_block=req_line,
        context=context,
    )

    result = call_llm_json(prompt, system_prompt=ANALYSIS_SYSTEM, max_tokens=4000)

    # Parse verdict
    item = result[0] if isinstance(result, list) and result else result
    if not isinstance(item, dict):
        item = {}

    urls_from_llm = _filter_urls(item.get("source_urls", []))
    combined_urls = list(dict.fromkeys(urls_from_llm + source_urls))

    return RequirementVerdict(
        requirement_id=req.id,
        section=req.section,
        requirement_text=req.text,
        category=req.category,
        verdict=item.get("verdict", "needs_clarification"),
        confidence=float(item.get("confidence", 0.5)),
        reasoning=item.get("reasoning", "Не удалось получить оценку"),
        evidence=item.get("evidence", ""),
        recommendation=item.get("recommendation", "Требуется ручная проверка"),
        source_urls=combined_urls,
    )


def _analyze_batch(requirements: list[Requirement]) -> list[RequirementVerdict]:
    """Analyze a batch of requirements."""
    # Collect context from knowledge base for all requirements in batch
    all_context_parts = []
    # Track which URLs are relevant to which requirement
    req_source_urls: dict[int, list[str]] = {r.id: [] for r in requirements}

    for req in requirements:
        query = req.text
        if req.tables:
            query += "\n" + req.tables
        docs = search(query, k=cfg.TOP_K_RESULTS)
        for doc in docs:
            url = doc.metadata.get("url", doc.metadata.get("source", ""))
            title = doc.metadata.get("title", "")
            source_label = url if url.startswith("http") else doc.metadata.get("source", "unknown")
            all_context_parts.append(f"[Источник: {source_label}]\n[Заголовок: {title}]\n{doc.page_content}")
            if url.startswith("http") and url not in req_source_urls[req.id]:
                req_source_urls[req.id].append(url)

    context = "\n\n---\n\n".join(all_context_parts) if all_context_parts else "База знаний пуста или не содержит релевантной информации."

    # Format requirements block
    req_lines = []
    for req in requirements:
        line = f"ID={req.id} | Раздел: {req.section} | Категория: {req.category}\nТребование: {req.text}"
        if req.tables:
            line += f"\nТаблица:\n{req.tables}"
        req_lines.append(line)
    requirements_block = "\n\n".join(req_lines)

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        requirements_block=requirements_block,
        context=context,
    )

    result = call_llm_json(prompt, system_prompt=ANALYSIS_SYSTEM, max_tokens=8000)

    verdicts = []
    items = result if isinstance(result, list) else [result]

    # Build a lookup for requirements
    req_map = {r.id: r for r in requirements}

    for item in items:
        if not isinstance(item, dict):
            continue
        req_id = item.get("requirement_id", 0)
        req = req_map.get(req_id)
        # Collect source URLs from LLM response + search results, filter junk
        urls_from_llm = _filter_urls(item.get("source_urls", []))
        urls_from_search = req_source_urls.get(req_id, [])
        combined_urls = list(dict.fromkeys(urls_from_llm + urls_from_search))  # dedupe, preserve order

        verdicts.append(RequirementVerdict(
            requirement_id=req_id,
            section=req.section if req else "",
            requirement_text=req.text if req else "",
            category=req.category if req else "other",
            verdict=item.get("verdict", "needs_clarification"),
            confidence=float(item.get("confidence", 0.5)),
            reasoning=item.get("reasoning", ""),
            evidence=item.get("evidence", ""),
            recommendation=item.get("recommendation", ""),
            source_urls=combined_urls,
        ))

    # Add verdicts for requirements not returned by LLM
    returned_ids = {v.requirement_id for v in verdicts}
    for req in requirements:
        if req.id not in returned_ids:
            verdicts.append(RequirementVerdict(
                requirement_id=req.id,
                section=req.section,
                requirement_text=req.text,
                category=req.category,
                verdict="needs_clarification",
                confidence=0.0,
                reasoning="Не удалось получить оценку от LLM",
                evidence="",
                recommendation="Требуется ручная проверка",
            ))

    return verdicts


def _generate_summary(report: AnalysisReport) -> str:
    """Generate a text summary of the report."""
    top_mismatches = []
    for v in report.verdicts:
        if v.verdict == "mismatch":
            top_mismatches.append(f"- [{v.section}] {v.requirement_text[:100]}... — {v.reasoning}")
    top_mismatches_text = "\n".join(top_mismatches[:10]) if top_mismatches else "Нет"

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        doc_name=report.document_name,
        total=report.total,
        match_count=report.match_count,
        partial_count=report.partial_count,
        mismatch_count=report.mismatch_count,
        clarification_count=report.clarification_count,
        compliance_pct=report.compliance_percentage,
        top_mismatches=top_mismatches_text,
    )

    return call_llm(prompt, system_prompt=SUMMARY_SYSTEM, max_tokens=2000)
