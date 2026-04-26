"""Compliance analyzer using Cloud.ru Managed RAG context."""

from __future__ import annotations

import logging
from urllib.parse import urlparse
from src.managed_rag.client import ManagedRagResult, retrieve_generate
from src.models import AnalysisReport, Requirement, RequirementVerdict
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
    search_mode: str = "managed_rag",
    batch_size: int = cfg.MAX_REQUIREMENTS_PER_BATCH,
    progress_callback=None,
) -> AnalysisReport:
    """Analyze all requirements against Cloud.ru Managed RAG context.

    Args:
        search_mode: kept for API compatibility. Managed RAG is always used.
        progress_callback: optional callable(done, total) for progress updates.
    """
    report = AnalysisReport(document_name=document_name)

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


def _managed_rag_query(req: Requirement) -> str:
    query = "\n".join(
        [
            "Нужно проверить возможность Cloud.ru выполнить требование из ТЗ.",
            f"Пункт ТЗ: {req.section or req.id}",
            f"Категория: {req.category}",
            f"Требование: {req.text}",
        ]
    )
    if req.tables:
        query += f"\nТаблица:\n{req.tables}"
    return query


def _analyze_batch(requirements: list[Requirement]) -> list[RequirementVerdict]:
    """Analyze a batch of requirements."""
    all_context_parts = []
    req_rag_results: dict[int, ManagedRagResult] = {}
    req_source_urls: dict[int, list[str]] = {r.id: [] for r in requirements}

    for req in requirements:
        try:
            rag_result = retrieve_generate(_managed_rag_query(req), number_of_results=cfg.TOP_K_RESULTS)
            req_rag_results[req.id] = rag_result
            req_source_urls[req.id].extend(_filter_urls(rag_result.source_labels))
            all_context_parts.append(
                "\n".join(
                    [
                        f"ID требования: {req.id}",
                        f"Пункт ТЗ: {req.section}",
                        rag_result.as_context(),
                    ]
                )
            )
        except Exception as exc:
            logger.warning("Managed RAG failed for requirement %s: %s", req.id, exc)
            all_context_parts.append(
                "\n".join(
                    [
                        f"ID требования: {req.id}",
                        f"Пункт ТЗ: {req.section}",
                        "Managed RAG не вернул контекст для этого требования.",
                    ]
                )
            )

    context = "\n\n---\n\n".join(all_context_parts) if all_context_parts else "Managed RAG не вернул релевантной информации."

    # Format requirements block
    req_lines = []
    for req in requirements:
        line = f"ID={req.id} | Раздел: {req.section} | Категория: {req.category}\nТребование: {req.text}"
        if req.tables:
            line += f"\nТаблица:\n{req.tables}"
        req_lines.append(line)
    requirements_block = "\n\n".join(req_lines)

    from src.prompt_store import get_prompt

    analysis_template = get_prompt("analysis_user_template")
    analysis_system = get_prompt("analysis_system")
    prompt = analysis_template.format(
        requirements_block=requirements_block,
        context=context,
    )

    result = call_llm_json(prompt, system_prompt=analysis_system, max_tokens=8000)

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
        rag_result = req_rag_results.get(req_id)
        source_note = ""
        if rag_result and rag_result.source_labels:
            source_note = "\n\nДокументы Managed RAG: " + ", ".join(rag_result.source_labels[:5])

        verdicts.append(RequirementVerdict(
            requirement_id=req_id,
            section=req.section if req else "",
            requirement_text=req.text if req else "",
            category=req.category if req else "other",
            verdict=item.get("verdict", "needs_clarification"),
            confidence=float(item.get("confidence", 0.5)),
            reasoning=item.get("reasoning", ""),
            evidence=(item.get("evidence", "") or "") + source_note,
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

    from src.prompt_store import get_prompt

    summary_template = get_prompt("summary_user_template")
    summary_system = get_prompt("summary_system")
    prompt = summary_template.format(
        doc_name=report.document_name,
        total=report.total,
        match_count=report.match_count,
        partial_count=report.partial_count,
        mismatch_count=report.mismatch_count,
        clarification_count=report.clarification_count,
        compliance_pct=report.compliance_percentage,
        top_mismatches=top_mismatches_text,
    )

    return call_llm(prompt, system_prompt=summary_system, max_tokens=2000)
