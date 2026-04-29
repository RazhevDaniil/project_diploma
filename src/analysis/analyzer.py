"""Compliance analyzer using Cloud.ru Managed RAG context."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from urllib.parse import urlparse
from src.managed_rag.client import ManagedRagResult, retrieve_generate
from src.models import AnalysisReport, PlatformAssessment, Requirement, RequirementVerdict
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
            "Приоритет поиска: 1) документация по платформам Cloud.ru; 2) документация по внешним услугам/подрядчикам.",
            "Если платформенной документации нет, явно ищи документы по внешним услугам, ПНР, ПСИ и подрядным работам.",
            f"Пункт ТЗ: {req.section or req.id}",
            f"Категория: {req.category}",
            f"Требование: {req.text}",
        ]
    )
    if req.tables:
        query += f"\nТаблица:\n{req.tables}"
    return query


def _value_to_text(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_value_to_text(item) for item in value if item not in (None, ""))
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_value_to_text(val)}" for key, val in value.items())
    return str(value)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dedupe_strings(values: list) -> list[str]:
    result = []
    for value in values:
        text = _value_to_text(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _result_metadata(item: dict) -> dict:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    result = {}
    result.update({str(k): v for k, v in item.items() if k != "metadata"})
    result.update({str(k): v for k, v in metadata.items()})
    return result


def _metadata_value(metadata: dict, keys: tuple[str, ...]) -> str:
    normalized = {key.lower(): value for key, value in metadata.items()}
    for key in keys:
        value = normalized.get(key.lower())
        if value not in (None, ""):
            return _value_to_text(value).strip()
    return ""


def _result_label(item: dict, idx: int) -> str:
    metadata = _result_metadata(item)
    return (
        _metadata_value(metadata, ("title", "document_name", "filename", "source", "url", "document_id", "id"))
        or f"Документ {idx}"
    )


def _result_content(item: dict) -> str:
    metadata = _result_metadata(item)
    return _metadata_value(metadata, ("content", "text", "chunk", "page_content", "document_text"))


def _result_url(item: dict) -> str:
    metadata = _result_metadata(item)
    return _metadata_value(metadata, ("url", "source_url", "link", "source"))


def _looks_external_service(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "external",
        "contractor",
        "subcontractor",
        "outsourc",
        "подряд",
        "внешн",
        "пнр",
        "пси",
        "услуг",
    )
    return any(marker in lowered for marker in markers)


def _source_type_from_result(item: dict) -> str:
    metadata = _result_metadata(item)
    explicit = _metadata_value(
        metadata,
        ("meta_source_type", "source_type", "doc_type", "meta_doc_type", "document_type", "category", "meta_category"),
    )
    label = _result_label(item, 0)
    platform = _platform_from_result(item, 0)
    blob = " ".join([explicit, label, platform]).strip()
    if _looks_external_service(blob):
        return "external_service"
    if platform:
        return "platform"
    return "unknown"


def _platform_from_result(item: dict, idx: int) -> str:
    metadata = _result_metadata(item)
    explicit = _metadata_value(
        metadata,
        (
            "meta_platform",
            "platform",
            "platform_name",
            "service",
            "service_name",
            "product",
            "product_name",
            "meta_service",
            "meta_product",
        ),
    )
    if explicit:
        return explicit
    title = _metadata_value(metadata, ("title", "document_name", "filename"))
    if title:
        return title.split("|", 1)[0].split(" — ", 1)[0].split(" - ", 1)[0].strip()
    return f"Cloud.ru источник {idx}"


def _format_rag_context(req: Requirement, rag_result: ManagedRagResult | None, max_chars_per_result: int = 1400) -> str:
    if not rag_result:
        return "\n".join(
            [
                f"ID требования: {req.id}",
                f"Пункт ТЗ: {req.section}",
                "Managed RAG не вернул контекст для этого требования.",
            ]
        )

    parts = [
        f"ID требования: {req.id}",
        f"Пункт ТЗ: {req.section}",
        f"Ответ Managed RAG: {rag_result.answer or 'нет ответа'}",
    ]
    for idx, source in enumerate(rag_result.results, start=1):
        title = _result_label(source, idx)
        platform = _platform_from_result(source, idx)
        source_type = _source_type_from_result(source)
        url = _result_url(source)
        content = _result_content(source)
        parts.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"Название документа: {title}",
                    f"Тип источника: {source_type}",
                    f"Платформа/услуга: {platform}",
                    f"URL/источник: {url or title}",
                    f"Фрагмент: {content[:max_chars_per_result] if content else 'нет текста'}",
                ]
            )
        )
    return "\n\n".join(parts)


def _safe_float(value, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "да", "нужно", "required"}
    return bool(value)


def _normalize_verdict(value: str) -> str:
    verdict = _value_to_text(value).strip().lower()
    if verdict in {"match", "partial", "mismatch", "needs_clarification"}:
        return verdict
    if verdict in {"yes", "true", "соответствует", "+", "ok"}:
        return "match"
    if verdict in {"no", "false", "не соответствует", "-"}:
        return "mismatch"
    if "partial" in verdict or "част" in verdict:
        return "partial"
    return "needs_clarification"


def _assessment_from_item(item: dict, rag_result: ManagedRagResult | None, idx: int) -> PlatformAssessment:
    source_urls = _filter_urls(_as_list(item.get("source_urls")))
    source_titles = _dedupe_strings(_as_list(item.get("source_titles")))
    source_type = _value_to_text(item.get("source_type", "platform")).strip() or "platform"
    platform_name = _value_to_text(item.get("platform_name", "")).strip()

    if rag_result and idx <= len(rag_result.results):
        source = rag_result.results[idx - 1]
        if not source_urls:
            source_urls = _filter_urls([_result_url(source)])
        title = _result_label(source, idx)
        if title and title not in source_titles:
            source_titles.append(title)
        if not platform_name:
            platform_name = _platform_from_result(source, idx)
        if source_type == "platform":
            source_type = _source_type_from_result(source)

    return PlatformAssessment(
        platform_name=platform_name or "Cloud.ru (платформа не определена)",
        verdict=_normalize_verdict(item.get("verdict", "needs_clarification")),
        confidence=_safe_float(item.get("confidence"), 0.5),
        reasoning=_value_to_text(item.get("reasoning", "")),
        evidence_refs=_dedupe_strings(_as_list(item.get("evidence_refs"))),
        source_urls=source_urls,
        source_titles=source_titles,
        source_type=source_type if source_type in {"platform", "external_service", "unknown"} else "unknown",
        recommendation=_value_to_text(item.get("recommendation", "")),
    )


def _platform_assessments_from_llm(
    item: dict,
    rag_result: ManagedRagResult | None,
    req: Requirement | None,
    combined_urls: list[str],
) -> list[PlatformAssessment]:
    raw_items = item.get("platform_assessments", [])
    assessments = []
    if isinstance(raw_items, list):
        for idx, raw_item in enumerate(raw_items, start=1):
            if isinstance(raw_item, dict):
                assessments.append(_assessment_from_item(raw_item, rag_result, idx))

    if assessments:
        return assessments

    source_titles = []
    platform_name = "Cloud.ru (документация не найдена)"
    source_type = "platform"
    if rag_result and rag_result.results:
        first = rag_result.results[0]
        platform_name = _platform_from_result(first, 1)
        source_type = _source_type_from_result(first)
        source_titles = [_result_label(first, 1)]

    return [
        PlatformAssessment(
            platform_name=platform_name,
            verdict=_normalize_verdict(item.get("verdict", "mismatch")),
            confidence=_safe_float(item.get("confidence"), 0.3),
            reasoning=_value_to_text(item.get("reasoning", "Оценка сформирована по общему выводу LLM.")),
            evidence_refs=_dedupe_strings(_as_list(item.get("evidence_refs"))) or (["[1]"] if source_titles else []),
            source_urls=combined_urls,
            source_titles=source_titles,
            source_type=source_type,
            recommendation=_value_to_text(item.get("recommendation", "")),
        )
    ]


def _fetch_managed_rag(req: Requirement) -> tuple[int, ManagedRagResult | None, str | None]:
    try:
        result = retrieve_generate(_managed_rag_query(req), number_of_results=cfg.MANAGED_RAG_RESULTS)
        return req.id, result, None
    except Exception as exc:
        return req.id, None, str(exc)


def _analyze_batch(requirements: list[Requirement]) -> list[RequirementVerdict]:
    """Analyze a batch of requirements."""
    all_context_parts = []
    req_rag_results: dict[int, ManagedRagResult] = {}
    req_source_urls: dict[int, list[str]] = {r.id: [] for r in requirements}
    req_map = {r.id: r for r in requirements}

    max_workers = max(1, min(cfg.MANAGED_RAG_CONCURRENCY, len(requirements)))
    logger.info("Fetching Managed RAG context for %d requirements (parallel=%d)", len(requirements), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_managed_rag, req) for req in requirements]
        for future in as_completed(futures):
            req_id, rag_result, error = future.result()
            req = req_map[req_id]
            if error or rag_result is None:
                logger.warning("Managed RAG failed for requirement %s: %s", req_id, error)
                all_context_parts.append(_format_rag_context(req, None))
                continue

            req_rag_results[req.id] = rag_result
            req_source_urls[req.id].extend(_filter_urls(rag_result.source_labels))
            for idx, source in enumerate(rag_result.results, start=1):
                req_source_urls[req.id].extend(_filter_urls([_result_url(source)]))
            all_context_parts.append(_format_rag_context(req, rag_result))

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

    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            req_id = int(item.get("requirement_id", 0))
        except (TypeError, ValueError):
            req_id = 0
        if req_id == 0:
            logger.warning("Skipping verdict without requirement_id: %s", item)
            continue
        req = req_map.get(req_id)
        # Collect source URLs from LLM response + search results, filter junk
        urls_from_llm = _filter_urls(item.get("source_urls", []))
        urls_from_search = req_source_urls.get(req_id, [])
        combined_urls = list(dict.fromkeys(urls_from_llm + urls_from_search))  # dedupe, preserve order
        rag_result = req_rag_results.get(req_id)
        platform_assessments = _platform_assessments_from_llm(item, rag_result, req, combined_urls)
        source_note = ""
        if rag_result and rag_result.source_labels:
            source_note = "\n\nДокументы Managed RAG: " + ", ".join(rag_result.source_labels[:5])

        verdicts.append(RequirementVerdict(
            requirement_id=req_id,
            section=req.section if req else "",
            requirement_text=req.text if req else "",
            category=req.category if req else "other",
            verdict=_normalize_verdict(item.get("verdict", "needs_clarification")),
            confidence=_safe_float(item.get("confidence"), 0.5),
            reasoning=_value_to_text(item.get("reasoning", "")),
            evidence=_value_to_text(item.get("evidence", "")) + source_note,
            recommendation=_value_to_text(item.get("recommendation", "")),
            source_urls=combined_urls,
            platform_assessments=platform_assessments,
            requires_external_service=_safe_bool(item.get("requires_external_service"))
            or any(a.source_type == "external_service" for a in platform_assessments),
            external_service_notes=_value_to_text(item.get("external_service_notes", "")),
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
                platform_assessments=[
                    PlatformAssessment(
                        platform_name="Cloud.ru (документация не найдена)",
                        verdict="mismatch",
                        confidence=0.0,
                        reasoning="Managed RAG/LLM не вернули подтверждение по платформенной документации.",
                        source_type="platform",
                        recommendation="Проверить вручную или добавить релевантную документацию в RAG.",
                    )
                ],
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
