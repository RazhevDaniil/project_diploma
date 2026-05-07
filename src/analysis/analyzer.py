"""Compliance analyzer using Cloud.ru Managed RAG context."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import threading
from urllib.parse import urlparse
from src.managed_rag.client import ManagedRagResult, retrieve_generate
from src.models import AnalysisReport, PlatformAssessment, Requirement, RequirementVerdict
from src.llm.client import call_llm, call_llm_json

import config as cfg

logger = logging.getLogger(__name__)

# Domains considered relevant for Cloud.ru analysis
_TRUSTED_DOMAINS = {"cloud.ru", "cloudru.tech", "sbercloud.ru",
                    "fstec.ru", "rkn.gov.ru", "consultant.ru", "garant.ru"}
KNOWN_PLATFORM_PATTERNS = (
    ("гособлак", "ГосОблако"),
    ("гос облак", "ГосОблако"),
    ("goscloud", "ГосОблако"),
    ("vmware", "Облако VMware"),
    ("vcloud", "Облако VMware"),
    ("облако vmware", "Облако VMware"),
    ("advanced", "Advanced"),
    ("evolution", "Evolution"),
)
RERANK_TOP_K = 3


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

    batches = [
        (i // batch_size, requirements[i:i + batch_size])
        for i in range(0, len(requirements), batch_size)
    ]
    total_batches = len(batches)
    completed_requirements = 0
    progress_lock = threading.Lock()
    batch_results: dict[int, list[RequirementVerdict]] = {}

    def analyze_one_batch(batch_index: int, batch: list[Requirement]) -> tuple[int, list[RequirementVerdict], int]:
        logger.info("Analyzing batch %d/%d (%d requirements)", batch_index + 1, total_batches, len(batch))
        return batch_index, _analyze_batch(batch), len(batch)

    max_workers = max(1, min(cfg.ANALYSIS_BATCH_CONCURRENCY, total_batches))
    logger.info("Analyzing %d batches (parallel=%d)", total_batches, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(analyze_one_batch, batch_index, batch)
            for batch_index, batch in batches
        ]
        for future in as_completed(futures):
            batch_index, verdicts, batch_len = future.result()
            batch_results[batch_index] = verdicts
            if progress_callback:
                with progress_lock:
                    completed_requirements += batch_len
                    progress_callback(min(completed_requirements, len(requirements)), len(requirements))

    for batch_index in range(total_batches):
        report.verdicts.extend(batch_results.get(batch_index, []))

    report.summary = _generate_summary(report)
    return report


def _managed_rag_query(req: Requirement) -> str:
    profile = _requirement_search_profile(req)
    query = "\n".join(
        [
            "Нужно проверить возможность Cloud.ru выполнить требование из ТЗ.",
            "Приоритет поиска: 1) документация по платформам Cloud.ru; 2) документация по внешним услугам/подрядчикам.",
            "Если платформенной документации нет, явно ищи документы по внешним услугам, ПНР, ПСИ и подрядным работам.",
            f"Профиль требования: {profile['cluster']}",
            f"Целевые поисковые термины: {', '.join(profile['terms'])}",
            f"Предпочтительная/вероятная платформа: {profile['platform_hint'] or 'не определена'}",
            f"Пункт ТЗ: {req.section or req.id}",
            f"Категория: {req.category}",
            f"Требование: {req.text}",
        ]
    )
    if req.tables:
        query += f"\nТаблица:\n{req.tables}"
    return query


def _managed_rag_batch_query(requirements: list[Requirement]) -> str:
    lines = [
        "Нужно проверить возможность Cloud.ru выполнить группу требований из ТЗ.",
        "Верни релевантные документы по платформам Cloud.ru и, если нужно, по внешним услугам/подрядчикам.",
        "Приоритет поиска: 1) документация по платформам Cloud.ru; 2) документация по внешним услугам.",
        "Для каждого требования учитывай профиль поиска: безопасность, лимиты ВМ, сеть, SLA, ЦОД/колокация, личный кабинет/IAM или внешние услуги.",
        "Требования:",
    ]
    for req in requirements:
        requirement_text = req.text[:700]
        profile = _requirement_search_profile(req)
        lines.append(
            "\n".join(
                [
                    f"ID={req.id}",
                    f"Пункт ТЗ: {req.section or req.id}",
                    f"Категория: {req.category}",
                    f"Профиль поиска: {profile['cluster']}",
                    f"Поисковые термины: {', '.join(profile['terms'])}",
                    f"Вероятная платформа: {profile['platform_hint'] or 'не определена'}",
                    f"Требование: {requirement_text}",
                ]
            )
        )
    return "\n\n---\n\n".join(lines)


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


def _known_platform_from_text(text: str) -> str:
    lowered = (text or "").lower()
    for marker, platform_name in KNOWN_PLATFORM_PATTERNS:
        if marker in lowered:
            return platform_name
    return ""


def _canonical_platform_name(value: str) -> str:
    text = (value or "").strip()
    return _known_platform_from_text(text) or text


def _requirement_search_profile(req: Requirement) -> dict[str, object]:
    text = " ".join([req.section or "", req.category or "", req.text or "", req.tables or ""]).lower()
    cluster = "general"
    terms: list[str] = []
    platform_hint = ""

    if any(token in text for token in ("гис", "к1", "уз-1", "152-фз", "фстэк", "фсб", "аттестат", "модель угроз", "зокии")):
        cluster = "security_certification"
        terms.extend(["ГИС К1", "ИСПДн УЗ-1", "ФСТЭК", "аттестат соответствия", "модель угроз"])
        platform_hint = "ГосОблако"
    if any(token in text for token in ("ram", "vcpu", "cpu", "диск", "iops", "bps", "виртуальн", "вм", "ssd")):
        cluster = "vm_limits"
        terms.extend(["конфигурации виртуальных машин", "лимиты ВМ", "vCPU", "RAM", "диски", "IOPS", "BPS"])
    if any(token in text for token in ("sla", "доступност", "инцидент", "время решения", "техническ", "поддержк", "24/7")):
        cluster = "sla_support"
        terms.extend(["SLA", "техническая поддержка", "инциденты", "время решения"])
    if any(token in text for token in ("интернет", "ip-адрес", "публичн", "потер", "задержк", "мбит", "сеть")):
        cluster = "network"
        terms.extend(["VPC", "публичный IP", "интернет", "сетевая задержка", "пропускная способность"])
    if any(token in text for token in ("цод", "tier", "колокац", "размещени", "2u", "стойк", "питани", "физический доступ")):
        cluster = "datacenter_colocation"
        terms.extend(["ЦОД", "TIER III", "колокация", "размещение оборудования", "физический доступ"])
    if any(token in text for token in ("личный кабинет", "2fa", "двухфактор", "ролевая", "логирован", "пользовател")):
        cluster = "console_iam"
        terms.extend(["личный кабинет", "2FA", "ролевая модель", "IAM", "аудит действий"])
    if _looks_external_service(text):
        terms.extend(["внешние услуги", "подрядчики", "ПНР", "ПСИ"])

    if not terms:
        terms.extend(["Cloud.ru документация", req.category or "требование"])
    terms = list(dict.fromkeys(terms))
    return {"cluster": cluster, "terms": terms[:10], "platform_hint": platform_hint}


def _rank_tokens(text: str) -> set[str]:
    stop_words = {
        "для",
        "или",
        "при",
        "что",
        "как",
        "над",
        "под",
        "без",
        "это",
        "исполнитель",
        "заказчик",
        "услуга",
        "услуг",
        "должен",
        "должна",
        "должно",
        "должны",
    }
    return {
        token
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9_-]{3,}", (text or "").lower())
        if token not in stop_words
    }


def _rank_numbers(text: str) -> set[str]:
    numbers = set()
    for raw in re.findall(r"\d+(?:[,.]\d+)?", text or ""):
        normalized = raw.replace(",", ".").lstrip("0")
        numbers.add(normalized or "0")
    return numbers


def _is_trusted_url(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return False
    return any(hostname == domain or hostname.endswith("." + domain) for domain in _TRUSTED_DOMAINS)


def _score_rag_source(req: Requirement, source: dict, idx: int) -> tuple[float, list[str]]:
    profile = _requirement_search_profile(req)
    title = _result_label(source, idx)
    content = _result_content(source)
    url = _result_url(source)
    platform = _platform_from_result(source, idx)
    source_type = _source_type_from_result(source)
    haystack = " ".join([title, content[:5000], url, platform, source_type]).lower()

    score = 0.0
    reasons: list[str] = []

    req_tokens = _rank_tokens(" ".join([req.section or "", req.category or "", req.text or "", req.tables or ""]))
    source_tokens = _rank_tokens(haystack)
    overlap = sorted(req_tokens & source_tokens)
    if overlap:
        score += min(len(overlap), 14) * 0.22
        reasons.append("совпали термины: " + ", ".join(overlap[:6]))

    matched_terms = []
    for term in profile.get("terms", []):
        term_text = str(term).lower()
        term_tokens = _rank_tokens(term_text)
        if term_text in haystack or (term_tokens and term_tokens.issubset(source_tokens)):
            matched_terms.append(str(term))
    if matched_terms:
        score += min(len(matched_terms), 4) * 0.9
        reasons.append("совпал профиль поиска: " + ", ".join(matched_terms[:4]))

    req_numbers = _rank_numbers(req.text + " " + (req.tables or ""))
    source_numbers = _rank_numbers(title + " " + content[:5000])
    number_overlap = sorted(req_numbers & source_numbers)
    if number_overlap:
        score += min(len(number_overlap), 4) * 0.8
        reasons.append("совпали числовые значения: " + ", ".join(number_overlap[:4]))

    platform_hint = str(profile.get("platform_hint") or "")
    if platform_hint and _canonical_platform_name(platform) == _canonical_platform_name(platform_hint):
        score += 1.5
        reasons.append(f"совпала целевая платформа: {platform_hint}")

    cluster = str(profile.get("cluster", "general"))
    if source_type == "platform" and cluster != "external_service":
        score += 0.45
        reasons.append("платформенный источник")
    if source_type == "external_service" and _looks_external_service(req.text):
        score += 0.9
        reasons.append("источник по внешним услугам")
    if _is_trusted_url(url):
        score += 0.35
        reasons.append("доверенный домен")
    if not content:
        score -= 0.7
        reasons.append("нет текстового фрагмента")

    return round(score, 3), reasons or ["слабая лексическая релевантность"]


def _rerank_rag_result(req: Requirement, rag_result: ManagedRagResult | None) -> ManagedRagResult | None:
    if not rag_result or not rag_result.results:
        return rag_result

    ranked = []
    for idx, source in enumerate(rag_result.results, start=1):
        score, reasons = _score_rag_source(req, source, idx)
        annotated = dict(source)
        annotated["_rerank"] = {
            "original_rank": idx,
            "score": score,
            "reasons": reasons,
        }
        ranked.append((score, idx, annotated))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [item[2] for item in ranked[:RERANK_TOP_K]]
    source_labels = [_result_label(item, idx) for idx, item in enumerate(selected, start=1)]
    return ManagedRagResult(
        answer=rag_result.answer,
        results=selected,
        reasoning_content=rag_result.reasoning_content,
        source_labels=list(dict.fromkeys(source_labels)),
    )


def _trace_source_summary(source: dict, idx: int) -> dict:
    rerank = source.get("_rerank", {}) if isinstance(source.get("_rerank"), dict) else {}
    content = _result_content(source)
    return {
        "rank": idx,
        "original_rank": rerank.get("original_rank", idx),
        "score": rerank.get("score", 0.0),
        "reasons": rerank.get("reasons", []),
        "title": _result_label(source, idx),
        "url": _result_url(source),
        "platform": _platform_from_result(source, idx),
        "source_type": _source_type_from_result(source),
        "excerpt": content[:500] if content else "",
    }


def _build_analysis_trace(
    req: Requirement,
    rag_mode: str,
    rag_query: str,
    rag_result: ManagedRagResult | None,
    rag_error: str | None = None,
) -> dict:
    profile = _requirement_search_profile(req)
    selected_sources = []
    if rag_result and rag_result.results:
        selected_sources = [_trace_source_summary(source, idx) for idx, source in enumerate(rag_result.results, start=1)]
    return {
        "rag_mode": rag_mode,
        "profile": profile,
        "rag_query": rag_query[:3000],
        "rag_error": rag_error or "",
        "managed_rag_answer": (rag_result.answer[:1000] if rag_result and rag_result.answer else ""),
        "selected_sources": selected_sources,
    }


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
    if platform and not platform.lower().startswith("cloud.ru источник"):
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
        return _canonical_platform_name(explicit)
    title = _metadata_value(metadata, ("title", "document_name", "filename"))
    platform_from_title = _known_platform_from_text(title)
    if platform_from_title:
        return platform_from_title
    content = _result_content(item)
    platform_from_content = _known_platform_from_text(content[:1000])
    if platform_from_content:
        return platform_from_content
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
        rerank = source.get("_rerank", {}) if isinstance(source.get("_rerank"), dict) else {}
        rerank_reasons = "; ".join(rerank.get("reasons", [])[:4]) if isinstance(rerank.get("reasons"), list) else ""
        parts.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"Название документа: {title}",
                    f"Тип источника: {source_type}",
                    f"Платформа/услуга: {platform}",
                    f"RAG-rerank score: {rerank.get('score', 'n/a')}; причины: {rerank_reasons or 'не указаны'}",
                    f"URL/источник: {url or title}",
                    f"Фрагмент: {content[:max_chars_per_result] if content else 'нет текста'}",
                ]
            )
        )
    return "\n\n".join(parts)


def _format_batch_rag_context(requirements: list[Requirement], rag_result: ManagedRagResult | None) -> str:
    req_lines = []
    for req in requirements:
        req_lines.append(f"ID требования: {req.id}; пункт ТЗ: {req.section}; категория: {req.category}")
    if not rag_result:
        return "\n".join(
            [
                "Контекст Managed RAG для группы требований.",
                "\n".join(req_lines),
                "Managed RAG не вернул контекст для этой группы.",
            ]
        )

    parts = [
        "Контекст Managed RAG для группы требований.",
        "\n".join(req_lines),
        f"Ответ Managed RAG: {rag_result.answer or 'нет ответа'}",
    ]
    for idx, source in enumerate(rag_result.results, start=1):
        title = _result_label(source, idx)
        platform = _platform_from_result(source, idx)
        source_type = _source_type_from_result(source)
        url = _result_url(source)
        content = _result_content(source)
        rerank = source.get("_rerank", {}) if isinstance(source.get("_rerank"), dict) else {}
        rerank_reasons = "; ".join(rerank.get("reasons", [])[:4]) if isinstance(rerank.get("reasons"), list) else ""
        parts.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"Название документа: {title}",
                    f"Тип источника: {source_type}",
                    f"Платформа/услуга: {platform}",
                    f"RAG-rerank score: {rerank.get('score', 'n/a')}; причины: {rerank_reasons or 'не указаны'}",
                    f"URL/источник: {url or title}",
                    f"Фрагмент: {content[:1600] if content else 'нет текста'}",
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
        source_platform = _platform_from_result(source, idx)
        if not source_urls:
            source_urls = _filter_urls([_result_url(source)])
        title = _result_label(source, idx)
        if title and title not in source_titles:
            source_titles.append(title)
        if not item.get("evidence_refs"):
            item = {**item, "evidence_refs": [f"[{idx}]"]}
        if source_type == "platform":
            source_type = _source_type_from_result(source)
        if source_type == "platform" and source_platform:
            platform_name = source_platform
        elif not platform_name:
            platform_name = source_platform

    return PlatformAssessment(
        platform_name=_canonical_platform_name(platform_name) or "Cloud.ru (платформа не определена)",
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


def _assessment_has_source(assessment: PlatformAssessment) -> bool:
    return bool(assessment.source_urls or assessment.source_titles)


def _assessment_has_ref(assessment: PlatformAssessment) -> bool:
    return bool(assessment.evidence_refs)


def _refs_from_text(text: str) -> list[int]:
    refs = []
    for raw in re.findall(r"\[(\d+)\]", text or ""):
        try:
            refs.append(int(raw))
        except ValueError:
            continue
    return refs


def _refs_match_selected_sources(refs: list[int], selected_source_count: int) -> bool:
    return bool(selected_source_count > 0 and any(1 <= ref <= selected_source_count for ref in refs))


def _evidence_quote_text(evidence: str) -> str:
    return (evidence or "").split("\n\nВыбранные документы Managed RAG:", 1)[0].strip()


def _verdict_has_evidence_quote(verdict: RequirementVerdict, selected_source_count: int) -> bool:
    text = _evidence_quote_text(verdict.evidence)
    return bool(len(text) >= 12 and _refs_match_selected_sources(_refs_from_text(text), selected_source_count))


def _verdict_has_cited_evidence(verdict: RequirementVerdict) -> bool:
    has_source = bool(verdict.source_urls) or any(_assessment_has_source(item) for item in verdict.platform_assessments)
    selected_source_count = len((verdict.trace or {}).get("selected_sources") or [])
    return has_source and _verdict_has_evidence_quote(verdict, selected_source_count)


def _apply_evidence_contract(verdict: RequirementVerdict) -> RequirementVerdict:
    notes = list(verdict.evidence_contract_notes or [])
    original_verdict = verdict.verdict
    selected_source_count = len((verdict.trace or {}).get("selected_sources") or [])

    for assessment in verdict.platform_assessments:
        if assessment.verdict not in {"match", "partial"}:
            continue
        assessment_refs_valid = _refs_match_selected_sources(
            _refs_from_text(" ".join(assessment.evidence_refs)),
            selected_source_count,
        )
        if _assessment_has_source(assessment) and _assessment_has_ref(assessment) and assessment_refs_valid:
            continue
        if _assessment_has_source(assessment) and _assessment_has_ref(assessment) and not assessment_refs_valid:
            assessment.verdict = "needs_clarification"
            assessment.reasoning = (
                assessment.reasoning.rstrip()
                + " Evidence contract: сноска не соответствует выбранным RAG-фрагментам."
            ).strip()
            notes.append(f"{assessment.platform_name}: сноска не соответствует выбранным RAG-фрагментам")
            continue
        if _assessment_has_source(assessment) and not _assessment_has_ref(assessment):
            assessment.verdict = "needs_clarification"
            assessment.reasoning = (
                assessment.reasoning.rstrip()
                + " Evidence contract: источник найден, но нет явной сноски на подтверждающий фрагмент."
            ).strip()
            notes.append(f"{assessment.platform_name}: нет явной сноски на источник")
        else:
            assessment.verdict = "needs_clarification"
            assessment.reasoning = (
                assessment.reasoning.rstrip()
                + " Evidence contract: нет подтверждающего источника из RAG."
            ).strip()
            notes.append(f"{assessment.platform_name}: нет подтверждающего источника")

    if verdict.verdict in {"match", "partial"} and not _verdict_has_cited_evidence(verdict):
        verdict.verdict = "needs_clarification"
        verdict.confidence = min(verdict.confidence, 0.5)
        if not (verdict.trace or {}).get("selected_sources"):
            notes.append("Нет выбранного RAG-фрагмента")
        if not (bool(verdict.source_urls) or any(_assessment_has_source(item) for item in verdict.platform_assessments)):
            notes.append("Нет подтверждающего источника")
        if not _verdict_has_evidence_quote(verdict, selected_source_count):
            notes.append("В evidence нет содержательной цитаты/доказательства с валидной сноской [n]")
        notes.append(
            f"Вердикт {original_verdict} понижен: нет связки источник + сноска + выбранный RAG-фрагмент"
        )

    if verdict.verdict in {"match", "partial"}:
        verdict.evidence_status = "confirmed"
    elif notes:
        verdict.evidence_status = "downgraded" if original_verdict != verdict.verdict else "weak"
    elif not (verdict.trace or {}).get("selected_sources"):
        verdict.evidence_status = "missing"
    else:
        verdict.evidence_status = "weak" if verdict.verdict == "needs_clarification" else "confirmed"

    if notes:
        prefix = "Evidence contract: "
        verdict.reasoning = (verdict.reasoning.rstrip() + "\n" + prefix + "; ".join(dict.fromkeys(notes))).strip()
    verdict.evidence_contract_notes = list(dict.fromkeys(notes))
    if verdict.trace is not None:
        verdict.trace["evidence_contract"] = {
            "original_verdict": original_verdict,
            "final_verdict": verdict.verdict,
            "status": verdict.evidence_status,
            "notes": verdict.evidence_contract_notes,
        }
    return verdict


def _fetch_managed_rag(req: Requirement) -> tuple[int, ManagedRagResult | None, str | None]:
    try:
        result = retrieve_generate(_managed_rag_query(req), number_of_results=cfg.MANAGED_RAG_RESULTS)
        return req.id, result, None
    except Exception as exc:
        return req.id, None, str(exc)


def _fetch_batch_managed_rag(requirements: list[Requirement], query: str | None = None) -> tuple[ManagedRagResult | None, str | None]:
    try:
        result = retrieve_generate(query or _managed_rag_batch_query(requirements), number_of_results=cfg.MANAGED_RAG_RESULTS)
        return result, None
    except Exception as exc:
        return None, str(exc)


def _analyze_batch(requirements: list[Requirement]) -> list[RequirementVerdict]:
    """Analyze a batch of requirements."""
    all_context_parts = []
    req_rag_results: dict[int, ManagedRagResult] = {}
    req_source_urls: dict[int, list[str]] = {r.id: [] for r in requirements}
    req_traces: dict[int, dict] = {}
    req_map = {r.id: r for r in requirements}

    if cfg.ANALYSIS_RAG_MODE == "grouped":
        logger.info("Fetching grouped Managed RAG context for %d requirements", len(requirements))
        batch_query = _managed_rag_batch_query(requirements)
        rag_result, error = _fetch_batch_managed_rag(requirements, batch_query)
        if error or rag_result is None:
            logger.warning("Grouped Managed RAG failed: %s", error)
            for req in requirements:
                req_traces[req.id] = _build_analysis_trace(req, "grouped", batch_query, None, error)
                all_context_parts.append(_format_rag_context(req, None))
        else:
            for req in requirements:
                reranked = _rerank_rag_result(req, rag_result)
                req_rag_results[req.id] = reranked
                req_traces[req.id] = _build_analysis_trace(req, "grouped", batch_query, reranked)
                source_urls = []
                for idx, source in enumerate((reranked.results if reranked else []), start=1):
                    source_urls.extend(_filter_urls([_result_url(source)]))
                req_source_urls[req.id].extend(list(dict.fromkeys(source_urls)))
                all_context_parts.append(_format_rag_context(req, reranked, max_chars_per_result=900))
    else:
        max_workers = max(1, min(cfg.MANAGED_RAG_CONCURRENCY, len(requirements)))
        logger.info("Fetching Managed RAG context for %d requirements (parallel=%d)", len(requirements), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fetch_managed_rag, req) for req in requirements]
            for future in as_completed(futures):
                req_id, rag_result, error = future.result()
                req = req_map[req_id]
                if error or rag_result is None:
                    logger.warning("Managed RAG failed for requirement %s: %s", req_id, error)
                    req_traces[req_id] = _build_analysis_trace(req, "per_requirement", _managed_rag_query(req), None, error)
                    all_context_parts.append(_format_rag_context(req, None))
                    continue

                reranked = _rerank_rag_result(req, rag_result)
                req_rag_results[req.id] = reranked
                req_traces[req.id] = _build_analysis_trace(req, "per_requirement", _managed_rag_query(req), reranked)
                for idx, source in enumerate((reranked.results if reranked else []), start=1):
                    req_source_urls[req.id].extend(_filter_urls([_result_url(source)]))
                all_context_parts.append(_format_rag_context(req, reranked, max_chars_per_result=900))

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
            source_note = "\n\nВыбранные документы Managed RAG: " + ", ".join(rag_result.source_labels[:5])
        trace = dict(req_traces.get(req_id, {}))
        trace["llm_response"] = {
            "verdict": _value_to_text(item.get("verdict", "")),
            "confidence": item.get("confidence"),
            "source_urls": item.get("source_urls", []),
            "evidence": _value_to_text(item.get("evidence", ""))[:1000],
        }

        verdict = RequirementVerdict(
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
            trace=trace,
        )
        verdicts.append(_apply_evidence_contract(verdict))

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
                evidence_status="missing",
                evidence_contract_notes=["LLM не вернула вердикт по требованию"],
                trace=req_traces.get(req.id, {}),
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
