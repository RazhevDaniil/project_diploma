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
from src.runtime_config import RuntimeSettings, build_runtime_settings

logger = logging.getLogger(__name__)

# v12.1: side-channel из _local_rag_supplement_for_batch — нужно, чтобы
# downstream-код (assemble verdicts) мог записать в trace, какие local_rag
# URL'ы реально были подмешаны в контекст. Без этого после прогона невозможно
# понять, активировался ли local_rag вообще, и каждый отчёт приходится
# дебажить по логам контейнера.
_LAST_LOCAL_RAG_HITS_BY_REQ: dict[str, list[dict]] = {}


def _take_and_clear_local_rag_hits() -> dict[str, list[dict]]:
    """Возвращает snapshot текущего side-channel и чистит его. Вызывается
    сразу после _format_batch_rag_context, чтобы при следующем batch'е
    данные не перепутались."""
    snapshot = dict(_LAST_LOCAL_RAG_HITS_BY_REQ)
    _LAST_LOCAL_RAG_HITS_BY_REQ.clear()
    return snapshot

# Domains considered relevant for Cloud.ru analysis
_TRUSTED_DOMAINS = {"cloud.ru", "cloudru.tech", "sbercloud.ru",
                    "fstec.ru", "rkn.gov.ru", "consultant.ru", "garant.ru"}

# Источники, которые в KB всплывают в топ-3 на широком классе запросов и
# при этом НЕ содержат продуктовой информации (страницы про AI-ассистента
# Cloudia, GigaChat, маркетинговые landing'и). LLM начинает использовать их
# как «универсальные источники» и галлюцинирует цитаты. Жёстко штрафуем —
# они не должны выигрывать у целевых страниц документации продуктов.
_LOW_RELEVANCE_SOURCE_MARKERS = (
    "ai_assistant",
    "ai-assistant",
    "cloudia",
    "gigachat",
    "guides__doc_research",
    "гига-помощник",
    "гига помощник",
    "гигачат",
    "ai-помощник",
)


def _is_low_relevance_source(title: str, url: str, content: str) -> bool:
    """True, если чанк относится к нерелевантным «универсальным» страницам."""
    haystack = " ".join([title or "", url or "", (content or "")[:300]]).lower()
    return any(marker in haystack for marker in _LOW_RELEVANCE_SOURCE_MARKERS)
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


# Патч 11 (ZK10). Эвристика для SLA-классификации: процедурное описание
# приоритетов / категорий / типов инцидентов БЕЗ конкретных числовых таргетов.
# Такие пункты — это регламент взаимодействия (Service Desk-таблица), а не
# техническое требование, оценимое анализатором. Уходят сразу в out_of_scope.
_SLA_CLASSIFICATION_MARKERS = (
    "подраздел",
    "подразделяют",
    "подразделяет",
    "подразделя",
    "классификац",
    "матриц приоритет",
    "матрица приоритет",
    "уровней критич",
    "уровней приоритет",
    "категори инцидент",
    "категории инцидент",
    "типов инцидент",
    "типы инцидент",
    "виды инцидент",
)
# Если хоть один из маркеров есть И в тексте отсутствуют конкретные числовые
# SLA-параметры (часы, проценты, минуты, секунды, дни) — это классификация.
_SLA_METRIC_RE = re.compile(
    r"\b\d+([.,]\d+)?\s*(%|час|часов|часа|ч\b|мин|минут|сек|секунд|"
    r"д\b|день|дней|раб(?:\.|очих)?|суток)",
    re.IGNORECASE,
)


def _is_sla_classification(text: str) -> bool:
    """True, если требование — это SLA-классификация (категории/приоритеты)
    без числовых SLA-таргетов."""
    if not text:
        return False
    lowered = text.lower()
    if not any(marker in lowered for marker in _SLA_CLASSIFICATION_MARKERS):
        return False
    if "приоритет" not in lowered and "инцидент" not in lowered and "критич" not in lowered:
        return False
    # Если в тексте есть конкретные числовые SLA-таргеты, это уже не просто
    # классификация — оцениваем по сути (LLM-вызов нужен).
    if _SLA_METRIC_RE.search(text):
        return False
    return True


def analyze_requirements(
    requirements: list[Requirement],
    document_name: str,
    search_mode: str = "managed_rag",
    batch_size: int | None = None,
    progress_callback=None,
    extraction_summary: dict | None = None,
    settings: RuntimeSettings | dict | None = None,
) -> AnalysisReport:
    """Analyze all requirements against Cloud.ru Managed RAG context.

    Args:
        search_mode: kept for API compatibility. Managed RAG is always used.
        progress_callback: optional callable(done, total) for progress updates.
    """
    runtime_settings = build_runtime_settings(settings)
    effective_batch_size = max(1, batch_size or runtime_settings.max_requirements_per_batch)
    report = AnalysisReport(document_name=document_name, extraction_summary=extraction_summary or {})

    # Процедурные пункты закупки (ОКПД, цена, обеспечение заявки, антикоррупция
    # и т.п.) НЕ оцениваются LLM/RAG-пайплайном — они вне технического scope.
    # Создаём для них synthetic-verdict со специальным verdict='out_of_scope'.
    # Счётчики AnalysisReport их исключают из знаменателя compliance%.
    procedural_reqs = [r for r in requirements if (r.category or "").lower() == "procedural"]
    # Патч 11 (ZK10). SLA-классификации («инциденты должны подразделяться на
    # четыре приоритета», «матрица приоритетов» без конкретных SLA-таргетов)
    # — это процедурный регламент, а не техническое требование к Cloud.ru.
    # Раньше анализатор тратил LLM-вызов и ставил partial без основания (A-5).
    sla_classification_reqs = [
        r for r in requirements
        if (r.category or "").lower() != "procedural"
        and _is_sla_classification(r.text or "")
    ]
    sla_classification_ids = {r.id for r in sla_classification_reqs}
    technical_reqs = [
        r for r in requirements
        if (r.category or "").lower() != "procedural"
        and r.id not in sla_classification_ids
    ]
    procedural_verdicts: list[RequirementVerdict] = []
    for req in procedural_reqs:
        procedural_verdicts.append(
            RequirementVerdict(
                requirement_id=req.id,
                section=req.section,
                requirement_text=req.text,
                category="procedural",
                verdict="out_of_scope",
                confidence=1.0,
                reasoning=(
                    "Процедурный пункт закупки (ОКПД/цена/обеспечение заявки/"
                    "антикоррупция/идентификация участников) — вне технической "
                    "оценки Cloud.ru. Заполняется коммерческой/правовой командой "
                    "Cloud.ru при подготовке коммерческого предложения."
                ),
                evidence="",
                recommendation="",
                source_urls=[],
                platform_assessments=[],
                requires_external_service=False,
                external_service_notes="",
                evidence_status="out_of_scope",
            )
        )
    for req in sla_classification_reqs:
        procedural_verdicts.append(
            RequirementVerdict(
                requirement_id=req.id,
                section=req.section,
                requirement_text=req.text,
                category="sla_classification",
                verdict="out_of_scope",
                confidence=1.0,
                reasoning=(
                    "SLA-классификация (приоритеты инцидентов / матрица "
                    "категорий / описание уровней критичности без конкретных "
                    "числовых SLA-таргетов) — это процедурный регламент "
                    "взаимодействия, а не техническая возможность Cloud.ru. "
                    "Согласуется в проектном SLA индивидуально."
                ),
                evidence="",
                recommendation="",
                source_urls=[],
                platform_assessments=[],
                requires_external_service=False,
                external_service_notes="",
                evidence_status="out_of_scope",
            )
        )
    if procedural_verdicts:
        logger.info(
            "Out-of-scope shortcut: %d procedural + %d sla_classification "
            "marked as out_of_scope (skipped LLM) из %d требований",
            len(procedural_reqs), len(sla_classification_reqs), len(requirements),
        )

    batches = [
        (i // effective_batch_size, technical_reqs[i:i + effective_batch_size])
        for i in range(0, len(technical_reqs), effective_batch_size)
    ]
    total_batches = len(batches)
    completed_requirements = 0
    progress_lock = threading.Lock()
    batch_results: dict[int, list[RequirementVerdict]] = {}

    def analyze_one_batch(batch_index: int, batch: list[Requirement]) -> tuple[int, list[RequirementVerdict], int]:
        logger.info("Analyzing batch %d/%d (%d requirements)", batch_index + 1, total_batches, len(batch))
        return batch_index, _analyze_batch(batch, runtime_settings), len(batch)

    if total_batches > 0:
        max_workers = max(1, min(runtime_settings.analysis_batch_concurrency, total_batches))
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
    # Дополняем процедурными verdict'ами (вне LLM-цикла) и пересортировываем
    # по requirement_id, чтобы порядок в отчёте совпадал с порядком в ТЗ.
    if procedural_verdicts:
        report.verdicts.extend(procedural_verdicts)
        if progress_callback:
            completed_requirements += len(procedural_verdicts)
            progress_callback(min(completed_requirements, len(requirements)), len(requirements))
    report.verdicts.sort(key=lambda v: v.requirement_id)

    # Патчи 7 + 10 (ZK10): cross-verdict post-process. Дедуп reasoning'а
    # между разными requirement_id и дискретизация confidence.
    post_stats = _post_process_verdicts(report.verdicts, runtime_settings)
    # Сохраняем для последующего отображения в quality-секции отчёта (patch 14).
    if isinstance(report.extraction_summary, dict):
        report.extraction_summary.setdefault("analysis_quality", {}).update(post_stats)

    # Генерируем РАЗДЕЛЬНЫЕ резюме под два режима шапки UI:
    # • portfolio — best-case по портфелю (исторический режим, остаётся
    #   в `summary` для обратной совместимости).
    # • platform  — только по рекомендуемой платформе (счётчики и процент
    #   считаются по platform_assessments этой платформы).
    portfolio_summary = _generate_summary(report, runtime_settings, mode="portfolio")
    report.summary_portfolio = portfolio_summary
    report.summary = portfolio_summary  # обратная совместимость
    try:
        report.summary_platform = _generate_summary(report, runtime_settings, mode="platform")
    except Exception as exc:
        logger.warning("Не удалось сгенерировать platform-summary: %s", exc)
        report.summary_platform = portfolio_summary
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

    # Use the relevance score returned by Managed RAG (~0..1) as the starting
    # point and let the heuristic bonuses below stack on top.  Without this,
    # the reranker ignores a free signal from the server and relies entirely
    # on token overlap, which loses to noise on short or paraphrased queries.
    raw_server_score = source.get("score")
    try:
        server_score = float(raw_server_score)
    except (TypeError, ValueError):
        server_score = 0.0
    if server_score > 0:
        bonus = round(server_score * 3.0, 3)
        score += bonus
        reasons.append(f"server score: {server_score:.3f}")

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

    # Жёсткий штраф для «универсальных» страниц (AI-ассистент Cloudia,
    # GigaChat, гайды по поиску в документации). Эти страницы упоминают все
    # продукты сразу и часто всплывают в топе по широким запросам, после чего
    # LLM использует их как «доказательство» произвольных лимитов и SLA.
    if _is_low_relevance_source(title, url, content):
        score -= 5.0
        reasons.append("низкорелевантный источник (AI-ассистент / Cloudia)")

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


# v12.2: словарь синонимов для query expansion перед BM25-поиском.
# Русскоязычные страницы cloud.ru/docs часто описывают фичи не на «канцелярите
# ТЗ» (WORM / Write Once / S3 hosting), а через прикладные термины
# (защита от удаления, retention, политики). Без расширения BM25 не поднимает
# нужную страницу в топ.
_LOCAL_RAG_QUERY_SYNONYMS: tuple[tuple[tuple[str, ...], str], ...] = (
    # WORM / Object Lock семейство
    (("worm", "write once", "неизменност", "защита от удаления", "защита от перезапис"),
     "WORM защита удаления перезаписи retention compliance governance object lock"),
    (("object lock",),
     "object lock retention legal hold compliance governance mode worm"),
    (("retention",),
     "retention срок хранения compliance governance worm object lock"),
    (("legal hold",),
     "legal hold юридическое удержание retention compliance"),
    # Versioning
    (("versioning", "версионир", "версия объект"),
     "versioning версионирование версии объектов откат восстановление prevversion"),
    # PFS / SFTP / FUSE / s3fs
    (("sftp",),
     "sftp файловый протокол доступ передача"),
    (("pfs", "обектная файловая", "параллельная файловая"),
     "PFS параллельная файловая система object storage"),
    (("s3fs", "obsfs", "fuse"),
     "s3fs obsfs fuse mountpoint монтирование bucket файловая"),
    # Lifecycle
    (("lifecycle", "жизненн", "управлен сроком жизни"),
     "lifecycle жизненный цикл срок хранения автоматическое удаление политика"),
    # Шифрование
    (("sse-c", "byok"),
     "sse-c byok шифрование клиентский ключ encryption"),
    # Tiers и аттестации
    (("tier iii", "tier 3"),
     "Tier III TIER 3 Uptime Institute сертификация ЦОД"),
    (("уз-1", "уз1", "уровень защищенности"),
     "УЗ-1 УЗ-2 УЗ-3 ФСТЭК класс защищенности персональные данные"),
    # Биллинг
    (("поминутн", "посекундн"),
     "поминутная посекундная тарификация биллинг pay-as-you-go"),
    # OBS / S3
    (("obs", "object storage", "s3"),
     "Object Storage Service OBS S3 объектное хранилище bucket"),
)


def _expand_query_for_local_rag(query: str) -> str:
    """v12.2: query expansion для BM25 — добавляет русские/английские
    синонимы фич, если они упоминаются в запросе. Это повышает recall
    на специфических страницах cloud.ru/docs (guides__worm и т.п.),
    где локальная лексика отличается от формулировок ТЗ."""
    if not query:
        return query
    q_lower = query.lower().replace("ё", "е")
    extras: list[str] = []
    seen_keys: set[str] = set()
    for triggers, expansion in _LOCAL_RAG_QUERY_SYNONYMS:
        for trig in triggers:
            if trig in q_lower:
                key = trig
                if key in seen_keys:
                    break
                seen_keys.add(key)
                extras.append(expansion)
                break
    if not extras:
        return query
    return query + " " + " ".join(extras)


def _local_rag_supplement_for_single(
    req: Requirement,
    rag_result: ManagedRagResult | None,
) -> str:
    """v12.1: per-requirement версия local_rag-supplement (раньше была
    только batch-версия, которая в per_requirement-режиме не вызывалась
    вовсе — отсюда «local_rag молчит в production»).

    Подмешивает чанки из local_rag (BM25 по 225 страницам cloud.ru/docs)
    для одного требования. Включаем, если Managed-RAG слаб
    (top score < 0.65), пустой или требование содержит маркер
    специфической фичи. Снимок хитов кладётся в side-channel
    _LAST_LOCAL_RAG_HITS_BY_REQ для последующей записи в v.trace.
    """
    try:
        from src.local_rag import get_default_search
    except Exception as exc:
        logger.debug("local_rag не доступен: %s", exc)
        return ""

    is_weak = (rag_result is None) or (not rag_result.results)
    text_for_check = " ".join([req.section or "", req.text or "", req.tables or ""])
    has_specific = _has_specific_feature_marker(text_for_check)
    if not is_weak and rag_result and rag_result.results and not has_specific:
        top_score = 0.0
        try:
            for src in rag_result.results[:3]:
                rerank = src.get("_rerank", {}) if isinstance(src.get("_rerank"), dict) else {}
                score = float(rerank.get("score", 0) or 0)
                if score > top_score:
                    top_score = score
        except Exception:
            top_score = 0.0
        if top_score >= 0.65:
            return ""

    try:
        search = get_default_search()
    except Exception as exc:
        logger.debug("local_rag.get_default_search упал: %s", exc)
        return ""
    if not search.chunks:
        return ""

    query = " ".join([req.text or "", req.section or "", req.tables or ""])
    # v12.2: расширяем запрос синонимами фич перед BM25-поиском.
    expanded_query = _expand_query_for_local_rag(query)
    hits = search.search(expanded_query, k=5, min_score=1.0)
    if not hits:
        return ""

    seen: set[str] = set()
    unique_hits = []
    for h in hits:
        if h.url in seen:
            continue
        seen.add(h.url)
        unique_hits.append(h)
        if len(unique_hits) >= 3:
            break

    chunks_text = "\n\n".join(
        h.to_context_line(i + 1) for i, h in enumerate(unique_hits)
    )
    header = (
        "=========================================================\n"
        "АВТОРИТЕТНЫЕ ФРАГМЕНТЫ ИЗ cloud.ru/docs (BM25 по локальному индексу)\n"
        "=========================================================\n"
        "Это ПЕРВИЧНЫЙ источник — прямой текст ОФИЦИАЛЬНОЙ документации\n"
        "Cloud.ru. Если фрагмент дословно подтверждает фичу (имя фичи или\n"
        "её работа описана) — ставь **match** и цитируй URL в evidence/\n"
        "source_urls. Если фрагмент явно отрицает («не поддерживается») —\n"
        "ставь mismatch. Если по фиче во фрагментах ничего нет —\n"
        "needs_clarification."
    )
    body = f"--- ID требования {req.id} (пункт {req.section}) ---\n{chunks_text}"

    # Записываем хиты в side-channel для последующей трассировки в v.trace.
    _LAST_LOCAL_RAG_HITS_BY_REQ[str(req.id)] = [
        {"url": h.url, "score": float(getattr(h, "score", 0) or 0), "title": getattr(h, "title", "") or ""}
        for h in unique_hits
    ]
    logger.info(
        "local_rag (per_req): подмешано к req=%s; URL: %s",
        req.id,
        ", ".join(h.url for h in unique_hits),
    )

    return "\n\n".join([header, body])


def _format_rag_context(req: Requirement, rag_result: ManagedRagResult | None, max_chars_per_result: int = 2000) -> str:
    local_rag_block = _local_rag_supplement_for_single(req, rag_result)

    if not rag_result:
        parts = [
            f"ID требования: {req.id}",
            f"Пункт ТЗ: {req.section}",
            "Managed RAG не вернул контекст для этого требования.",
        ]
        if local_rag_block:
            parts.append(local_rag_block)
        return "\n".join(parts) if not local_rag_block else "\n\n".join(parts)

    parts = [
        f"ID требования: {req.id}",
        f"Пункт ТЗ: {req.section}",
    ]
    # v12.1: local_rag блок размещаем СРАЗУ ПОСЛЕ заголовка, ДО Managed RAG.
    # Это сигнализирует LLM, что авторитетные фрагменты идут первыми и
    # имеют приоритет над общими страницами Managed RAG.
    if local_rag_block:
        parts.append(local_rag_block)
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
    if rag_result.answer:
        parts.append(
            f"Управленческая интерпретация Managed RAG (опционально):\n{rag_result.answer}"
        )
    return "\n\n".join(parts)


# Маркеры специфических фич, для которых curated_facts подмешиваются
# ВСЕГДА (независимо от RAG-score). RAG может вернуть общий чанк про OBS
# со score 0.6, но без дословного упоминания WORM/Versioning/PFS — и
# модель ставит NC. Curated_fact с URL `cloud.ru/docs/obs/ug/topics/...`
# даёт явное подтверждение и URL для цитирования.
_SPECIFIC_FEATURE_MARKERS = (
    "worm", "object lock", "versioning", "версионир",
    "sftp", "файловый протокол", "файловый доступ",
    "s3fs", "obsfs", "pfs", "mountpoint", "fuse",
    "retention", "legal hold", "compliance mode", "governance mode",
    "lifecycle", "жизненн", "управлен сроком жизни",
    "поминутн", "посекундн",
    "cold", "холодное", "архивн", "долгосрочное хранение",
    "multi-az", "зон доступност", "репликация между зон",
    "tier iii", "tier 3",
    "cross-region", "межрегиональн", "репликация в dr",
    "sse-c", "byok", "клиентское шифрование",
    "presigned", "pre-signed",
    "event notification", "веб-хук", "вебхук",
    "object tagging", "bucket tagging", "теги объект",
    "ip allow-list", "определенной подсети", "white-list",
    "virtual-hosted", "путь-style", "кастомный домен",
    "obs", "object storage service",
)


def _has_specific_feature_marker(text: str) -> bool:
    """True, если требование содержит маркер специфической фичи —
    тогда curated_facts подмешиваются всегда, независимо от RAG score."""
    if not text:
        return False
    t = text.lower().replace("ё", "е")
    return any(m in t for m in _SPECIFIC_FEATURE_MARKERS)


def _local_rag_supplement_for_batch(
    requirements: list[Requirement],
    rag_result: ManagedRagResult | None,
) -> str:
    """Подмешивает чанки из local_rag (crawled cloud.ru/docs) когда основной
    RAG слаб ИЛИ требование специфическое.

    Local_rag — это BM25-поиск по 225 предварительно скачанным страницам
    cloud.ru/docs (см. local_rag/raw/). Это решает проблему «Managed RAG
    возвращает общую страницу OBS вместо страницы про WORM».

    v12.1: условие включения ослаблено — подмешиваем почти всегда, потому
    что Managed RAG для big-batch часто отдаёт overview/sla страницы с
    высоким score, но без конкретики (WORM, Object Lock, SFTP):
      • rag_result отсутствует / пуст → ВКЛЮЧАЕМ;
      • top Managed-RAG score < 0.65 (было 0.5) → ВКЛЮЧАЕМ;
      • есть marker специфической фичи в любом требовании → ВКЛЮЧАЕМ;
      • иначе пропускаем.

    Дополнительно: side-channel `_LAST_LOCAL_RAG_HITS_BY_REQ` пишет хиты
    в module-level dict, чтобы вызывающий код мог положить их в trace
    каждого verdict.
    """
    try:
        from src.local_rag import get_default_search
    except Exception as exc:
        logger.debug("local_rag не доступен: %s", exc)
        return ""

    is_weak = (rag_result is None) or (not rag_result.results)
    has_specific = any(
        _has_specific_feature_marker(" ".join([r.section or "", r.text or "", r.tables or ""]))
        for r in requirements
    )
    if not is_weak and rag_result and rag_result.results and not has_specific:
        top_score = 0.0
        try:
            for src in rag_result.results[:3]:
                rerank = src.get("_rerank", {}) if isinstance(src.get("_rerank"), dict) else {}
                score = float(rerank.get("score", 0) or 0)
                if score > top_score:
                    top_score = score
        except Exception:
            top_score = 0.0
        if top_score >= 0.65:
            return ""

    try:
        search = get_default_search()
    except Exception as exc:
        logger.debug("local_rag.get_default_search упал: %s", exc)
        return ""
    if not search.chunks:
        return ""

    blocks: list[str] = []
    # v12.1: пишем в module-level mapping, какие URL local_rag вернул для
    # каждого требования — будем класть в v.trace позже.
    hits_by_req: dict[str, list[dict]] = {}
    for req in requirements:
        query = " ".join([req.text or "", req.section or "", req.tables or ""])
        # v12.1: min_score снижен с 2.0 до 1.0 — BM25 может давать низкие
        # score'ы на коротких запросах, при этом релевантные страницы.
        # v12.2: query expansion синонимами фич — поднимает guides__worm
        # и т.п. в топ BM25 для русскоязычных страниц cloud.ru/docs.
        expanded_query = _expand_query_for_local_rag(query)
        hits = search.search(expanded_query, k=5, min_score=1.0)
        if not hits:
            continue
        # Уникальные URL'ы — топ-3 по убыванию score.
        seen: set[str] = set()
        unique_hits = []
        for h in hits:
            if h.url in seen:
                continue
            seen.add(h.url)
            unique_hits.append(h)
            if len(unique_hits) >= 3:
                break
        chunks_text = "\n\n".join(
            h.to_context_line(i + 1) for i, h in enumerate(unique_hits)
        )
        blocks.append(
            f"--- ID требования {req.id} (пункт {req.section}) ---\n{chunks_text}"
        )
        hits_by_req[str(req.id)] = [
            {"url": h.url, "score": float(getattr(h, "score", 0) or 0), "title": getattr(h, "title", "") or ""}
            for h in unique_hits
        ]
    if hits_by_req:
        _LAST_LOCAL_RAG_HITS_BY_REQ.clear()
        _LAST_LOCAL_RAG_HITS_BY_REQ.update(hits_by_req)
        logger.info(
            "local_rag активирован: подмешано блоков=%d, требований с хитами=%d (топ URL: %s)",
            len(blocks),
            len(hits_by_req),
            ", ".join(
                sorted({h["url"] for hs in hits_by_req.values() for h in hs})[:3]
            ),
        )

    if not blocks:
        return ""
    header = (
        "=========================================================\n"
        "АВТОРИТЕТНЫЕ ФРАГМЕНТЫ ИЗ cloud.ru/docs (прямой текст страниц документации)\n"
        "=========================================================\n"
        "Это ПЕРВИЧНЫЙ источник — фрагменты получены через BM25-поиск по 225\n"
        "проиндексированным страницам ОФИЦИАЛЬНОЙ документации Cloud.ru\n"
        "(cloud.ru/docs/...). Каждый фрагмент — реальный текст из этой\n"
        "документации, не пересказ и не общее описание.\n\n"
        "ПРАВИЛА ИСПОЛЬЗОВАНИЯ (ОБЯЗАТЕЛЬНЫЕ):\n"
        "1) Если фрагмент дословно подтверждает фичу (есть имя фичи или\n"
        "   её работа описана) → **match** с обязательным цитированием URL\n"
        "   из этого фрагмента в evidence/source_urls. НЕ пиши \"в документации\n"
        "   Cloud.ru нет упоминания\", если фрагмент здесь это упоминание содержит.\n"
        "2) Если фрагмент явно отрицает («не поддерживается», «недоступно»,\n"
        "   «не входит в портфель») → **mismatch**.\n"
        "3) Если по фиче во фрагментах ничего НЕТ (включая поиск синонимов и\n"
        "   близких терминов) — это сигнал, что фича не публично подтверждена\n"
        "   → **needs_clarification** с reasoning «уточнить у клиентского\n"
        "   менеджера Cloud.ru».\n\n"
        "ПРИОРИТЕТ ИСТОЧНИКОВ: фрагменты ниже имеют ПРИОРИТЕТ над Managed RAG\n"
        "(которые часто возвращают общие страницы). Если local-docs подтверждает,\n"
        "а Managed RAG не упоминает фичу — доверяй local-docs.\n"
        "================================================================="
    )
    return "\n\n".join([header, *blocks])


def _curated_supplement_for_batch(requirements: list[Requirement], rag_result: ManagedRagResult | None) -> str:
    """Подмешивает curated_facts в контекст.

    Условие срабатывания:
      • rag_result отсутствует или пуст;
      • верхний rerank-score < 0.4 (слабая релевантность);
      • ИЛИ хотя бы одно требование в batch'е содержит маркер специфической
        фичи (WORM, Object Lock, PFS, поминутный биллинг и т.п.) — для них
        даже сильный RAG-score может не покрыть конкретику.

    Для КАЖДОГО требования берём топ-3 релевантных factа и форматируем
    блоком «Дополнительный контекст». Это снижает риск, что LLM начнёт
    угадывать по тренировочным данным, и даёт цитируемый URL.
    """
    try:
        from src.knowledge import find_relevant_facts, format_facts_for_prompt
    except Exception:
        return ""
    is_weak = (rag_result is None) or (not rag_result.results)
    has_specific = any(
        _has_specific_feature_marker(" ".join([r.section or "", r.text or "", r.tables or ""]))
        for r in requirements
    )
    if not is_weak and rag_result.results and not has_specific:
        top_score = 0.0
        try:
            for src in rag_result.results[:3]:
                rerank = src.get("_rerank", {}) if isinstance(src.get("_rerank"), dict) else {}
                score = float(rerank.get("score", 0) or 0)
                if score > top_score:
                    top_score = score
        except Exception:
            top_score = 0.0
        if top_score >= 0.4:
            return ""
    chunks: list[str] = []
    for req in requirements:
        text_for_search = " ".join([req.section or "", req.text or "", req.tables or ""])
        facts = find_relevant_facts(text_for_search, platform=None, limit=3)
        if not facts:
            continue
        snippet = format_facts_for_prompt(facts)
        if snippet:
            chunks.append(f"--- ID требования {req.id} (пункт {req.section}) ---\n{snippet}")
    if not chunks:
        return ""
    # Усиленное напутствие модели: если в требовании есть специфическая
    # фича — curated_fact с URL имеет приоритет над «RAG не нашёл явного
    # упоминания», и модель должна ставить match вместо NC при наличии
    # curated_fact.
    if has_specific:
        header = (
            "===========================================================\n"
            "ПЕРВИЧНЫЙ ИСТОЧНИК — ПРОВЕРЕННЫЕ ФАКТЫ CLOUD.RU\n"
            "===========================================================\n"
            "Группа содержит требования по СПЕЦИФИЧЕСКИМ ФИЧАМ (WORM, "
            "Object Lock, Retention, Legal Hold, Versioning, PFS, "
            "Lifecycle, SFTP, поминутная тарификация, Multi-AZ и т.п.).\n\n"
            "ЭТИ ФАКТЫ ИМЕЮТ ПРИОРИТЕТ над RAG-чанками. Каждый факт ниже "
            "вручную сверен по документации Cloud.ru и имеет цитируемый URL.\n\n"
            "ПРАВИЛА ИСПОЛЬЗОВАНИЯ:\n"
            "• Если curated_fact ПОДТВЕРЖДАЕТ фичу — это **match** с обязательным "
            "цитированием URL в evidence. НЕ ставь needs_clarification только потому "
            "что RAG-чанк не нашёл дословного упоминания.\n"
            "• Если curated_fact явно говорит 'не подтверждено' / 'требует "
            "уточнения' / 'не публично' — следуй ему, ставь partial или "
            "needs_clarification.\n"
            "• Если curated_fact относится к подгруппе (например, Compliance Mode "
            "подтверждён, а Governance — нет) — отвечай ИМЕННО по подгруппе, "
            "не обобщай.\n"
            "• Платформа из поля 'platforms' факта = на этой платформе он "
            "действует. 'all' = на всех 4 платформах Cloud.ru.\n"
            "==========================================================="
        )
    else:
        header = (
            "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ — проверенные факты Cloud.ru с URL "
            "(curated knowledge base). Managed RAG вернул слабый контекст "
            "по этой группе. Используй curated_facts как первичный источник. "
            "Цитируй URL в evidence."
        )
    return "\n\n".join([header, *chunks])


def _format_batch_rag_context(requirements: list[Requirement], rag_result: ManagedRagResult | None) -> str:
    req_lines = []
    for req in requirements:
        req_lines.append(f"ID требования: {req.id}; пункт ТЗ: {req.section}; категория: {req.category}")
    local_rag_block = _local_rag_supplement_for_batch(requirements, rag_result)
    curated_supplement = _curated_supplement_for_batch(requirements, rag_result)
    if not rag_result:
        no_rag_block = "\n".join(
            [
                "Контекст Managed RAG для группы требований.",
                "\n".join(req_lines),
                "Managed RAG не вернул контекст для этой группы.",
            ]
        )
        extra = []
        if local_rag_block:
            extra.append(local_rag_block)
        if curated_supplement:
            extra.append(curated_supplement)
        if extra:
            return no_rag_block + "\n\n" + "\n\n".join(extra)
        return no_rag_block

    parts = [
        "Контекст Managed RAG для группы требований.",
        "\n".join(req_lines),
    ]
    # Local_rag (BM25 по crawled cloud.ru/docs) — ПЕРЕД RAG-чанками.
    # Это даёт LLM реальный текст из официальной документации, который
    # часто отсутствует в Managed RAG KB (например, страницы про WORM,
    # Object Lock, PFS не были загружены в KB).
    if local_rag_block:
        parts.append(local_rag_block)
    # Curated facts — резерв на случай если ни Managed RAG, ни local_rag
    # не покрыли требование (например, корпоративные атрибуты Cloud.ru,
    # которых нет на отдельной странице).
    if curated_supplement:
        parts.append(curated_supplement)
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
                    f"Фрагмент: {content[:2000] if content else 'нет текста'}",
                ]
            )
        )
    if rag_result.answer:
        parts.append(
            f"Управленческая интерпретация Managed RAG (опционально):\n{rag_result.answer}"
        )
    # curated_supplement уже добавлен в начало parts (перед RAG-чанками),
    # чтобы LLM видел его первым.
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
    reasoning_text = _value_to_text(item.get("reasoning", ""))

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
        # ВАЖНО: НЕ перезаписываем platform_name значением из RAG-чанка по
        # порядковому индексу. Раньше для batch'а из 4 platform_assessments
        # (по 4 платформам Cloud.ru) подсасывались первые 4 RAG-чанка, и если
        # они все были про «Облако VMware», то все 4 элемента получали имя
        # «Облако VMware» — даже когда reasoning говорил про Advanced /
        # Evolution / ГосОблако. Уважаем имя, которое вернул LLM.
        if not platform_name:
            platform_name = source_platform

    # Если LLM не указал platform_name, но в reasoning явно фигурирует одна
    # из канонических платформ — выводим её из reasoning. Это страхует от
    # ситуации, когда модель забыла заполнить поле или вернула пустую строку.
    if not platform_name and reasoning_text:
        platform_name = _known_platform_from_text(reasoning_text)

    canonical = _canonical_platform_name(platform_name)
    # Дополнительная страховка: если canonical формально валидный (например
    # «Облако VMware»), но в reasoning упоминается ДРУГАЯ платформа на первой
    # позиции — доверяем reasoning'у. Это лечит случаи, когда LLM проставил
    # одно имя во все элементы (mode collapse), а суть пишет в reasoning.
    if reasoning_text:
        from_reasoning = _known_platform_from_text(reasoning_text[:200])
        if from_reasoning and canonical and from_reasoning != canonical:
            # reasoning явно говорит про другую платформу — берём её
            canonical = from_reasoning

    # КОРЕНЬ БАГА: LLM иногда ставит source_type="external_service" для
    # канонической платформы Cloud.ru (ГосОблако/VMware/Advanced/Evolution).
    # Это путает «партнёрская услуга» и «внутренний сервис Cloud.ru».
    # Для канонических платформ source_type обязан быть "platform" —
    # это внутренняя инфраструктура Cloud.ru, а не партнёр.
    # Без этой нормализации:
    #  • assessment теряется в матрице (UI фильтрует external_service);
    #  • verdict ошибочно помечается requires_external_service=true и
    #    попадает в раздел «Внешние услуги / подрядчики».
    if canonical in CANONICAL_PLATFORMS_ORDER:
        if source_type != "platform":
            logger.debug(
                "Forcing source_type=platform for canonical platform %s "
                "(LLM сказал %r — игнорируем)",
                canonical,
                source_type,
            )
        source_type = "platform"

    normalized_source_type = (
        source_type if source_type in {"platform", "external_service", "unknown"} else "unknown"
    )

    return PlatformAssessment(
        platform_name=canonical or "Cloud.ru (платформа не определена)",
        verdict=_normalize_verdict(item.get("verdict", "needs_clarification")),
        confidence=_safe_float(item.get("confidence"), 0.5),
        reasoning=_value_to_text(item.get("reasoning", "")),
        evidence_refs=_dedupe_strings(_as_list(item.get("evidence_refs"))),
        source_urls=source_urls,
        source_titles=source_titles,
        source_type=normalized_source_type,
        recommendation=_value_to_text(item.get("recommendation", "")),
    )


CANONICAL_PLATFORMS_ORDER = ["ГосОблако", "Облако VMware", "Advanced", "Evolution"]

# Маркеры платформенно-специфичных тем — для таких требований НЕ пропагируем
# match с одной платформы на остальные. ГосОблако-специфика: К1/УЗ-1/ФСТЭК.
# Advanced-специфика: экстремальные ёмкости одной ВМ. VMware-специфика: SLA
# 99.982%, vSphere-функции.
_GOSCLOUD_SPECIFIC_RE = re.compile(
    r"\b(ФСТЭК|ФСБ|ГИС|УЗ[-\s]?1|УЗ[-\s]?2|К1|К2|"
    r"152[-\s]?ФЗ|187[-\s]?ФЗ|аттестат|модель угроз|"
    r"гос\s*информ|муниципальн|реестр\s+отечественн)",
    re.IGNORECASE,
)
_HIGH_CAPACITY_RE = re.compile(
    # Только ЯВНЫЕ требования сверх 1 ТБ на ВМ. Не цепляемся за упоминания
    # «до 1 ТБ» в capability-описании (это лимит, а не требование).
    # Покрываем формулировки в любом порядке слов: «более 5 ТБ», «не менее
    # 2 ТБ», «5 ТБ диск», «диск … 5 ТБ», «RAM … 2 ТБ».
    r"(\bбол(?:ьше|ее)\s+(?:[2-9]|1[0-9]+|1)\s*ТБ\b|"
    r"\bсвыше\s+(?:[2-9]|1[0-9]+|1)\s*ТБ\b|"
    r"\bне\s+мен(?:ее|ьше)\s+(?:[2-9]|1[0-9]+)\s*ТБ\b|"
    r"\bот\s+(?:[2-9]|1[0-9]+)\s*ТБ\b|"
    r"\b(?:[2-9]|1[0-9]+)\s*ТБ\s+(?:RAM|памяти|диск(?:ового)?)\b|"
    r"\b(?:RAM|памят[ьи]|диск(?:ового)?)\s.{0,40}?(?:[2-9]|1[0-9]+)\s*ТБ\b)",
    re.IGNORECASE | re.DOTALL,
)
_VMWARE_SPECIFIC_RE = re.compile(
    r"\b(vSphere|vCenter|VMware\s+Tools|99[,.]982\s*%|99[,.]999\s*%)",
    re.IGNORECASE,
)


def _is_platform_specific(req: Requirement | None, base_assessment: PlatformAssessment | None) -> str | None:
    """Возвращает имя платформы, к которой требование уникально привязано
    (или None, если требование общепортфельное).

    Используется при достройке недостающих platform_assessments — чтобы НЕ
    пропагировать match на платформы, которые объективно не закрывают такое
    требование (например, ФСТЭК/ГИС закрывает только ГосОблако).

    ВАЖНО: смотрим ТОЛЬКО на текст требования, НЕ на reasoning донора.
    Иначе capability-фразы вроде «Облако VMware поддерживает до 1 ТБ» в
    reasoning донора триггерят high-capacity маркер, и требование «RAM ≥
    512 ГБ» ошибочно помечается как Advanced-специфичное. Маркеры в req.text
    отражают реальные требования заказчика.
    """
    if req is None or not (req.text or "").strip():
        return None
    haystack = req.text or ""

    if _GOSCLOUD_SPECIFIC_RE.search(haystack):
        return "ГосОблако"
    if _VMWARE_SPECIFIC_RE.search(haystack):
        return "Облако VMware"
    if _HIGH_CAPACITY_RE.search(haystack):
        return "Advanced"
    return None


def _fill_missing_canonical_platforms(
    assessments: list[PlatformAssessment],
    req: Requirement | None,
) -> list[PlatformAssessment]:
    """Достраивает platform_assessments до 4 канонических платформ.

    Логика:
    1. Среди существующих оценок выбираем «донора» — наиболее представительный
       вердикт по приоритету: match > partial > mismatch > needs_clarification.
       При равенстве — донор с наибольшей confidence. Это отражает то, что
       пресейл уже знает про требование на других платформах.
    2. Платформенно-специфичные темы (ФСТЭК — ГосОблако; SLA 99.982 — VMware;
       экстремальные ёмкости — Advanced) — если донор=match, то на остальные
       платформы НЕ пропагируем match (там реально другая ситуация). На их
       место ставим NC.
    3. Если донор = partial / mismatch — пропагируем как есть (это корректно:
       если 3 платформы дали partial, то и 4-я скорее всего partial; если все
       3 mismatch, как с Colocation, то и 4-я mismatch с пометкой партнёра).
    4. Если донор = needs_clarification (LLM везде сказал «уточнить») — то и
       missing получают NC с тем же reasoning, что и донор.

    Это убирает UX-проблему: раньше для требований, которые модель оценила
    одинаково по всем известным ей платформам, отсутствующие 4-я платформа
    показывалась как `?` без сноски — пресейл думал, что мнения нет, хотя
    оно очевидно (то же, что и у других платформ Cloud.ru).
    """
    if not assessments:
        return assessments

    canonical_present = {a.platform_name for a in assessments if a.platform_name in CANONICAL_PLATFORMS_ORDER}
    missing = [p for p in CANONICAL_PLATFORMS_ORDER if p not in canonical_present]
    if not missing:
        return assessments

    # Приоритет вердиктов для выбора донора. Match — самый «информативный»,
    # NC — самый слабый. При равенстве priority берём с большей confidence.
    verdict_rank = {"match": 4, "partial": 3, "mismatch": 2, "needs_clarification": 1}

    canonical_assessments = [a for a in assessments if a.platform_name in CANONICAL_PLATFORMS_ORDER]
    if not canonical_assessments:
        return assessments  # нечего пропагировать

    donor: PlatformAssessment = max(
        canonical_assessments,
        key=lambda a: (verdict_rank.get(a.verdict, 0), a.confidence),
    )

    specific_to = _is_platform_specific(req, donor)

    extras: list[PlatformAssessment] = []
    for platform_name in missing:
        # Платформенно-специфичная тема + донор=match → на остальных платформах
        # match не пропагируется (там реально хуже / NC).
        if (
            donor.verdict == "match"
            and specific_to is not None
            and specific_to != platform_name
        ):
            extras.append(
                PlatformAssessment(
                    platform_name=platform_name,
                    verdict="needs_clarification",
                    confidence=0.4,
                    reasoning=(
                        f"Тема платформенно-специфична (закрывается на "
                        f"{specific_to}), для этой платформы требуется отдельная "
                        f"проверка у профильной команды Cloud.ru."
                    ),
                    evidence_refs=[],
                    source_urls=[],
                    source_titles=[],
                    source_type="platform",
                    recommendation="",
                )
            )
            continue

        # В остальных случаях — наследуем вердикт донора. Confidence снижаем
        # на 0.15, чтобы отметить, что это inference, а не прямая оценка.
        inherited_confidence = max(0.4, min(0.85, donor.confidence - 0.15))
        donor_label = donor.platform_name
        extras.append(
            PlatformAssessment(
                platform_name=platform_name,
                verdict=donor.verdict,
                confidence=inherited_confidence,
                reasoning=(
                    f"Унаследовано от {donor_label} (LLM не оценил эту "
                    f"платформу отдельно). "
                    f"{(donor.reasoning or '').strip()[:200]}"
                ).strip(),
                evidence_refs=list(donor.evidence_refs or []),
                source_urls=list(donor.source_urls or []),
                source_titles=list(donor.source_titles or []),
                source_type=donor.source_type or "platform",
                recommendation=donor.recommendation or "",
            )
        )

    # Сохраняем порядок: сначала исходные, потом достроенные. Дальше по
    # пайплайну порядок не важен — UI рендерит по PREFERRED_PLATFORM_ORDER.
    return assessments + extras


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
        # Дедупликация по platform_name. LLM иногда повторяет одно имя в
        # нескольких элементах (mode collapse). Если у двух записей совпадает
        # platform_name, но в reasoning одной из них упоминается ДРУГАЯ
        # каноническая платформа — переименуем в эту другую.
        canonical_priority = ["ГосОблако", "Облако VMware", "Advanced", "Evolution"]
        seen_names: set[str] = set()
        for a in assessments:
            if a.platform_name in seen_names and a.reasoning:
                # Ищем в reasoning имя платформы, которое ещё не использовалось.
                for candidate in canonical_priority:
                    if candidate in a.reasoning and candidate not in seen_names:
                        a.platform_name = candidate
                        break
            seen_names.add(a.platform_name)
        # Гарантия 4 канонических платформ. Если LLM вернул < 4 элементов
        # (типичный кейс: только «Облако VMware = match» по vCPU/RAM/диску
        # без оценки других платформ), достраиваем недостающие, унаследовав
        # вердикт от наиболее уверенного match'а — но только для тем, где
        # капабилити общие для портфеля Cloud.ru. Платформенно-специфичные
        # темы (ФСТЭК/ГИС/К1/УЗ-1, экстремальные ёмкости) НЕ пропагируются.
        assessments = _fill_missing_canonical_platforms(assessments, req)
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


# v15 (S1): Авторитетные домены Cloud.ru. Любой URL из этого списка в
# source_urls platform-assessment'а считается самостоятельным
# доказательством — даже если у LLM нет валидной [N]-сноски на selected_sources.
# Системная причина существования: curated_facts.url подмешивают цитаты
# из cloud.ru/docs (WORM, Object Lock, Versioning, Lifecycle), но они НЕ
# попадают в trace.selected_sources. LLM при этом отвечает refs вида
# `[Local-DOC 1]`, которые регекс `\[(\d+)\]` не ловит. Без этого
# исключения evidence-contract стабильно понижает match→partial на
# подтверждённых фичах OBS/S3 (см. Quality_baseline_v0.md, проблема S1).
_AUTHORITATIVE_CLOUD_DOMAINS = (
    "cloud.ru/docs",
    "cloud.ru/products",
    "cloud.ru/documents",
    "cloud.ru/about",
    "cloud.ru/services",
)


def _is_authoritative_cloud_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(d in u for d in _AUTHORITATIVE_CLOUD_DOMAINS)


def _assessment_has_authoritative_cloud_url(assessment: PlatformAssessment) -> bool:
    """True, если у assessment есть source_url из домена cloud.ru/docs (или
    смежных). Используется как самостоятельный сигнал доказательства —
    закрывает дыру между [Local-DOC N] цитатами curated_facts и
    селективным numeric ref-check."""
    for url in (assessment.source_urls or []):
        if _is_authoritative_cloud_url(str(url)):
            return True
    return False


def _refs_from_text(text: str) -> list[int]:
    refs = []
    for raw in re.findall(r"\[(\d+)\]", text or ""):
        try:
            refs.append(int(raw))
        except ValueError:
            continue
    return refs


def _has_named_doc_ref(text: str) -> bool:
    """True, если в evidence_refs/тексте есть «именные» сноски вида
    `[Local-DOC N]`, `[Curated N]`, `[Doc N]`, `[L-N]`. Это сноски на
    curated_supplement и local_rag-хиты — они валидны параллельно с
    numeric refs на selected_sources."""
    if not text:
        return False
    return bool(
        re.search(r"\[(?:local[\s\-]?doc|curated|cloud[\s\-]?doc|doc|l)\s*[-:]?\s*\d+\]",
                  text.lower())
    )


def _refs_match_selected_sources(refs: list[int], selected_source_count: int) -> bool:
    return bool(selected_source_count > 0 and any(1 <= ref <= selected_source_count for ref in refs))


def _assessment_refs_valid(
    assessment: PlatformAssessment, selected_source_count: int
) -> bool:
    """Расширённая проверка валидности сносок для одного assessment.

    Старое поведение: только `[1]..[N]` в `evidence_refs` и N≤selected_source_count.
    Новое поведение (v15, S1): дополнительно засчитываем «именные» сноски
    типа `[Local-DOC N]` ПРИ УСЛОВИИ, что у assessment есть авторитетный
    cloud.ru URL в source_urls. Это закрывает FALSE PARTIAL для фич,
    подтверждённых curated_facts (WORM, Object Lock, Versioning, …)."""
    refs_blob = " ".join(assessment.evidence_refs or [])
    if _refs_match_selected_sources(_refs_from_text(refs_blob), selected_source_count):
        return True
    # Cloud.ru URL + именная сноска — это полное доказательство.
    if _has_named_doc_ref(refs_blob) and _assessment_has_authoritative_cloud_url(assessment):
        return True
    return False


def _evidence_quote_text(evidence: str) -> str:
    return (evidence or "").split("\n\nВыбранные документы Managed RAG:", 1)[0].strip()


def _verdict_has_evidence_quote(verdict: RequirementVerdict, selected_source_count: int) -> bool:
    text = _evidence_quote_text(verdict.evidence)
    return bool(len(text) >= 12 and _refs_match_selected_sources(_refs_from_text(text), selected_source_count))


def _verdict_has_cited_evidence(verdict: RequirementVerdict) -> bool:
    has_source = bool(verdict.source_urls) or any(_assessment_has_source(item) for item in verdict.platform_assessments)
    selected_source_count = len((verdict.trace or {}).get("selected_sources") or [])
    return has_source and _verdict_has_evidence_quote(verdict, selected_source_count)


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def _normalize_for_match(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation, для подстрочного сравнения."""
    if not text:
        return ""
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(tokens)


def _extract_quote_candidates(quote: str) -> list[str]:
    """Берёт цитату из evidence и режет на куски, по которым можно проверить
    вхождение в источники. Использует 3-граммы по словам — достаточно
    специфично, чтобы поймать галлюцинации (3 слова подряд редко совпадают
    случайно), и достаточно толерантно к лёгкому парафразу LLM.
    """
    normalized = _normalize_for_match(quote)
    words = normalized.split()
    if len(words) < 3:
        return [normalized] if normalized else []
    grams: list[str] = []
    for i in range(len(words) - 2):
        grams.append(" ".join(words[i : i + 3]))
    return grams


_NUMBER_TOKEN_RE = re.compile(r"\d[\d.,]*")


def _extract_number_tokens(text: str) -> set[str]:
    """Числовые токены из текста — для проверки числовых галлюцинаций."""
    if not text:
        return set()
    tokens = set()
    for raw in _NUMBER_TOKEN_RE.findall(text):
        norm = raw.replace(",", ".").rstrip(".")
        if norm and len(norm) <= 12:
            tokens.add(norm)
    return tokens


def _retrieved_content_blob(verdict: RequirementVerdict) -> str:
    """Склеивает все excerpt'ы из retrieved chunks в одну нормализованную строку,
    по которой можно искать substring."""
    sources = (verdict.trace or {}).get("selected_sources") or []
    parts: list[str] = []
    for src in sources:
        if isinstance(src, dict):
            excerpt = src.get("excerpt") or ""
            if excerpt:
                parts.append(_normalize_for_match(excerpt))
    return " | ".join(parts)


# Стоп-слова для bag-of-words проверки grounded'ности. Эти слова почти
# всегда встречаются в любом тексте про Cloud.ru, поэтому их совпадение
# не даёт информации — игнорируем.
_GROUND_STOPWORDS = frozenset({
    "cloud", "ru", "облако", "облака", "облаке", "сервис", "сервиса", "сервисов",
    "услуга", "услуги", "услуг", "услугу", "должен", "должна", "должны", "должно",
    "обеспечить", "обеспечивает", "предоставляет", "предоставить", "предоставление",
    "это", "что", "как", "при", "для", "или", "над", "под", "без", "также",
    "включая", "согласно", "соответствии", "соответствует", "требования", "требований",
    "клиент", "клиенту", "заказчик", "заказчика", "пользоват",
    "платформа", "платформе", "платформы", "данных", "данные",
})


def _evidence_is_grounded_in_sources(verdict: RequirementVerdict) -> tuple[bool, float]:
    """Проверяет, что цитата в verdict.evidence действительно опирается на
    содержимое retrieved RAG-фрагментов.

    Возвращает (grounded, ratio):
    - grounded — True, если evidence закреплён в источниках.
    - ratio — доля совпавших значимых слов (для логирования).

    Стратегия (gentle, чтобы не давить реальные match'и LLM):
    1. Числовая проверка. Если в evidence есть значимые числа (>= 10),
       которых нет ни в requirement_text, ни в excerpt'ах источников —
       это галлюцинация цифр, понижаем независимо от ratio.
    2. Bag-of-words. Доля «значимых» слов цитаты (после удаления
       стоп-слов), которые встречаются в excerpt'ах источников. Должна
       быть ≥ 35%. Этот критерий толерантен к перестановкам слов и
       парафразу, при этом ловит цитаты, не имеющие отношения к
       источнику.

    Короткие evidence (≤6 значимых слов) пропускаем — нечего проверять.
    """
    quote = _evidence_quote_text(verdict.evidence or "")
    quote = re.sub(r"\[\d+\]", "", quote)
    quote_stripped = quote.strip()
    if len(quote_stripped) < 12:
        return True, 1.0

    blob = _retrieved_content_blob(verdict)
    if not blob:
        # Нет excerpt'ов — нечего сравнивать, не делаем выводов.
        return True, 1.0

    # Критерий 1: проверка чисел.
    quote_numbers = _extract_number_tokens(quote_stripped)
    # Игнорируем тривиальные числа (1, 2, 3 и т.п.).
    significant_numbers = {n for n in quote_numbers if not (n.isdigit() and int(n) < 10)}
    if significant_numbers:
        req_text = (verdict.requirement_text or "")
        req_numbers = _extract_number_tokens(req_text)
        numbers_to_verify = significant_numbers - req_numbers
        if numbers_to_verify:
            blob_numbers = _extract_number_tokens(blob)
            unverified = [n for n in numbers_to_verify if n not in blob_numbers]
            if unverified:
                return False, 0.0

    # Критерий 2: bag-of-words по значимым словам с префиксным matching'ом
    # (первые 5 символов), чтобы морфология русских слов не мешала:
    # «личный» / «личном» / «личного» считаем одним и тем же.
    quote_norm = _normalize_for_match(quote_stripped)
    blob_norm = blob  # уже нормализован в _retrieved_content_blob
    quote_words = [w for w in quote_norm.split() if len(w) >= 4 and w not in _GROUND_STOPWORDS]
    if len(quote_words) < 3:
        # Слишком мало значимых слов — нечего серьёзно проверять.
        return True, 1.0

    def _stem(w: str) -> str:
        return w[:5] if len(w) >= 5 else w

    blob_stems = {_stem(w) for w in blob_norm.split() if len(w) >= 4}
    matched = sum(1 for w in quote_words if _stem(w) in blob_stems)
    ratio = matched / len(quote_words)
    # Порог 15% — снижен с 25% для коротких/детальных формулировок ТЗ.
    # На больших ТЗ (НИИОЗМ: 873 пункта) гранулярные требования вида
    # «группировка правил по CSRF» имели overlap 10-20% с обзорным RAG-
    # чанком про WAF и массово ложно понижались в NC. 15% сохраняет защиту
    # от выдуманных цитат, но не режет реальные match'и на детальных
    # подпунктах подтверждённой капабилити.
    return ratio >= 0.15, ratio


_VERDICT_RANK = {"match": 3, "partial": 2, "needs_clarification": 1, "mismatch": 0}


# Маркеры colocation. ВАЖНО: должны срабатывать только на ЯВНЫЕ требования
# физического размещения оборудования заказчика. Раньше регексы были
# слишком широкими: «размещени» ловило «размещение ГИС» в Целях контракта,
# «физическ» ловило «физическая безопасность данных», «круглосуточн»
# ловило «круглосуточную техподдержку». В итоге 44/82 требований ТЗ
# Калугаинформтех помечались как партнёрские (по эталону должно быть 7).
#
# Сейчас два типа маркеров:
#  • _STRONG_COLOCATION_PHRASES — однозначные фразы, которые сами по себе
#    означают colocation (например «размещение оборудования заказчика»,
#    «2U», «акт сдачи-приёмки оборудования»).
#  • _SOFT_*_MARKERS — слабые маркеры, которые срабатывают ТОЛЬКО в паре
#    (например «оборудование» + «заказчик», «физический доступ» + «ЦОД»).
_STRONG_COLOCATION_PHRASES = (
    "colocation",
    "колокац",
    "размещение оборудовани",       # «размещение оборудования заказчика»
    "размещения оборудовани",       # «для размещения оборудования»
    "разместить оборудовани",
    "размещения сетевого оборудовани",
    "сетевого оборудования заказчик",
    "акт сдачи-приёмки оборудовани",
    "акт сдачи-приемки оборудовани",
    "акт сдачи оборудовани",
    "передача оборудования заказчик",
    "передачи оборудования заказчик",
    "сохранност.*оборудовани.*заказчик",  # regex
    " 2u ",
    "вт на каждый юнит",
    "на каждый юнит",
    "юнит",
    "стойк",
    "оборудование, размещённое в",
    "оборудование, размещенное в",
    "оборудование заказчика, размещ",         # «Оборудование Заказчика, размещённое в ЦОД, подключается…»
    "оборудования заказчика, размещ",
    "физический доступ",                       # 7.2.4 «физический доступ сотрудников Заказчика»
    "физического доступа",
    "физическому доступу",
    "администрирование оборудовани",          # 7.2.5 «администрирование оборудования силами Заказчика»
    "администрирования оборудовани",
)

# Слабые маркеры для requirements_text — должны встретиться ВДВОЁМ.
# Это страхует от срабатывания на общих фразах типа «Цель — размещение
# ГИС» (оборудования нет) или «Cloud.ru обеспечивает физическую
# безопасность данных» (физический доступ Заказчика не упомянут).
_SOFT_EQUIPMENT_MARKERS = ("оборудовани заказчик", "оборудования заказчик")
_SOFT_PHYSICAL_ACCESS_MARKERS = (
    "физический доступ",
    "физического доступа",
    "физическому доступу",
    "доступ сотрудников заказчик",
)

# Маркеры экзотической эксплуатации ЦОД, которые тоже считаются партнёрской
# услугой (24/7 нахождение технического персонала Заказчика в здании ЦОД и
# т.п.). СТРОГО узко — иначе ловит «круглосуточную техподдержку» из
# гарантийных пунктов 10.x, что по эталону НЕ партнёрская услуга
# (Cloud.ru сам предоставляет техподдержку 24/7).
_DC_PHYSICAL_ATTENDANCE_PHRASES = (
    "круглосуточное присутствие.*в здании цод",
    "присутстви.*персонал.*заказчик.*цод",
    "нахождени.*персонал.*заказчик.*цод",
    "сотрудники заказчика.*в здании цод",
)

# Требования, на которые НИ ПРИ КАКИХ обстоятельствах не ставится
# requires_external_service. Покрывает раздел 8 (площадки ЦОД, в т.ч. 8.5
# — круглосуточное присутствие СЕРВИСНОГО персонала, по эталону match
# без партнёра), раздел 9 (сертификаты — это документы Cloud.ru, не
# партнёр) и раздел 10 (гарантия и техподдержка — Cloud.ru сам).
_NEVER_EXTERNAL_PHRASES = (
    "круглосуточн.{0,30}техподдержк",
    "круглосуточн.{0,30}поддержк",
    "выделенный менеджер",
    "сертификат",
    "аттестат",
    "лицензи",
    "регистрац.*заявок",
    "поступающ.*запрос",
    "система мониторинга",
    # SLA-шкалы и штрафные клаузы — это договорное условие, решается
    # индивидуальным проектным договором с Cloud.ru, НЕ через colocation.
    # Без этого исключения шкалы доступности (Таблица 6 в типовом ТЗ)
    # попадают в Сбер colocation, что вводит tech-sales в заблуждение.
    "коэффициент доступности",
    "размер неустойки",
    "размер компенсации",
    "размер штрафа",
    "несоблюдение уровня",
    "несоблюдение заявленного",
    "уровн.{0,15}доступност",
    "доступности услуг",
    "kpi.{0,15}ниже",
    "уплачивает штраф",
    "штраф.{0,30}размере",
    "etap.{0,15}показателем",
    # Процедурные пункты карточки закупки — НЕ colocation:
    # • «Место поставки услуги/работы: <адрес заказчика>» — это юридический
    #   атрибут карточки 44-ФЗ, не запрос на физ. размещение в ЦОД.
    # • «Транспортные/инсталляционные расходы за счёт Поставщика» —
    #   это условие 44-ФЗ, не услуга доставки оборудования.
    # • «Адрес юридический/фактический/почтовый» — реквизиты заказчика.
    r"место поставки",
    r"место оказания услуг",
    r"место выполнения работ",
    r"транспортн.{0,15}расход",
    r"инсталляционн.{0,15}расход",
    r"расходы за счет",
    r"расходы за счёт",
    r"юридический адрес",
    r"фактический адрес",
    r"почтовый адрес",
    r"банковск.{0,15}реквизит",
    r"наименование заказчика",
    r"информация о заказчике",
)


def _looks_like_dc_operations(req_text: str) -> bool:
    """True, если требование явно про физическое присутствие персонала
    Заказчика в здании ЦОД (или аналогичную узкую семантику). Не должно
    срабатывать на общую фразу «круглосуточная техподдержка» или
    «обслуживающий персонал ЦОД» (это match для Cloud.ru, не партнёр).
    """
    if not req_text:
        return False
    lowered = " " + req_text.lower() + " "
    if any(re.search(p, lowered) for p in _NEVER_EXTERNAL_PHRASES):
        return False
    return any(re.search(p, lowered) for p in _DC_PHYSICAL_ATTENDANCE_PHRASES)


def _looks_like_external_colocation(req_text: str) -> bool:
    """True, если требование явно про физическое размещение оборудования
    Заказчика / физический доступ / акт сдачи-приёмки — то, что
    Cloud.ru закрывает только через партнёрскую услугу colocation в ЦОД.

    Условия (любое из):
      1. Сильная фраза из _STRONG_COLOCATION_PHRASES.
      2. Слабая пара: маркер «оборудование заказчика» + физический контекст
         (доступ / 2U / Вт / юнит / акт / стойка).

    Защита: если в тексте есть фраза из _NEVER_EXTERNAL_PHRASES — флаг не
    ставится, даже если совпал маркер colocation.
    """
    if not req_text:
        return False
    lowered = " " + req_text.lower() + " "

    # Чёрный список — если в тексте есть «техподдержка», «сертификат»,
    # «лицензия», «регистрация заявок» и т.п. — это НЕ colocation.
    if any(re.search(p, lowered) for p in _NEVER_EXTERNAL_PHRASES):
        return False

    # Сильные фразы — однозначно colocation.
    for marker in _STRONG_COLOCATION_PHRASES:
        if "*" in marker:  # regex
            if re.search(marker, lowered):
                return True
        else:
            if marker in lowered:
                return True

    # Слабые маркеры в паре: «оборудование заказчика» + физический контекст.
    has_equipment = any(m in lowered for m in _SOFT_EQUIPMENT_MARKERS)
    has_physical = any(m in lowered for m in _SOFT_PHYSICAL_ACCESS_MARKERS)
    if has_equipment and has_physical:
        return True

    return False


def _enforce_external_service_partial(verdict: RequirementVerdict) -> list[str]:
    """Для требований, закрываемых ТОЛЬКО партнёрской услугой Cloud.ru
    (colocation, физ. доступ в ЦОД, акт сдачи-приёмки, 24/7 присутствие в
    ЦОД) overall не может быть mismatch — это всегда partial с
    requires_external_service=true. Mismatch в этом случае означает, что
    LLM не вспомнил про партнёрскую услугу, хотя в промпте мы это требуем.

    Также проставляем requires_external_service / external_service_notes.
    """
    notes: list[str] = []
    text = verdict.requirement_text or ""
    is_colocation = _looks_like_external_colocation(text)
    is_dc_ops = _looks_like_dc_operations(text)
    if not (is_colocation or is_dc_ops):
        return notes
    if verdict.verdict == "mismatch":
        verdict.verdict = "partial"
        verdict.confidence = max(verdict.confidence, 0.55)
        notes.append(
            "Overall verdict синхронизирован: требование закрывается партнёрской услугой Cloud.ru / Сбер ЦОД"
        )
    if not verdict.requires_external_service:
        verdict.requires_external_service = True
    if not verdict.external_service_notes:
        if is_colocation:
            verdict.external_service_notes = "Cloud.ru / Сбер colocation в партнёрском ЦОД"
        else:
            verdict.external_service_notes = "Cloud.ru / партнёрский ЦОД ГК Сбер с 24/7 присутствием инженеров"
    return notes


_ENFORCE_RANK = {
    "match": 3,
    "partial": 2,
    "needs_clarification": 1,
    "mismatch": 0,
}


def _apply_curated_enforcement(verdict: RequirementVerdict) -> list[str]:
    """Programmatic enforcement: применяем curated_fact.enforce_verdict
    к platform_assessments и overall, если LLM не следовал curated.

    С v11 — local_rag (BM25 по cloud.ru/docs) даёт LLM реальный контекст,
    и enforcement в большинстве случаев излишен (LLM сама ставит match
    по конкретной странице). Чтобы не создавать конфликт между
    локальным RAG и enforcement-правилами, поведение управляется
    переменной окружения CURATED_ENFORCEMENT (по умолчанию off).

    Логика для каждой platform_assessment:
      • Находим curated_fact с enforce_verdict, применимый к (platform, text).
      • Сравниваем с текущим vердиктом по _ENFORCE_RANK.
      • Если enforce строже («match» vs LLM «NC», или «partial» vs LLM
        «match») — переписываем verdict + reasoning + source_urls.

    Логика для overall:
      • После платформ — пересчитываем overall: если хотя бы одна
        каноническая платформа имеет match, а overall = NC — поднимаем.
      • Если curated явно говорит partial (Governance Mode), а LLM
        ставил match — понижаем.

    Procedural verdicts пропускаем (они уже out_of_scope).
    """
    # С v11 enforcement выключен по умолчанию — local_rag даёт реальный
    # контекст и LLM ставит match сама. Включить можно через env.
    import os
    if os.getenv("CURATED_ENFORCEMENT", "off").lower() not in {"1", "on", "true", "yes"}:
        return []
    try:
        from src.knowledge import find_enforce_verdict
    except Exception:
        return []
    if (verdict.category or "").lower() == "procedural":
        return []
    if verdict.verdict == "out_of_scope":
        return []

    notes: list[str] = []
    req_text = verdict.requirement_text or ""

    for assessment in verdict.platform_assessments or []:
        platform = (assessment.platform_name or "").strip()
        if not platform:
            continue
        enforce_result = find_enforce_verdict(req_text, platform)
        if not enforce_result:
            continue
        enforce_v, fact = enforce_result
        cur_rank = _ENFORCE_RANK.get(assessment.verdict, -1)
        new_rank = _ENFORCE_RANK.get(enforce_v, -1)
        # Случай 1: enforce строже текущего (match при текущем NC/partial,
        # или partial при текущем NC) — поднимаем.
        # Случай 2: enforce явно partial, а LLM ставил match — понижаем
        # (только если фактически partial-факт, чтобы не сломать match-кейсы).
        should_apply = False
        if enforce_v == "match" and assessment.verdict in {"needs_clarification", "mismatch"}:
            should_apply = True
        elif enforce_v == "partial" and assessment.verdict in {"match"}:
            should_apply = True
        elif enforce_v == "needs_clarification" and assessment.verdict in {"match", "partial"}:
            # Curated прямо говорит «не подтверждено» — а LLM ставит match.
            should_apply = True
        if not should_apply:
            continue
        old_verdict = assessment.verdict
        assessment.verdict = enforce_v
        if enforce_v == "match":
            assessment.confidence = max(assessment.confidence, 0.85)
        elif enforce_v == "partial":
            assessment.confidence = min(max(assessment.confidence, 0.55), 0.7)
        else:
            assessment.confidence = min(assessment.confidence, 0.5)
        addition = (
            f" Curated knowledge base: {fact.title}. См. {fact.url}."
        )
        if addition.strip() not in (assessment.reasoning or ""):
            assessment.reasoning = (assessment.reasoning or "").rstrip() + addition
        # Подмешиваем URL в source_urls, если его там нет.
        if fact.url and fact.url not in (assessment.source_urls or []):
            assessment.source_urls = list(assessment.source_urls or []) + [fact.url]
        if fact.title and fact.title not in (assessment.source_titles or []):
            assessment.source_titles = list(assessment.source_titles or []) + [fact.title]
        notes.append(
            f"{platform}: {old_verdict} → {enforce_v} (curated_fact: {fact.title})"
        )

    # Overall: если есть match на канонической платформе после enforcement,
    # а overall был NC — поднимаем overall до match.
    canonical = {"ГосОблако", "Облако VMware", "Advanced", "Evolution"}
    best_canonical_verdict = None
    best_canonical_rank = -1
    for a in verdict.platform_assessments or []:
        if (a.platform_name or "").strip() not in canonical:
            continue
        r = _ENFORCE_RANK.get(a.verdict, -1)
        if r > best_canonical_rank:
            best_canonical_rank = r
            best_canonical_verdict = a.verdict
    if best_canonical_verdict and best_canonical_rank > _ENFORCE_RANK.get(verdict.verdict, -1):
        old_overall = verdict.verdict
        verdict.verdict = best_canonical_verdict
        verdict.confidence = max(verdict.confidence, 0.7)
        notes.append(
            f"Overall: {old_overall} → {best_canonical_verdict} (curated_enforcement по платформам)"
        )

    return notes


def _reconcile_overall_with_platforms(verdict: RequirementVerdict) -> list[str]:
    """Если overall_verdict жёстче, чем то, что говорят platform_assessments,
    выравниваем overall по агрегату по платформам.

    Это страхует от случаев, когда LLM возвращает overall=mismatch, но в
    platform_assessments все элементы — match/partial (классический логический
    сбой: модель в reasoning пишет «исправление: verdict=match», но JSON-полю
    overall_verdict уже выставила mismatch). Также покрывает обратную ошибку:
    overall=match при нескольких mismatch по платформам.

    Возвращает список заметок, которые попадут в evidence_contract_notes.
    """
    if not verdict.platform_assessments:
        return []
    platform_verdicts = [a.verdict for a in verdict.platform_assessments]
    # Ранг — лучший вердикт среди всех платформ.
    best = max(platform_verdicts, key=lambda v: _VERDICT_RANK.get(v, -1))
    overall_rank = _VERDICT_RANK.get(verdict.verdict, -1)
    best_rank = _VERDICT_RANK.get(best, -1)

    notes: list[str] = []
    # Если overall = mismatch, но хотя бы одна платформа даёт match/partial —
    # это противоречие. Поднимаем overall до уровня лучшей платформы.
    if verdict.verdict == "mismatch" and best in {"match", "partial"}:
        notes.append(
            f"Overall verdict синхронизирован: было mismatch, по платформам максимум — {best}"
        )
        verdict.verdict = best
        return notes

    # Если overall = match, но ни одна платформа не даёт match — понижаем до
    # лучшего по платформам.
    if verdict.verdict == "match" and best != "match":
        notes.append(
            f"Overall verdict синхронизирован: было match, по платформам максимум — {best}"
        )
        verdict.verdict = best
        return notes

    # Если overall = partial, а среди платформ есть match — поднимаем до match.
    # Это лечит случай, когда anti-hallucination понизил overall по тонкой
    # причине, а в platform_assessments несколько платформ дают match со
    # своими источниками. Доверяем платформам.
    #
    # ИСКЛЮЧЕНИЕ: если downgrade был сделан из-за overlap=0% (флаг
    # overlap_downgrade_locked в trace), НЕ поднимаем обратно — это
    # сигнал «RAG ничего не подтвердил, требуется уточнение». Без этого
    # исключения reconcile перетирает downgrade и снова даёт ложный match.
    locked_by_overlap = bool((verdict.trace or {}).get("overlap_downgrade_locked"))
    if verdict.verdict == "partial" and best == "match" and not locked_by_overlap:
        notes.append(
            "Overall verdict синхронизирован: было partial, но по платформам — match"
        )
        verdict.verdict = "match"
        return notes

    # Если overall = needs_clarification, а есть match/partial по платформам —
    # тоже поднимаем (часто LLM пишет nc в overall, а в platforms — match).
    if (
        verdict.verdict == "needs_clarification"
        and best in {"match", "partial"}
    ):
        notes.append(
            f"Overall verdict синхронизирован: было needs_clarification, по платформам — {best}"
        )
        verdict.verdict = best
        return notes

    return notes


# v16 (S2/S3): паттерны конкретных численных характеристик в требованиях.
# Используется для grounding-check: если LLM выдал match, но число в источниках
# не встречается — понижаем до partial.
#
# Захватываем число в обоих порядках:
#   - «<число> <единица>»: «5000 IOPS», «99,982 %», «10 000 Мбит/с»
#   - «<единица> ... <число>» — для конструкций «ядер не менее 456»,
#     «частота не менее 2.6 ГГц», поскольку в нормативных требованиях
#     число часто следует за единицей.
# Также поддерживаем space-разделитель разрядов (10 000 → одно число).
_NUMERIC_UNITS = (
    r"ггц|мгц|ghz|mhz|tflops|tflop|tops|iops|"
    r"мбит/?с|гбит/?с|mbit/?s|gbit/?s|"
    r"мс\b|ms\b|кбит/?с|кбит|"
    r"тензорн\w*\s+ядер|cuda\W+ядер|ядер\b|потоков\b|"
    r"вольт\b|hz|"
    r"тб\b|гб\b|tb\b|gb\b|"
    r"%"
)
_NUMBER = r"\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_NUMERIC_UNIT_RE = re.compile(
    rf"(?<![\w.,])({_NUMBER})\s*({_NUMERIC_UNITS})",
    re.IGNORECASE,
)
# Обратный порядок: «<единица> ... <число>». Допускаем небольшое окно слов
# между единицей и числом («ядер не менее 456»).
_UNIT_THEN_NUMBER_RE = re.compile(
    rf"({_NUMERIC_UNITS})[^\d]{{1,60}}?(?<![\w.,])({_NUMBER})",
    re.IGNORECASE,
)
# Маркеры «договорного обещания» — фразы, которыми LLM пытается обосновать
# match без реального источника из cloud.ru/docs. Если verdict=match И нет
# конкретного URL cloud.ru/docs И reasoning содержит эти маркеры — понижаем.
_CONTRACTUAL_PROMISE_PHRASES = (
    "готов(?:а)? зафиксировать",
    "готов(?:а)? оформлять",
    "готов(?:а)? согласовать",
    "согласовать индивидуально",
    "в рамках проектного (?:договор|sla)",
    "в индивидуальном (?:договор|sla)",
    "в рамках индивидуального",
    "договорн[ая]\\w* условие",
    "в рамках договора",
    "оформляться в договоре",
    "фиксируется в договоре",
)
_CONTRACTUAL_PROMISE_RE = re.compile("|".join(_CONTRACTUAL_PROMISE_PHRASES), re.IGNORECASE)


def _norm_number(raw: str) -> str:
    """5 000 → 5000; 99,982 → 99.982; 67.5 → 67.5."""
    if not raw:
        return raw
    return raw.replace(" ", "").replace(" ", "").replace(",", ".")


def _extract_numeric_claims(text: str) -> list[tuple[str, str]]:
    """Возвращает список пар (число, единица) из текста. Подхватывает оба
    порядка: «число + единица» и «единица … число»."""
    if not text:
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _NUMERIC_UNIT_RE.finditer(text):
        num = _norm_number(m.group(1))
        unit = m.group(2).lower().strip()
        key = (num, unit)
        if key not in seen:
            seen.add(key)
            out.append(key)
    for m in _UNIT_THEN_NUMBER_RE.finditer(text):
        unit = m.group(1).lower().strip()
        num = _norm_number(m.group(2))
        key = (num, unit)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _numeric_value_in_text(num: str, text: str) -> bool:
    """True, если число `num` (в формате '5000' или '99.95') встречается в text.
    Допускаем варианты с запятой/точкой и пробелом-разделителем разрядов:
    5000 ≡ 5 000 ≡ 5,000."""
    if not text or not num:
        return False
    text_l = text.lower()
    candidates = [num]
    if "." in num:
        candidates.append(num.replace(".", ","))
    if num.isdigit() and len(num) >= 4:
        # 5000 → 5 000 / 5,000
        head, tail = num[:-3], num[-3:]
        if head:
            candidates.append(f"{head} {tail}")
            candidates.append(f"{head},{tail}")
            candidates.append(f"{head}.{tail}")
    for c in candidates:
        if c.lower() in text_l:
            return True
    return False


def _verdict_evidence_blob(verdict: RequirementVerdict) -> str:
    """Собирает текст всех источников verdict'а — evidence, source_urls/titles,
    platform_assessments.* + trace.selected_sources contents + curated supplement
    + local_rag content. Используется для numeric grounding."""
    parts: list[str] = []
    parts.append(verdict.evidence or "")
    parts.extend(verdict.source_urls or [])
    for a in verdict.platform_assessments or []:
        parts.append(a.reasoning or "")
        parts.extend(a.source_urls or [])
        parts.extend(a.source_titles or [])
        parts.extend(a.evidence_refs or [])
    trace = verdict.trace or {}
    for src in trace.get("selected_sources") or []:
        if isinstance(src, dict):
            parts.append(str(src.get("content", "")))
            parts.append(str(src.get("title", "")))
            parts.append(str(src.get("url", "")))
        else:
            parts.append(str(src))
    for hit in trace.get("local_rag") or []:
        if isinstance(hit, dict):
            parts.append(str(hit.get("content", "")))
            parts.append(str(hit.get("title", "")))
            parts.append(str(hit.get("url", "")))
    return " ".join(p for p in parts if p)


def _apply_numeric_grounding_check(verdict: RequirementVerdict) -> list[str]:
    """v16 (S2/S3): если в `requirement_text` есть конкретные числа с единицами
    (ГГц, ТБ, IOPS, мс, %, Мбит/с, тензорные ядра и т.п.), а в источниках
    verdict'а (evidence + source_urls + platform_assessments + selected_sources
    + local_rag chunks) ни одного из этих чисел не встречается — это сигнал,
    что LLM мог дать `match` по числу, которое сам придумал.

    Действие:
    • verdict=match → понижаем до `partial` с конкретной формулировкой.
    • verdict=partial → оставляем как есть, но добавляем notes.
    • verdict=needs_clarification/mismatch/out_of_scope — не трогаем.

    Исключение: процедурные/SLA-классификационные verdict'ы. У них требования
    типа «KPI ≥ 0.65» — это часто числа из ТЗ, не привязанные к Cloud.ru
    конкретному источнику; они уже обработаны как out_of_scope/процедурный.

    Также игнорируем число «0», «1», «2», «3» — это слишком короткие
    значения, чтобы дать осмысленный сигнал.
    """
    if verdict.verdict not in {"match", "partial"}:
        return []
    if (verdict.category or "").lower() == "procedural":
        return []
    req_text = (verdict.requirement_text or "")
    claims = _extract_numeric_claims(req_text)
    # Отфильтровываем короткие/тривиальные числа.
    meaningful = [
        (num, unit) for num, unit in claims
        if not (num in {"0", "1", "2", "3"} and unit in {"в", "ггц"})
        and float(num.replace(",", ".") or 0) >= 4
    ]
    if not meaningful:
        return []
    evidence_blob = _verdict_evidence_blob(verdict)
    grounded_count = 0
    ungrounded_pairs: list[tuple[str, str]] = []
    for num, unit in meaningful:
        if _numeric_value_in_text(num, evidence_blob):
            grounded_count += 1
        else:
            ungrounded_pairs.append((num, unit))
    notes: list[str] = []
    # Если ВСЕ числа не подтверждены — однозначный downgrade.
    # Если есть хотя бы одно подтверждённое — оставляем (это означает, что
    # ключевая цифра прошла grounding).
    if grounded_count > 0:
        return notes
    if not ungrounded_pairs:
        return notes
    sample = ", ".join(f"{n}{u}" for n, u in ungrounded_pairs[:3])
    if verdict.verdict == "match":
        verdict.verdict = "partial"
        verdict.confidence = min(verdict.confidence, 0.55)
        if verdict.trace is None:
            verdict.trace = {}
        verdict.trace["numeric_grounding_downgrade"] = True
        notes.append(
            f"Numeric grounding: match понижен до partial — конкретные числа "
            f"требования ({sample}) не подтверждены в источниках Cloud.ru/docs"
        )
    else:  # partial — оставляем, но trace и note
        if verdict.trace is None:
            verdict.trace = {}
        verdict.trace["numeric_grounding_warning"] = sample
        notes.append(
            f"Numeric grounding warning: числа ({sample}) не найдены в источниках Cloud.ru/docs"
        )
    return notes


def _apply_contractual_promise_check(verdict: RequirementVerdict) -> list[str]:
    """v16: если verdict=match построен на «договорном обещании»
    («Cloud.ru готов зафиксировать», «согласовать индивидуально» и т.п.)
    без авторитетного cloud.ru/docs URL — это не доказательство, а
    готовность к индивидуальному обсуждению. Понижаем до partial с пометкой.

    Не трогаем, если:
    • в source_urls verdict'а есть cloud.ru/docs URL,
    • или ХОТЯ БЫ ОДНА platform_assessment.source_urls содержит cloud.ru/docs.
    """
    if verdict.verdict != "match":
        return []
    if (verdict.category or "").lower() == "procedural":
        return []
    blob = " ".join([
        verdict.reasoning or "",
        verdict.evidence or "",
    ])
    if not _CONTRACTUAL_PROMISE_RE.search(blob):
        return []
    # Проверка наличия авторитетного источника.
    def _has_doc_url(urls):
        return any(_is_authoritative_cloud_url(str(u)) for u in (urls or []))
    if _has_doc_url(verdict.source_urls):
        return []
    if any(_has_doc_url(a.source_urls) for a in (verdict.platform_assessments or [])):
        return []
    # Нет источника, при этом reasoning — «договорное обещание». Понижаем.
    verdict.verdict = "partial"
    verdict.confidence = min(verdict.confidence, 0.5)
    if verdict.trace is None:
        verdict.trace = {}
    verdict.trace["contractual_promise_downgrade"] = True
    return [
        "Contractual promise: match понижен до partial — обоснование сводится к "
        "«готовы согласовать в договоре», без подтверждающего cloud.ru/docs URL"
    ]


def _apply_evidence_contract(verdict: RequirementVerdict) -> RequirementVerdict:
    notes = list(verdict.evidence_contract_notes or [])
    # Защита от антипаттерна «исключить из ТЗ» для colocation-требований:
    # если требование явно про физическое размещение / акт сдачи-приёмки —
    # это partial + Сбер colocation, не mismatch.
    notes.extend(_enforce_external_service_partial(verdict))
    # Сверка overall с platform_assessments. Если LLM «передумал» в
    # процессе генерации, мы это здесь подхватим до того, как сработает
    # evidence-контракт.
    notes.extend(_reconcile_overall_with_platforms(verdict))
    original_verdict = verdict.verdict
    selected_source_count = len((verdict.trace or {}).get("selected_sources") or [])

    for assessment in verdict.platform_assessments:
        if assessment.verdict not in {"match", "partial"}:
            continue
        # v15 (S1): расширенная проверка — numeric [N] на selected_sources
        # ИЛИ именная сноска [Local-DOC N] на фоне cloud.ru/docs URL.
        assessment_refs_valid = _assessment_refs_valid(assessment, selected_source_count)
        # Если есть и источник, и валидная сноска — оставляем как есть.
        if _assessment_has_source(assessment) and _assessment_has_ref(assessment) and assessment_refs_valid:
            continue
        # Дополнительно: даже если refs пустые, наличие cloud.ru/docs URL
        # самостоятельно даёт право не понижать match — это прямая ссылка
        # на авторитетную страницу. Иначе мы стабильно теряем подтверждения
        # из curated_facts (см. S1 в Quality_baseline_v0.md).
        if _assessment_has_authoritative_cloud_url(assessment):
            continue
        # Нет источника совсем — это самое серьёзное, понижаем в nc.
        if not _assessment_has_source(assessment):
            assessment.verdict = "needs_clarification"
            assessment.reasoning = (
                assessment.reasoning.rstrip()
                + " Evidence contract: нет подтверждающего источника из RAG."
            ).strip()
            notes.append(f"{assessment.platform_name}: нет подтверждающего источника")
            continue
        # Есть источник (но не cloud.ru/docs), и сноска неполная или невалидная.
        # Для match — понижаем до partial. Для partial — оставляем, помечаем.
        if assessment.verdict == "match":
            assessment.verdict = "partial"
            assessment.confidence = min(assessment.confidence, 0.6)
            assessment.reasoning = (
                assessment.reasoning.rstrip()
                + " Evidence contract: match понижен до partial — источник найден, но связка со сноской не подтверждена."
            ).strip()
            notes.append(f"{assessment.platform_name}: match → partial (нет валидной сноски)")
        # partial с источником — оставляем как есть, без понижения в nc.

    has_source = bool(verdict.source_urls) or any(
        _assessment_has_source(item) for item in verdict.platform_assessments
    )

    # Anti-hallucination evidence-контракт. Это ОСНОВНАЯ проверка: цитата в
    # evidence должна реально встречаться в excerpt'ах retrieved chunks.
    # Запускаем ДО старого жёсткого `[n]`-контракта, потому что её результат
    # используется как разрешение «оставить match без идеальной [n]-сноски».
    grounded, ratio = (True, 1.0)
    if verdict.verdict in {"match", "partial"}:
        grounded, ratio = _evidence_is_grounded_in_sources(verdict)
        if not grounded:
            # Если все platform_assessments дают match — у нас коллективное
            # доказательство (каждая платформа с собственными источниками),
            # и понижать overall из-за overlap одной цитаты не нужно. Это
            # исключает кейсы, когда LLM написал в overall.evidence короткий
            # парафраз, но в platform_assessments — детальные ссылки.
            platform_verdicts = [
                a.verdict for a in (verdict.platform_assessments or [])
            ]
            all_platforms_match = (
                bool(platform_verdicts)
                and all(v == "match" for v in platform_verdicts)
            )
            # ЗАЩИТА: если overlap почти нулевой (< 5%), это означает, что
            # ни overall, ни platform_assessments не подтверждены RAG-данными.
            # «Все platforms match» в этом случае — это не доказательство,
            # а коллективное угадывание модели. Принудительно понижаем
            # overall до partial с пометкой «требуется подтверждение».
            # Исключения:
            #   • процедурные пункты — у них RAG legitimно пуст;
            #   • verdict.evidence содержит конкретный URL cloud.ru/docs —
            #     это «цитата по ссылке», не текстовый overlap. Низкий
            #     overlap в этом случае не означает галлюцинацию.
            is_procedural = (verdict.category or "").lower() == "procedural"
            virtually_no_overlap = ratio < 0.05
            evidence_blob = " ".join([
                verdict.evidence or "",
                " ".join(verdict.source_urls or []),
            ]).lower()
            has_hard_url = (
                "cloud.ru/docs" in evidence_blob
                or "cloud.ru/products" in evidence_blob
                or "cloud.ru/about" in evidence_blob
                or "cloud.ru/documents" in evidence_blob
            )
            if is_procedural:
                pass  # procedural verdicts уже обработаны отдельно
            elif has_hard_url:
                # Есть URL из cloud.ru — это уже доказательство, низкий
                # текстовый overlap не означает галлюцинацию. Не понижаем.
                notes.append(
                    f"Overlap низкий ({ratio:.0%}), но evidence ссылается на "
                    f"cloud.ru/docs или cloud.ru/products — verdict оставлен match"
                )
            elif virtually_no_overlap and verdict.verdict == "match":
                # Не доверяем «все matches», если RAG не вернул ничего.
                verdict.verdict = "partial"
                verdict.confidence = min(verdict.confidence, 0.5)
                # Помечаем флагом, чтобы финальный reconcile НЕ поднимал
                # обратно до match. Без этого флага reconcile видит
                # «все platforms — match» и переписывает наш downgrade.
                if verdict.trace is None:
                    verdict.trace = {}
                verdict.trace["overlap_downgrade_locked"] = True
                notes.append(
                    f"Цитата не подтверждена ни в overall, ни в RAG (overlap={ratio:.0%}); "
                    f"verdict понижен до partial — требуется подтверждение у клиентского "
                    f"менеджера Cloud.ru"
                )
            elif all_platforms_match:
                # «Все платформы match» — это коллективное доказательство ТОЛЬКО
                # если у платформ есть конкретные источники (source_urls /
                # source_titles). Иначе это коллективное угадывание модели,
                # которое не должно перебивать слабый overlap. Требуем хотя бы
                # одну платформу с непустым source_urls / evidence_refs —
                # иначе downgrade'им до partial + lock.
                any_platform_with_source = any(
                    _assessment_has_source(a) or _assessment_has_ref(a)
                    for a in (verdict.platform_assessments or [])
                )
                if any_platform_with_source:
                    notes.append(
                        f"Цитата overall слабо подтверждена (overlap={ratio:.0%}), но все "
                        f"platform_assessments — match. Overall не понижается."
                    )
                else:
                    # Нет ни одного подтверждающего источника — это галлюцинация.
                    verdict.verdict = "partial"
                    verdict.confidence = min(verdict.confidence, 0.5)
                    if verdict.trace is None:
                        verdict.trace = {}
                    verdict.trace["overlap_downgrade_locked"] = True
                    notes.append(
                        f"Цитата слабо подтверждена (overlap={ratio:.0%}), при этом ни одна "
                        f"платформа не приводит конкретный источник — verdict понижен до "
                        f"partial: требуется подтверждение у клиентского менеджера Cloud.ru"
                    )
            elif verdict.verdict == "match":
                verdict.verdict = "partial"
                verdict.confidence = min(verdict.confidence, 0.55)
                # Любой downgrade по overlap должен блокировать обратный
                # подъём в reconcile_overall_with_platforms — иначе reconcile
                # видит «все platforms match» и снова даёт ложный match.
                if verdict.trace is None:
                    verdict.trace = {}
                verdict.trace["overlap_downgrade_locked"] = True
                notes.append(
                    f"Цитата не подтверждена RAG-источниками (overlap={ratio:.0%}); вердикт match понижен до partial"
                )
            else:
                verdict.confidence = min(verdict.confidence, 0.45)
                notes.append(
                    f"Цитата слабо подтверждена RAG-источниками (overlap={ratio:.0%})"
                )

    # Старая «строгая» проверка на наличие [n]-сноски. Раньше она понижала
    # ЛЮБОЙ match/partial без идеальной связки [n]→retrieved_chunk в
    # needs_clarification, что после ужесточения промпта (LLM перестал
    # пересказывать требование в evidence) валило 60%+ требований в nc.
    #
    # Теперь мы ослабляем её следующим образом:
    # 1. Если есть source (URL или title) И цитата подтверждена RAG (grounded
    #    по anti-galloo) — оставляем вердикт как есть.
    # 2. Если есть source, но цитаты нет в evidence — для partial оставляем
    #    (источник имеется, доверяем).
    # 3. Только match без grounded цитаты И без source понижается в nc.
    if verdict.verdict == "match" and not has_source:
        verdict.verdict = "needs_clarification"
        verdict.confidence = min(verdict.confidence, 0.4)
        notes.append(
            f"Вердикт {original_verdict} понижен: match без подтверждающего источника"
        )
    elif verdict.verdict == "match" and has_source and not grounded:
        # Уже понизили в partial выше anti-galloo проверкой — нечего больше.
        pass

    # PROGRAMMATIC ENFORCEMENT через curated_facts.
    # Если для требования есть curated_fact с enforce_verdict (например,
    # WORM/Versioning/PFS = match, Governance Mode = partial), а LLM
    # поставил «слабее» (NC при ожидаемом match, или match при ожидаемом
    # partial для Governance) — программно поправляем. Это страхует от
    # того, что LLM проигнорировал curated_supplement в контексте.
    notes.extend(_apply_curated_enforcement(verdict))

    # Финальная сверка overall с platform_assessments — после всех понижений
    # выше. Если overall просел ниже, чем максимум по платформам (например,
    # match понижен до partial anti-galloo, а в platforms все match), то
    # доверяем платформам и поднимаем overall обратно.
    notes.extend(_reconcile_overall_with_platforms(verdict))

    # Патч 9 (ZK10). requires_external_service=True несовместимо с overall
    # match: если требование закрывается ТОЛЬКО внешней услугой/подрядчиком,
    # сам Cloud.ru напрямую его не выполняет — это partial с пометкой
    # «via external». В ZK10-прогоне найдено 7 verdict'ов, где LLM поставил
    # match И одновременно requires_external_service=True — для пресейла это
    # вводит в заблуждение.
    if verdict.requires_external_service and verdict.verdict == "match":
        old = verdict.verdict
        verdict.verdict = "partial"
        verdict.confidence = min(verdict.confidence, 0.7)
        if not verdict.recommendation:
            verdict.recommendation = (
                "Покрытие через внешнюю услугу / партнёра Cloud.ru — "
                "включить условие в КП и согласовать с клиентским менеджером."
            )
        notes.append(
            f"Overall: {old} → partial (requires_external_service=True; "
            "требуется партнёрская услуга или внешний подрядчик)"
        )
        if verdict.trace is None:
            verdict.trace = {}
        verdict.trace["external_service_demoted"] = True

    # v16 (S2/S3): numeric grounding + договорный антипаттерн.
    # Это отдельный последний шаг — после curated enforcement и финального reconcile.
    # Защищает от false match'ей класса «LLM назвал конкретное число (ГГц/ТБ/TFLOPS/
    # IOPS/мс/%/Мбит/Гбит) но в источниках Cloud.ru его нет» и «Cloud.ru готов
    # зафиксировать в договоре — поэтому match».
    notes.extend(_apply_numeric_grounding_check(verdict))
    notes.extend(_apply_contractual_promise_check(verdict))

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


# ---------------------------------------------------------------------------
# Cross-verdict post-process (patches 7, 10).
# Запускается ОДИН РАЗ после того, как все батчи завершили работу — внутри
# `analyze_requirements`, после сортировки verdict'ов и до генерации summary.
# Делает две вещи: дедуп reasoning (patch 7) и дискретизация confidence
# (patch 10). Порядок важен: дедуп опирается на сырые confidence; дискретизация
# затем сглаживает значения после понижения.
# ---------------------------------------------------------------------------

# Buckets для патча 10. Тройки (нижний порог, дискретное значение, label).
_CONFIDENCE_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.85, 0.95),
    (0.60, 0.75),
    (0.45, 0.55),
    (0.0,  0.25),
)


def _snap_confidence(value: float | None, has_reasoning: bool) -> float:
    """Привести confidence в дискретное значение.

    Если value укладывается в бакет 0.45–0.6 (низкая уверенность) и при этом
    у вердикта НЕТ полноценного reasoning'а (≤ 40 симв.) — снапаем в 0.4
    (auto-downgrade-в-NC, см. `_discretize_confidence_inplace`). Это патч 10
    (ZK10): жёстко избавляемся от бимодального распределения 0.5 «по умолчанию».
    """
    if value is None:
        return 0.4
    for threshold, bucket_value in _CONFIDENCE_BUCKETS:
        if value >= threshold:
            if bucket_value == 0.55 and not has_reasoning:
                return 0.4
            return bucket_value
    return 0.25


def _detect_and_penalize_duplicate_reasoning(
    verdicts: list[RequirementVerdict],
    threshold: int = 3,
) -> int:
    """Patch 7 (ZK10). Найти verdict'ы с идентичным reasoning между разными
    requirement_id и понизить им confidence + evidence_status.

    Алгоритм:
      • группируем по нормализованному префиксу reasoning'а (300 симв.).
      • группы с size ≥ threshold помечаем как «template hallucination».
      • выбираем «канонический» по (len(source_urls), confidence) — у него
        наибольшее подтверждение, ему доверяем.
      • остальным понижаем confidence на 0.2 и переводим evidence_status
        в weak; если confidence упала < 0.5 и verdict=match → partial.

    Возвращает число изменённых verdict'ов.
    """
    from collections import defaultdict

    if threshold < 2:
        threshold = 2
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, v in enumerate(verdicts):
        if v.verdict not in {"match", "partial"}:
            continue
        if (v.category or "").lower() == "procedural":
            continue
        if not v.reasoning:
            continue
        key = _normalize_for_match(v.reasoning[:300])
        # Слишком короткие префиксы дают ложные совпадения общих фраз
        # (например, «Cloud.ru предоставляет…»). Требуем разумную длину.
        if not key or len(key) < 80:
            continue
        groups[key].append(idx)

    changed = 0
    for key, indices in groups.items():
        if len(indices) < threshold:
            continue

        def score(idx: int) -> tuple[int, float]:
            v = verdicts[idx]
            return (len(v.source_urls or []), v.confidence)

        canonical = max(indices, key=score)
        canonical_req_id = verdicts[canonical].requirement_id
        for idx in indices:
            if idx == canonical:
                continue
            v = verdicts[idx]
            old_verdict = v.verdict
            old_conf = v.confidence
            v.confidence = max(0.0, v.confidence - 0.2)
            if v.confidence < 0.5 and v.verdict == "match":
                v.verdict = "partial"
            if v.evidence_status == "confirmed":
                v.evidence_status = "weak"
            note = (
                f"Dup reasoning (×{len(indices)}): совпадает с requirement_id="
                f"{canonical_req_id}. Confidence {old_conf:.2f}→{v.confidence:.2f}"
            )
            if old_verdict != v.verdict:
                note += f", verdict {old_verdict}→{v.verdict}"
            existing_notes = list(v.evidence_contract_notes or [])
            existing_notes.append(note)
            v.evidence_contract_notes = list(dict.fromkeys(existing_notes))
            if v.trace is None:
                v.trace = {}
            v.trace["dup_reasoning_canonical_id"] = canonical_req_id
            changed += 1
    return changed


def _discretize_confidence_inplace(verdicts: list[RequirementVerdict]) -> int:
    """Patch 10 (ZK10): дискретизация confidence до {0.95, 0.75, 0.55, 0.4, 0.25}.

    Дополнительное правило: если confidence снапнулась в 0.4 (низкая
    уверенность БЕЗ reasoning'а) и verdict оставался match/partial — это
    индикатор «default 0.5 без обоснования» из A-2; принудительно
    переводим verdict в needs_clarification.

    Возвращает число изменённых verdict'ов.
    """
    changed = 0
    for v in verdicts:
        if (v.category or "").lower() == "procedural":
            continue
        if v.verdict == "out_of_scope":
            continue
        has_reasoning = bool(v.reasoning) and len(v.reasoning) >= 40
        new_overall = _snap_confidence(v.confidence, has_reasoning)
        if new_overall == 0.4 and v.verdict in {"match", "partial"}:
            old = v.verdict
            v.verdict = "needs_clarification"
            existing = list(v.evidence_contract_notes or [])
            existing.append(
                f"Auto-downgrade: confidence snapped to 0.4 без обоснования → "
                f"verdict {old}→needs_clarification"
            )
            v.evidence_contract_notes = list(dict.fromkeys(existing))
        if abs(new_overall - (v.confidence or 0.0)) > 1e-6:
            v.confidence = new_overall
            changed += 1
        for assessment in v.platform_assessments or []:
            a_has_reasoning = bool(assessment.reasoning) and len(assessment.reasoning) >= 40
            new_p = _snap_confidence(assessment.confidence, a_has_reasoning)
            if abs(new_p - (assessment.confidence or 0.0)) > 1e-6:
                assessment.confidence = new_p
                changed += 1
    return changed


def _verdict_has_authoritative_docs_url(v: RequirementVerdict) -> bool:
    """True, если у verdict в source_urls (или в platform_assessments) есть
    хотя бы один URL из cloud.ru/docs/... — это «авторитетный» источник из
    официальной документации. Такие verdict'ы НЕ трогаем в URL overuse и
    sync_verdict_with_confidence каскадах: если LLM нашёл подтверждение в
    реальной документации, мы ему доверяем."""
    def _has_docs(urls) -> bool:
        if not urls:
            return False
        for u in urls:
            if not u:
                continue
            ul = u.lower()
            if "cloud.ru/docs/" in ul:
                return True
        return False

    if _has_docs(v.source_urls):
        return True
    for a in (v.platform_assessments or []):
        if _has_docs(a.source_urls):
            return True
    return False


def _is_authoritative_docs_url(url: str) -> bool:
    """True для URL'ов вида https://cloud.ru/docs/... — авторитетная
    документация, которая может цитироваться часто без штрафа за overuse."""
    if not url:
        return False
    return "cloud.ru/docs/" in url.lower()


def _penalize_url_overuse(
    verdicts: list[RequirementVerdict],
    threshold: int = 5,
) -> tuple[int, list[dict]]:
    """Patch 13 (ZK10). Если один URL цитируется > threshold раз в одном
    отчёте, у verdict'ов с этим URL — кроме top-3 по relevance (score из
    selected_sources) — понижаем confidence и помечаем weak.

    ИСКЛЮЧЕНИЕ (v12.1): URL'ы из cloud.ru/docs/ — авторитетная документация,
    которая обоснованно повторяется (S3 Object Lock, OBS guides и т.п.).
    Они НЕ считаются overused: повторное цитирование — норма для официальных
    страниц, а не сигнал галлюцинации.

    Возвращает (число изменённых verdict'ов, список overused URL для отчёта).
    """
    from collections import defaultdict

    if threshold < 2:
        threshold = 2

    url_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, v in enumerate(verdicts):
        if v.verdict not in {"match", "partial"}:
            continue
        if (v.category or "").lower() == "procedural":
            continue
        urls: set[str] = set()
        urls.update(u for u in (v.source_urls or []) if u)
        for a in v.platform_assessments or []:
            urls.update(u for u in (a.source_urls or []) if u)
        for url in urls:
            # v12.1: cloud.ru/docs/ — авторитетные URL, не штрафуем
            if _is_authoritative_docs_url(url):
                continue
            url_to_indices[url].append(idx)

    changed_set: set[int] = set()
    overused: list[dict] = []
    for url, indices in url_to_indices.items():
        if len(indices) < threshold:
            continue
        overused.append({"url": url, "uses": len(indices)})

        def relevance_score(idx: int) -> float:
            v = verdicts[idx]
            sources = (v.trace or {}).get("selected_sources") or []
            top = 0.0
            for s in sources:
                if isinstance(s, dict):
                    try:
                        sc = float(s.get("score", 0) or 0)
                    except (TypeError, ValueError):
                        sc = 0.0
                    if sc > top:
                        top = sc
            return top

        ordered = sorted(indices, key=relevance_score, reverse=True)
        keep = set(ordered[:3])
        for idx in indices:
            if idx in keep:
                continue
            v = verdicts[idx]
            old_conf = v.confidence
            v.confidence = max(0.0, v.confidence - 0.15)
            if v.evidence_status == "confirmed":
                v.evidence_status = "weak"
            existing = list(v.evidence_contract_notes or [])
            existing.append(
                f"URL overuse: {url} цитируется в {len(indices)} verdict'ах; "
                f"релевантность к этому требованию требует ручной проверки. "
                f"Confidence {old_conf:.2f}→{v.confidence:.2f}"
            )
            v.evidence_contract_notes = list(dict.fromkeys(existing))
            if v.trace is None:
                v.trace = {}
            v.trace.setdefault("url_overuse_flags", []).append(url)
            changed_set.add(idx)
    return len(changed_set), overused


def _sync_verdict_with_confidence(
    verdicts: list[RequirementVerdict],
) -> int:
    """Patch 15. Если после всех downgrade-каскадов confidence упал низко,
    но verdict остался match — это противоречие («match при confidence 25%»).
    Синхронизируем verdict с confidence:
      • confidence < 0.30: match/partial → needs_clarification
      • 0.30 ≤ confidence < 0.50: match → partial (partial оставляем)
      • confidence ≥ 0.50: оставляем как есть

    Procedural и out_of_scope не трогаем.
    Verdicts, помеченные curated_enforced=True (т.е. _apply_curated_enforcement
    подтвердил verdict через curated knowledge base), тоже НЕ трогаем —
    у них verdict обоснован URL'ом из cloud.ru/docs, а не overlap'ом цитат.
    URL overuse cascade может занизить confidence, но это не повод
    перетирать enforced match.

    v12.1: verdicts, у которых в source_urls есть авторитетный URL из
    cloud.ru/docs/..., тоже НЕ трогаем — даже если confidence упал.
    Реальная cloud.ru/docs ссылка — это сильнее, чем формальный confidence:
    LLM нашёл подтверждение в официальной документации, не убиваем match
    из-за URL overuse каскада.

    Возвращает число изменённых verdict'ов.
    """
    changed = 0
    for v in verdicts:
        if v.verdict in {"out_of_scope"}:
            continue
        if (v.category or "").lower() == "procedural":
            continue
        # Защита: если verdict подтверждён curated_fact (через
        # _apply_curated_enforcement), мы доверяем ему даже при низкой
        # confidence после URL overuse cascade.
        if (v.trace or {}).get("curated_enforced"):
            continue
        # v12.1: иммунитет для verdicts с авторитетным URL cloud.ru/docs
        if _verdict_has_authoritative_docs_url(v):
            if v.trace is None:
                v.trace = {}
            v.trace["confidence_sync_skipped_authoritative_url"] = True
            continue
        old = v.verdict
        if v.confidence < 0.30 and v.verdict in {"match", "partial"}:
            v.verdict = "needs_clarification"
        elif v.confidence < 0.50 and v.verdict == "match":
            v.verdict = "partial"
        if old != v.verdict:
            changed += 1
            existing = list(v.evidence_contract_notes or [])
            existing.append(
                f"Verdict синхронизирован с confidence: {old} → {v.verdict} "
                f"(confidence={v.confidence:.2f}, < порога 0.50/0.30)"
            )
            v.evidence_contract_notes = list(dict.fromkeys(existing))
            if v.trace is None:
                v.trace = {}
            v.trace["confidence_verdict_synced"] = True
    return changed


def _post_process_verdicts(
    verdicts: list[RequirementVerdict],
    settings: RuntimeSettings,
) -> dict:
    """Cross-verdict post-process для патчей 7, 10, 13, 15.

    Порядок (важен):
      1. Dedup reasoning (patch 7) — снижение confidence у дубликатов.
      2. URL overuse (patch 13) — снижение у verdict'ов с переиспользованным URL.
      3. Discretize confidence (patch 10) — финальное снапание.
      4. Sync verdict с confidence (patch 15) — если confidence упал
         после cascade'ов, понижаем verdict до partial/NC.

    Возвращает stats для report.quality_metrics (patch 14).
    """
    stats: dict = {
        "dedup_reasoning_downgrades": 0,
        "url_overuse_downgrades": 0,
        "confidence_snaps": 0,
        "verdict_synced": 0,
        "url_overused": [],
    }
    threshold_dedup = max(
        2,
        int(getattr(settings, "analysis_duplicate_reasoning_threshold", 3) or 3),
    )
    stats["dedup_reasoning_downgrades"] = _detect_and_penalize_duplicate_reasoning(
        verdicts, threshold=threshold_dedup
    )

    threshold_url = max(
        2,
        int(getattr(settings, "analysis_url_overuse_threshold", 5) or 5),
    )
    url_changed, overused = _penalize_url_overuse(verdicts, threshold=threshold_url)
    stats["url_overuse_downgrades"] = url_changed
    stats["url_overused"] = overused

    if getattr(settings, "analysis_discrete_confidence", True):
        stats["confidence_snaps"] = _discretize_confidence_inplace(verdicts)

    # Patch 15 — после всех downgrade-каскадов синхронизируем verdict
    # с confidence. Это устраняет противоречие «match при confidence 25%».
    stats["verdict_synced"] = _sync_verdict_with_confidence(verdicts)

    if any(stats.get(k) for k in ("dedup_reasoning_downgrades", "url_overuse_downgrades", "confidence_snaps", "verdict_synced")):
        logger.info(
            "Post-process: dedup=%d, url_overuse=%d, snaps=%d, verdict_synced=%d",
            stats["dedup_reasoning_downgrades"],
            stats["url_overuse_downgrades"],
            stats["confidence_snaps"],
            stats["verdict_synced"],
        )
    return stats


def _fetch_managed_rag(
    req: Requirement,
    settings: RuntimeSettings,
) -> tuple[int, ManagedRagResult | None, str | None]:
    try:
        result = retrieve_generate(
            _managed_rag_query(req),
            number_of_results=settings.managed_rag_results,
            settings=settings,
        )
        return req.id, result, None
    except Exception as exc:
        return req.id, None, str(exc)


def _fetch_batch_managed_rag(
    requirements: list[Requirement],
    settings: RuntimeSettings,
    query: str | None = None,
) -> tuple[ManagedRagResult | None, str | None]:
    try:
        result = retrieve_generate(
            query or _managed_rag_batch_query(requirements),
            number_of_results=settings.managed_rag_results,
            settings=settings,
        )
        return result, None
    except Exception as exc:
        return None, str(exc)


def _analyze_batch(requirements: list[Requirement], settings: RuntimeSettings) -> list[RequirementVerdict]:
    """Analyze a batch of requirements."""
    all_context_parts = []
    req_rag_results: dict[int, ManagedRagResult] = {}
    req_source_urls: dict[int, list[str]] = {r.id: [] for r in requirements}
    req_traces: dict[int, dict] = {}
    req_map = {r.id: r for r in requirements}

    if settings.analysis_rag_mode == "grouped":
        logger.info("Fetching grouped Managed RAG context for %d requirements", len(requirements))
        batch_query = _managed_rag_batch_query(requirements)
        rag_result, error = _fetch_batch_managed_rag(requirements, settings, batch_query)
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
                all_context_parts.append(_format_rag_context(req, reranked, max_chars_per_result=1800))
    else:
        max_workers = max(1, min(settings.managed_rag_concurrency, len(requirements)))
        logger.info("Fetching Managed RAG context for %d requirements (parallel=%d)", len(requirements), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fetch_managed_rag, req, settings) for req in requirements]
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
                all_context_parts.append(_format_rag_context(req, reranked, max_chars_per_result=1800))

    context = "\n\n---\n\n".join(all_context_parts) if all_context_parts else "Managed RAG не вернул релевантной информации."

    # v12.1: забираем snapshot local_rag-хитов, собранных в side-channel при
    # вызовах _format_rag_context для каждого требования этого батча.
    # Положим эти данные в trace каждого verdict ниже — чтобы в отчёте
    # было видно «local_rag активирован, URLs: [...]».
    local_rag_snapshot = _take_and_clear_local_rag_hits()
    if local_rag_snapshot:
        logger.info(
            "local_rag: snapshot для батча — %d требований получили хиты",
            len(local_rag_snapshot),
        )

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

    # 48000 — расширенный запас под cheat sheet, compliance-правила и
    # объёмные reasoning'и для 4 platform_assessments × 8 требований.
    # Qwen3-Next-80B-A3B-Instruct поддерживает context window 128K и
    # max_completion_tokens до ~64K. Реальный расход в большинстве случаев
    # ~10-15K, остаток — буфер безопасности. Оплата идёт по факту, не по
    # выделенному пределу.
    # История значений:
    #   8000  → батч=12 регулярно обрезался, JSON ломался;
    #   12000 → раздел 7.2-9.x всё ещё уходил с confidence=0%;
    #   24000 → стабильно работало, но при добавлении cheat sheet/правил
    #           иногда LLM не успевала закончить детальный reasoning;
    #   48000 → промпт+local_rag-блок суммарно >35К символов системного +
    #           ~30К символов user-prompt'a; иногда LLM «ужимала»
    #           reasoning отдельных вердиктов;
    #   80000 → запас для local_rag-фрагментов в контексте + детальных
    #           reasoning'ов с цитированием URL. Qwen3-Next-80B имеет
    #           128K контекста, так что 80К на output безопасно.
    result = call_llm_json(prompt, system_prompt=analysis_system, max_tokens=80000, settings=settings)

    verdicts: list[RequirementVerdict] = []

    def _process_items(items_list) -> set[int]:
        """Парсит JSON-объекты от LLM, формирует RequirementVerdict-ы и
        складывает их в `verdicts`. Возвращает множество requirement_id,
        для которых вердикт получен.
        """
        ids_done: set[int] = set()
        for item in items_list:
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
            urls_from_llm = _filter_urls(item.get("source_urls", []))
            urls_from_search = req_source_urls.get(req_id, [])
            combined_urls = list(dict.fromkeys(urls_from_llm + urls_from_search))
            rag_result = req_rag_results.get(req_id)
            platform_assessments = _platform_assessments_from_llm(item, rag_result, req, combined_urls)
            source_note = ""
            if rag_result and rag_result.source_labels:
                source_note = "\n\nВыбранные документы Managed RAG: " + ", ".join(rag_result.source_labels[:5])
            trace = dict(req_traces.get(req_id, {}))
            # v12.1: записываем local_rag-хиты в trace, чтобы в отчёте было
            # видно, что local_rag реально подмешал URLs из cloud.ru/docs.
            local_rag_hits = local_rag_snapshot.get(str(req_id), [])
            if local_rag_hits:
                trace["local_rag"] = {
                    "activated": True,
                    "hits_count": len(local_rag_hits),
                    "urls": [h["url"] for h in local_rag_hits],
                    "top_score": max((h.get("score", 0) for h in local_rag_hits), default=0.0),
                }
            else:
                trace["local_rag"] = {"activated": False}
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
                or any(
                    a.source_type == "external_service"
                    and a.platform_name not in CANONICAL_PLATFORMS_ORDER
                    for a in platform_assessments
                ),
                external_service_notes=_value_to_text(item.get("external_service_notes", "")),
                trace=trace,
            )
            verdicts.append(_apply_evidence_contract(verdict))
            ids_done.add(req_id)
        return ids_done

    primary_items = result if isinstance(result, list) else [result]
    returned_ids = _process_items(primary_items)

    # RETRY ДЛЯ ПРОПУЩЕННЫХ. Если LLM вернула меньше вердиктов чем требований
    # (типичная причина — обрыв JSON на лимите max_tokens или невалидный JSON
    # на одном из объектов), повторяем дозапрос только для тех requirement_id,
    # которых не оказалось в первом ответе. Делаем суб-батчами по 4 — это
    # гарантированно укладывается в любой разумный max_tokens.
    missing_reqs = [r for r in requirements if r.id not in returned_ids]
    if missing_reqs:
        logger.warning(
            "LLM не вернула вердикт для %d/%d требований первого батча; делаем retry батчами по 4",
            len(missing_reqs),
            len(requirements),
        )
        retry_chunk_size = 4
        for chunk_start in range(0, len(missing_reqs), retry_chunk_size):
            chunk = missing_reqs[chunk_start:chunk_start + retry_chunk_size]
            chunk_lines = []
            for r in chunk:
                line = f"ID={r.id} | Раздел: {r.section} | Категория: {r.category}\nТребование: {r.text}"
                if r.tables:
                    line += f"\nТаблица:\n{r.tables}"
                chunk_lines.append(line)
            retry_prompt = analysis_template.format(
                requirements_block="\n\n".join(chunk_lines),
                context=context,
            )
            try:
                retry_result = call_llm_json(
                    retry_prompt,
                    system_prompt=analysis_system,
                    # Retry-чанк = до 4 требований; промпт + local_rag
                    # фрагменты могут давать тяжёлый контекст. 48K — двойной
                    # запас, гарантия что reasoning не обрежется.
                    max_tokens=48000,
                    settings=settings,
                )
                retry_items = retry_result if isinstance(retry_result, list) else [retry_result]
                got_ids = _process_items(retry_items)
                returned_ids |= got_ids
                logger.info(
                    "Retry chunk %d-%d: получили вердикты для %d/%d",
                    chunk_start,
                    chunk_start + len(chunk),
                    len(got_ids),
                    len(chunk),
                )
            except Exception as exc:
                logger.error(
                    "Retry chunk %d-%d failed: %s — оставляем placeholder",
                    chunk_start,
                    chunk_start + len(chunk),
                    exc,
                )

    # Add verdicts for requirements not returned by LLM. Создаём 4
    # placeholder-assessment'а — по одному на каждую каноническую платформу
    # Cloud.ru, чтобы такое требование корректно попало в матрицу с
    # ячейками `?` (needs_clarification), а не выпало с пустыми ячейками
    # `-`. Раньше создавался один placeholder с именем «Cloud.ru
    # (документация не найдена)» — он не попадал в by_platform, и в матрице
    # все 4 ячейки рендерились как mismatch для 12 строк (когда сломался
    # один LLM-батч).
    returned_ids = {v.requirement_id for v in verdicts}
    for req in requirements:
        if req.id not in returned_ids:
            placeholder_assessments = [
                PlatformAssessment(
                    platform_name=platform_name,
                    verdict="needs_clarification",
                    confidence=0.0,
                    reasoning=(
                        "Managed RAG/LLM не вернули вердикт по требованию "
                        "(вероятно, сломан батч анализа). Требуется повторный "
                        "прогон или ручная проверка."
                    ),
                    source_type="platform",
                    recommendation="Перезапустить анализ или проверить вручную.",
                )
                for platform_name in CANONICAL_PLATFORMS_ORDER
            ]
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
                platform_assessments=placeholder_assessments,
            ))

    return verdicts


def _generate_summary(
    report: AnalysisReport,
    settings: RuntimeSettings,
    mode: str = "portfolio",
) -> str:
    """Generate a text summary of the report.

    mode == "portfolio" — счётчики и процент считаются best-case по
                          портфелю (overall_verdict каждого требования).
    mode == "platform"  — счётчики и процент берутся ТОЛЬКО по
                          platform_assessments рекомендуемой платформы;
                          требование, у которого этой платформы нет,
                          выпадает из знаменателя.

    Резюме под разные режимы хранится отдельно (summary_platform /
    summary_portfolio в AnalysisReport), UI выбирает нужное по тогглу
    в шапке.
    """
    # Топ блокеров: сначала все mismatch, потом partial с самой низкой
    # уверенностью (т.е. наиболее проблемные partial). Передаём ПОЛНЫЙ текст
    # требования, чтобы LLM не выдумывал блокеры из своих представлений.
    mismatch_lines: list[str] = []
    for v in report.verdicts:
        if v.verdict == "mismatch":
            mismatch_lines.append(
                f"- [{v.section}] ТРЕБОВАНИЕ: {v.requirement_text[:240]}\n  "
                f"ПРИЧИНА: {(v.reasoning or '')[:240]}"
            )
    partial_sorted = sorted(
        [v for v in report.verdicts if v.verdict == "partial"],
        key=lambda v: (v.confidence, v.requirement_id),
    )
    partial_lines: list[str] = []
    for v in partial_sorted[:8]:
        partial_lines.append(
            f"- [{v.section}] ТРЕБОВАНИЕ: {v.requirement_text[:240]}\n  "
            f"ОБОСНОВАНИЕ: {(v.reasoning or '')[:240]}"
        )
    blockers_lines: list[str] = []
    if mismatch_lines:
        blockers_lines.append("MISMATCH (несоответствия):")
        blockers_lines.extend(mismatch_lines[:10])
    if partial_lines:
        blockers_lines.append("PARTIAL (частичные / требуют доработки):")
        blockers_lines.extend(partial_lines)
    top_mismatches_text = "\n".join(blockers_lines) if blockers_lines else "Нет"

    # Aggregate platform_assessments across all verdicts so the summary LLM
    # has actual statistics and не «придумывает» рекомендацию платформы.
    canonical_platforms = ("ГосОблако", "Облако VMware", "Advanced", "Evolution")
    platform_counts: dict[str, dict[str, int]] = {
        name: {"match": 0, "partial": 0, "mismatch": 0, "needs_clarification": 0}
        for name in canonical_platforms
    }
    for v in report.verdicts:
        for a in (v.platform_assessments or []):
            name = _canonical_platform_name(a.platform_name or "")
            if name not in platform_counts:
                continue
            verdict = a.verdict if a.verdict in {"match", "partial", "mismatch", "needs_clarification"} else "needs_clarification"
            platform_counts[name][verdict] += 1

    matrix_lines = ["| Платформа | Match | Partial | Mismatch | NC |", "|---|---|---|---|---|"]
    for name in canonical_platforms:
        c = platform_counts[name]
        matrix_lines.append(
            f"| {name} | {c['match']} | {c['partial']} | {c['mismatch']} | {c['needs_clarification']} |"
        )
    platform_matrix_text = "\n".join(matrix_lines)

    # Best platform: most match'ей, при равенстве — больший рейтинг
    # match*2 + partial. Если у всех 0 match — возвращаем «Не определена».
    def _platform_score(name: str) -> int:
        c = platform_counts[name]
        return c["match"] * 2 + c["partial"]

    # Используем единый источник истины — report.recommended_platform.
    # Он считается тем же образом, и ровно так же отображается в шапке отчёта.
    recommended_platform = report.recommended_platform or "Не определена (требует ручной проработки)"
    recommended_compliance = report.recommended_platform_compliance

    # Список внешних услуг, которые упоминались в requires_external_service
    external_services = []
    for v in report.verdicts:
        if v.requires_external_service and v.external_service_notes:
            external_services.append(v.external_service_notes)
    external_services_text = ", ".join(sorted(set(external_services))) if external_services else "Нет"

    # Подбираем счётчики и процент под режим. Для platform-mode цифры
    # считаются ТОЛЬКО по platform_assessments рекомендуемой платформы;
    # требования без оценки этой платформы не входят в знаменатель.
    if mode == "platform" and recommended_platform != "Не определена (требует ручной проработки)":
        platform_total = 0
        pcounts = {"match": 0, "partial": 0, "mismatch": 0, "needs_clarification": 0}
        platform_score = 0
        for v in report.verdicts:
            chosen_verdict = None
            for a in (v.platform_assessments or []):
                if _canonical_platform_name(a.platform_name or "") == recommended_platform:
                    chosen_verdict = a.verdict
                    break
            if chosen_verdict is None:
                continue
            if chosen_verdict not in pcounts:
                chosen_verdict = "needs_clarification"
            pcounts[chosen_verdict] += 1
            platform_total += 1
            if chosen_verdict == "match":
                platform_score += 2
            elif chosen_verdict == "partial":
                platform_score += 1
        platform_max = platform_total * 2 if platform_total else 0
        platform_pct = (
            round(platform_score / platform_max * 100, 1) if platform_max else 0.0
        )
        prompt_total = platform_total
        prompt_match = pcounts["match"]
        prompt_partial = pcounts["partial"]
        prompt_mismatch = pcounts["mismatch"]
        prompt_nc = pcounts["needs_clarification"]
        prompt_pct = platform_pct
        mode_note = (
            f"\nРЕЖИМ РЕЗЮМЕ: «По платформе ({recommended_platform})». "
            f"Счётчики выше посчитаны ТОЛЬКО по platform_assessments этой платформы. "
            f"Требования без оценки {recommended_platform} в знаменатель не входят."
        )
    else:
        prompt_total = report.total
        prompt_match = report.match_count
        prompt_partial = report.partial_count
        prompt_mismatch = report.mismatch_count
        prompt_nc = report.clarification_count
        prompt_pct = report.compliance_percentage
        mode_note = (
            "\nРЕЖИМ РЕЗЮМЕ: «Best-case по портфелю Cloud.ru». "
            "Берётся лучший вердикт по каждому требованию из всех 4 платформ — "
            "так считается потенциал гибридного предложения."
        )

    from src.prompt_store import get_prompt

    summary_template = get_prompt("summary_user_template")
    summary_system = get_prompt("summary_system")
    prompt = summary_template.format(
        doc_name=report.document_name,
        total=prompt_total,
        match_count=prompt_match,
        partial_count=prompt_partial,
        mismatch_count=prompt_mismatch,
        clarification_count=prompt_nc,
        compliance_pct=prompt_pct,
        top_mismatches=top_mismatches_text,
        platform_matrix=platform_matrix_text,
        recommended_platform=recommended_platform,
        recommended_platform_compliance=recommended_compliance,
        external_services=external_services_text,
    ) + mode_note

    return call_llm(prompt, system_prompt=summary_system, max_tokens=4000, settings=settings)
