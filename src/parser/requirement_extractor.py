"""Requirement extractor — uses LLM to split document into structured requirements."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re

from src.models import Requirement
from src.llm.client import call_llm_json
from src.parser.document_parser import ParsedBlock, ParsedDocument
from src.runtime_config import RuntimeSettings, build_runtime_settings

logger = logging.getLogger(__name__)

KEY_REQUIREMENT_SIGNALS = (
    # Патч 5 (ZK10). Расширенная схема ключевых сигналов:
    #   • needles — «сильные» термины, дословные маркеры требования.
    #     1 hit → direct match.
    #   • weak_needles — «слабые», часто живут в глоссарии или мелькают
    #     случайно (E-1 ложные срабатывания: «ddos», «waf», «фстэк» как 1
    #     упоминание не означают, что в ТЗ есть требование к этой капабилити).
    #   • synonyms — альтернативные формулировки заказчика (E-2:
    #     «гипервизор», «портал управления виртуальной инфраструктурой»
    #     означают то же, что «VMware vCloud Director»).
    #   • confirming — слова, подтверждающие контекст (нужны для weak/synonym).
    #   • min_weak_count — сколько упоминаний weak_needles должно быть, чтобы
    #     это считалось direct-match'ем (по умолчанию 2). Считается ПО ВСЕМ
    #     weak-terms сразу.
    # Strength: direct → синтез реален; synonym → есть основание, но
    # нужна проверка; absent → не найдено.
    {
        "id": "vmware_vcloud",
        "label": "VMware vCloud Director / ВЦОД",
        "needles": ("vmware vcloud director", "vcloud director", "vcd"),
        "synonyms": (
            "гипервизор",
            "vsphere",
            "vmware esxi",
            "портал управления виртуальной",
            "виртуализация на базе vmware",
            "vmware tools",
        ),
        "confirming": ("vmware", "vsphere", "vcloud", "esxi", "виртуальной инфраструктур"),
        "critical": True,
    },
    {
        "id": "capacity",
        "label": "Количественные мощности ВЦОД",
        "needles": ("164", "656", "8 300", "8300", "7 000", "7000"),
        "weak_needles": ("vram", "vcpu"),
        "confirming": ("вцод", "виртуальн", "ресурс"),
        "min_weak_count": 1,
        "critical": True,
    },
    {
        "id": "monitoring",
        "label": "vRealize Operations / мониторинг",
        "needles": ("vrealize operations", "расширенного мониторинга"),
        "synonyms": ("мониторинг", "vrealize"),
        "confirming": ("vmware", "виртуальн", "инфраструктур", "log", "metric"),
        "critical": False,
    },
    {
        "id": "s3",
        "label": "S3-хранилище и лимиты API",
        "needles": (
            "s3 совместим",
            "s3 api",
            "10 000 запросов",
            "трех зонах доступности",
            "трёх зонах доступности",
        ),
        "weak_needles": ("s3", "5 гб", "5 тб"),
        "confirming": ("хранилищ", "объектн", "bucket", "ведро"),
        "min_weak_count": 2,
        "critical": True,
    },
    {
        "id": "ddos_waf",
        "label": "DDoS/WAF",
        "needles": (
            "точек очистки",
            "защита web-приложений",
            "защита веб-приложений",
            "ddos-атак",
            "ддос-атак",
            "анти-ddos",
            "antiddos",
            "межсетевой экран web",
            "наложенное средство защиты",
        ),
        "weak_needles": ("ddos", "waf"),
        "confirming": ("атак", "защит", "межсетев", "трафик", "ботнет"),
        "min_weak_count": 3,  # 1-2 случайных хитов = глоссарный шум.
        "critical": True,
    },
    {
        "id": "ddos_waf_advanced",
        "label": "DDoS/WAF расширенные параметры",
        "needles": (
            "ja3",
            "websocket",
            "let’s encrypt",
            "lets encrypt",
            "let's encrypt",
            "50мс",
            "50 мс",
            "captcha",
        ),
        "critical": False,
    },
    {
        "id": "ngfw_usergate",
        "label": "UserGate / NGFW",
        "needles": (
            "usergate",
            "межсетевого экрана нового поколения",
            "межсетевому экрану нового поколения",
        ),
        "weak_needles": ("ngfw",),
        "synonyms": ("checkpoint", "fortinet"),
        "confirming": ("экран", "трафик", "vpn", "защит", "межсетев"),
        "min_weak_count": 2,
        "critical": True,
    },
    {
        "id": "fstec",
        "label": "ФСТЭК / отечественное ПО",
        "needles": (
            "реестр российского по",
            "реестр отечественного по",
            "сертификат соответствия фстэк",
            "лицензи фстэк",
            "приказ фстэк",
            "приказа фстэк",
            "тзки",
        ),
        "weak_needles": ("фстэк",),
        "confirming": ("безопасн", "защит", "сертифик", "лиценз", "несанкционирован"),
        "min_weak_count": 3,
        "critical": True,
    },
    {
        "id": "uz2_datacenter",
        "label": "ЦОД РФ / УЗ-2 / аттестация",
        "needles": (
            "уровня защищенности не ниже",
            "аттестация цод",
            "аттестован цод",
            "адрес нахождения цод",
        ),
        "weak_needles": ("уз-2", "уз-1"),
        "confirming": ("защищенност", "аттестат", "пдн", "152-фз", "цод"),
        "min_weak_count": 2,
        "critical": True,
    },
)

EXTRACTION_SYSTEM_PROMPT = """Ты — эксперт по анализу технических заданий (ТЗ) и тендерной документации.
Твоя задача — извлечь из текста документа отдельные требования.

Для каждого требования укажи:
- id: порядковый номер
- section: ОБЯЗАТЕЛЬНО укажи точный номер пункта из оригинального документа (например "7.2.4", "9.9.3", "10.5"). Если пункт не пронумерован, укажи название раздела или заголовок, под которым он находится. НИКОГДА не оставляй section пустым.
- text: полный текст требования
- category: одна из категорий: "technical", "sla", "legal", "commercial", "security", "other"
- tables: если требование связано с таблицей, включи её в формате markdown

ВАЖНО:
- Сохраняй оригинальную нумерацию пунктов документа — это критически важно для навигации.
- Если у пункта есть номер (7.2.4, 10.5 и т.п.), используй именно его в поле section.
- Если у пункта нет номера, используй ближайший заголовок раздела.
- Каждое отдельное требование — отдельный элемент, даже если они в одном пункте.
- Таблицы SLA, матрицы приоритетов, формулы — это отдельные требования.
- Юридические пункты (антикоррупция, ПДн, ИБ) тоже выделяй.

Верни JSON массив объектов. Без пояснений, только JSON."""

EXTRACTION_PROMPT_TEMPLATE = """Извлеки все требования из следующего текста документа:

---
{document_text}
---

Верни JSON массив."""


def _extract_field(item: dict, keys: list[str]) -> str:
    """Try multiple possible key names to extract a field from an LLM response.

    LLMs don't always use the exact field names we asked for.
    This tries each key (case-insensitive) and returns the first non-empty match.
    """
    # First try exact matches
    for key in keys:
        val = item.get(key)
        if val and str(val).strip():
            return str(val).strip()

    # Try case-insensitive match against all item keys
    lower_keys = {k.lower(): k for k in item.keys()}
    for key in keys:
        real_key = lower_keys.get(key.lower())
        if real_key:
            val = item.get(real_key)
            if val and str(val).strip():
                return str(val).strip()

    return ""


def _extract_chunk_items(
    chunk_index: int,
    total_chunks: int,
    chunk: str,
    settings: RuntimeSettings,
) -> tuple[int, list[dict]]:
    logger.info("Extracting requirements from chunk %d/%d", chunk_index + 1, total_chunks)
    from src.prompt_store import get_prompt

    prompt_template = get_prompt("parser_user_template")
    system_prompt = get_prompt("parser_system")
    prompt = prompt_template.format(document_text=chunk)
    result = call_llm_json(prompt, system_prompt=system_prompt, max_tokens=16000, settings=settings)

    items = result if isinstance(result, list) else result.get("requirements", result.get("raw", []))
    if isinstance(items, str):
        logger.warning("LLM returned string instead of list for chunk %d", chunk_index + 1)
        return chunk_index, []
    if not isinstance(items, list):
        logger.warning("LLM returned unsupported payload for chunk %d: %s", chunk_index + 1, type(items).__name__)
        return chunk_index, []

    if items and isinstance(items[0], dict):
        logger.info(
            "LLM response keys for chunk %d: %s (first item sample: %s)",
            chunk_index + 1,
            list(items[0].keys()),
            {k: str(v)[:50] for k, v in items[0].items()},
        )
    return chunk_index, [item for item in items if isinstance(item, dict)]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_blob(text: str) -> str:
    return _normalize_text(text).lower().replace("ё", "е")


def _category_counts(requirements: list[Requirement]) -> dict[str, int]:
    return dict(Counter(req.category or "other" for req in requirements))


def _signal_present(blob: str, needles: tuple[str, ...]) -> bool:
    """Legacy helper — оставлен ради внешних использований/тестов."""
    return any(_normalize_blob(needle) in blob for needle in needles)


def _signal_evidence(blob: str, signal: dict) -> tuple[str, list[str], int]:
    """Возвращает (strength, hits, total_count) для одного сигнала.

    Патч 5 (ZK10). Используется ранжированная схема:
      • direct  — есть «сильный» needle ИЛИ ≥ min_weak_count weak-needles
        + confirming-контекст.
      • synonym — есть синоним / weak-needle + confirming, но недостаточно
        для direct. Сигнал стоит уточнить, но не считать «потерянным».
      • absent  — ни needle, ни synonym, либо нет подтверждающего контекста.
    """

    def count(term: str) -> int:
        n = _normalize_blob(term)
        return blob.count(n) if n else 0

    needles = tuple(signal.get("needles", ()))
    weak = tuple(signal.get("weak_needles", ()))
    syns = tuple(signal.get("synonyms", ()))
    confirming = tuple(signal.get("confirming", ()))
    min_weak = int(signal.get("min_weak_count", 2))

    needle_counts = {n: count(n) for n in needles}
    needle_hits = [n for n, c in needle_counts.items() if c > 0]
    weak_counts = {n: count(n) for n in weak}
    weak_total = sum(weak_counts.values())
    weak_hits = [n for n, c in weak_counts.items() if c > 0]
    syn_counts = {n: count(n) for n in syns}
    syn_hits = [n for n, c in syn_counts.items() if c > 0]
    confirming_present = (
        any(count(c) > 0 for c in confirming) if confirming else True
    )

    total = sum(needle_counts.values()) + weak_total + sum(syn_counts.values())

    # 1. Сильный needle → direct.
    if needle_hits:
        return "direct", needle_hits + weak_hits[:3] + syn_hits[:3], total
    # 2. Достаточно weak-упоминаний + подтверждение → direct.
    if weak_total >= min_weak and confirming_present:
        return "direct", weak_hits + syn_hits[:3], total
    # 3. Синонимы + подтверждение → synonym.
    if syn_hits and confirming_present:
        return "synonym", syn_hits + weak_hits[:3], total
    # 4. Weak-хиты без достаточной частоты, но подтверждение есть → synonym.
    if weak_hits and confirming_present:
        return "synonym", weak_hits, total
    return "absent", [], total


def _key_signal_coverage(source_text: str, requirements: list[Requirement]) -> list[dict]:
    source_blob = _normalize_blob(source_text)
    extracted_blob = _normalize_blob(
        "\n".join(" ".join([req.section or "", req.text or "", req.tables or ""]) for req in requirements)
    )
    coverage = []
    for signal in KEY_REQUIREMENT_SIGNALS:
        doc_strength, doc_hits, doc_total = _signal_evidence(source_blob, signal)
        ext_strength, ext_hits, ext_total = _signal_evidence(extracted_blob, signal)
        coverage.append(
            {
                "id": signal["id"],
                "label": signal["label"],
                "critical": bool(signal.get("critical")),
                "present_in_document": doc_strength != "absent",
                "present_in_extracted": ext_strength != "absent",
                "match_strength_doc": doc_strength,
                "match_strength_extracted": ext_strength,
                "doc_hits": doc_hits[:5],
                "doc_total_mentions": doc_total,
            }
        )
    return coverage


def _style_diversity(document: ParsedDocument | None) -> int:
    """Число уникальных непустых стилей среди блоков документа.

    Используется как sanity-сигнал для structured_fast: docx, прибывший после
    конверсии из .doc через soffice headless, часто содержит 2–4 разных стиля
    («Body Text», «Heading 1», «Heading 9», без списков и пр. подзаголовков).
    Парсер опирается на стили/уровни заголовков и на таких файлах возвращает
    0 требований. Эта метрика помогает понять, что виноват не документ, а его
    форматная «бедность» после конвертации.
    """
    if not isinstance(document, ParsedDocument):
        return 0
    styles = {(b.style or "").strip() for b in document.blocks}
    styles.discard("")
    return len(styles)


def _set_extraction_summary(
    document: ParsedDocument,
    parser_name: str,
    detected_requirements: list[Requirement],
    returned_requirements: list[Requirement],
    settings: RuntimeSettings,
) -> None:
    coverage = _key_signal_coverage(document.full_text, returned_requirements)
    # Патч 5 (ZK10). Различаем «реально потеряли» (direct в документе, нет в
    # извлечении) от «возможно, шум» (только synonym-match'и). Первое требует
    # внимания пресейла; второе — пометка «уточнить, действительно ли это
    # требование в ТЗ».
    missing_signals = [
        item["label"]
        for item in coverage
        if item.get("match_strength_doc") == "direct"
        and not item["present_in_extracted"]
    ]
    false_positives_suspected = [
        {
            "label": item["label"],
            "strength_in_doc": item.get("match_strength_doc"),
            "hits": item.get("doc_hits", []),
            "total_mentions": item.get("doc_total_mentions", 0),
        }
        for item in coverage
        if item.get("match_strength_doc") == "synonym"
        and not item["present_in_extracted"]
    ]
    document.metadata["requirements_extraction"] = {
        "parser": parser_name,
        "requirements_detected_before_cap": len(detected_requirements),
        "requirements_returned": len(returned_requirements),
        "cap": settings.parser_fast_max_requirements,
        "cap_applied": len(detected_requirements) > len(returned_requirements),
        "requirements_omitted_by_cap": max(0, len(detected_requirements) - len(returned_requirements)),
        "category_counts_detected": _category_counts(detected_requirements),
        "category_counts_returned": _category_counts(returned_requirements),
        "key_signal_coverage": coverage,
        "missing_key_signals_after_extraction": missing_signals,
        # Патч 5 (ZK10). Сигналы, помеченные «present_in_document» только
        # через synonym-match (без сильного needle), вынесены отдельно —
        # это потенциальные false-positives детектора, не реально потерянные
        # требования.
        "key_signals_false_positives_suspected": false_positives_suspected,
        "table_count": len(document.tables),
        "block_count": len(document.blocks),
        "style_diversity": _style_diversity(document),
    }


def _category_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in (
        "персональн",
        "152-фз",
        "фстэк",
        "фсб",
        "скзи",
        "шифр",
        "защит",
        "иб",
        "антивирус",
        "аутентификац",
        "двухфактор",
        "аттестат",
        "модель угроз",
        "к1",
        "уз-1",
        "несанкционирован",
        "ddos",
        "waf",
        "ngfw",
        "usergate",
        "межсетев",
        "фильтрац",
        "ips",
        "ids",
        "soc",
        "siem",
    )):
        return "security"
    if any(token in lowered for token in ("sla", "доступност", "время реакции", "время решения", "rto", "rpo", "простой", "инцидент", "компенсац")):
        return "sla"
    if any(token in lowered for token in ("сервер", "виртуаль", "виртуальная машина", "вм", "кластер", "сеть", "хранилищ", "cpu", "vcpu", "vram", "vhdd", "ram", "iops", "bps", "ssd", "hdd", "ip-адрес", "мбит", "интернет", "api", "s3", "резервн", "backup", "veeam", "мониторинг", "личный кабинет", "цод", "vmware", "vcloud", "nsx", "vrealize")):
        return "technical"
    if any(token in lowered for token in ("штраф", "неустой", "оплат", "стоимост", "цена", "договор", "контракт")):
        return "commercial"
    if any(token in lowered for token in ("закон", "лиценз", "сертифик", "соответств", "право", "персональных данных")):
        return "legal"
    return "other"


def _section_from_line(line: str) -> tuple[str, str] | None:
    match = re.match(
        r"^\s*(?:п\.|пункт\s*)?(\d+(?:\s*\.\s*\d+){1,6})[.)]?\s+(.+)$",
        line,
        re.IGNORECASE,
    )
    if not match:
        return None
    section, rest = match.groups()
    section = re.sub(r"\s*\.\s*", ".", section).strip(".")
    rest = rest.strip()
    if len(rest) < 8:
        return None
    return section, rest


def _is_list_style(style: str) -> bool:
    lowered = (style or "").lower()
    if "list" in lowered or "спис" in lowered or "марк" in lowered:
        return True
    # Локальные стили docx часто называются «Стиль1», «Стиль2», «Стиль3» —
    # это, как правило, bullet'ы с кастомной разметкой. Если стиль
    # содержит «стиль» + цифру, считаем list-style. (Эвристика — если
    # окажется не bullet, фильтр на _is_requirement_text всё равно отсеет
    # лишнее.)
    if re.match(r"стиль\s*\d", lowered):
        return True
    return False


def _is_table_caption(text: str) -> bool:
    return bool(re.match(r"^табл(?:ица)?\.?\s*(?:№|n)?\s*\d+", text.strip(), re.IGNORECASE))


# Bridge-фразы, которые ссылаются на таблицу без собственного содержания.
# Сами строки таблицы парсер выделяет как отдельные требования; bridge-предложение
# становится дублирующим шумом и попадает в анализ как пустое «оцените таблицу».
_TABLE_REFERENCE_PATTERNS = (
    r"в\s+объ[её]м(?:ах|е)\s*,?\s*указанн(?:ых|ом)\s+в\s+табл",
    r"согласно\s+табл",
    r"в\s+соответствии\s+с\s+табл",
    r"приведённ(?:ых|ом)\s+в\s+табл",
    r"приведенн(?:ых|ом)\s+в\s+табл",
    r"представленн(?:ых|ом)\s+в\s+табл",
    r"определ[её]нн(?:ых|ом)\s+в\s+табл",
    r"перечисленн(?:ых|ом)\s+в\s+табл",
)
_TABLE_REFERENCE_RE = re.compile("|".join(_TABLE_REFERENCE_PATTERNS), re.IGNORECASE)


def _heading_needs_parent_intro(text: str) -> bool:
    """True, если короткий Heading-текст требует склейки с родительской
    intro-фразой для понятности.

    Пример «SSD – до 5 000 IOPS на 1 Тб» — без склейки непонятно, про что речь
    (про IOPS какой метрики? производительности? задержки?). Поэтому склеиваем
    с родительским «Производительность каждого диска Виртуальной машины:».

    Пример «Исполнитель должен обеспечить Доступность Услуги не менее 99,982%» —
    самодостаточен, intro не нужен.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) > 100:
        return False
    lowered = stripped.lower()
    # Если есть явный субъект-исполнитель и модальный глагол — это полное
    # требование, intro не приклеиваем.
    has_subject = any(
        marker in lowered
        for marker in (
            "исполнитель",
            "заказчик",
            "помещения",
            "инфраструктура",
            "качество",
            "решения",
            "панель управления",
            "услуга",
            "услуги",
            "услугу",
        )
    )
    has_modal = any(
        marker in lowered
        for marker in (
            "должен",
            "должна",
            "должно",
            "должны",
            "обязан",
            "обязана",
            "обязуется",
            "необходимо",
            "не имеет права",
        )
    )
    if has_subject and has_modal:
        return False
    return True


def _is_pure_table_reference(text: str) -> bool:
    """True, если фраза — это только обзорная ссылка на Таблицу №N без собственного
    содержания.

    Пример: «Исполнитель должен предоставить Услуги в объемах, указанных в Таблице №1».
    Сама Таблица №1 распарсивается как 6 отдельных требований — bridge-предложение
    дублирует их и засоряет анализ.
    """
    if not text:
        return False
    stripped = text.strip()
    # Не давим длинные требования, у которых есть и ссылка, и собственное содержание.
    if len(stripped) > 160:
        return False
    if not _TABLE_REFERENCE_RE.search(stripped):
        return False
    # Должно содержать слово «Таблиц» — это якорь.
    if "табл" not in stripped.lower():
        return False
    return True


def _caption_from_block(block: ParsedBlock, fallback: str = "") -> str:
    return (block.caption or fallback or "").strip()


def _cell_paragraphs(text: str) -> list[str]:
    return [_normalize_text(part) for part in re.split(r"[\r\n]+", text or "") if _normalize_text(part)]


def _row_section_and_title(block: ParsedBlock) -> tuple[str, str, str] | None:
    if len(block.cells) < 3:
        return None
    raw_section = _normalize_text(block.cells[0]).strip(".")
    title = _normalize_text(block.cells[1])
    body = block.cells[2] or ""
    if not re.fullmatch(r"\d+(?:\.\d+){0,6}", raw_section or ""):
        return None
    if not title or len(_normalize_text(body)) < 12:
        return None
    return raw_section, title, body


def _is_bullet_line(text: str) -> bool:
    return bool(re.match(r"^\s*(?:[-–—•·▪▫*]|\(?[a-zа-я]\)|\d+[.)])\s+", text, re.IGNORECASE))


def _strip_bullet(text: str) -> str:
    return re.sub(r"^\s*(?:[-–—•·▪▫*]|\(?[a-zа-я]\)|\d+[.)])\s+", "", text, flags=re.IGNORECASE).strip()


def _is_embedded_child_section(section: str, parent_section: str) -> bool:
    if not section or not parent_section:
        return False
    return section.startswith(parent_section + ".")


def _rich_cell_should_split(text: str) -> bool:
    paragraphs = _cell_paragraphs(text)
    if len(paragraphs) >= 4:
        return True
    return bool(re.search(r"(?:^|[\r\n])\s*\d+(?:\.\d+){1,6}\.\s+\S+", text or ""))


def _with_heading(heading: str, text: str) -> str:
    heading = _normalize_text(heading).rstrip(".")
    text = _normalize_text(text)
    if not heading or heading.lower() in text.lower()[:160]:
        return text
    return f"{heading}. {text}"


def _is_terms_context(context: str) -> bool:
    lowered = context.lower()
    return "термины и определения" in lowered or "термины и сокращения" in lowered


def _is_definition_only(text: str) -> bool:
    stripped = _normalize_text(text)
    if not stripped or _is_requirement_text(stripped):
        return False
    return bool(
        re.match(
            r"^[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9()«»\"'\s/,.-]{1,90}\s+[–-]\s+\S+",
            stripped,
        )
    )


def _pseudo_heading_level(text: str) -> int | None:
    stripped = _normalize_text(text).strip(" .")
    if not stripped or len(stripped) > 170:
        return None
    lowered = stripped.lower()
    if re.match(r"^приложение\s*№?\s*\d+", lowered):
        return 1
    if lowered.startswith(("соглашение об уровне", "регламент взаимодействия")):
        return 2
    if lowered in {"отчетность"}:
        return 3
    if stripped.endswith((".", ";", ":")):
        return None
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) < 8:
        return None
    uppercase_ratio = sum(1 for char in letters if char.upper() == char) / len(letters)
    if uppercase_ratio >= 0.82 and len(stripped.split()) <= 14:
        return 3
    return None


def _context_label(path_stack: dict[int, str]) -> str:
    return " > ".join(path_stack[level] for level in sorted(path_stack))


def _short_context_label(path_stack: dict[int, str]) -> str:
    """Short version of the heading path — last two non-empty levels.

    Long breadcrumb-style sections (used as `req.section`) crowd out the actual
    parameter when the report or matrix truncates to ~150 chars; analyzers
    benefit from a tighter parent label like 'Требования к Услугам > Услуга
    Виртуальная вычислительная инфраструктура' instead of the full path.
    """
    ordered = [path_stack[level] for level in sorted(path_stack) if path_stack.get(level)]
    if not ordered:
        return ""
    return " > ".join(ordered[-2:])


def _synthetic_section(context: str, block_index: int, suffix: str = "") -> str:
    label = (context or "").strip() or "Без раздела"
    if len(label) > 120:
        label = label[:117].rstrip() + "…"
    return f"{label} / {suffix or f'блок {block_index}'}"


def _is_section_title_only(text: str) -> bool:
    lowered = text.lower().strip(" .:")
    if lowered.startswith("требования к ") or lowered.startswith("требования по "):
        return True
    if lowered in {"общие требования оказания услуг", "требования к услугам"}:
        return True
    return False


def _is_requirement_text(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "должен",
        "должна",
        "должно",
        "должны",
        "обязан",
        "обязана",
        "обязуется",
        "необходимо",
        "требуется",
        "предоставить",
        "предоставляет",
        "предоставляться",
        "обеспечить",
        "обеспечивает",
        "соответствовать",
        "соответствует",
        "не ниже",
        "не менее",
        "не более",
        "круглосуточ",
        "двухфактор",
        "лицензи",
        "сертификат",
        "аттестат",
        "аттестаци",
        "уровня защищенности",
        "реестр",
        "российская федерация",
    )
    if any(marker in lowered for marker in markers):
        return True
    if re.search(r"\b(?:до|от)\s+\d", lowered):
        return True
    # Короткие bullet'ы с критичными техническими терминами (Татспиртпром:
    # «Поддержка SFTP;», «Доступ по протоколу S3») — должны считаться
    # требованиями, даже без модального глагола. Используем тот же
    # whitelist, что и в _FILTER_IMPORTANT.
    technical_signals = (
        "sftp", "worm", "s3 ", "rest api", "object lock", "versioning",
        "tier iii", "tier 3", "vmware", "vcloud", "kubernetes", "openstack",
        "ngfw", "waf", "ddos", "siem", "multipart",
        "поддержка ", "доступ по протоколу", "доступ к ",
    )
    if any(sig in lowered for sig in technical_signals):
        # Дополнительная защита от bag-of-words: текст должен быть
        # коротким (≤ 100 симв.) и заканчиваться на `;` `.` `:` — это
        # типичный признак bullet-пункта.
        if len(text) <= 100 and text.rstrip().endswith((";", ".", ":")):
            return True
    return False


def _is_intro_requirement(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered.endswith(":"):
        return False
    return any(
        marker in lowered
        for marker in (
            "следующ",
            "требование",
            "возможность",
            "в том числе",
            "должен обеспечить",
            "должны обеспечиваться",
            "должен предоставить",
            "обязан предоставить",
            "предоставить",
            "обеспечить",
            "обеспечива",       # «обеспечивает», «обеспечивающий»
            "включа",            # «включающее в себя», «включает»
            "состоит из",
        )
    )


def _is_requirement_table(caption: str, headers: list[str]) -> bool:
    blob = " ".join([caption, *headers]).lower()
    if "описание приоритетов" in blob or "категории запросов" in blob:
        return False
    if any(token in blob for token in (
        "контактные данные",
        "контактное лицо",
        "реквизит",
        "адрес электронной",
        "инн",
        "кпп",
        "огрн",
        "наименование банка",
        "подпись",
    )):
        return False
    table_markers = (
        # Стандартные подписи таблиц требований
        "наименование",
        "кол-во",
        "количество",
        "единица измерения",
        "параметр",
        "объем",
        "объём",
        "целев",
        "показатель",
        "время решения",
        "sla",
        "услуг",
        # Расширенный набор маркеров — встречается в ТЗ заказчиков
        # с двухколоночной структурой «требование / комментарий»:
        "функциональн",       # «Функциональные требования»
        "функция",
        "характеристик",
        "опция",
        "опции",
        "возможност",         # «Возможности», «Функциональные возможности»
        "требование",
        "требования",
        "состав",
        "значение",           # «Параметр / Значение»
        "описание",           # «Опция / Описание»
        "комментарий",        # «Требование / Комментарий»
        "тип",
        "критерий",
        "технические",
        "техническое",
        "условие",
        "условия",
        "содержание",         # «Наименование / Содержание»
        "состав услуг",
    )
    return any(marker in blob for marker in table_markers)


_TABLE_SKIP_HEADERS = {"№", "n", "№ п/п", "n п/п", "п/п", "no", "no.", "пп"}

# Универсальный паттерн «intro: a, b, c, ..., z» — для разворачивания
# одной ячейки таблицы на N отдельных требований. Срабатывает, если в
# тексте есть двоеточие и за ним список через запятые с длиной ≥ 4 элементов.
# Применяется к ТЗ, где функциональные требования собраны строкой
# (Татспиртпром: «Поддержка S3 со следующими опциями: list-buckets,
# head-bucket, list-objects, ..., create-multipart-upload»).
_LIST_INTRO_HINTS_RE = re.compile(
    r"(следующ|опц|опции|включа|поддерж|перечн|перечис|операц|команд|"
    r"метод|возможност|функц|должн[аы]?\s+(?:быть|обеспеч|поддерж)|"
    r"в составе|состав)",
    re.IGNORECASE,
)


def _split_cell_into_list_items(text: str, min_items: int = 6) -> list[str] | None:
    """Если ячейка таблицы — «intro: a, b, c, ..., n» с N ≥ min_items
    элементов, возвращает [intro, a, b, c, ..., n].

    Применяется к таблицам функциональных требований ТЗ, где одна ячейка
    содержит длинный список опций/команд/возможностей. Без этого парсер
    создаёт ОДНО требование вида «Поддержка X со следующими опциями: …»,
    а LLM ставит match по поверхностному совпадению ключевого слова и
    пропускает реальные специфические фичи внутри списка.

    Условия применения (ВСЕ должны выполняться):
      1. Есть двоеточие в первой половине текста.
      2. После двоеточия — список через запятые/точки с запятой/«и».
      3. Элементов ≥ min_items (4 по умолчанию).
      4. Каждый элемент — короткий (≤ 80 символов), без вложенных предложений.
      5. Intro содержит маркер из _LIST_INTRO_HINTS_RE (следующие/опции/…).

    Возвращает None если условия не выполнены — тогда ячейка остаётся
    одной строкой требования.
    """
    if not text or len(text) < 30:
        return None
    # Ищем ПОСЛЕДНЕЕ двоеточие в первой половине строки (плюс люфт 20 симв).
    # Так корректно отрабатывается двухсложный intro вида
    # «Функциональные требования: Поддержка S3 со следующими опциями:
    # list-buckets, head-bucket, ...» — нам нужно второе двоеточие, после
    # «опциями», а не первое после «требования».
    boundary = len(text) // 2 + 20
    colon_idx = -1
    for i, ch in enumerate(text[:boundary]):
        if ch == ":":
            colon_idx = i
    if colon_idx < 5:
        return None
    intro = text[:colon_idx].strip()
    tail = text[colon_idx + 1:].strip()
    if not tail:
        return None
    # Cutoff: всё, что после "; Header:" (например "; Комментарий:",
    # "; Описание:", "; Значение:") — это уже следующая ячейка merged-row'а,
    # её не нужно включать в список опций. Обрезаем по первому такому
    # вхождению.
    cutoff_match = re.search(r";\s*[A-ЯЁ]\w+\s*:", tail)
    if cutoff_match:
        tail = tail[:cutoff_match.start()].strip()
    # Список через `,`, `;` или `\n`. Точка не используется — она часть номеров.
    parts_raw = re.split(r"\s*[;,]\s*|\s*\n\s*", tail)
    parts = [_normalize_text(p) for p in parts_raw if _normalize_text(p)]
    # Объединяем кейс «a, b, c и d» — последнее «и» может склеить.
    if len(parts) >= 2 and " и " in parts[-1] and len(parts[-1]) <= 120:
        last = parts[-1]
        for fragment in re.split(r"\s+и\s+", last):
            fragment = fragment.strip()
            if fragment and fragment != last:
                parts.append(fragment)
    if len(parts) < min_items:
        return None
    # Все элементы должны быть короткими: ≤ 80 символов, без двоеточий
    # внутри (иначе это уже сложное вложенное требование).
    if any(len(p) > 80 or p.count(":") > 0 for p in parts):
        return None
    # Intro должен содержать сигнальное слово.
    if not _LIST_INTRO_HINTS_RE.search(intro):
        return None
    # Технический сигнал: хотя бы часть элементов должна выглядеть как
    # API/опции, а не общие слова (видео, картинки, документы). Признаки —
    # дефис-в-середине (list-buckets), цифра в имени (s3v2), латиница ≥3
    # подряд (REST, JSON, API). Это убирает false-positive разворот
    # перечислений типов контента и доменных категорий.
    def looks_technical(item: str) -> bool:
        lowered = item.lower()
        # Дефис между букв (API-команда: list-buckets, head-bucket)
        if re.search(r"[a-zа-яё]-[a-zа-яё]", lowered):
            return True
        # Латиница длиной ≥ 3 подряд (REST, API, JSON, S3)
        if re.search(r"[a-z]{3,}", lowered):
            return True
        # Цифра внутри слова (s3v2, http2)
        if re.search(r"[a-zа-яё]\d|\d[a-zа-яё]", lowered):
            return True
        return False
    technical_items = sum(1 for p in parts if looks_technical(p))
    # Хотя бы половина элементов должна быть технической.
    if technical_items * 2 < len(parts):
        return None
    return [intro] + parts


def _table_requirement_text(
    caption: str,
    block: ParsedBlock,
    *,
    row_label_col: int | None = None,
    strip_prefix: str = "",
) -> tuple[str, str]:
    """Build (section_suffix, body) from a table row.

    The cell content (header: value) is concentrated at the start of `body` so that
    truncation in the report or in the analyzer prompt preserves the actual
    parameter being described. Long heading-path captions (e.g.
    'Общие требования ... > Таблица №1. ...') are reduced to a compact suffix
    in parentheses, since the section field already carries the navigation path.

    Патч 4 (ZK10): если `row_label_col` задан, ячейка этой колонки уезжает в
    `section_suffix` (идентификатор «о чём строка»: «vCPU», «RAM», «SSD»),
    а в body остаются только «data»-ячейки. Так в анализатор летит чище
    контекст: section «Таблица 3 / vCPU», text «Значение: 8», а не
    «Параметр: vCPU; Значение: 8».

    Патч 3 (ZK10): если `strip_prefix` задан (для data_uniform-таблиц с
    повторяющейся первой колонкой типа «Тип заказа: Виртуальные машины»),
    префикс удаляется из body как избыточный шум.

    Возвращает кортеж (section_suffix, body). section_suffix может быть "".
    """
    pairs: list[str] = []
    section_suffix = ""
    for index, cell in enumerate(block.cells):
        cell_text = (cell or "").strip()
        if not cell_text:
            continue
        header_clean = ""
        if index < len(block.headers):
            header_clean = (block.headers[index] or "").strip()
        if header_clean and header_clean.lower() in _TABLE_SKIP_HEADERS:
            continue
        # row-label колонка — извлекаем как section_suffix, в body не кладём.
        if row_label_col is not None and index == row_label_col:
            section_suffix = cell_text
            continue
        if header_clean and header_clean.lower() not in {"колонка", "column"}:
            pairs.append(f"{header_clean}: {cell_text}")
        else:
            pairs.append(cell_text)

    body = "; ".join(pairs).strip()

    # Удаление общего префикса (для data_uniform — все строки начинаются
    # с одинакового «Header: Value», который не несёт информации).
    if strip_prefix:
        sp = strip_prefix.strip()
        if sp:
            lower_body = body.lower()
            lower_sp = sp.lower()
            if lower_body.startswith(lower_sp):
                body = body[len(sp):].lstrip(" ;").strip()
            else:
                # Префикс мог совпасть только до знака «:» (когда значение
                # отличается, а заголовок одинаков). Срежем по схеме
                # "<header>: <X>; rest" → "rest" если <header>: на месте.
                header_part = sp.split(":", 1)[0].strip().lower() if ":" in sp else ""
                if header_part:
                    pattern_prefix = f"{header_part}:"
                    if lower_body.startswith(pattern_prefix):
                        # Найти конец первого "; " после префикса.
                        rest_idx = body.find(";", len(pattern_prefix))
                        if rest_idx != -1:
                            body = body[rest_idx + 1:].lstrip(" ;").strip()

    if not body:
        body = (caption or "").strip() or f"Таблица {block.table_index}"

    # Short tail of the caption (e.g. just "Таблица №1") for context.
    short_caption = ""
    if caption:
        match = re.search(r"(Табл(?:ица)?\s*№?\s*\d+[^>]*)", caption)
        if match:
            short_caption = match.group(1).strip()
        else:
            tail = caption.rsplit(">", 1)[-1].strip()
            short_caption = tail
        if len(short_caption) > 80:
            short_caption = short_caption[:80].rstrip() + "…"

    if short_caption and short_caption.lower() not in body.lower():
        body = f"{body} ({short_caption})"

    return section_suffix, body


def _patch_merged_caption_headers(blocks: list) -> None:
    """Если у таблицы headers все одинаковые (merged-caption) — подменяем
    их на cells первой строки, и эту строку помечаем как уже использованную.

    Типовая ситуация в ТЗ заказчика: в исходном docx первая строка таблицы —
    merged-cell с названием раздела (например, «Облачное хранение файлов
    для АО Татспиртпром»), и parser dumps её как headers. Реальные заголовки
    («Функциональные требования / Комментарий») при этом оказываются в
    первой row таблицы. Без подмены _is_requirement_table возвращает False,
    и вся таблица функциональных требований теряется.

    Алгоритм:
      1. Группируем table_row блоки по table_index.
      2. Для каждой группы: если все headers одинаковые ИЛИ headers пусты,
         И первая row имеет cells с короткими «header-like» строками (≤80
         симв., без чисел, без модальности) — берём cells первой row как
         новые headers, и проставляем флаг `_skip_as_pseudo_header` на эту
         row (она будет пропущена при обработке).
      3. Все остальные row группы получают обновлённые headers.

    Мутирует blocks in-place.
    """
    if not blocks:
        return
    from collections import defaultdict
    table_groups: dict[int, list] = defaultdict(list)
    for block in blocks:
        if block.kind == "table_row" and block.table_index is not None:
            table_groups[block.table_index].append(block)

    from collections import Counter

    for tidx, rows in table_groups.items():
        if not rows:
            continue
        headers = rows[0].headers or []
        first_row = rows[0]
        if not first_row.cells:
            continue

        # Условие срабатывания «full merge»: пустые headers, либо все одинаковые.
        is_full_merged = (
            not headers
            or (len(headers) >= 2 and len(set(headers)) == 1)
        )
        # Патч 3 (ZK10). Дополнительный кейс: «№ п/п» + N одинаковых других
        # столбцов («Тип заказа» × 6 в 0373100062625000096). Skip-headers
        # игнорируем, если оставшиеся одинаковы — это тоже merged-caption.
        if not is_full_merged and len(headers) >= 3:
            non_skip = [
                (h or "").strip()
                for h in headers
                if (h or "").strip().lower() not in _TABLE_SKIP_HEADERS
            ]
            non_skip = [h for h in non_skip if h]
            if len(non_skip) >= 2 and len(set(non_skip)) == 1:
                is_full_merged = True

        # Патч 3 (ZK10): partial-merge. Headers содержат N (≥3) одинаковых
        # значений + ещё какие-то отдельные («Тип заказа»×6 + «Ед. изм.» +
        # «Кол-во»). Патчим только дублирующиеся позиции, оставляя
        # уникальные headers как есть.
        partial_positions: list[int] = []
        if not is_full_merged and len(headers) >= 4:
            cnt = Counter((h or "").strip() for h in headers if (h or "").strip())
            for value, count in cnt.most_common(1):
                if (
                    value
                    and value.lower() not in _TABLE_SKIP_HEADERS
                    and count >= 3
                    and count >= max(3, int(0.5 * len(headers)))
                ):
                    partial_positions = [
                        i for i, h in enumerate(headers)
                        if (h or "").strip() == value
                    ]

        if not is_full_merged and not partial_positions:
            continue

        if is_full_merged:
            target_positions = list(range(len(first_row.cells)))
        else:
            target_positions = partial_positions

        # Cells первой row в target позициях должны выглядеть как заголовки —
        # короткие, без модальности и без длинных чисел.
        candidate = [(c or "").strip() for c in first_row.cells]
        target_cells = [
            candidate[i] if i < len(candidate) else ""
            for i in target_positions
        ]
        # Минимум валидных заменителей (для partial — ≥ 60% позиций).
        valid_count = sum(
            1 for v in target_cells
            if v and len(v) <= 90 and not _FILTER_MODAL_RE.search(v)
        )
        if is_full_merged:
            if valid_count < len(target_cells):
                continue
        else:
            min_required = max(3, int(0.6 * len(target_positions)))
            if valid_count < min_required:
                continue

        # Подменяем headers на всех row'ах группы — только в target позициях.
        new_headers = list(headers) if headers else [""] * len(first_row.cells)
        # Если длина headers меньше cells, расширяем.
        while len(new_headers) < len(first_row.cells):
            new_headers.append("")
        for pos in target_positions:
            if pos >= len(candidate):
                continue
            new_value = candidate[pos]
            if new_value and len(new_value) <= 90 and not _FILTER_MODAL_RE.search(new_value):
                new_headers[pos] = new_value
        for r in rows:
            r.headers = list(new_headers)
        # Помечаем первую row как «псевдо-заголовок» — пропустим её
        # при обработке.
        first_row._skip_as_pseudo_header = True  # type: ignore[attr-defined]


# Заголовки столбцов, означающие «row label» — короткая метка, которая
# идентифицирует строку («параметр», «vCPU», «название услуги»), а не несёт
# самостоятельное требование. Содержимое такой колонки переезжает в section
# вместо text — патч 4 (ZK10).
_ROW_LABEL_HEADER_TOKENS = {
    "параметр",
    "показатель",
    "наименование",
    "наименование услуги",
    "наименование подсистемы",
    "наименование подсистемы или компонента",
    "наименование ресурса",
    "услуга",
    "услуги",
    "метрика",
    "характеристика",
    "уровень предоставления услуг",
    "тип заказа",
    "тип услуги",
    "термин",
    "сокращение",
}

# Заголовки/маркеры контекста для глоссарных таблиц.
_GLOSSARY_HEADER_TOKENS = {
    "термин", "термины", "определение", "определения",
    "сокращение", "сокращения", "термины и определения",
}


def _classify_table_role(rows: list) -> tuple[str, dict]:
    """Determine the role of a table given its row blocks.

    Returns ("glossary"|"data_uniform"|"normal", extras_dict).

    Используется патчами 3–4 (ZK10), чтобы:
    1. Не превращать каждую строку словаря терминов в отдельное «требование»
       (типовая утечка: 69 «требований» из глоссария «Аварийные работы» в
       0881500000124000009).
    2. Не дублировать в каждое требование одно и то же значение первой
       колонки («Тип заказа: Виртуальные машины» × 30 в 0373100062625000096).
    3. Перенести row-label из text в section, чтобы LLM видел чистый
       контекст: section «Таблица 3 / vCPU», text «Значение: 8».

    Extras для glossary: {} (классификатор подскажет вызывающему сгенерировать
    summary через `_build_glossary_summary`).
    Extras для data_uniform: {"common_prefix": str, "row_label_col": int|None}.
    Extras для normal: {"row_label_col": int|None}.
    """
    from collections import Counter

    if not rows:
        return "normal", {}

    headers = list(rows[0].headers or [])
    headers_low = [(h or "").strip().lower() for h in headers]

    # ----- Glossary detection -----
    # H1: заголовки колонок — это «Термин» / «Определение».
    has_glossary_header = any(
        h_low in _GLOSSARY_HEADER_TOKENS for h_low in headers_low
    )
    # H2: контекст таблицы (caption или соседние heading'и) — глоссарий.
    caption_lower = ((rows[0].caption or "") or "").strip().lower()
    context_glossary = any(
        marker in caption_lower
        for marker in ("термин", "определени", "сокращени", "глоссар")
    )
    # H3: статистика по содержимому — короткая первая колонка + длинное
    # описание во второй колонке.
    short_no_modal_count = 0
    long_definition_count = 0
    valid_count = 0
    for r in rows:
        cells = r.cells or []
        if not cells:
            continue
        valid_count += 1
        c0 = (cells[0] or "").strip()
        c0_words = len(c0.split())
        if 1 <= c0_words <= 8 and not _FILTER_MODAL_RE.search(c0) and not _FILTER_NUMERIC_RE.search(c0):
            short_no_modal_count += 1
            if len(cells) >= 2:
                c1 = (cells[1] or "").strip()
                if len(c1) >= 30:
                    long_definition_count += 1
            elif 1 <= c0_words <= 4:
                long_definition_count += 1  # single-column glossary

    if valid_count >= 5:
        short_ratio = short_no_modal_count / valid_count
        def_ratio = long_definition_count / valid_count
        if has_glossary_header or context_glossary or (short_ratio >= 0.8 and def_ratio >= 0.7):
            return "glossary", {}

    # ----- Data-uniform detection (одно и то же значение в первой колонке) -----
    cells0_clean = [
        ((r.cells[0] if r.cells else "") or "").strip()
        for r in rows
    ]
    cells0_clean = [c for c in cells0_clean if c]
    common_prefix = ""
    if cells0_clean:
        top_value, top_count = Counter(cells0_clean).most_common(1)[0]
        if top_count >= max(5, int(0.7 * len(cells0_clean))):
            first_header = headers[0] if headers else ""
            if first_header:
                common_prefix = f"{first_header}: {top_value}"
            else:
                common_prefix = top_value

    # ----- Row-label column detection (для normal и data_uniform) -----
    row_label_col = None
    for idx, h_low in enumerate(headers_low):
        if not h_low:
            continue
        if h_low not in _ROW_LABEL_HEADER_TOKENS:
            continue
        # Сверяем: значения в этой колонке короткие и разнообразные.
        vals = [
            ((r.cells[idx] if r.cells and idx < len(r.cells) else "") or "").strip()
            for r in rows
        ]
        non_empty = [v for v in vals if v]
        if not non_empty:
            continue
        short_count = sum(
            1
            for v in non_empty
            if 1 <= len(v.split()) <= 10 and not _FILTER_MODAL_RE.search(v)
        )
        unique_vals = len(set(non_empty))
        # Для data_uniform значение в row_label_col одинаковое (≈ 1 уникальное),
        # поэтому уникальность не требуем для idx==0; для остальных колонок
        # ожидаем варьирование.
        if short_count >= int(0.7 * len(non_empty)) and (unique_vals >= 3 or idx == 0 and common_prefix):
            row_label_col = idx
            break

    if common_prefix:
        return "data_uniform", {"common_prefix": common_prefix, "row_label_col": row_label_col}
    return "normal", {"row_label_col": row_label_col}


def _build_glossary_summary(rows: list) -> str:
    """Build a single "Термины и определения" summary text for a glossary table.

    Сворачивает N глоссарных строк в один блок-требование вида
    «Термины и определения таблицы: term1 — def1; term2 — def2; …».
    Длинные определения обрезаются до ~120 симв., список ограничен 30
    терминами (если больше — добавляется пометка «всего N»).
    """
    parts: list[str] = []
    for r in rows[:30]:
        cells = r.cells or []
        if not cells:
            continue
        c0 = (cells[0] or "").strip()
        if not c0:
            continue
        if len(cells) >= 2 and (cells[1] or "").strip():
            c1 = (cells[1] or "").strip()
            if len(c1) > 120:
                c1 = c1[:120].rstrip() + "…"
            parts.append(f"{c0} — {c1}")
        else:
            parts.append(c0)
    if not parts:
        return ""
    extra = ""
    if len(rows) > 30:
        extra = f" Всего терминов в таблице: {len(rows)}."
    return "Термины и определения таблицы. " + "; ".join(parts) + "." + extra


def _classify_and_mark_table_roles(blocks: list) -> dict[int, str]:
    """Pre-scan table_row blocks, classify each table, mark blocks with attrs.

    Используется в `_extract_requirements_from_blocks` сразу после
    `_patch_merged_caption_headers`. Мутирует blocks:
      • r._table_role: "glossary"|"data_uniform"|"normal"
      • r._skip_as_glossary_row: True (для всех row глоссария, КРОМЕ первой)
      • r._glossary_summary: str (только на первой row глоссария)
      • r._row_label_col: int|None (индекс «label»-колонки для row)
      • r._strip_prefix: str (общий префикс для data_uniform)

    Возвращает dict {table_index: role} — для отладочного логирования.
    """
    from collections import defaultdict

    if not blocks:
        return {}

    table_groups: dict[int, list] = defaultdict(list)
    for b in blocks:
        if b.kind != "table_row":
            continue
        if b.table_index is None:
            continue
        if getattr(b, "_skip_as_pseudo_header", False):
            continue
        table_groups[b.table_index].append(b)

    roles: dict[int, str] = {}
    for tidx, rows in table_groups.items():
        if not rows:
            continue
        role, extras = _classify_table_role(rows)
        roles[tidx] = role
        for r in rows:
            r._table_role = role  # type: ignore[attr-defined]

        if role == "glossary":
            summary_text = _build_glossary_summary(rows)
            if summary_text:
                rows[0]._glossary_summary = summary_text  # type: ignore[attr-defined]
                for r in rows[1:]:
                    r._skip_as_glossary_row = True  # type: ignore[attr-defined]
        elif role == "data_uniform":
            prefix = extras.get("common_prefix", "")
            row_label_col = extras.get("row_label_col")
            for r in rows:
                if prefix:
                    r._strip_prefix = prefix  # type: ignore[attr-defined]
                if row_label_col is not None:
                    r._row_label_col = row_label_col  # type: ignore[attr-defined]
        else:  # normal
            row_label_col = extras.get("row_label_col")
            if row_label_col is not None:
                for r in rows:
                    r._row_label_col = row_label_col  # type: ignore[attr-defined]

    if roles:
        from collections import Counter
        role_counts = Counter(roles.values())
        logger.info(
            "Table role classification: %s (total tables=%d)",
            dict(role_counts), len(roles),
        )
    return roles


def _extract_requirements_from_blocks(document: ParsedDocument, settings: RuntimeSettings) -> list[Requirement]:
    """Fast parser that uses DOCX structure: headings, lists and table rows."""
    requirements: list[Requirement] = []
    path_stack: dict[int, str] = {}
    current_intro = ""
    current_table_caption = ""
    current_table_index: int | None = None
    # Counter of synthetic items per section signature, so list bullets get
    # local IDs ("пункт 1", "пункт 2") instead of global block_index ("пункт 70").
    section_counter: Counter[str] = Counter()
    # Препроцессинг: распознать таблицы с merged-caption-headers и подменить
    # headers на cells первой строки. Это типичный паттерн в ТЗ Татспиртпром
    # (Таблица 2: «Функциональные требования / Комментарий») — без подмены
    # _is_requirement_table не срабатывает на «Облачное хранение файлов».
    _patch_merged_caption_headers(document.blocks)
    # Патчи 3–4 (ZK10): классифицируем таблицы (глоссарий / data_uniform /
    # normal) и проставляем теги row-ам. См. `_classify_and_mark_table_roles`.
    _classify_and_mark_table_roles(document.blocks)

    def next_local_index(short_context: str) -> int:
        section_counter[short_context] += 1
        return section_counter[short_context]

    def update_context(level: int, label: str) -> None:
        normalized = _normalize_text(label)
        if not normalized:
            return
        for existing_level in list(path_stack):
            if existing_level >= level:
                del path_stack[existing_level]
        path_stack[level] = normalized
        # Bullets/blocks under a fresh subsection should renumber locally.
        section_counter.clear()

    def add_requirement(section: str, text: str, tables: str = "", force: bool = False) -> None:
        normalized = _normalize_text(text)
        if len(normalized) < 12 or (_is_probably_heading(normalized) and not force):
            return
        if _is_definition_only(normalized):
            return
        # Bridge-предложение типа «указанных в Таблице №1» — отдельные строки
        # таблицы уже парсятся как требования, не дублируем.
        if _is_pure_table_reference(normalized):
            return
        requirements.append(
            Requirement(
                id=len(requirements) + 1,
                section=section,
                text=normalized,
                category=_category_from_text(normalized),
                tables=tables,
            )
        )

    def add_rich_cell_requirements(parent_section: str, title: str, cell_text: str, block_index: int) -> int:
        added_before = len(requirements)
        current_section = parent_section
        current_heading = title
        current_intro = ""

        paragraphs = _cell_paragraphs(cell_text)
        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            if _is_terms_context(paragraph):
                current_intro = ""
                continue
            if _is_section_title_only(paragraph):
                current_heading = paragraph
                current_intro = paragraph if _is_intro_requirement(paragraph) else ""
                continue

            parsed_section = _section_from_line(paragraph)
            if parsed_section and _is_embedded_child_section(parsed_section[0], parent_section):
                current_section, current_heading = parsed_section
                current_intro = current_heading if _is_intro_requirement(current_heading) else ""
                if _is_requirement_text(current_heading) and not _is_section_title_only(current_heading):
                    add_requirement(current_section, current_heading)
                continue

            if parsed_section and current_section:
                local_section, rest = parsed_section
                section = f"{current_section}/{local_section}"
                add_requirement(section, _with_heading(current_heading, rest))
                current_intro = rest if _is_intro_requirement(rest) else ""
                continue

            if _is_intro_requirement(paragraph):
                current_intro = paragraph
                continue

            if _is_bullet_line(paragraph):
                bullet = _strip_bullet(paragraph)
                if not bullet:
                    continue
                if current_intro:
                    add_requirement(
                        f"{current_section or parent_section} / пункт списка {paragraph_index}",
                        f"{current_intro.rstrip(':')}: {bullet}",
                        force=True,
                    )
                elif _is_requirement_text(bullet):
                    add_requirement(
                        f"{current_section or parent_section} / пункт списка {paragraph_index}",
                        _with_heading(current_heading, bullet),
                        force=True,
                    )
                continue

            if current_intro and len(paragraph) >= 4:
                add_requirement(
                    f"{current_section or parent_section} / пункт списка {paragraph_index}",
                    f"{current_intro.rstrip(':')}: {paragraph}",
                    force=True,
                )
                continue

            if _is_requirement_text(paragraph) and not _is_section_title_only(paragraph):
                add_requirement(
                    f"{current_section or parent_section} / блок {paragraph_index}",
                    _with_heading(current_heading, paragraph),
                )

        return len(requirements) - added_before

    for block_index, block in enumerate(document.blocks, start=1):
        text = _normalize_text(block.text)
        if not text:
            continue

        if block.kind == "table_row":
            # Пропускаем псевдо-headers, подменённые из cells первой row
            # (см. _patch_merged_caption_headers).
            if getattr(block, "_skip_as_pseudo_header", False):
                if block.table_index != current_table_index:
                    if block.caption:
                        current_table_caption = block.caption
                    current_table_index = block.table_index
                continue
            # Патч 3 (ZK10): глоссарные строки уже свёрнуты в один summary
            # на первой row таблицы. Все последующие row пропускаем.
            if getattr(block, "_skip_as_glossary_row", False):
                if block.table_index != current_table_index:
                    if block.caption:
                        current_table_caption = block.caption
                    current_table_index = block.table_index
                continue
            if block.table_index != current_table_index:
                if block.caption:
                    current_table_caption = block.caption
                elif current_table_index is not None:
                    current_table_caption = ""
                current_table_index = block.table_index
            context = _context_label(path_stack)
            if _is_terms_context(context):
                continue
            # Патч 3 (ZK10): глоссарий — эмитим ОДИН summary-блок на таблицу
            # вместо N «требований»-определений. Существенно режет шум для
            # таблиц «Термины и определения», «Аварийные работы», etc.
            glossary_summary = getattr(block, "_glossary_summary", "")
            if glossary_summary:
                caption_g = _caption_from_block(block, current_table_caption)
                if context and caption_g and caption_g not in context:
                    label_g = f"{context} > {caption_g}"
                elif context and not caption_g:
                    label_g = f"{context} > Таблица {block.table_index}"
                else:
                    label_g = caption_g or f"Таблица {block.table_index}"
                section_g = f"{label_g} (термины и определения)"
                add_requirement(section_g, glossary_summary, force=True)
                continue
            row_section = _row_section_and_title(block)
            if row_section:
                section, title, body = row_section
                if _rich_cell_should_split(body):
                    added = add_rich_cell_requirements(section, title, body, block_index)
                    if added:
                        continue
                add_requirement(section, f"{title}. {body}", force=True)
                continue

            caption = _caption_from_block(block, current_table_caption)
            if not _is_requirement_table(caption, block.headers):
                continue
            if context and caption and caption not in context:
                table_label = f"{context} > {caption}"
            elif context and not caption:
                table_label = f"{context} > Таблица {block.table_index}"
            else:
                table_label = caption or f"Таблица {block.table_index}"
            # Патч 4 (ZK10): row-label колонка едет в section, а не в text.
            # Патч 3 (ZK10): общий префикс data_uniform-таблиц вырезается из body.
            row_label_col = getattr(block, "_row_label_col", None)
            strip_prefix = getattr(block, "_strip_prefix", "")
            section_suffix, row_text = _table_requirement_text(
                table_label,
                block,
                row_label_col=row_label_col,
                strip_prefix=strip_prefix,
            )
            section = table_label
            if section_suffix:
                section = f"{section} / {section_suffix}"
            section = f"{section}, строка {block.row_index}"
            # Если ячейка содержит длинный список («intro: a, b, c, ..., n»)
            # — разворачиваем в отдельные требования. Иначе одно общее.
            list_parts = _split_cell_into_list_items(row_text)
            if list_parts and len(list_parts) >= 7:
                intro = list_parts[0].rstrip(":")
                for sub_idx, item in enumerate(list_parts[1:], start=1):
                    sub_section = f"{section} / опция {sub_idx}"
                    add_requirement(sub_section, f"{intro}: {item}", force=True)
            else:
                add_requirement(section, row_text, force=True)
            continue

        if _is_table_caption(text):
            current_table_caption = text
            continue

        parsed_section = _section_from_line(text)
        style = block.style or ""
        is_list = _is_list_style(style)
        level = block.level

        if level:
            # Heading может быть И реальным требованием. В этом ТЗ так оформлены
            # ключевые пункты ВВИ: «Доступность 99,982%», «SSD до 5000 IOPS на 1 Тб»,
            # «SSD до 5 мс», «100 Мбит/с гарантированно», «не менее 2U/300 Вт» и др.
            # Если текст Heading'а — это требование (содержит должен/обязан/числа
            # с единицами) и не похож на голый раздел («Требования к ...»), то
            # одновременно регистрируем requirement И обновляем контекст.
            is_actual_requirement = (
                level >= 2
                and _is_requirement_text(text)
                and not _is_section_title_only(text)
                and not _is_intro_requirement(text)
            )
            if is_actual_requirement:
                short_context = _short_context_label(path_stack) or _context_label(path_stack)
                # Section: используем родительский контекст, без добавления
                # heading-текста (он попадёт в req.text).
                section = short_context or "Без раздела"
                if len(section) > 120:
                    section = section[:117].rstrip() + "…"
                # Heading-требование часто короткое и непонятное без контекста.
                # Например «SSD – до 5 000 IOPS на 1 Тб.» — без родительской
                # фразы «Производительность каждого диска Виртуальной машины:»
                # неясно, про что речь. Но intro нужен ТОЛЬКО если текст не
                # самодостаточный — без явного субъекта-исполнителя.
                # «Исполнитель должен обеспечить Доступность 99,982%» — в
                # intro не нуждается, а вот «SSD – до 5 мс» — нуждается.
                req_text = text
                if _heading_needs_parent_intro(text):
                    parent_intro = ""
                    if current_intro and current_intro.rstrip(":").strip() != text.strip():
                        parent_intro = current_intro
                    if not parent_intro:
                        for parent_level in sorted(path_stack):
                            if parent_level >= level:
                                break
                            candidate = path_stack.get(parent_level) or ""
                            if _is_intro_requirement(candidate) or candidate.rstrip().endswith(":"):
                                parent_intro = candidate
                    if parent_intro:
                        req_text = f"{parent_intro.rstrip(':').strip()}: {text}"
                add_requirement(section, req_text, force=True)
            # Heading-уровни 1-2 — это смена крупного раздела. Поэтому сбрасываем
            # current_intro, чтобы intro из предыдущего раздела не «прилип»
            # к bullet'ам нового раздела (например, intro из 9.9.3 «защита
            # информации» прилипал к bullet'ам гарантийного техсопровождения).
            if level <= 2:
                current_intro = ""
            update_context(level, text)
            current_intro = text if _is_intro_requirement(text) else current_intro
            continue

        pseudo_level = _pseudo_heading_level(text)
        if pseudo_level:
            update_context(pseudo_level, text)
            current_intro = text if _is_intro_requirement(text) else ""
            continue

        context = _context_label(path_stack)
        if _is_terms_context(context or text):
            current_intro = ""
            continue

        if parsed_section:
            section, rest = parsed_section
            if _is_section_title_only(rest) or _pseudo_heading_level(rest):
                update_context(10 + section.count("."), f"{section} {rest}")
                current_intro = rest if _is_intro_requirement(rest) else ""
                continue
            # Если фраза заканчивается на «:» — это intro для последующих
            # bullet'ов (например «9.9.3 В рамках оказания услуг… в том
            # числе:»). Сами bullet'ы уже придут отдельными требованиями
            # с приклеенным intro. Дублировать intro как отдельный пункт
            # не нужно.
            if _is_intro_requirement(rest):
                update_context(10 + section.count("."), f"{section} {rest}")
                current_intro = rest
                continue
            add_requirement(section, rest)
            current_intro = rest if _is_intro_requirement(rest) else ""
            continue

        if is_list:
            if text.endswith(":"):
                current_intro = text
                continue
            short_context = _short_context_label(path_stack) or context
            if current_intro and not _is_terms_context(current_intro):
                local_n = next_local_index(short_context)
                section = _synthetic_section(short_context, block_index, f"пункт {local_n}")
                parent = current_intro.rstrip(":")
                add_requirement(section, f"{parent}: {text}")
            elif _is_requirement_text(text):
                local_n = next_local_index(short_context)
                section = _synthetic_section(short_context, block_index, f"пункт {local_n}")
                add_requirement(section, text)
            continue

        if _is_intro_requirement(text):
            current_intro = text
            continue

        if _is_requirement_text(text) and not _is_section_title_only(text):
            short_context = _short_context_label(path_stack) or context
            local_n = next_local_index(short_context + "/blk")
            section = _synthetic_section(short_context, block_index, f"абзац {local_n}")
            add_requirement(section, text)
            current_intro = text if _is_intro_requirement(text) else ""
        elif level:
            current_intro = text if _is_intro_requirement(text) else ""

    detected = _dedupe_requirements(requirements)
    result = _cap_requirements(detected, settings)
    _set_extraction_summary(document, "structured_fast", detected, result, settings)
    logger.info("Structured parser extracted %d requirements", len(result))
    return result


def _is_probably_heading(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) > 140:
        return False
    if stripped.endswith((".", ";", ":")):
        return False
    words = stripped.split()
    return len(words) <= 10


def _dedupe_requirements(requirements: list[Requirement]) -> list[Requirement]:
    seen = set()
    result = []
    for req in requirements:
        key = re.sub(r"[^a-zа-яё0-9]+", " ", req.text.lower()).strip()
        key = key[:500]
        if key in seen:
            continue
        seen.add(key)
        result.append(req)
    for idx, req in enumerate(result, start=1):
        req.id = idx
    return result


def _requirement_importance(req: Requirement) -> int:
    text = " ".join([req.section or "", req.text or "", req.tables or ""]).lower()
    score = 0
    high_value_terms = (
        "vmware",
        "vcloud",
        "nsx",
        "vrealize",
        "veeam",
        "usergate",
        "ngfw",
        "waf",
        "ddos",
        "s3",
        "фстэк",
        "фсб",
        "тзки",
        "уз-2",
        "152-фз",
        "персональн",
        "аттестац",
        "сертификат",
        "реестр российского по",
        "реестр отечественного по",
        "точек очистки",
        "websocket",
        "bgp",
        "ssl гост",
        "rest api",
        "личный кабинет",
        "ja3",
        "captcha",
        "let’s encrypt",
        "lets encrypt",
        "точки очистки",
        "точек очистки",
        "multicast",
        "siem",
        "soc",
        "tier3",
        "tier 3",
    )
    for term in high_value_terms:
        if term in text:
            score += 3
    numeric_with_units = (
        r"\b\d+\s*(?:мбит/с|гбит/с|гб|gb|тб|tb|мб|mb|rps|pps|bps|час|часов|мин|%)",
        r"\b\d+\s*(?:vcpu|vram|vhdd)",
    )
    for pattern in numeric_with_units:
        score += min(len(re.findall(pattern, text)), 6)
    if req.section.startswith("3"):
        score += 2
    if any(req.section.startswith(prefix) for prefix in ("3.6", "3.7", "3.8", "3.9", "3.10")):
        score += 4
    if "состав услуг" in text or "приложение №1" in text:
        score += 2
    if "термины" in text or "определения" in text:
        score -= 8
    if _is_definition_only(req.text):
        score -= 10
    return score


def _cap_requirements(requirements: list[Requirement], settings: RuntimeSettings) -> list[Requirement]:
    max_items = max(1, settings.parser_fast_max_requirements)
    if len(requirements) <= max_items:
        return requirements

    priority = {
        "technical": 0,
        "security": 1,
        "sla": 2,
        "legal": 3,
        "commercial": 4,
        "other": 5,
    }
    sorted_items = sorted(
        requirements,
        key=lambda req: (
            priority.get(req.category, 9),
            -_requirement_importance(req),
            0 if len(req.text) > 60 else 1,
            req.id,
        ),
    )
    kept = sorted(sorted_items[:max_items], key=lambda req: req.id)
    for idx, req in enumerate(kept, start=1):
        req.id = idx
    logger.info("Capped fast parser requirements from %d to %d", len(requirements), len(kept))
    return kept


def _extract_requirements_fast(document_text: str, settings: RuntimeSettings) -> list[Requirement]:
    """Fast local parser for numbered TZ clauses.

    It avoids LLM extraction by using stable section numbering from procurement TZs.
    """
    lines = [_normalize_text(line) for line in document_text.splitlines()]
    lines = [line for line in lines if line]
    candidates: list[tuple[str, str]] = []
    current_section = ""
    current_parts: list[str] = []

    def flush_current() -> None:
        nonlocal current_section, current_parts
        if not current_section or not current_parts:
            current_section = ""
            current_parts = []
            return
        text = _normalize_text(" ".join(current_parts))
        if len(text) >= 25 and not _is_probably_heading(text):
            candidates.append((current_section, text))
        current_section = ""
        current_parts = []

    for line in lines:
        parsed = _section_from_line(line)
        if parsed:
            flush_current()
            current_section, first_text = parsed
            current_parts = [first_text]
            continue
        if current_section:
            if _section_from_line(line):
                flush_current()
            elif len(line) > 8:
                current_parts.append(line)
    flush_current()

    requirements = [
        Requirement(
            id=index,
            section=section,
            text=text,
            category=_category_from_text(text),
            tables="",
        )
        for index, (section, text) in enumerate(candidates, start=1)
    ]
    requirements = _dedupe_requirements(requirements)
    requirements = _cap_requirements(requirements, settings)
    logger.info("Fast parser extracted %d requirements", len(requirements))
    return requirements


# ---------------------------------------------------------------------------
# Quality filters — пост-обработка для удаления заголовков-маркеров,
# гранулярных полей форм, дублей и мета-вступлений. Кросс-валидировано
# на 4 ТЗ (Калугаинформтех, НИИОЗМ, Мосгорэкспертиза, Татспиртпром):
# 0 потерь важных терминов (ФСТЭК, К1, vCPU, RAM, IOPS, SLA и т.п.).
# ---------------------------------------------------------------------------

_FILTER_IMPORTANT = (
    'фстэк', 'фсб', 'к1', 'к2', 'кии', 'уз-1', 'уз-2', '152-фз', '187-фз',
    'аттестат', 'сертификат', 'лицензи',
    'vcpu', 'cpu', 'ram', 'озу', 'диск',
    'ssd', 'sas', 'iops', 'мбит', 'гбит',
    'sla', 'доступност', 'uptime', '99.9', '99,9', '99,98',
    'astra', 'red os', 'роскомнадзор',
    # S3 / объектное хранилище — критичные технические термины,
    # которые часто идут короткими строками bullet-списка (Татспиртпром:
    # «Поддержка SFTP;», «Доступ по протоколу S3»). Без whitelist'а они
    # выпадают по фильтру A (короткие без модальности).
    's3', 'sftp', 'obs', 'worm', 'rest api', 'object lock', 'versioning',
    'multipart', 'bucket', 'бакет', 'корзин',
    # VMware / OpenStack ключевые термины
    'vmware', 'vcloud', 'nsx', 'openstack', 'evs', 'ecs', 'kubernetes',
    'k8s', 'docker', 'managed',
    # Сеть и безопасность
    'ngfw', 'waf', 'ddos', 'ids', 'ips', 'siem', 'soc',
    'tier iii', 'tier 3', 'uptime institute',
)

_FILTER_MODAL_RE = re.compile(
    r'\b(должен|должна|должно|должны|обязан|обязана|обязаны|обязано|'
    r'обеспеч|предостав|гарантир|поддержив|использов|применя|содержат|'
    r'осуществл|выполн|допуска|разреш|соответствов|реализ)',
    re.IGNORECASE,
)

_FILTER_NUMERIC_RE = re.compile(r'\d')

# Маркеры табличных строк (SLA-шкалы, ресурсы, штрафы) — для них дедуп
# НЕ применяется: они различаются числами и значениями, а не словами.
_FILTER_TABLE_MARKERS = (
    'приоритет:', 'время реакции', 'время решения',
    'размер компенсации', 'размер неустойки', 'размер штрафа',
    'окно предоставления сервиса', 'фактическая доступность',
    'коэффициент доступности', 'отчетный период',
    'наименование услуги:', 'параметр услуги и единица',
    'единица измерения:', 'параметр услуги', 'наименование:',
)

_FILTER_INTRO_MARKERS = (
    'настоящ', 'в данном разделе', 'в дальнейшем', 'далее по тексту',
)

_FILTER_CUSTOMER_PREFIXES = (
    'Заказчик ', 'Заказчик,', 'Заказчиком ',
    'Сторона Заказчика ', 'Стороны ',
)

_FILTER_NUMBER_RE = re.compile(r'\d[\d,.\s]*')

# H: Маркеры процедурных разделов закупки (44-ФЗ / 223-ФЗ / 218-ФЗ).
# Эти пункты НЕ оценивают технические возможности Cloud.ru — это
# административно-правовая обвязка тендера: коды ОКПД, цена контракта,
# обеспечение заявки, антикоррупция, требования к участникам закупки и т.п.
# Их оставляем в отчёте как отдельную категорию `procedural`, но исключаем
# из знаменателя compliance — иначе compliance%-я искусственно завышается.
_FILTER_PROCEDURAL_MARKERS = (
    # Классификаторы закупки
    'окпд2', 'окпд 2', 'код окпд', 'оквэд2', 'оквэд 2', 'код оквэд',
    # Цена контракта и обеспечение заявки
    'нмцк', 'начальная (максимальная) цена', 'начальной (максимальной) цены',
    'начальная максимальная цена', 'обеспечение заявки',
    'обеспечение исполнения договора', 'обеспечение исполнения контракта',
    'обеспечение гарантийных обязательств',
    'банковская гарантия', 'размер обеспечения',
    'извещение о закупке', 'извещение об осуществлении закупки',
    'участник закупки', 'единственный участник', 'участники конкурса',
    'идентификационный код закупки', 'икз',
    'идентификация заказчика', 'идентификация исполнителя',
    'идентификация участников',
    'реквизиты сторон', 'банковские реквизиты',
    'порядок заключения договора', 'порядок подписания контракта',
    'срок заключения договора', 'срок подписания контракта',
    'место исполнения контракта' ,
    # Антикоррупционные / 273-ФЗ
    'антикоррупционн', '273-фз', 'противодействие коррупции',
    'конфликт интересов',
    # Закупки 44/223/218-ФЗ
    '44-фз', '223-фз', '218-фз',
    'единая информационная система', 'еис',
    'госзакупки', 'электронная торговая площадка',
    # Декларации участника
    'декларация о соответствии', 'декларация участника закупки',
    'требования к участникам', 'соответствие требованиям статьи 31',
    # Бухгалтерия / документооборот тендера
    'счет-фактур', 'универсальный передаточный документ', 'упд',
    'акт сдачи-приёмки услуг', 'акт сдачи-приемки',
    # Сроки оказания услуг — если только дата без технической метрики
)
# Маркеры, которые ВЫКЛЮЧАЮТ procedural-флаг даже если другой маркер был найден.
# Если есть техническая или SLA-метрика, пункт остаётся техническим.
_FILTER_PROCEDURAL_OVERRIDE_TECH = (
    'sla', 'доступност', 'rpo', 'rto', 'iops', 'мбит', 'гбит',
    'vcpu', 'cpu', 'ram', 'озу', 'диск',
    'фстэк', 'фсб', 'к1', 'уз-1', 'уз-2', '152-фз', 'кии',
    'аттестат', 'сертификат',
    'vmware', 'vcloud', 'nsx', 'openstack', 's3', 'ngfw', 'waf', 'ddos',
    'личный кабинет', 'мониторинг', 'резервн', 'backup',
)


def _filter_is_procedural(text: str) -> bool:
    """H: True, если текст требования — это процедурный пункт закупки
    (ОКПД, цена, обеспечение заявки, антикоррупция, идентификация).
    Override: если в тексте есть техническая метрика — это НЕ процедурное,
    оцениваем по содержанию."""
    if not text:
        return False
    t = text.lower()
    # Override: технический контент важнее процедурного.
    if any(tech in t for tech in _FILTER_PROCEDURAL_OVERRIDE_TECH):
        return False
    return any(marker in t for marker in _FILTER_PROCEDURAL_MARKERS)


def _filter_has_important(text: str) -> bool:
    t = (text or '').lower()
    return any(kw in t for kw in _FILTER_IMPORTANT)


def _filter_is_table_row(text: str) -> bool:
    """True если требование выглядит как строка таблицы (SLA, шкала
    штрафов, ресурсы). Такие строки исключаются из дедупа."""
    t = (text or '').lower()
    return any(m in t for m in _FILTER_TABLE_MARKERS)


def _filter_short_no_modal(r: 'Requirement') -> bool:
    """A: убрать короткие пункты без модальных глаголов и без чисел
    (вроде 'Имя и фамилию.', 'Запрос на обслуживание.')."""
    t = (r.text or '').strip()
    if len(t) >= 50:
        return False
    if _FILTER_MODAL_RE.search(t):
        return False
    if _FILTER_NUMERIC_RE.search(t):
        return False
    if _filter_has_important(t):
        return False
    return True


def _filter_list_header(r: 'Requirement') -> bool:
    """B: убрать заголовки/intro-фразы списков и осколочные bullet-фрагменты.

    Патч 6 (ZK10). Расширенная логика:
    - `text:` без модальных и чисел — как раньше.
    - `<Subject> модальный:` (≤ 5 слов) — это тоже intro («Исполнитель
      обязан:», «Отчет должен содержать:»). Раньше пропускалось из-за
      наличия модального глагола и попадало в анализ как полноценное
      требование.
    - короткий фрагмент с `;`-терминатором без модальных и чисел —
      обрывок списка («Отражение DDoS-атак;») без самостоятельного смысла;
      родительский intro обычно даёт контекст соседним bullet'ам.
    """
    t = (r.text or '').strip()
    if not t:
        return False
    word_count = len(t.split())

    # 1. Короткий ;-терминированный фрагмент без модальности и чисел.
    if t.endswith(';') and len(t) <= 60 and word_count <= 6:
        if not _FILTER_MODAL_RE.search(t) and not _FILTER_NUMERIC_RE.search(t):
            return True

    if not t.endswith(':'):
        return False

    # Original logic для ':' без модальных и чисел.
    if not _FILTER_MODAL_RE.search(t):
        if _FILTER_NUMERIC_RE.search(t):
            return False
        if _filter_has_important(t):
            return False
        if len(t) > 80:
            return False
        return True

    # 2. «<Subject> модальный:» — короткое intro с модальным глаголом.
    # Признак: ≤ 5 слов, заканчивается на ":". Например «Исполнитель обязан:»,
    # «Отчет должен содержать:».
    if word_count <= 5 and len(t) <= 60:
        return True
    return False


def _filter_meta_intro(r: 'Requirement') -> bool:
    """G: убрать мета-вступления типа 'Настоящий регламент устанавливает...'
    без модальных глаголов."""
    t = (r.text or '').lower().strip()
    if len(t) > 200:
        return False
    if not any(m in t[:50] for m in _FILTER_INTRO_MARKERS):
        return False
    return not _FILTER_MODAL_RE.search(t)


def _filter_customer_obligation(r: 'Requirement') -> bool:
    """F: убрать обязательства Заказчика (не требования к Cloud.ru)."""
    t = (r.text or '').strip()
    if not any(t.startswith(s) for s in _FILTER_CUSTOMER_PREFIXES):
        return False
    if not _FILTER_MODAL_RE.search(t[:80]):
        return False
    return True


def _filter_words_for_jaccard(text: str) -> set:
    return set(re.findall(r'[A-Za-zА-Яа-я]{4,}', (text or '').lower()))


def _filter_jaccard(a: str, b: str) -> float:
    wa = _filter_words_for_jaccard(a)
    wb = _filter_words_for_jaccard(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _filter_extract_numbers(text: str) -> set:
    """Извлекает все числовые значения для проверки 'разные числа = разные
    требования'."""
    nums = set()
    for m in _FILTER_NUMBER_RE.findall(text or ''):
        n = m.replace(' ', '').replace(',', '.').strip('.')
        if n and any(c.isdigit() for c in n):
            nums.add(n)
    return nums


def _filter_different_numbers(a: str, b: str) -> bool:
    na = _filter_extract_numbers(a)
    nb = _filter_extract_numbers(b)
    if not na or not nb:
        return False
    if len(na | nb) >= 2 and (na - nb or nb - na):
        return True
    return False


def _filter_tail_diff(a: str, b: str, ratio: float = 0.3) -> float:
    """Сравнение хвостов: если последние 30% сильно различаются, тексты
    разные несмотря на общий префикс."""
    a_tail = a[int(len(a) * (1 - ratio)):]
    b_tail = b[int(len(b) * (1 - ratio)):]
    return _filter_jaccard(a_tail, b_tail)


_SECTION_NUMBER_RE = re.compile(r'^\s*(\d+(?:\.\d+){0,6})')


def _filter_section_number(section: str) -> str:
    """Извлекает иерархический номер раздела из строки section.

    Поддерживает форматы:
      '3.7'                          → '3.7'
      '3.7 / пункт списка 53'        → '3.7'
      '3.7.4'                        → '3.7.4'
      'Требования к Услугам > ...'   → ''   (нет нумерации)
    """
    if not section:
        return ""
    m = _SECTION_NUMBER_RE.match(section)
    return m.group(1) if m else ""


def _filter_sections_belong_to_same_clause(a: str, b: str) -> bool:
    """True, если две секции относятся к одному и тому же пункту ТЗ.

    Используется как защита от ложного слияния похожих требований из
    соседних подразделов (например, 3.7 «8 точек очистки» и 3.8 «UserGate»).
    Считаем «один пункт», если:
      • один номер — префикс другого ('3.7' и '3.7.4' → один пункт);
      • либо у обоих секций нет номеров и текстовые префиксы совпадают.
    """
    num_a = _filter_section_number(a)
    num_b = _filter_section_number(b)
    if num_a and num_b:
        if num_a == num_b:
            return True
        # Префиксная проверка с границей на точке, чтобы '3.7' не считалось
        # префиксом '3.70'.
        return (num_a + ".").startswith(num_b + ".") or (num_b + ".").startswith(num_a + ".")
    if not num_a and not num_b:
        # Нет номеров — сравниваем по текстовым префиксам breadcrumb.
        prefix_a = (a or "").split(" / ")[0].strip().lower()
        prefix_b = (b or "").split(" / ")[0].strip().lower()
        if not prefix_a or not prefix_b:
            return True  # совсем нет информации → не блокируем
        return prefix_a == prefix_b
    # Одна секция с номером, другая без — это разные пункты.
    return False


def _filter_safe_dedup(
    reqs: list,
    threshold: float = 0.7,
    min_len: int = 150,
) -> tuple[list, list]:
    """D-safe: дедуп почти-идентичных требований. 4 защиты:
       1. Пропускаем строки таблиц (SLA, шкалы, ресурсы).
       2. Не дедуплицируем при разных числовых значениях.
       3. Не дедуплицируем при сильно отличающихся хвостах.
       4. Не дедуплицируем требования из разных пунктов ТЗ (3.7 vs 3.8) —
          иначе теряем клаузы из соседних подразделов.
    Возвращает (keep, dropped).
    """
    processed = set()
    keep_indices = list(range(len(reqs)))
    drop_set = set()
    for i, r1 in enumerate(reqs):
        if i in processed:
            continue
        if len(r1.text) < min_len:
            continue
        if _filter_is_table_row(r1.text):
            continue
        for j in range(i + 1, len(reqs)):
            if j in processed:
                continue
            r2 = reqs[j]
            if len(r2.text) < min_len:
                continue
            if _filter_is_table_row(r2.text):
                continue
            if _filter_jaccard(r1.text, r2.text) < threshold:
                continue
            if _filter_different_numbers(r1.text, r2.text):
                continue
            if _filter_tail_diff(r1.text, r2.text) < 0.5:
                continue
            # Защита 4: разные section_path = разные клаузы ТЗ, не дедуп.
            if not _filter_sections_belong_to_same_clause(r1.section, r2.section):
                continue
            # Дубль — удаляем j, оставляем представителя i.
            drop_set.add(j)
            processed.add(j)
        processed.add(i)
    keep = [r for k, r in enumerate(reqs) if k not in drop_set]
    drop = [r for k, r in enumerate(reqs) if k in drop_set]
    return keep, drop


# v15 (S4): паттерны явных ссылок на нормативные документы.
# Используется в _collapse_normative_list_requirements.
_NORMATIVE_DOC_RE = re.compile(
    r"^\s*[\"«»«»]?\s*("
    r"гост(?:\s+р)?\b"               # ГОСТ, ГОСТ Р
    r"|приказ(?:а|ом)?\b"             # Приказ ФСБ/ФСТЭК/Минцифры
    r"|указ(?:а|ом)?\s+президент"
    r"|постановлени[еия]\s+правительств"
    r"|пп\s*рф\b"
    r"|распоряжени[еия]\s+правительств"
    r"|федеральн(?:ый|ого)\s+закон"
    r"|ф\s*з\s+от\b|\bфз\s*[№\-]?\s*\d"
    r"|санпин\b|сп\s+\d|сн\s+\d"
    r"|свод\s+правил"
    r"|стандарт\b"
    r"|приказ(?:у)?\s+минцифр"
    r"|приказ(?:у)?\s+минфин"
    r"|приказ(?:у)?\s+фст"
    r"|приказ(?:у)?\s+фсб"
    r"|регламент\s+правительств"
    r")",
    re.IGNORECASE,
)


def _looks_like_normative_doc_reference(tail: str) -> bool:
    """Эвристика: фрагмент похож на чистую ссылку на НПА.

    Истина, если строка:
    • явно начинается с маркера ГОСТ/Приказ/Указ/ПП/ФЗ/СанПиН/СП,
    • или содержит характерный паттерн «от DD.MM.YYYY N <num>».
    """
    if not tail:
        return False
    t = tail.strip().strip('".,;:')
    if _NORMATIVE_DOC_RE.search(t):
        return True
    # Дата + номер: «от 10.07.2014 N 378»
    if re.search(r"от\s+\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4}\s+(?:N|№|\.|N\.)\s*\d", t, re.IGNORECASE):
        return True
    return False


def _split_intro_and_tail(text: str) -> tuple[str, str]:
    """Разделяет «<intro>: <tail>» по первому двоеточию. Если двоеточия
    нет — intro='' и tail=text."""
    if ":" not in text:
        return "", text
    intro, tail = text.split(":", 1)
    return intro.strip(), tail.strip()


def _section_root(section: str) -> str:
    """Нормализует section для сравнения соседних bullet-пунктов.

    Парсер прицепляет к section хвост вида `/ пункт NN` или `/ пункт списка NN`,
    из-за чего соседние bullet'ы выглядят как разные секции. Для склейки
    нормативного листа нам важна именно «корневая» секция документа."""
    if not section:
        return ""
    for sep in (" / пункт списка ", " / пункт ", " / блок "):
        if sep in section:
            return section.split(sep, 1)[0].strip()
    return section.strip()


def _collapse_normative_list_requirements(reqs: list) -> list:
    """v15 (S4): склеивает последовательные requirements'ы, у которых:
      • одинаковый section,
      • одинаковый intro-prefix (часть до первого «:»),
      • tail (часть после «:») — это чистая ссылка на нормативный документ
        (ГОСТ Р…, Приказ ФСБ N…, Указ Президента, ПП РФ, ФЗ, СанПиН…).

    Цель — устранить «фантомные требования» из bullet-списков нормативной
    базы. Без склейки 17 пунктов листа ГОСТов превращаются в 17 NC-вердиктов
    и раздувают знаменатель compliance%. С склейкой это один пункт
    «соответствие нормативной базе».

    Безопасность:
      • Склеиваются ТОЛЬКО соседние requirements (без перепрыгивания между
        несвязанными секциями).
      • Если хотя бы один tail в группе НЕ выглядит как НПА — группа не
        склеивается (защита от потери реальных требований).
      • Группа должна быть из ≥3 элементов с одинаковым intro — точечные
        одиночные ссылки на НПА остаются как есть.
    """
    if not reqs or len(reqs) < 3:
        return reqs

    result: list = []
    i = 0
    collapsed_groups = 0
    collapsed_dropped = 0
    while i < len(reqs):
        ri = reqs[i]
        intro_i, tail_i = _split_intro_and_tail(ri.text or "")
        # Кандидат для группы только если у текущего есть intro и tail —
        # нормативный документ.
        if intro_i and _looks_like_normative_doc_reference(tail_i):
            j = i + 1
            tails = [tail_i]
            root_i = _section_root(ri.section or "")
            while j < len(reqs):
                rj = reqs[j]
                if _section_root(rj.section or "") != root_i:
                    break
                intro_j, tail_j = _split_intro_and_tail(rj.text or "")
                if intro_j != intro_i:
                    break
                if not _looks_like_normative_doc_reference(tail_j):
                    break
                tails.append(tail_j)
                j += 1
            if len(tails) >= 3:
                # Склеиваем в один req: сохраняем id, section, category первого,
                # а tails объединяем перечислением.
                merged_text = f"{intro_i}: " + "; ".join(tails)
                ri.text = merged_text
                # category — оставляем 'legal' / 'security' / category первого,
                # это уже учитывается анализатором.
                result.append(ri)
                collapsed_groups += 1
                collapsed_dropped += (j - i - 1)
                i = j
                continue
        result.append(ri)
        i += 1
    if collapsed_groups:
        logger.info(
            "Collapse normative lists: %d группы, удалено %d дублирующих требований",
            collapsed_groups, collapsed_dropped,
        )
    return result


def _apply_quality_filters(reqs: list) -> list:
    """Post-processing после извлечения. Удаляет 5 типов «не-требований»
    и маркирует 1 тип «процедурных» (H):
       A — короткие без модальных глаголов и чисел  (DROP)
       B — заголовки-маркеры списков (text:)        (DROP)
       D — дубли блоков с защитами (SLA, числа, хвост) (DROP)
       F — обязательства Заказчика                  (DROP)
       G — мета-вступления                          (DROP)
       H — процедурные пункты закупки (ОКПД, цена)  (MARK as category='procedural')
       + v15 (S4): collapse normative-document bullet lists

    Безопасность проверена на 4 ТЗ: 0 потерь важных терминов.
    """
    # v15 (S4): сначала склеиваем bullet-списки НПА в одно требование,
    # потом применяем остальные фильтры.
    reqs = _collapse_normative_list_requirements(reqs)
    if not reqs:
        return reqs

    before = len(reqs)
    drop_ids = set()
    a_n = b_n = f_n = g_n = h_n = 0
    for r in reqs:
        if _filter_short_no_modal(r):
            drop_ids.add(id(r))
            a_n += 1
        elif _filter_list_header(r):
            drop_ids.add(id(r))
            b_n += 1
        elif _filter_customer_obligation(r):
            drop_ids.add(id(r))
            f_n += 1
        elif _filter_meta_intro(r):
            drop_ids.add(id(r))
            g_n += 1
        else:
            # H: помечаем процедурные пункты НОВОЙ категорией, не удаляем.
            # Анализатор пропустит их через LLM и создаст synthetic-verdict
            # `out_of_scope`. Из compliance% они исключаются.
            joined = " ".join([r.section or "", r.text or "", r.tables or ""])
            if _filter_is_procedural(joined):
                r.category = "procedural"
                h_n += 1

    cleaned = [r for r in reqs if id(r) not in drop_ids]
    cleaned, dedup_drops = _filter_safe_dedup(cleaned)
    d_n = len(dedup_drops)

    after = len(cleaned)
    logger.info(
        "Quality filters: %d → %d (A=%d, B=%d, D=%d, F=%d, G=%d, H-marked=%d)",
        before, after, a_n, b_n, d_n, f_n, g_n, h_n,
    )
    return cleaned


def extract_requirements(
    document: str | ParsedDocument,
    max_chunk_size: int | None = None,
    settings: RuntimeSettings | dict | None = None,
) -> list[Requirement]:
    """Extract structured requirements from document text using LLM.

    Splits long documents into chunks and processes each separately.
    """
    runtime_settings = build_runtime_settings(settings)
    document_text = document.full_text if isinstance(document, ParsedDocument) else document

    if runtime_settings.parser_mode in {"fast", "hybrid"}:
        if isinstance(document, ParsedDocument) and document.blocks:
            fast_requirements = _extract_requirements_from_blocks(document, runtime_settings)
        else:
            fast_requirements = _extract_requirements_fast(document_text, runtime_settings)

        # Патч 1 (ZK10). Раньше при parser_mode=="fast" мы возвращали
        # fast_requirements ВСЕГДА — даже если их 0. Это и есть «тихий провал»
        # на закупке 0318100047925000210: 34 КБ исходного текста → 0
        # требований без единого warning'а. Теперь:
        #   * если требований достаточно (≥ min) — отдаём fast (как раньше);
        #   * если их мало и parser_fallback_to_llm=True — громко логируем,
        #     помечаем metadata.requirements_extraction.fallback_triggered и
        #     уходим в LLM-ветку ниже;
        #   * если fallback отключён — возвращаем sparse-результат, но
        #     помечаем metadata.warning и логируем WARNING, чтобы не было
        #     «100% completed, 0 требований».
        min_req = runtime_settings.parser_fast_min_requirements
        if len(fast_requirements) >= min_req:
            return _apply_quality_filters(fast_requirements)

        style_div = _style_diversity(document if isinstance(document, ParsedDocument) else None)
        if runtime_settings.parser_fallback_to_llm:
            logger.warning(
                "Fast parser returned %d requirements (< min %d). "
                "Falling back to LLM extraction. style_diversity=%d, parser_mode=%s",
                len(fast_requirements), min_req, style_div, runtime_settings.parser_mode,
            )
            if isinstance(document, ParsedDocument):
                doc_meta = document.metadata.setdefault("requirements_extraction", {})
                doc_meta["fallback_triggered"] = "llm"
                doc_meta["fallback_reason"] = (
                    f"structured_fast={len(fast_requirements)} < min={min_req}"
                    f" (style_diversity={style_div})"
                )
        else:
            logger.warning(
                "Fast parser returned %d requirements (< min %d), "
                "parser_fallback_to_llm=False. Returning sparse result as-is. "
                "style_diversity=%d",
                len(fast_requirements), min_req, style_div,
            )
            if isinstance(document, ParsedDocument):
                doc_meta = document.metadata.setdefault("requirements_extraction", {})
                doc_meta["warning"] = (
                    f"structured_fast returned {len(fast_requirements)} "
                    f"(< min {min_req}); fallback disabled, style_diversity={style_div}"
                )
            return _apply_quality_filters(fast_requirements)

    chunks = _split_text(document_text, max_chunk_size or runtime_settings.parser_chunk_size)
    all_requirements: list[Requirement] = []
    global_id = 1
    chunk_items: dict[int, list[dict]] = {}

    max_workers = max(1, min(runtime_settings.parser_concurrency, len(chunks)))
    logger.info("Extracting requirements from %d chunks (parallel=%d)", len(chunks), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_extract_chunk_items, index, len(chunks), chunk, runtime_settings)
            for index, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            chunk_index, items = future.result()
            chunk_items[chunk_index] = items

    for i in range(len(chunks)):
        items = chunk_items.get(i, [])
        for item in items:
            text = _extract_field(item, ["text", "requirement", "requirement_text",
                                         "description", "content", "требование",
                                         "текст", "текст_требования"])
            section = _extract_field(item, ["section", "paragraph", "point", "clause",
                                            "пункт", "раздел", "номер", "номер_пункта",
                                            "number", "item_number", "section_number"])
            category = _extract_field(item, ["category", "категория", "type", "тип"])
            tables = _extract_field(item, ["tables", "table", "таблица", "таблицы"])

            # Validate category
            valid_categories = {"technical", "sla", "legal", "commercial", "security", "other"}
            if category.lower() not in valid_categories:
                category = "other"
            else:
                category = category.lower()

            # Skip items with no meaningful text
            if not text or len(text.strip()) < 3:
                logger.debug("Skipping item with empty text: %s", item)
                continue

            req = Requirement(
                id=global_id,
                section=section,
                text=text,
                category=category,
                tables=tables,
            )
            all_requirements.append(req)
            global_id += 1

    logger.info("Extracted %d requirements total", len(all_requirements))

    # Патч 1 (ZK10): если попали сюда после fallback от structured_fast,
    # либо изначально шли по LLM-ветке без structured_fast — обновим/проставим
    # metadata, чтобы в отчёте честно отображался реальный parser.
    if isinstance(document, ParsedDocument):
        doc_meta = document.metadata.setdefault("requirements_extraction", {})
        previous_parser = doc_meta.get("parser", "")
        if previous_parser:
            doc_meta["parser"] = f"{previous_parser}→llm_fallback"
        else:
            doc_meta["parser"] = "llm"
        doc_meta["requirements_returned_llm"] = len(all_requirements)
        if "style_diversity" not in doc_meta:
            doc_meta["style_diversity"] = _style_diversity(document)
        if "block_count" not in doc_meta:
            doc_meta["block_count"] = len(document.blocks)
        if "table_count" not in doc_meta:
            doc_meta["table_count"] = len(document.tables)
    return _apply_quality_filters(all_requirements)


def _split_text(text: str, max_size: int) -> list[str]:
    """Split text into chunks, trying to break at paragraph boundaries."""
    if len(text) <= max_size:
        return [text]
    chunks = []
    current = ""
    for paragraph in text.split("\n"):
        if len(current) + len(paragraph) + 1 > max_size and current:
            chunks.append(current)
            current = paragraph
        else:
            current = current + "\n" + paragraph if current else paragraph
    if current:
        chunks.append(current)
    return chunks
