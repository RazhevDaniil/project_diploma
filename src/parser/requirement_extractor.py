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
    {
        "id": "vmware_vcloud",
        "label": "VMware vCloud Director / ВЦОД",
        "needles": ("vmware vcloud director", "vcloud director"),
        "critical": True,
    },
    {
        "id": "capacity",
        "label": "Количественные мощности ВЦОД",
        "needles": ("164", "656", "8 300", "8300", "7 000", "7000", "vram", "vcpu"),
        "critical": True,
    },
    {
        "id": "monitoring",
        "label": "vRealize Operations / мониторинг",
        "needles": ("vrealize operations", "расширенного мониторинга", "мониторинг"),
        "critical": False,
    },
    {
        "id": "s3",
        "label": "S3-хранилище и лимиты API",
        "needles": ("s3 совместим", "s3 api", "5 гб", "5 тб", "10 000", "трех зонах доступности"),
        "critical": True,
    },
    {
        "id": "ddos_waf",
        "label": "DDoS/WAF",
        "needles": ("ddos", "waf", "точек очистки", "защита web-приложений", "защита веб-приложений"),
        "critical": True,
    },
    {
        "id": "ddos_waf_advanced",
        "label": "DDoS/WAF расширенные параметры",
        "needles": ("ja3", "websocket", "let’s encrypt", "lets encrypt", "50мс", "50 мс", "captcha"),
        "critical": False,
    },
    {
        "id": "ngfw_usergate",
        "label": "UserGate / NGFW",
        "needles": ("usergate", "ngfw", "межсетевого экрана нового поколения", "межсетевому экрану"),
        "critical": True,
    },
    {
        "id": "fstec",
        "label": "ФСТЭК / отечественное ПО",
        "needles": ("фстэк", "реестр российского по", "реестр отечественного по", "сертификат соответствия"),
        "critical": True,
    },
    {
        "id": "uz2_datacenter",
        "label": "ЦОД РФ / УЗ-2 / аттестация",
        "needles": ("уз-2", "уровня защищенности не ниже", "аттестация цод", "адрес нахождения цод"),
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
    result = call_llm_json(prompt, system_prompt=system_prompt, max_tokens=8000, settings=settings)

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
    return any(_normalize_blob(needle) in blob for needle in needles)


def _key_signal_coverage(source_text: str, requirements: list[Requirement]) -> list[dict]:
    source_blob = _normalize_blob(source_text)
    extracted_blob = _normalize_blob(
        "\n".join(" ".join([req.section or "", req.text or "", req.tables or ""]) for req in requirements)
    )
    coverage = []
    for signal in KEY_REQUIREMENT_SIGNALS:
        needles = tuple(signal["needles"])
        in_document = _signal_present(source_blob, needles)
        in_extracted = _signal_present(extracted_blob, needles)
        coverage.append(
            {
                "id": signal["id"],
                "label": signal["label"],
                "critical": bool(signal.get("critical")),
                "present_in_document": in_document,
                "present_in_extracted": in_extracted,
            }
        )
    return coverage


def _set_extraction_summary(
    document: ParsedDocument,
    parser_name: str,
    detected_requirements: list[Requirement],
    returned_requirements: list[Requirement],
    settings: RuntimeSettings,
) -> None:
    coverage = _key_signal_coverage(document.full_text, returned_requirements)
    missing_signals = [
        item["label"]
        for item in coverage
        if item["present_in_document"] and not item["present_in_extracted"]
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
        "table_count": len(document.tables),
        "block_count": len(document.blocks),
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
    return "list" in lowered or "спис" in lowered or "марк" in lowered


def _is_table_caption(text: str) -> bool:
    return bool(re.match(r"^табл(?:ица)?\.?\s*(?:№|n)?\s*\d+", text.strip(), re.IGNORECASE))


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


def _synthetic_section(context: str, block_index: int, suffix: str = "") -> str:
    label = context.strip() if context else "Без раздела"
    if len(label) > 120:
        label = label[:117].rstrip() + "..."
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
    return bool(re.search(r"\b(?:до|от)\s+\d", lowered))


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
    )
    return any(marker in blob for marker in table_markers)


def _table_requirement_text(caption: str, block: ParsedBlock) -> str:
    pairs = []
    for index, cell in enumerate(block.cells):
        if not cell:
            continue
        header = block.headers[index] if index < len(block.headers) and block.headers[index] else f"Колонка {index + 1}"
        pairs.append(f"{header}: {cell}")
    prefix = caption or f"Таблица {block.table_index}"
    return f"{prefix}. " + "; ".join(pairs)


def _extract_requirements_from_blocks(document: ParsedDocument, settings: RuntimeSettings) -> list[Requirement]:
    """Fast parser that uses DOCX structure: headings, lists and table rows."""
    requirements: list[Requirement] = []
    path_stack: dict[int, str] = {}
    current_intro = ""
    current_table_caption = ""
    current_table_index: int | None = None

    def update_context(level: int, label: str) -> None:
        normalized = _normalize_text(label)
        if not normalized:
            return
        for existing_level in list(path_stack):
            if existing_level >= level:
                del path_stack[existing_level]
        path_stack[level] = normalized

    def add_requirement(section: str, text: str, tables: str = "", force: bool = False) -> None:
        normalized = _normalize_text(text)
        if len(normalized) < 12 or (_is_probably_heading(normalized) and not force):
            return
        if _is_definition_only(normalized):
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
            if block.table_index != current_table_index:
                if block.caption:
                    current_table_caption = block.caption
                elif current_table_index is not None:
                    current_table_caption = ""
                current_table_index = block.table_index
            context = _context_label(path_stack)
            if _is_terms_context(context):
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
            section = table_label
            section = f"{section}, строка {block.row_index}"
            add_requirement(section, _table_requirement_text(table_label, block), force=True)
            continue

        if _is_table_caption(text):
            current_table_caption = text
            continue

        parsed_section = _section_from_line(text)
        style = block.style or ""
        is_list = _is_list_style(style)
        level = block.level

        if level:
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
            add_requirement(section, rest)
            current_intro = rest if _is_intro_requirement(rest) else ""
            continue

        if is_list:
            if text.endswith(":"):
                current_intro = text
                continue
            if current_intro and not _is_terms_context(current_intro):
                section = _synthetic_section(context, block_index, f"пункт списка {block_index}")
                parent = current_intro.rstrip(":")
                add_requirement(section, f"{parent}: {text}")
            elif _is_requirement_text(text):
                section = _synthetic_section(context, block_index, f"пункт списка {block_index}")
                add_requirement(section, text)
            continue

        if _is_intro_requirement(text):
            current_intro = text
            continue

        if _is_requirement_text(text) and not _is_section_title_only(text):
            section = _synthetic_section(context, block_index)
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
        if runtime_settings.parser_mode == "fast" or len(fast_requirements) >= runtime_settings.parser_fast_min_requirements:
            return fast_requirements
        logger.info(
            "Fast parser found only %d requirements, falling back to LLM extraction",
            len(fast_requirements),
        )

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
    return all_requirements


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
