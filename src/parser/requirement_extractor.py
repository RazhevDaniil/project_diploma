"""Requirement extractor — uses LLM to split document into structured requirements."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re

import config as cfg
from src.models import Requirement
from src.llm.client import call_llm_json

logger = logging.getLogger(__name__)

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


def _extract_chunk_items(chunk_index: int, total_chunks: int, chunk: str) -> tuple[int, list[dict]]:
    logger.info("Extracting requirements from chunk %d/%d", chunk_index + 1, total_chunks)
    from src.prompt_store import get_prompt

    prompt_template = get_prompt("parser_user_template")
    system_prompt = get_prompt("parser_system")
    prompt = prompt_template.format(document_text=chunk)
    result = call_llm_json(prompt, system_prompt=system_prompt, max_tokens=8000)

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


def _category_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("sla", "доступност", "время реакции", "rto", "rpo", "простой", "инцидент")):
        return "sla"
    if any(token in lowered for token in ("персональн", "152-фз", "фстэк", "фсб", "скзи", "шифр", "доступ", "защит", "иб", "антивирус")):
        return "security"
    if any(token in lowered for token in ("штраф", "неустой", "оплат", "стоимост", "цена", "договор", "контракт")):
        return "commercial"
    if any(token in lowered for token in ("закон", "лиценз", "сертифик", "соответств", "право", "персональных данных")):
        return "legal"
    if any(token in lowered for token in ("сервер", "виртуаль", "кластер", "сеть", "хранилищ", "cpu", "ram", "api", "резервн", "мониторинг")):
        return "technical"
    return "other"


def _section_from_line(line: str) -> tuple[str, str] | None:
    match = re.match(r"^\s*(?:п\.|пункт\s*)?(\d+(?:\.\d+){1,6})[.)]?\s+(.+)$", line, re.IGNORECASE)
    if not match:
        return None
    section, rest = match.groups()
    rest = rest.strip()
    if len(rest) < 8:
        return None
    return section, rest


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


def _cap_requirements(requirements: list[Requirement]) -> list[Requirement]:
    max_items = max(1, cfg.PARSER_FAST_MAX_REQUIREMENTS)
    if len(requirements) <= max_items:
        return requirements

    priority = {
        "technical": 0,
        "sla": 1,
        "security": 2,
        "legal": 3,
        "commercial": 4,
        "other": 5,
    }
    sorted_items = sorted(
        requirements,
        key=lambda req: (
            priority.get(req.category, 9),
            0 if len(req.text) > 60 else 1,
            req.id,
        ),
    )
    kept = sorted(sorted_items[:max_items], key=lambda req: req.id)
    for idx, req in enumerate(kept, start=1):
        req.id = idx
    logger.info("Capped fast parser requirements from %d to %d", len(requirements), len(kept))
    return kept


def _extract_requirements_fast(document_text: str) -> list[Requirement]:
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
    requirements = _cap_requirements(requirements)
    logger.info("Fast parser extracted %d requirements", len(requirements))
    return requirements


def extract_requirements(document_text: str, max_chunk_size: int | None = None) -> list[Requirement]:
    """Extract structured requirements from document text using LLM.

    Splits long documents into chunks and processes each separately.
    """
    if cfg.PARSER_MODE in {"fast", "hybrid"}:
        fast_requirements = _extract_requirements_fast(document_text)
        if cfg.PARSER_MODE == "fast" or len(fast_requirements) >= cfg.PARSER_FAST_MIN_REQUIREMENTS:
            return fast_requirements
        logger.info(
            "Fast parser found only %d requirements, falling back to LLM extraction",
            len(fast_requirements),
        )

    chunks = _split_text(document_text, max_chunk_size or cfg.PARSER_CHUNK_SIZE)
    all_requirements: list[Requirement] = []
    global_id = 1
    chunk_items: dict[int, list[dict]] = {}

    max_workers = max(1, min(cfg.PARSER_CONCURRENCY, len(chunks)))
    logger.info("Extracting requirements from %d chunks (parallel=%d)", len(chunks), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_extract_chunk_items, index, len(chunks), chunk)
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
