"""Requirement extractor — uses LLM to split document into structured requirements."""

from __future__ import annotations

import logging

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


def extract_requirements(document_text: str, max_chunk_size: int = 6000) -> list[Requirement]:
    """Extract structured requirements from document text using LLM.

    Splits long documents into chunks and processes each separately.
    """
    chunks = _split_text(document_text, max_chunk_size)
    all_requirements: list[Requirement] = []
    global_id = 1

    for i, chunk in enumerate(chunks):
        logger.info("Extracting requirements from chunk %d/%d", i + 1, len(chunks))
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(document_text=chunk)
        result = call_llm_json(prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT, max_tokens=8000)

        items = result if isinstance(result, list) else result.get("requirements", result.get("raw", []))
        if isinstance(items, str):
            logger.warning("LLM returned string instead of list for chunk %d", i + 1)
            continue

        # Log first item keys to help debug field name mismatches
        if items and isinstance(items[0], dict):
            logger.info("LLM response keys for chunk %d: %s (first item sample: %s)",
                        i + 1, list(items[0].keys()),
                        {k: str(v)[:50] for k, v in items[0].items()})

        for item in items:
            if not isinstance(item, dict):
                continue

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
