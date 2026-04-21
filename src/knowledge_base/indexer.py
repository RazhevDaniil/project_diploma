"""Knowledge base indexer — loads documents from various sources into the vector store."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from src.knowledge_base.store import create_or_update_vectorstore

logger = logging.getLogger(__name__)


def index_text_file(file_path: str | Path, metadata: dict | None = None) -> int:
    """Index a plain text or markdown file."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    meta = {"source": str(path), "filename": path.name}
    if metadata:
        meta.update(metadata)
    docs = [Document(page_content=text, metadata=meta)]
    vs = create_or_update_vectorstore(docs)
    return vs.index.ntotal


def index_directory(
    dir_path: str | Path,
    extensions: tuple[str, ...] = (".txt", ".md", ".html"),
    metadata: dict | None = None,
) -> int:
    """Index all matching files from a directory."""
    dir_path = Path(dir_path)
    docs = []
    for path in sorted(dir_path.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        meta = {"source": str(path), "filename": path.name}
        if metadata:
            meta.update(metadata)
        docs.append(Document(page_content=text, metadata=meta))
        logger.info("Loaded %s (%d chars)", path.name, len(text))

    if not docs:
        logger.warning("No documents found in %s", dir_path)
        return 0

    vs = create_or_update_vectorstore(docs)
    return vs.index.ntotal


def index_raw_texts(texts: list[dict]) -> int:
    """Index a list of {text, source, ...} dicts.

    Each dict must have 'text' key; all other keys become metadata.
    """
    docs = []
    for item in texts:
        text = item.pop("text", "")
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata=item))

    if not docs:
        return 0

    vs = create_or_update_vectorstore(docs)
    return vs.index.ntotal


def index_parsed_document(parsed_doc) -> int:
    """Index a ParsedDocument (from our parser) into the knowledge base."""
    text = parsed_doc.full_text
    docs = [Document(
        page_content=text,
        metadata={"source": parsed_doc.filename, "filename": parsed_doc.filename},
    )]
    vs = create_or_update_vectorstore(docs)
    return vs.index.ntotal
