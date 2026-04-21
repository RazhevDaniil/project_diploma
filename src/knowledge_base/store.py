"""Vector store — FAISS-based storage for knowledge base chunks."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_gigachat import GigaChatEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config as cfg

logger = logging.getLogger(__name__)

_embeddings: Embeddings | None = None
_vectorstore: FAISS | None = None


def get_embeddings() -> Embeddings:
    """Singleton embeddings model (GigaChat API)."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading GigaChat embeddings (model=%s)", cfg.GIGACHAT_EMBEDDING_MODEL)
        _embeddings = GigaChatEmbeddings(
            credentials=cfg.GIGACHAT_CREDENTIALS,
            scope=cfg.GIGACHAT_SCOPE,
            model=cfg.GIGACHAT_EMBEDDING_MODEL,
            verify_ssl_certs=False,
        )
    return _embeddings


def get_vectorstore() -> FAISS | None:
    """Get existing vectorstore or None."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    index_path = cfg.FAISS_INDEX_DIR / "index.faiss"
    if index_path.exists():
        logger.info("Loading FAISS index from %s", cfg.FAISS_INDEX_DIR)
        _vectorstore = FAISS.load_local(
            str(cfg.FAISS_INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def invalidate_cached_runtime():
    """Reset in-memory embeddings/vectorstore objects after config changes."""
    global _embeddings, _vectorstore
    _embeddings = None
    _vectorstore = None


def get_persisted_vector_count() -> int:
    """Read the persisted FAISS index size without loading LangChain objects."""
    index_path = cfg.FAISS_INDEX_DIR / "index.faiss"
    if not index_path.exists():
        return 0
    import faiss

    index = faiss.read_index(str(index_path))
    return index.ntotal


# GigaChat Embeddings API has a strict request size limit.
# We batch chunks to stay well under it.
EMBEDDING_BATCH_SIZE = 50


def create_or_update_vectorstore(documents: list[Document]) -> FAISS:
    """Create or update the vectorstore with new documents."""
    global _vectorstore

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))

    embeddings = get_embeddings()

    # Index in small batches to avoid 413 from GigaChat API
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
        logger.info(
            "Embedding batch %d/%d (%d chunks)",
            i // EMBEDDING_BATCH_SIZE + 1,
            (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE,
            len(batch),
        )
        if _vectorstore is None:
            _vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            _vectorstore.add_documents(batch)

    _vectorstore.save_local(str(cfg.FAISS_INDEX_DIR))
    logger.info("Saved FAISS index with %d vectors", _vectorstore.index.ntotal)
    return _vectorstore


def search(query: str, k: int = cfg.TOP_K_RESULTS) -> list[Document]:
    """Search the vectorstore for relevant chunks."""
    vs = get_vectorstore()
    if vs is None:
        logger.warning("No vectorstore available — returning empty results")
        return []
    return vs.similarity_search(query, k=k)


def search_with_scores(query: str, k: int = cfg.TOP_K_RESULTS) -> list[tuple[Document, float]]:
    """Search with similarity scores (lower = more similar for L2)."""
    vs = get_vectorstore()
    if vs is None:
        return []
    return vs.similarity_search_with_score(query, k=k)


def reset_vectorstore():
    """Clear the vectorstore."""
    global _vectorstore
    _vectorstore = None
    import shutil
    if cfg.FAISS_INDEX_DIR.exists():
        shutil.rmtree(cfg.FAISS_INDEX_DIR)
        cfg.FAISS_INDEX_DIR.mkdir(exist_ok=True)
    logger.info("Vectorstore reset")
