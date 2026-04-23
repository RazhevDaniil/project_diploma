"""Vector store — FAISS-based storage for knowledge base chunks."""

from __future__ import annotations

import logging
import time

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

import config as cfg

logger = logging.getLogger(__name__)

_embeddings: Embeddings | None = None
_vectorstore: FAISS | None = None


class FoundationModelsEmbeddings(Embeddings):
    """Embeddings adapter for Cloud.ru Foundation Models."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def _embed_batch(self, texts: list[str], attempt: int = 1) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
            if len(texts) > 1:
                midpoint = max(1, len(texts) // 2)
                logger.warning(
                    "Embeddings batch failed for %d texts (%s). Retrying as split batches of %d and %d.",
                    len(texts),
                    exc.__class__.__name__,
                    midpoint,
                    len(texts) - midpoint,
                )
                return self._embed_batch(texts[:midpoint]) + self._embed_batch(texts[midpoint:])

            if attempt >= cfg.OPENAI_EMBEDDING_MAX_RETRIES:
                raise

            delay = min(8.0, 2 ** (attempt - 1))
            logger.warning(
                "Embedding request failed for a single text (%s), retry %d/%d in %.1fs",
                exc.__class__.__name__,
                attempt,
                cfg.OPENAI_EMBEDDING_MAX_RETRIES,
                delay,
            )
            time.sleep(delay)
            return self._embed_batch(texts, attempt + 1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    """Singleton embeddings model for Cloud.ru Foundation Models."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading Foundation Models embeddings (model=%s)", cfg.OPENAI_EMBEDDING_MODEL)
        _embeddings = FoundationModelsEmbeddings(
            api_key=cfg.OPENAI_API_KEY,
            base_url=cfg.OPENAI_API_BASE,
            model=cfg.OPENAI_EMBEDDING_MODEL,
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


# Remote embeddings APIs have request size limits.
# We batch chunks to stay within a reasonable payload size.
EMBEDDING_BATCH_SIZE = cfg.OPENAI_EMBEDDING_BATCH_SIZE


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

    # Index in small batches to avoid oversized requests to the embeddings API
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
