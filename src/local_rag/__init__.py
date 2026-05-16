"""Local RAG: BM25-поиск по crawled cloud.ru/docs страницам.

Используется как fallback / дополнение к Managed RAG. Когда основной RAG
вернул слабый контекст или требование специфическое (WORM/SFTP/PFS),
local_rag добавляет реальные тексты из официальной документации
cloud.ru.

Public API:
    LocalDocSearch.from_dir(raw_dir) → объект, выдающий top-K chunks
    LocalDocSearch.search(query, k=5) → list[SearchHit]
"""

from src.local_rag.search import LocalDocSearch, SearchHit, get_default_search

__all__ = ["LocalDocSearch", "SearchHit", "get_default_search"]
