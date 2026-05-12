"""BM25-поиск по локальной коллекции crawled JSON-страниц cloud.ru/docs.

Дизайн:
- Один процесс — один индекс в памяти (loaded once, переиспользуется).
- Чанкируем каждую страницу по абзацам (~400-800 символов) для точного
  retrieval — иначе длинные страницы про OBS «съедают» запрос про SFTP.
- BM25 (k1=1.5, b=0.75) — стандартные параметры.
- Токенизация: lowercase, латиница + кириллица + цифры, 3+ символа.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# Корень всех crawled JSON: монтированный путь в контейнере / на хосте.
# Совместимо и с docker-compose (где монтируется ./local_rag) и с
# локальным запуском.
_DEFAULT_RAW_DIRS = [
    Path("local_rag/raw"),                                    # из workdir
    Path("/work/local_rag/raw"),                               # в docker
    Path(__file__).resolve().parents[2] / "local_rag" / "raw", # absolute
]


@dataclass
class SearchHit:
    """Один результат поиска — один чанк страницы."""

    url: str
    title: str
    text: str
    score: float
    chunk_index: int = 0

    def to_context_line(self, idx: int) -> str:
        """Форматирование для подмеса в LLM-контекст."""
        return (
            f"[Local-DOC {idx}] {self.title}\n"
            f"URL: {self.url}\n"
            f"Фрагмент: {self.text}"
        )


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9-]{2,}", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Простая токенизация: lowercase, ё→е, слова ≥3 символов."""
    if not text:
        return []
    text = text.lower().replace("ё", "е")
    return _TOKEN_RE.findall(text)


def _chunk_text(text: str, target: int = 600, overlap: int = 100) -> list[str]:
    """Бьём страницу на чанки по абзацам, ориентируясь на target символов.

    Чанк ≥ 200 символов — иначе сливаем со следующим. Overlap нужен,
    чтобы граничный термин не «потерялся» между чанками.
    """
    if not text or len(text) < target:
        return [text.strip()] if text and text.strip() else []
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 < target:
            current = f"{current}\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # Если параграф сам по себе длиннее target — режем грубо.
            if len(para) > target * 1.5:
                pos = 0
                while pos < len(para):
                    chunks.append(para[pos:pos + target])
                    pos += target - overlap
                current = ""
            else:
                # Overlap: возьмём хвост current'a, чтобы граничная инфа
                # повторилась в следующем чанке.
                tail = current[-overlap:] if current and overlap else ""
                current = (tail + "\n" + para).strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 80]


@dataclass
class _IndexedChunk:
    url: str
    title: str
    text: str
    tokens: list[str]
    chunk_index: int
    doc_freqs: Counter = field(default_factory=Counter)
    length: int = 0


class LocalDocSearch:
    """BM25-индекс по chunks. Lazy-loaded singleton по умолчанию."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.chunks: list[_IndexedChunk] = []
        self.idf: dict[str, float] = {}
        self.avg_len: float = 0.0
        self._built = False

    @classmethod
    def from_dir(cls, raw_dir: Path) -> "LocalDocSearch":
        inst = cls(raw_dir)
        inst.build()
        return inst

    def build(self) -> None:
        """Загружаем все .json, чанкируем, считаем idf."""
        if not self.raw_dir.exists():
            logger.warning("local_rag raw_dir не существует: %s", self.raw_dir)
            self._built = True
            return
        files = sorted(self.raw_dir.glob("*.json"))
        for f in files:
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("Не удалось прочитать %s: %s", f, exc)
                continue
            url = doc.get("url", "")
            title = doc.get("title", "")
            text = doc.get("text", "")
            if not text or len(text) < 100:
                continue
            for i, chunk in enumerate(_chunk_text(text)):
                # title в каждый чанк, чтобы запрос «WORM Cloud.ru» дотянул
                # до релевантного фрагмента независимо от того, есть ли
                # слово «WORM» в самом чанке.
                tokens = _tokenize(f"{title}\n{chunk}")
                if not tokens:
                    continue
                ic = _IndexedChunk(
                    url=url, title=title, text=chunk,
                    tokens=tokens, chunk_index=i,
                    doc_freqs=Counter(tokens), length=len(tokens),
                )
                self.chunks.append(ic)
        if not self.chunks:
            self._built = True
            return
        # idf
        df: Counter[str] = Counter()
        for c in self.chunks:
            df.update(set(c.tokens))
        N = len(self.chunks)
        self.idf = {
            term: math.log(1 + (N - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }
        self.avg_len = sum(c.length for c in self.chunks) / N
        self._built = True
        logger.info(
            "local_rag: %d страниц → %d чанков, avg_len=%.1f токенов",
            len(files), N, self.avg_len,
        )

    def search(self, query: str, k: int = 5,
               min_score: float = 0.5) -> list[SearchHit]:
        """BM25 top-K по query."""
        if not self._built:
            self.build()
        if not self.chunks or not query:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        k1, b = 1.5, 0.75
        scores: list[tuple[float, _IndexedChunk]] = []
        for c in self.chunks:
            s = 0.0
            for term in q_tokens:
                tf = c.doc_freqs.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                norm = 1 - b + b * (c.length / self.avg_len)
                s += idf * tf * (k1 + 1) / (tf + k1 * norm)
            if s >= min_score:
                scores.append((s, c))
        scores.sort(key=lambda x: -x[0])
        return [
            SearchHit(url=c.url, title=c.title, text=c.text,
                      score=round(s, 3), chunk_index=c.chunk_index)
            for s, c in scores[:k]
        ]


# Глобальный singleton, lazily инициализируется при первом запросе.
_DEFAULT_INSTANCE: LocalDocSearch | None = None
_INSTANCE_LOCK = threading.Lock()


def get_default_search() -> LocalDocSearch:
    """Возвращает общий индекс. Lazy-init на первый вызов."""
    global _DEFAULT_INSTANCE
    if _DEFAULT_INSTANCE is not None:
        return _DEFAULT_INSTANCE
    with _INSTANCE_LOCK:
        if _DEFAULT_INSTANCE is not None:
            return _DEFAULT_INSTANCE
        raw_dir = None
        for cand in _DEFAULT_RAW_DIRS:
            if cand.exists():
                raw_dir = cand
                break
        if raw_dir is None:
            raw_dir = _DEFAULT_RAW_DIRS[0]
        _DEFAULT_INSTANCE = LocalDocSearch.from_dir(raw_dir)
        return _DEFAULT_INSTANCE
