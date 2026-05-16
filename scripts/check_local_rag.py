"""Диагностика local_rag для docker production.

Запускается ВНУТРИ контейнера:
    docker exec cloudru-tz-backend python /app/scripts/check_local_rag.py

или с хоста:
    docker exec cloudru-tz-backend python scripts/check_local_rag.py

Проверяет 4 уровня:
  1. Файлы JSON на диске (volume подмонтирован?)
  2. Импорт модуля src.local_rag (image содержит код?)
  3. Сборка индекса (1243 чанка ожидается)
  4. Полный pipeline: _format_batch_rag_context подмешивает Local-DOC?
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Гарантируем, что src/ виден как корень пакета (для запуска из scripts/)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def step(n, title):
    print(f"\n{'=' * 70}")
    print(f"  [{n}] {title}")
    print("=" * 70)


def main():
    # Шаг 1: файлы
    step(1, "Файлы в /app/local_rag/raw")
    for cand in [Path("/app/local_rag/raw"), Path("local_rag/raw"),
                 Path(__file__).resolve().parents[1] / "local_rag" / "raw"]:
        if cand.exists():
            files = list(cand.glob("*.json"))
            print(f"  Найден {cand}: {len(files)} JSON-файлов")
            if len(files) < 200:
                print(f"  ⚠️  МАЛО ФАЙЛОВ — volume не подмонтирован или crawl не завершён")
            break
    else:
        print(f"  ✗ Папка local_rag/raw НЕ НАЙДЕНА")
        print(f"  Проверьте docker-compose.yml: должна быть строка '- ./local_rag:/app/local_rag'")
        sys.exit(2)

    # Шаг 2: импорт модуля
    step(2, "Импорт src.local_rag")
    try:
        from src.local_rag import get_default_search, LocalDocSearch
        print(f"  ✓ Импорт ОК")
    except Exception:
        print(f"  ✗ ImportError:")
        traceback.print_exc()
        print(f"\n  ➡ Это значит docker image устарел. Нужен 'docker compose up -d --build'")
        sys.exit(3)

    # Шаг 3: сборка индекса
    step(3, "Сборка BM25-индекса")
    import time
    t0 = time.time()
    try:
        idx = get_default_search()
        dt = time.time() - t0
    except Exception:
        print(f"  ✗ Ошибка сборки индекса:")
        traceback.print_exc()
        sys.exit(4)
    print(f"  Чанков:   {len(idx.chunks)}")
    print(f"  Терминов: {len(idx.idf)}")
    print(f"  Время:    {dt:.2f}s")
    if len(idx.chunks) < 100:
        print(f"  ⚠️ Слишком мало чанков — данные не загружены")
        sys.exit(5)

    # Шаг 4: smoke-search
    step(4, "Smoke-search по 3 проблемным запросам")
    cases = [
        ("WORM защита от удаления", 10.0),
        ("Object Lock Retention Compliance Mode", 20.0),
        ("Поддержка SFTP объектное хранилище", 3.0),
    ]
    for q, expect_min in cases:
        hits = idx.search(q, k=2)
        if not hits:
            print(f"  ✗ '{q[:40]}': 0 hits")
            continue
        top = hits[0]
        ok = "✓" if top.score >= expect_min else "⚠️"
        print(f"  {ok} '{q[:40]}': top={top.score:.2f} (ожидаем ≥{expect_min})")
        print(f"       URL: {top.url[-60:]}")

    # Шаг 5: full pipeline
    step(5, "_format_batch_rag_context даёт Local-DOC в контексте?")
    from src.analysis.analyzer import _format_batch_rag_context
    from src.models import Requirement
    test_reqs = [
        Requirement(id=1, section="1.5", category="technical",
                    text="Доступ к объектному хранилищу через файловый протокол доступа"),
        Requirement(id=2, section="1.9", category="security",
                    text="Функция WORM (Write Once, Read Many) для защиты от изменений и удаления"),
    ]
    ctx = _format_batch_rag_context(test_reqs, None)
    has_marker = "ЛОКАЛЬНЫЙ ИНДЕКС" in ctx
    has_doc = "Local-DOC" in ctx
    print(f"  Размер контекста: {len(ctx)} символов")
    print(f"  Содержит 'ЛОКАЛЬНЫЙ ИНДЕКС': {has_marker}")
    print(f"  Содержит 'Local-DOC':       {has_doc}")
    if has_marker and has_doc:
        print(f"\n  ✓✓✓ ВСЁ РАБОТАЕТ — local_rag активен в production")
        sys.exit(0)
    else:
        print(f"\n  ✗ Контекст НЕ содержит Local-DOC, хотя индекс собран.")
        print(f"  Возможно условие активации не сработало для тестового batch.")
        print(f"  Покажу первые 800 символов ctx:")
        print(ctx[:800])
        sys.exit(6)


if __name__ == "__main__":
    main()
