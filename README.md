# Cloud.ru TZ Analyzer

AI-бот для автоматической вычитки технических заданий (ТЗ) от клиентов и тендеров. Сопоставляет каждый пункт ТЗ с официальной документацией Cloud.ru и выдаёт структурированный отчёт: что соответствует, что нет, и где почитать подробнее.

Проект теперь можно запускать в двух вариантах:

- локально как Python-приложение;
- через Docker Compose как два сервиса: `backend` (FastAPI + агент) и `ui` (Streamlit).

## Что делает бот

1. Загружаете ТЗ (PDF, DOCX, XLSX, TXT) — бот извлекает из него отдельные требования
2. Каждое требование проверяется по базе знаний, построенной из документации [cloud.ru/docs](https://cloud.ru/docs)
3. На выходе — отчёт с процентом соответствия, вердиктом по каждому пункту и ссылками на документацию

---

## Быстрый старт

### 1. Установка зависимостей

Требуется Python 3.11+.

```bash
# Создать виртуальное окружение (если ещё нет)
python -m venv venv

# Активировать
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt
```

### 2. Настройка Foundation Models API

Проект использует OpenAI-совместимый API сервиса Foundation Models на Cloud.ru.

```bash
export OPENAI_API_BASE="https://foundation-models.api.cloud.ru/v1"
export OPENAI_API_KEY="ваш_api_key"
export OPENAI_MODEL="GigaChat/GigaChat-2-Max"
export OPENAI_EMBEDDING_MODEL="BAAI/bge-m3"
```

Для текстовых запросов можно использовать модели GigaChat, доступные в Foundation Models. Для RAG-поиска нужны отдельные embedding-модели, например `BAAI/bge-m3` или `Qwen/Qwen3-Embedding-0.6B`.

> **Важно:** после смены embedding-модели нужно пересобрать `faiss_index/`, иначе старый индекс будет несовместим с новыми векторами.

### 3. Наполнение базы знаний

Перед анализом ТЗ нужно проиндексировать документацию Cloud.ru. Краулер обходит [cloud.ru/docs](https://cloud.ru/docs) (около 7000 страниц), извлекает текст и загружает в векторную базу FAISS.

```bash
# Быстрый тест — 500 страниц (~2-3 мин)
python seed_knowledge_base.py --max-pages 500

# Полная индексация — все ~7000 страниц (~15-30 мин)
python seed_knowledge_base.py

# Переиндексация с нуля (очистить старый индекс)
python seed_knowledge_base.py --clear
```

Параметры:
- `--max-pages N` — ограничить количество страниц (0 = все)
- `--concurrency N` — параллельность запросов (по умолчанию 10)
- `--clear` — очистить существующий индекс перед краулингом

Краулер кеширует скачанные страницы в папке `crawl_cache/`. При повторном запуске ранее скачанные страницы берутся из кеша — это быстро. Чтобы обновить кеш, удалите папку `crawl_cache/` и запустите заново.

### 4. Запуск приложения

```bash
# Терминал 1: backend API
uvicorn backend_api:app --host 0.0.0.0 --port 8000

# Терминал 2: Streamlit UI
streamlit run app.py
```

После старта:

- UI: `http://localhost:8501`
- Backend API: `http://localhost:8000/health`

---

## Docker Compose

Проект разделён на два сервиса:

- `backend` — API, RAG, краулинг, парсинг, экспорт отчётов
- `ui` — Streamlit-интерфейс, который работает через backend по HTTP

### 1. Подготовьте `.env`

```bash
cp .env.example .env
```

Минимальный набор:

```env
BACKEND_API_URL=http://backend:8000
OPENAI_API_BASE=https://foundation-models.api.cloud.ru/v1
OPENAI_API_KEY=ваш_api_key
OPENAI_MODEL=GigaChat/GigaChat-2-Max
OPENAI_EMBEDDING_MODEL=BAAI/bge-m3
```

### 2. Поднимите сервисы

```bash
docker compose up --build
```

После старта:

- UI: [http://localhost:8501](http://localhost:8501)
- Backend healthcheck: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Остановите сервисы

```bash
docker compose down
```

### 4. Что важно про данные

- `faiss_index/`, `crawl_cache/`, `uploads/`, `reports/`, `knowledge_base_data/` подключаются в backend как volume.
- Если у вас уже есть готовый `faiss_index/`, backend подхватит его автоматически.
- `faiss_index/index.faiss` не стоит пушить в обычный GitHub-репозиторий: файл слишком большой. Его лучше переносить отдельно через `rsync`/`scp` или пересобирать на сервере.

---

## Как пользоваться (UI)

Приложение имеет три вкладки:

### Вкладка «Анализ ТЗ»

1. **Загрузите файлы ТЗ** — перетащите или выберите PDF, DOCX, XLSX, TXT. Можно загрузить несколько файлов
2. **Нажмите «Извлечь требования»** — бот разберёт документ на отдельные требования (технические, SLA, юридические, коммерческие, ИБ)
3. **Нажмите «Запустить анализ»** — каждое требование проверяется по базе знаний. Результат: вердикт (соответствует / частично / не соответствует / требует уточнения), обоснование и ссылки на документацию

### Вкладка «База знаний»

- **Краулинг cloud.ru/docs** — кнопка запускает краулинг прямо из UI. Можно задать количество страниц и параллельность
- **Дополнительные файлы** — можно вручную загрузить внутреннюю документацию (TXT, MD, HTML, PDF, DOCX) для расширения базы
- **Тестовый поиск** — проверить, что база знаний находит релевантные фрагменты по запросу
- **Сброс базы** — кнопка в боковой панели очищает индекс

### Вкладка «Отчёт»

Появляется после анализа. Содержит:
- Метрики: общий процент соответствия, количество по каждому вердикту
- Резюме для руководителя (генерируется LLM)
- Несоответствия — с причиной и ссылками на документацию
- Частичные соответствия — что есть, чего не хватает
- Детализация по всем требованиям с группировкой по категориям

Отчёт можно экспортировать в **Markdown**, **DOCX**, **PDF** и **Excel**.

---

## Настройка через переменные окружения

Все настройки задаются в `config.py` и переопределяются через переменные окружения:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `OPENAI_API_BASE` | `https://foundation-models.api.cloud.ru/v1` | URL Foundation Models API |
| `OPENAI_API_KEY` | — | API key сервисного аккаунта Cloud.ru |
| `OPENAI_MODEL` | `GigaChat/GigaChat-2-Max` | LLM-модель для чата и анализа |
| `OPENAI_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding-модель для RAG |
| `CHUNK_SIZE` | `500` | Размер чанка при разбиении текста |
| `CHUNK_OVERLAP` | `80` | Перекрытие чанков |
| `TOP_K_RESULTS` | `5` | Количество результатов из FAISS при поиске |
| `CRAWL_MAX_PAGES` | `0` | Макс. страниц для краулинга (0 = все) |
| `CRAWL_CONCURRENCY` | `10` | Параллельность запросов краулера |
| `CRAWL_DELAY` | `0.2` | Задержка между запросами (сек) |
| `BACKEND_API_URL` | `http://backend:8000` | URL backend API для UI |

---

## Структура проекта

```
project_diploma/
├── app.py                          # Streamlit UI-клиент для backend API
├── backend_api.py                  # FastAPI backend API
├── config.py                       # Все настройки приложения
├── docker-compose.yml              # Оркестрация backend + ui
├── Dockerfile.backend              # Контейнер backend
├── Dockerfile.ui                   # Контейнер UI
├── DEPLOY_VM.md                    # Пошаговый деплой на виртуальную машину
├── seed_knowledge_base.py          # CLI-скрипт для краулинга и индексации
├── requirements.txt                # Python-зависимости
│
├── src/
│   ├── models.py                   # Общие доменные модели
│   ├── crawler/
│   │   └── spider.py               # Краулер cloud.ru/docs (sitemap, async fetch, extraction)
│   ├── parser/
│   │   ├── document_parser.py      # Парсинг ТЗ: PDF, DOCX, XLSX, TXT → текст + таблицы
│   │   └── requirement_extractor.py # Извлечение требований из текста через LLM
│   ├── knowledge_base/
│   │   ├── store.py                # FAISS vector store (эмбеддинги, поиск, персистенция)
│   │   └── indexer.py              # Загрузка документов в vector store
│   ├── analysis/
│   │   ├── analyzer.py             # RAG-движок: поиск контекста + LLM-оценка каждого требования
│   │   └── prompts.py              # Промпты для LLM (извлечение, анализ, сводка)
│   └── report/
│       └── generator.py            # Генерация отчётов (Markdown, DOCX)
│
├── crawl_cache/                    # Кеш скачанных страниц (JSON)
├── faiss_index/                    # Персистентный FAISS-индекс
├── uploads/                        # Загруженные файлы ТЗ
├── reports/                        # Сгенерированные отчёты
└── knowledge_base_data/            # Дополнительная документация (ручная загрузка)
```

---

## Типичный сценарий использования

```bash
# 1. Активировать окружение
source venv/bin/activate

# 2. Задать параметры Foundation Models
export OPENAI_API_BASE="https://foundation-models.api.cloud.ru/v1"
export OPENAI_API_KEY="ваш_api_key"
export OPENAI_MODEL="GigaChat/GigaChat-2-Max"
export OPENAI_EMBEDDING_MODEL="BAAI/bge-m3"

# 3. Проиндексировать документацию (первый раз)
python seed_knowledge_base.py --max-pages 500

# 4. Запустить backend и UI
uvicorn backend_api:app --host 0.0.0.0 --port 8000
streamlit run app.py

# 5. В браузере:
#    - Загрузить ТЗ на вкладке «Анализ ТЗ»
#    - Нажать «Извлечь требования» → «Запустить анализ»
#    - Посмотреть результат на вкладке «Отчёт»
#    - Скачать отчёт в Markdown / DOCX / PDF / XLSX
```

---

## Развёртывание на VM

Подробный пошаговый сценарий для Cloud.ru VM описан в [DEPLOY_VM.md](DEPLOY_VM.md).
