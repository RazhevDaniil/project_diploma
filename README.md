# Cloud.ru TZ Analyzer

AI-бот для вычитки технических заданий и тендерной документации. Приложение извлекает требования из ТЗ, проверяет каждое требование через Cloud.ru Managed RAG и формирует отчеты с вердиктами, обоснованиями и рекомендациями для пресейла.

Проект запускается как два сервиса:

- `backend` — FastAPI API, агент анализа, интеграция с Foundation Models и Managed RAG
- `ui` — Streamlit-интерфейс

## Что изменилось в новой версии

- Локальный FAISS/RAG больше не используется.
- Вкладка наполнения базы знаний удалена.
- Проверка возможностей Cloud.ru выполняется через Managed RAG `retrieve_generate`.
- В UI можно выбрать одну из моделей Foundation Models:
  - `openai/gpt-oss-120b`
  - `zai-org/GLM-4.6`
  - `Qwen/Qwen3-235B-A22B-Instruct-2507`
  - `Qwen/Qwen3-Next-80B-A3B-Instruct`
- Добавлена вкладка «Промпты» с версиями для парсера, анализатора и summary.

## Быстрый старт

```bash
cp .env.example .env
docker compose up --build
```

После старта:

- UI: http://localhost:8501
- Backend healthcheck: http://localhost:8000/health

## `.env`

Минимальный пример:

```env
BACKEND_API_URL=http://backend:8000

OPENAI_API_BASE=https://foundation-models.api.cloud.ru/v1
OPENAI_API_KEY=your_foundation_models_api_key_here
OPENAI_MODEL=openai/gpt-oss-120b
OPENAI_TEMPERATURE=0.05

MANAGED_RAG_URL=https://e424a162-618c-4862-b789-b089abd81b46.managed-rag.inference.cloud.ru/api/v2/retrieve_generate
MANAGED_RAG_KB_VERSION=eb73eb63-ec91-47c9-851e-1c14949b7a14
MANAGED_RAG_API_KEY=your_managed_rag_api_key_here
MANAGED_RAG_RESULTS=2
MANAGED_RAG_CONTEXT_CHUNKS=3
MANAGED_RAG_MAX_TOKENS=256
MANAGED_RAG_TEMPERATURE=0.01
MANAGED_RAG_CONCURRENCY=4
MANAGED_RAG_CACHE_ENABLED=true
```

Если Managed RAG использует тот же ключ, что и Foundation Models, можно указать одинаковое значение в `OPENAI_API_KEY` и `MANAGED_RAG_API_KEY`.

## Как пользоваться

1. Откройте вкладку «Анализ ТЗ».
2. Загрузите PDF/DOCX/XLSX/TXT.
3. Нажмите «Извлечь требования».
4. Нажмите «Запустить анализ».
5. На вкладке «Отчёт» посмотрите оценку и summary, затем скачайте полный отчет.

Состояние обработки сохраняется в `runs/`: после обновления страницы UI подхватит текущий запуск, а на вкладке «История» можно открыть прошлые результаты.

## Промпты

Вкладка «Промпты» хранит версии в `prompt_versions/prompts.json`.

Доступные промпты:

- `parser_system`
- `parser_user_template`
- `analysis_system`
- `analysis_user_template`
- `summary_system`
- `summary_user_template`

При сохранении создается новая версия. Активная версия используется сразу для следующих запусков анализа.

## Обновление

```bash
git pull
docker compose up -d --build
```

## Структура

```text
project_diploma/
├── app.py
├── backend_api.py
├── config.py
├── docker-compose.yml
├── src/
│   ├── managed_rag/
│   │   └── client.py
│   ├── parser/
│   ├── analysis/
│   ├── report/
│   └── prompt_store.py
├── prompt_versions/
├── runs/
├── rag_cache/
├── uploads/
└── reports/
```
