# Cloud.ru TZ Analyzer

Cloud.ru TZ Analyzer - AI-система для первичной вычитки технических заданий,
тендерной документации и приложений к закупкам. Проект извлекает из документа
атомарные требования, проверяет их на соответствие возможностям Cloud.ru через
Managed RAG и Foundation Models, формирует интерактивный отчет для пресейла и
экспортирует результаты в Markdown, DOCX, PDF и XLSX.

Документ описывает проект с двух сторон:

- смысловой: какую задачу решает система, как устроен пользовательский сценарий,
  как читать результаты и какие ограничения учитывать;
- технической: архитектура сервисов, backend API, структура данных, настройки,
  пайплайн анализа, хранилища, деплой, эксплуатация и точки для доработок.

## Содержание

1. [Назначение проекта](#1-назначение-проекта)
2. [Что считается результатом анализа](#2-что-считается-результатом-анализа)
3. [Архитектура](#3-архитектура)
4. [Технологический стек](#4-технологический-стек)
5. [Структура проекта](#5-структура-проекта)
6. [Быстрый старт локально](#6-быстрый-старт-локально)
7. [Минимальный `.env`](#7-минимальный-env)
8. [Поддерживаемые модели Foundation Models](#8-поддерживаемые-модели-foundation-models)
9. [Пользовательский сценарий в UI](#9-пользовательский-сценарий-в-ui)
10. [Чат по отчету](#10-чат-по-отчету)
11. [Backend API](#11-backend-api)
12. [Доменные модели](#12-доменные-модели)
13. [Пайплайн обработки документа](#13-пайплайн-обработки-документа)
14. [Генерация отчетов](#14-генерация-отчетов)
15. [Runtime settings и переменные окружения](#15-runtime-settings-и-переменные-окружения)
16. [Хранилища и volume management](#16-хранилища-и-volume-management)
17. [Docker Compose](#17-docker-compose)
18. [Деплой на промышленный стенд](#18-деплой-на-промышленный-стенд)
19. [Операционная эксплуатация](#19-операционная-эксплуатация)
20. [Тюнинг качества и производительности](#20-тюнинг-качества-и-производительности)
21. [Типовые проблемы](#21-типовые-проблемы)
22. [Точки доработки](#22-точки-доработки)
23. [Разработка](#23-разработка)
24. [Безопасность](#24-безопасность)
25. [Production checklist](#25-production-checklist)
26. [Короткая схема данных запуска](#26-короткая-схема-данных-запуска)
27. [Главное, что нужно знать новой команде](#27-главное-что-нужно-знать-новой-команде)

## 1. Назначение проекта

Система нужна пресейлу и техническим архитекторам Cloud.ru для быстрой оценки
тендерного ТЗ:

1. Заказчик присылает PDF, DOCX, XLSX, TXT или старый DOC.
2. Приложение извлекает требования из текста, таблиц и структурированных блоков.
3. Для каждого требования ищется подтверждающий контекст в базе знаний Cloud.ru.
4. LLM оценивает, закрывает ли Cloud.ru требование полностью, частично, не
   закрывает или нужны уточнения.
5. Отчет показывает покрытие по платформам Cloud.ru, проблемные пункты,
   источники RAG, рекомендации и технические индикаторы качества анализа.

Проект не заменяет финальное коммерческое или юридическое согласование. Его
роль - ускорить первичную разметку ТЗ, подсветить блокеры, собрать аргументы
из документации и дать пресейлу основу для подготовки КП.

## 2. Что считается результатом анализа

Результатом является `AnalysisReport`, который включает:

- список требований с вердиктами;
- платформенную матрицу по четырем каноническим платформам Cloud.ru;
- общий best-case процент по портфелю Cloud.ru;
- покрытие на рекомендуемой платформе;
- краткое executive summary;
- список несоответствий, частичных соответствий и пунктов на уточнение;
- список требований, где нужны внешние услуги или подрядчики;
- трассировку RAG: какие источники были выбраны и почему;
- блок качества анализа: дубликаты reasoning, переиспользованные URL,
  низкая уверенность, потерянные ключевые сигналы;
- сведения о качестве извлечения: сколько требований найдено, сколько передано
  в анализ, применялся ли лимит, какие категории были отброшены.

Вердикты:

| Вердикт | Смысл | Баллы |
|---|---|---:|
| `match` | Требование подтверждено источниками и закрывается Cloud.ru | 2 |
| `partial` | Закрывается частично, на части платформ или требует доработки КП | 1 |
| `mismatch` | Есть явное противоречие или требование не входит в портфель | 0 |
| `needs_clarification` | Данных недостаточно, нужна ручная проверка | 0 |
| `out_of_scope` | Процедурный/закупочный пункт вне технической оценки | не учитывается |

Процент соответствия считается по техническим требованиям. Процедурные пункты
закупки исключаются из знаменателя, но не теряются: они выводятся отдельным
блоком в отчете.

## 3. Архитектура

Проект запускается как два сервиса:

- `backend` - FastAPI API, пайплайн парсинга и анализа, интеграция с Cloud.ru
  Foundation Models и Cloud.ru Managed RAG, генерация отчетов, хранение запусков,
  настроек и версий промптов.
- `ui` - Streamlit-интерфейс для загрузки документов, управления настройками,
  запуска анализа, просмотра отчетов, чата по результатам и редактирования
  промптов.

По умолчанию сервисы поднимаются через Docker Compose:

```text
browser
  |
  | http://<host>:8501
  v
Streamlit UI
  |
  | BACKEND_API_URL=http://backend:8000
  v
FastAPI backend
  |
  +-- Cloud.ru Foundation Models OpenAI-compatible API
  |
  +-- Cloud.ru Managed RAG /api/v2/retrieve или /api/v2/retrieve_generate
  |
  +-- local_rag BM25 fallback по локально сохраненным страницам cloud.ru/docs
  |
  +-- файловые хранилища: uploads, runs, reports, settings, prompt_versions, rag_cache
```

## 4. Технологический стек

Backend:

- Python 3.11
- FastAPI
- Uvicorn
- OpenAI Python SDK для OpenAI-compatible API Cloud.ru Foundation Models
- `requests` для Managed RAG
- `python-docx` для DOCX parsing/export
- `pdfplumber` для PDF parsing
- `openpyxl` и `pandas` для XLSX parsing/export
- `fpdf2` для PDF export

UI:

- Streamlit
- `requests`

Внешние сервисы:

- Cloud.ru Foundation Models: прямые LLM-вызовы.
- Cloud.ru Managed RAG: поиск по индексированной базе знаний.

Дополнительный локальный контекст:

- `local_rag` - BM25-поиск по заранее сохраненным JSON-страницам
  `cloud.ru/docs`. Это не основной RAG и не FAISS, а вспомогательный
  подмес, который включается при слабом Managed RAG или специфичных фичах.

## 5. Структура проекта

```text
project_diploma/
├── app.py                         # Streamlit UI
├── backend_api.py                 # FastAPI backend
├── config.py                      # env/default config и директории
├── docker-compose.yml             # backend + ui
├── Dockerfile.backend
├── Dockerfile.ui
├── requirements.backend.txt
├── requirements.ui.txt
├── DEPLOY_VM.md                   # короткий сценарий деплоя на VM
├── scripts/
│   └── check_local_rag.py         # диагностика локального BM25-индекса
├── src/
│   ├── analysis/
│   │   ├── analyzer.py            # RAG+LLM анализ требований
│   │   └── prompts.py             # базовые промпты
│   ├── llm/
│   │   └── client.py              # клиент Foundation Models
│   ├── managed_rag/
│   │   └── client.py              # клиент Cloud.ru Managed RAG
│   ├── parser/
│   │   ├── document_parser.py     # PDF/DOC/DOCX/XLS/XLSX/TXT parsing
│   │   └── requirement_extractor.py
│   ├── report/
│   │   └── generator.py           # Markdown/DOCX/PDF/XLSX export
│   ├── local_rag/
│   │   └── search.py              # BM25 по local_rag/raw
│   ├── knowledge/
│   │   └── curated_facts.py       # проверенные факты для сложных кейсов
│   ├── metrics/                   # вспомогательная оценка качества
│   ├── models.py                  # доменные dataclass-модели
│   ├── prompt_store.py            # версионирование промптов
│   ├── run_store.py               # файловое хранилище запусков
│   ├── runtime_config.py          # immutable settings snapshot
│   └── settings_store.py          # сохранение UI-настроек
├── local_rag/
│   ├── README.md
│   ├── sitemap_filtered.txt
│   └── raw/*.json                 # локальные страницы cloud.ru/docs
├── uploads/                       # загруженные пользователем файлы
├── runs/                          # JSON-состояние запусков
├── reports/                       # сгенерированные отчеты
├── prompt_versions/               # версии промптов
├── rag_cache/                     # кэш Managed RAG для фиксированных KB version
├── settings/                      # сохраненные UI-настройки
└── llm_cache/                     # опциональный dev/eval-кэш LLM
```

## 6. Быстрый старт локально

Создайте `.env`:

```bash
cp .env.example .env
```

Если `.env.example` отсутствует в поставке, создайте `.env` вручную по
примеру ниже.

Запустите:

```bash
docker compose up -d --build
```

Проверьте:

```bash
docker compose ps
curl http://127.0.0.1:8000/health
```

Откройте:

- UI: `http://localhost:8501`
- backend healthcheck: `http://localhost:8000/health`

## 7. Минимальный `.env`

```env
BACKEND_API_URL=http://backend:8000

OPENAI_API_BASE=https://foundation-models.api.cloud.ru/v1
OPENAI_API_KEY=your_foundation_models_api_key_here
OPENAI_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
OPENAI_TEMPERATURE=0.2
LLM_REQUEST_DELAY=0

PARSER_MODE=fast
PARSER_CHUNK_SIZE=6000
PARSER_CONCURRENCY=4
PARSER_FAST_MIN_REQUIREMENTS=20
PARSER_FAST_MAX_REQUIREMENTS=1000
MAX_REQUIREMENTS_PER_BATCH=8

ANALYSIS_RAG_MODE=per_requirement
ANALYSIS_BATCH_CONCURRENCY=4

MANAGED_RAG_URL=https://e424a162-618c-4862-b789-b089abd81b46.managed-rag.inference.cloud.ru/api/v2/retrieve
MANAGED_RAG_KB_VERSION=latest
MANAGED_RAG_API_KEY=your_managed_rag_api_key_here
MANAGED_RAG_RESULTS=6
MANAGED_RAG_CONTEXT_CHUNKS=6
MANAGED_RAG_MAX_TOKENS=2048
MANAGED_RAG_TEMPERATURE=0.2
MANAGED_RAG_CONCURRENCY=4
MANAGED_RAG_CACHE_ENABLED=true
```

Если Managed RAG и Foundation Models используют один API-ключ, укажите одно
и то же значение в `OPENAI_API_KEY` и `MANAGED_RAG_API_KEY`.

## 8. Поддерживаемые модели Foundation Models

Модели выбираются в UI и передаются в backend через runtime settings.

Текущий список в `app.py`:

| UI label | API model id |
|---|---|
| `gpt-oss-120b` | `openai/gpt-oss-120b` |
| `GLM-4.6` | `zai-org/GLM-4.6` |
| `GLM-4.7` | `zai-org/GLM-4.7` |
| `Qwen3-235B-A22B-Instruct-2507` | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| `Qwen3-Next-80B-A3B-Instruct` | `Qwen/Qwen3-Next-80B-A3B-Instruct` |

Для добавления новой модели нужно:

1. Добавить пару label/id в `MODEL_OPTIONS` в `app.py`.
2. При необходимости обновить README и дефолт `OPENAI_MODEL` в `config.py`,
   `docker-compose.yml` или `.env`.
3. Проверить, что модель поддерживает нужный размер контекста и JSON-ответы.

## 9. Пользовательский сценарий в UI

Основной экран состоит из трех вкладок:

- `📄 Анализ ТЗ`
- `🕘 История`
- `✍️ Промпты`

В боковой панели есть переключатель:

- `💬 Чат`
- `⚙️ Настройки`

### 9.1. Вкладка `Анализ ТЗ`

Назначение: загрузить документ, извлечь требования, запустить анализ и
посмотреть готовый отчет.

Поддерживаемые форматы:

- PDF
- DOCX
- DOC
- XLSX
- XLS
- TXT

Сценарий:

1. Загрузить один или несколько файлов.
2. Нажать `1. Запустить извлечение`.
3. Дождаться статуса `extracted`.
4. Нажать `2. Запустить анализ`.
5. Дождаться статуса `completed`.
6. Просмотреть отчет на этой же вкладке.
7. Подготовить и скачать нужные форматы отчета.

Если отчет готов, экран становится компактнее: показывается плашка
`Анализ готов`, имя документа, число требований и кнопка `Новый анализ`.
Блок загрузки сворачивается в expander.

### 9.2. Блок прогресса запуска

Для активного запуска UI показывает:

- ID запуска;
- имя документа;
- статус;
- stage;
- прогресс `progress_done / progress_total`;
- дату обновления;
- ошибку, если статус `failed`;
- кнопку ручного обновления статуса;
- опцию автообновления.

Статусы запуска:

| Статус | Смысл |
|---|---|
| `created` | запуск создан локально |
| `queued` | задача поставлена в очередь FastAPI BackgroundTasks |
| `extracting` | backend извлекает текст и требования |
| `extracted` | требования извлечены, можно запускать анализ |
| `analyzing` | backend выполняет RAG+LLM-анализ |
| `completed` | отчет готов |
| `failed` | ошибка парсинга или анализа |

### 9.3. Отчет в UI

Отчет отображается прямо на вкладке `Анализ ТЗ`.

Основные блоки:

- переключатель режима метрик;
- шапка с процентом покрытия;
- сводные счетчики;
- покрытие извлечения;
- резюме;
- сомнительные места;
- матрица по платформам;
- сноски RAG;
- детализация карточки требования;
- внешние услуги и подрядчики;
- блок скачивания отчетов.

Режимы метрик:

- `По рекомендуемой платформе` - процент и счетчики считаются по платформе,
  которую система предлагает как основную для КП.
- `Best-case по портфелю Cloud.ru` - по каждому требованию берется лучший
  вердикт из всех платформ Cloud.ru. Это показывает теоретический максимум,
  если можно комбинировать платформы.

Сводные счетчики:

- `Всего`
- `Соответствует`
- `Частично`
- `Не соответствует`
- `Уточнить`

Если в документе есть процедурные пункты закупки, UI показывает пояснение,
что они исключены из технического процента.

### 9.4. Матрица по платформам

Матрица показывает требования строками, а платформы Cloud.ru столбцами.

Канонические платформы:

- `ГосОблако`
- `Облако VMware`
- `Advanced`
- `Evolution`

Обозначения:

| Символ | Смысл |
|---|---|
| `+` | соответствует |
| `±` | частично |
| `-` | не подтверждено или не соответствует |
| `?` | требуется уточнение |
| `[N]` | номер RAG-источника |

В матрице есть фильтры по вердиктам и платформам. Клик по строке открывает
карточку требования с деталями:

- полный пункт ТЗ;
- текст требования;
- итоговый вердикт;
- категория;
- confidence;
- reasoning;
- evidence;
- recommendation;
- оценки по платформам;
- внешние услуги, если нужны;
- выбранные RAG-источники;
- кнопка, которая подставляет вопрос по этому пункту в чат.

### 9.5. Сомнительные места

Блок показывает пункты, которые требуют ручного внимания:

- `needs_clarification`;
- низкая уверенность;
- внешняя услуга или подрядчик;
- слабые или отсутствующие RAG-источники;
- evidence `missing`, `weak` или `downgraded`;
- технические предупреждения evidence contract.

В expander доступна трассировка RAG: профиль требования, выбранные источники,
score и причины выбора.

### 9.6. Скачивание отчетов

UI готовит файл через backend API и затем показывает кнопку скачивания.

Доступные форматы:

- Markdown (`md`)
- DOCX
- PDF
- Excel (`xlsx`)

Экспорт не хранится в session state как постоянный артефакт. Backend создает
файл в `reports/`, возвращает его как `FileResponse`, UI кладет содержимое
во временный `st.session_state.downloads`.

### 9.7. Вкладка `История`

История показывает последние запуски из директории `runs/`.

Для каждого запуска отображается:

- имя документа;
- ID запуска;
- статус;
- дата обновления;
- число извлеченных требований;
- ошибка, если есть;
- кнопка `Открыть`.

История переживает обновление страницы и перезапуск UI-контейнера, потому что
состояние хранится backend-ом в JSON-файлах.

### 9.8. Вкладка `Промпты`

Вкладка управляет версионированными промптами.

Доступные ключи:

- `parser_system`
- `parser_user_template`
- `analysis_system`
- `analysis_user_template`
- `summary_system`
- `summary_user_template`

Возможности:

- выбрать промпт;
- выбрать версию;
- посмотреть активную версию;
- сделать выбранную версию активной;
- отредактировать текст;
- сохранить новую версию;
- автоматически активировать новую версию.

Версии хранятся в `prompt_versions/prompts.json`.

Переменные шаблонов:

| Prompt key | Переменные |
|---|---|
| `parser_user_template` | `{document_text}` |
| `analysis_user_template` | `{requirements_block}`, `{context}` |
| `summary_user_template` | `{doc_name}`, `{total}`, `{match_count}`, `{partial_count}`, `{mismatch_count}`, `{clarification_count}`, `{compliance_pct}`, `{top_mismatches}`, `{platform_matrix}`, `{recommended_platform}`, `{recommended_platform_compliance}`, `{external_services}` |

### 9.9. Боковая панель `Настройки`

Настройки делятся на блоки:

- Backend API
- Foundation Models API
- Скорость обработки
- Managed RAG
- Сохранение настроек

#### Backend API

Поля:

- `Backend URL`

UI проверяет `/health` и показывает:

- доступен ли backend;
- какой RAG provider используется;
- какая LLM-модель активна;
- актуальна ли информация healthcheck.

#### Foundation Models API

Поля:

- `API Base URL`
- `API Key`
- `LLM Model`
- `Температура LLM`

API-ключи используются для текущей обработки, но не сохраняются в
`settings/ui_settings.json`. Для постоянной конфигурации задавайте их через
`.env` или секреты стенда.

`Температура LLM` влияет на:

- LLM-извлечение требований;
- JSON-анализ требований;
- генерацию summary;
- чат по отчету.

Рекомендуемое значение по умолчанию: `0.2`.

#### Скорость обработки

Параметры этого блока применяются к новым запускам.

| UI-поле | Runtime key | Назначение |
|---|---|---|
| `Режим извлечения требований` | `parser_mode` | `fast`, `hybrid` или `llm` |
| `Размер чанка парсера` | `parser_chunk_size` | размер куска текста для LLM-извлечения |
| `Параллельность парсера` | `parser_concurrency` | число параллельных LLM-чанков |
| `Мин. требований для fast/hybrid` | `parser_fast_min_requirements` | sanity-порог для fallback |
| `Макс. требований для анализа` | `parser_fast_max_requirements` | жесткий лимит требований, которые пойдут в анализ |
| `Требований в батче анализа` | `max_requirements_per_batch` | размер JSON-батча анализатора |
| `RAG для анализа` | `analysis_rag_mode` | один RAG-запрос на батч или на каждое требование |
| `Параллельность батчей анализа` | `analysis_batch_concurrency` | число параллельных batch-анализов |
| `Пауза между LLM-запросами, сек` | `llm_request_delay` | искусственная задержка для rate limit |

Режимы парсера:

- `fast` - локальный структурный парсер. Основной режим для больших DOCX/PDF.
- `hybrid` - сначала `fast`, но если требований меньше минимума, включается
  LLM fallback.
- `llm` - требования извлекаются LLM по чанкам.

Важное поведение лимита:

- лимит не берет первые N строк;
- перед применением лимита найденные требования ранжируются;
- выше идут категории `technical`, `security`, `sla`, `legal`, `commercial`,
  `other`;
- внутри категорий выше требования с маркерами VMware/vCloud, ИБ, SLA, S3,
  WORM/API, сетей, ЦОД, IAM, личного кабинета;
- дополнительный приоритет получают требования с числовыми параметрами;
- похожие на определения или глоссарий строки получают меньший приоритет;
- после отбора исходный порядок ТЗ восстанавливается;
- в отчете фиксируется, сколько требований отброшено, какие категории
  отброшены и примеры пунктов вне анализа.

Зачем нужен лимит:

- контроль стоимости LLM/RAG;
- контроль времени обработки;
- снижение риска обрыва JSON-ответов;
- читаемость отчета;
- защита от документов, где парсер извлек сотни процедурных или дублирующих
  строк.

#### Managed RAG

Поля:

| UI-поле | Runtime key | Назначение |
|---|---|---|
| `RAG URL` | `managed_rag_url` | endpoint Managed RAG |
| `Knowledge Base Version` | `managed_rag_kb_version` | версия индексированной базы |
| `RAG API Key` | `managed_rag_api_key` | ключ Managed RAG |
| `Кол-во результатов` | `managed_rag_results` | сколько документов запросить у RAG |
| `Чанков в контексте` | `managed_rag_context_chunks` | совместимость с кэшем и будущим API |
| `Макс. токенов RAG` | `managed_rag_max_tokens` | только для `/retrieve_generate` |
| `Температура RAG` | `managed_rag_temperature` | только для `/retrieve_generate` |
| `Параллельность RAG` | `managed_rag_concurrency` | число параллельных RAG-запросов в `per_requirement` |
| `Кэшировать RAG-ответы` | `managed_rag_cache_enabled` | кэш для фиксированных KB version |

`Knowledge Base Version` по умолчанию: `latest`.

Если версия равна `latest`, backend не читает и не пишет локальный RAG-кэш,
чтобы не вернуть результаты от старого индекса.

Поддерживаемые endpoint-режимы:

- `/api/v2/retrieve` - основной режим. Возвращает найденные чанки и metadata.
- `/api/v2/retrieve_generate` - дополнительно просит Managed RAG
  сгенерировать answer выбранной LLM. Этот answer добавляется в контекст
  анализатора как вспомогательная интерпретация.

#### Сохранение настроек

Сохраняются только несекретные UI/runtime настройки:

- модель;
- температуры;
- параметры скорости;
- параметры Managed RAG без ключа;
- версия базы знаний;
- URL backend/RAG.

Не сохраняются:

- `OPENAI_API_KEY`;
- `MANAGED_RAG_API_KEY`.

Хранилище: `settings/ui_settings.json`.

При загрузке сохраненных настроек backend мигрирует старые дефолты:

- legacy KB UUID заменяется на `latest`;
- legacy `OPENAI_TEMPERATURE=0.05` заменяется на текущий дефолт;
- legacy `MANAGED_RAG_TEMPERATURE=0.01` заменяется на текущий дефолт.

## 10. Чат по отчету

Чат находится в боковой панели `💬 Чат` и появляется после готового отчета.

Он отвечает на вопросы пользователя по контексту:

- итоговый отчет;
- релевантные verdict-ы;
- извлеченные требования;
- история последних вопросов;
- дополнительный Managed RAG-поиск по вопросу пользователя.

Backend endpoint: `POST /analysis/ask`.

Ответ включает:

- `answer` - текст ответа;
- `related_sections` - связанные пункты ТЗ;
- `source_urls` - источники из отчета.

Чат не должен выдумывать новые факты. Системный промпт требует отвечать по
переданному контексту, на русском языке, с явным указанием нехватки данных.

## 11. Backend API

Backend запускается как FastAPI-приложение в `backend_api.py`.

Базовый адрес в Docker Compose:

```text
http://backend:8000
```

Снаружи хоста:

```text
http://localhost:8000
```

### 11.1. Healthcheck

```http
GET /health
```

Ответ:

```json
{
  "status": "ok",
  "provider": "foundation_models",
  "rag_provider": "managed_rag",
  "llm_model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
  "llm_temperature": 0.2,
  "managed_rag_kb_version": "latest"
}
```

Используется UI для проверки доступности backend.

### 11.2. Настройки UI

```http
GET /settings
```

Возвращает сохраненные настройки и признаки наличия ключей в env:

```json
{
  "settings": {},
  "updated_at": "2026-05-16 12:00:00",
  "has_openai_api_key": true,
  "has_managed_rag_api_key": true
}
```

```http
POST /settings
Content-Type: application/json
```

Payload:

```json
{
  "settings": {
    "openai_model": "zai-org/GLM-4.7",
    "openai_temperature": 0.2,
    "managed_rag_kb_version": "latest"
  }
}
```

API-ключи игнорируются при сохранении, если не передавать `include_secrets`
на уровне внутренней функции. Публичный endpoint сохраняет только allowlist
из `settings_store.ALLOWED_SETTINGS_KEYS`.

### 11.3. Промпты

```http
GET /prompts
```

Возвращает все prompt definitions, версии и активные версии.

```http
POST /prompts/version
Content-Type: application/json
```

Payload:

```json
{
  "prompt_key": "analysis_system",
  "content": "Новый текст промпта",
  "label": "Эксперимент v2",
  "activate": true
}
```

Создает новую версию. Если `activate=true`, версия сразу становится активной.

```http
POST /prompts/activate
Content-Type: application/json
```

Payload:

```json
{
  "prompt_key": "analysis_system",
  "version_id": "default"
}
```

Делает указанную версию активной.

### 11.4. Запуски

```http
GET /runs?limit=50
```

Возвращает список последних запусков.

```http
GET /runs/{run_id}/status
```

Возвращает краткое состояние запуска.

```http
GET /runs/{run_id}
```

Возвращает полный JSON запуска:

- файлы;
- parsed files;
- requirements;
- report;
- settings;
- status/stage/progress/error.

```http
POST /runs/extract
Content-Type: multipart/form-data
```

Поля:

- `files` - один или несколько файлов;
- `llm_settings_json` - JSON-строка с runtime settings.

Endpoint:

1. сохраняет файлы в `uploads/`;
2. создает run в `runs/`;
3. ставит фоновую задачу `_run_extract_requirements`;
4. возвращает JSON запуска.

```http
POST /runs/{run_id}/analysis
Content-Type: application/json
```

Payload:

```json
{
  "llm_settings": {
    "analysis_rag_mode": "per_requirement"
  }
}
```

Endpoint:

1. проверяет, что run существует;
2. проверяет, что требования уже извлечены;
3. ставит фоновую задачу `_run_analyze_requirements`;
4. возвращает обновленный run.

### 11.5. Синхронные endpoints для интеграций

Эти endpoints удобны для внешних тестов, CLI или будущей интеграции без
Streamlit. Они выполняют работу в рамках HTTP-запроса.

```http
POST /requirements/extract
Content-Type: multipart/form-data
```

Извлекает требования без создания полноценного run lifecycle.

```http
POST /analysis/report
Content-Type: application/json
```

Payload:

```json
{
  "document_name": "tz.docx",
  "requirements": [
    {
      "id": 1,
      "section": "7.2.4",
      "text": "Исполнитель должен обеспечить...",
      "category": "technical",
      "tables": ""
    }
  ],
  "search_mode": "managed_rag",
  "extraction_summary": {},
  "llm_settings": {}
}
```

Возвращает `AnalysisReport`.

### 11.6. Чат по отчету

```http
POST /analysis/ask
Content-Type: application/json
```

Payload:

```json
{
  "question": "Почему пункт 7.2.4 получил partial?",
  "report": {},
  "requirements": [],
  "history": [],
  "search_mode": "managed_rag",
  "llm_settings": {}
}
```

Ответ:

```json
{
  "answer": "...",
  "related_sections": ["7.2.4"],
  "source_urls": ["https://cloud.ru/docs/..."]
}
```

### 11.7. Экспорт отчетов

```http
POST /reports/markdown
Content-Type: application/json
```

Возвращает:

```json
{
  "markdown": "# Отчет..."
}
```

```http
POST /reports/export/{format_name}
Content-Type: application/json
```

`format_name`:

- `md`
- `docx`
- `pdf`
- `xlsx`

Payload:

```json
{
  "report": {}
}
```

Возвращает файл через `FileResponse`.

## 12. Доменные модели

Модели описаны в `src/models.py`.

### 12.1. Requirement

```json
{
  "id": 1,
  "section": "7.2.4",
  "text": "Текст требования",
  "category": "technical",
  "tables": ""
}
```

Категории:

- `technical`
- `sla`
- `legal`
- `commercial`
- `security`
- `procedural`
- `sla_classification`
- `other`

### 12.2. PlatformAssessment

Оценка одного требования на одной платформе или группе услуг.

```json
{
  "platform_name": "Облако VMware",
  "verdict": "match",
  "confidence": 0.95,
  "reasoning": "...",
  "evidence_refs": ["[1]"],
  "source_urls": ["https://cloud.ru/docs/..."],
  "source_titles": ["..."],
  "source_type": "platform",
  "recommendation": "..."
}
```

`source_type`:

- `platform` - внутренняя платформа Cloud.ru;
- `external_service` - внешняя услуга/подрядчик;
- `unknown` - источник не классифицирован.

### 12.3. RequirementVerdict

Итоговая оценка требования.

Поля:

- `requirement_id`
- `section`
- `requirement_text`
- `category`
- `verdict`
- `confidence`
- `reasoning`
- `evidence`
- `recommendation`
- `source_urls`
- `platform_assessments`
- `requires_external_service`
- `external_service_notes`
- `evidence_status`
- `evidence_contract_notes`
- `trace`

`trace` содержит техническую трассировку:

- режим RAG;
- поисковый профиль;
- RAG query;
- RAG error;
- ответ Managed RAG, если был;
- выбранные источники;
- local_rag hits;
- часть LLM-ответа.

### 12.4. AnalysisReport

Полный отчет.

Ключевые вычисляемые метрики:

- `total`
- `total_with_procedural`
- `procedural_count`
- `match_count`
- `partial_count`
- `mismatch_count`
- `clarification_count`
- `score`
- `max_score`
- `compliance_percentage`
- `recommended_platform`
- `recommended_platform_compliance`
- `platform_summary`
- `suspicious_items`

`summary` оставлен для обратной совместимости и содержит portfolio-summary.
Новые поля:

- `summary_platform`
- `summary_portfolio`

## 13. Пайплайн обработки документа

### 13.1. Загрузка файла

Файлы сохраняются в `uploads/` с UUID-префиксом:

```text
uploads/<uuid>_<original_filename>
```

Это снижает риск коллизий при одновременной работе нескольких пользователей.

### 13.2. Парсинг документа

Модуль: `src/parser/document_parser.py`.

Поддержка форматов:

| Формат | Реализация |
|---|---|
| PDF | `pdfplumber` |
| DOCX | `python-docx` |
| DOC | headless `soffice` или `libreoffice` conversion в DOCX |
| XLS/XLSX | `openpyxl` |
| TXT | чтение UTF-8 с `errors=replace` |

Результат: `ParsedDocument`.

Поля:

- `filename`;
- `text`;
- `tables`;
- `metadata`;
- `blocks`.

Для DOCX дополнительно сохраняется структурный порядок блоков:

- paragraph;
- heading level;
- table rows;
- caption;
- headers/cells.

Это важно для fast-парсера: он извлекает требования не только из плоского
текста, но и из структуры документа и строк таблиц.

### 13.3. Извлечение требований

Модуль: `src/parser/requirement_extractor.py`.

Режимы:

- `fast` - структурные эвристики по блокам документа;
- `hybrid` - `fast`, затем fallback в LLM при подозрительно малом числе
  требований;
- `llm` - LLM-извлечение по чанкам.

LLM-извлечение использует:

- `parser_system`;
- `parser_user_template`;
- `call_llm_json`;
- `parser_chunk_size`;
- `parser_concurrency`.

После извлечения формируется `requirements_extraction` metadata:

- какой парсер использовался;
- сколько требований найдено до лимита;
- сколько вернулось после лимита;
- применялся ли cap;
- сколько требований отброшено;
- правила приоритезации cap;
- распределение по категориям;
- примеры отброшенных требований;
- покрытие ключевых сигналов;
- потерянные ключевые сигналы;
- подозрение на ложные потери;
- число таблиц;
- число блоков;
- разнообразие стилей DOCX.

### 13.4. Категоризация

Категория требования определяется по текстовым маркерам:

- security: ПДн, ФСТЭК, ФСБ, СКЗИ, DDoS, WAF, NGFW, SIEM, SOC и т.п.;
- sla: SLA, доступность, время реакции, RTO/RPO, инциденты;
- technical: ВМ, CPU, RAM, диски, сеть, API, S3, backup, ЦОД, VMware;
- commercial: цена, договор, штраф, оплата;
- legal: закон, лицензия, сертификат, соответствие;
- other: все остальное.

Процедурные закупочные пункты и SLA-классификации могут быть исключены из
LLM/RAG-анализа и получить `out_of_scope`.

### 13.5. RAG-контекст

Модуль: `src/managed_rag/client.py`.

Для каждого требования или батча строится query, где есть:

- задача проверки возможности Cloud.ru;
- профиль требования;
- целевые поисковые термины;
- вероятная платформа;
- номер пункта ТЗ;
- категория;
- текст требования;
- таблицы, если есть.

Режимы:

- `per_requirement` - RAG-запрос на каждое требование. Дольше, но лучше
  покрытие и меньше пропусков.
- `grouped` - один RAG-запрос на батч. Быстрее, но контекст менее точный.

Managed RAG response обогащается:

- title;
- URL;
- platform;
- service;
- source_type;
- clean content.

Если `/retrieve` возвращает `metadata.jq_metadata`, backend использует его как
источник истины. Если данные лежат HTML-escaped JSON внутри `content`,
backend распаковывает их.

### 13.6. Локальный BM25-подмес

Модуль: `src/local_rag/search.py`.

Назначение: дать LLM реальные фрагменты `cloud.ru/docs`, когда Managed RAG
вернул слишком общий контекст или требование содержит специфичные фичи.

Примеры специфичных фич:

- WORM;
- Object Lock;
- Versioning;
- SFTP;
- PFS;
- Lifecycle;
- Retention;
- Legal Hold;
- Multi-AZ;
- Tier III;
- SSE-C/BYOK;
- presigned URL;
- object tagging;
- поминутная/посекундная тарификация.

Индекс строится лениво при первом обращении. Данные лежат в `local_rag/raw`.
В Docker Compose каталог монтируется в backend:

```yaml
- ./local_rag:/app/local_rag
```

Диагностика:

```bash
docker exec cloudru-tz-backend python scripts/check_local_rag.py
```

### 13.7. Curated facts

Модуль: `src/knowledge/curated_facts.py`.

Это небольшой набор проверенных фактов для сложных или плохо индексируемых
тем. Они добавляются в контекст как дополнительный источник, особенно для
специфичных фич и слабого RAG.

В production curated facts стоит рассматривать как временный слой. Более
правильная долгосрочная доработка - переиндексировать Managed RAG с чистой
metadata и полным покрытием нужных страниц.

### 13.8. LLM-анализ

Модуль: `src/analysis/analyzer.py`.

Основные шаги:

1. Исключить `procedural` и SLA-классификации из технического анализа.
2. Разбить требования на батчи по `max_requirements_per_batch`.
3. Для каждого батча получить RAG-контекст.
4. Сформировать `requirements_block`.
5. Подставить `requirements_block` и `context` в `analysis_user_template`.
6. Вызвать `call_llm_json` с `analysis_system`.
7. Распарсить JSON.
8. Достроить platform assessments до четырех канонических платформ.
9. Применить evidence contract и постобработку.
10. Сгенерировать summary.

Первичный анализ использует `max_tokens=80000`. Если LLM не вернула вердикт
для части требований, backend делает retry по одному требованию с его
собственным RAG-контекстом и `max_tokens=16000`.

Если и retry не помог, создается placeholder:

- overall `needs_clarification`;
- confidence `0.0`;
- reasoning `Не удалось получить оценку от LLM`;
- четыре platform assessments с `needs_clarification`.

Это важно для матрицы: требование не пропадает, а явно видно как требующее
ручной проверки.

### 13.9. Evidence contract

Analyzer проверяет, что выводы не висят в воздухе:

- есть ли источники;
- валидны ли numeric refs `[1]`, `[2]`;
- есть ли authoritative Cloud.ru URL;
- не выдуманы ли числа в evidence;
- есть ли overlap между evidence и retrieved excerpts;
- не противоречит ли overall verdict оценкам платформ;
- корректно ли помечены внешние услуги.

По результатам может быть:

- снижен confidence;
- изменен verdict;
- добавлен `evidence_status`;
- добавлены `evidence_contract_notes`.

### 13.10. Постобработка качества

После анализа всех батчей выполняется cross-verdict post-process:

- дедупликация одинаковых reasoning;
- штраф за переиспользование одного URL слишком большим числом verdict-ов;
- дискретизация confidence в фиксированные уровни;
- синхронизация verdict с confidence.

Статистика записывается в `extraction_summary.analysis_quality` и затем
показывается в отчете.

### 13.11. Summary

Генерируются два summary:

- `summary_portfolio` - best-case по портфелю;
- `summary_platform` - по рекомендуемой платформе.

Поле `summary` равно `summary_portfolio` для обратной совместимости.

## 14. Генерация отчетов

Модуль: `src/report/generator.py`.

Форматы:

- Markdown;
- DOCX;
- PDF;
- XLSX.

### 14.1. Markdown

Markdown - самый полный текстовый формат. Он включает:

- шапку;
- методику оценки;
- сводку;
- покрытие извлечения;
- резюме;
- матрицу платформ;
- итоги по платформам;
- внешние услуги;
- сомнительные места;
- трассировку RAG;
- качество анализа;
- что проверить в первую очередь;
- mismatch/clarification/partial;
- важные подтвержденные соответствия;
- процедурные пункты;
- сноски RAG;
- детализацию по всем требованиям.

### 14.2. DOCX

DOCX строится через `python-docx`.

Особенности:

- альбомная ориентация;
- уменьшенные поля;
- компактные таблицы;
- Markdown summary конвертируется в параграфы и реальные DOCX-таблицы;
- markdown-синтаксис очищается в ячейках;
- ссылки приводятся к читаемому plain text виду;
- таблицы отчета рендерятся как Word tables.

### 14.3. PDF

PDF строится через `fpdf2`.

Особенности:

- пытается подключить Unicode font с поддержкой кириллицы;
- матрица платформ уходит на landscape-страницу;
- Markdown очищается в plain text;
- таблицы в summary сохраняются в читаемом pipe-separated виде там, где
  полноценная PDF-таблица не используется.

### 14.4. XLSX

Excel строится через `pandas` и `openpyxl`.

Листы:

- `Сводка`;
- `Матрица платформ`;
- `Платформы детально`;
- `Сноски RAG`;
- `Сомнительные места`;
- `Трассировка RAG`;
- `Покрытие извлечения`;
- `Ключевые сигналы`;
- `Все требования`;
- отдельные листы по вердиктам.

## 15. Runtime settings и переменные окружения

Backend использует immutable `RuntimeSettings`. Это важно: несколько запусков
могут идти параллельно, и настройки должны ехать вместе с конкретным run, а не
мутировать process-wide config.

### 15.1. Foundation Models

| Env | Default | UI | Описание |
|---|---|---|---|
| `OPENAI_API_BASE` | `https://foundation-models.api.cloud.ru/v1` | да | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | пусто | вводится, не сохраняется | API key |
| `OPENAI_MODEL` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | да | модель LLM |
| `OPENAI_TEMPERATURE` | `0.2` | да | температура прямых LLM-вызовов |
| `LLM_REQUEST_DELAY` | `0` | да | задержка после LLM-запроса |

### 15.2. Managed RAG

| Env | Default | UI | Описание |
|---|---|---|---|
| `MANAGED_RAG_URL` | Cloud.ru `/api/v2/retrieve` endpoint | да | endpoint RAG |
| `MANAGED_RAG_KB_VERSION` | `latest` | да | версия базы знаний |
| `MANAGED_RAG_API_KEY` | `OPENAI_API_KEY` | вводится, не сохраняется | API key RAG |
| `MANAGED_RAG_RESULTS` | `6` | да | число результатов |
| `MANAGED_RAG_CONTEXT_CHUNKS` | `6` | да | совместимый лимит chunks |
| `MANAGED_RAG_MAX_TOKENS` | `2048` | да | max tokens для `/retrieve_generate` |
| `MANAGED_RAG_TEMPERATURE` | `0.2` | да | temperature для `/retrieve_generate` |
| `MANAGED_RAG_RETRIEVAL_TYPE` | `SEMANTIC` | нет | тип retrieval |
| `MANAGED_RAG_CONCURRENCY` | `4` | да | параллельные RAG-запросы |
| `MANAGED_RAG_CACHE_ENABLED` | `true` | да | кэш RAG для фиксированных версий |

### 15.3. Парсер и анализ

| Env | Default | UI | Описание |
|---|---|---|---|
| `PARSER_MODE` | `fast` | да | режим извлечения |
| `PARSER_CHUNK_SIZE` | `6000` | да | размер чанка для LLM parser |
| `PARSER_CONCURRENCY` | `4` | да | параллельность LLM parser |
| `PARSER_FAST_MIN_REQUIREMENTS` | `20` | да | минимум для sanity fallback |
| `PARSER_FAST_MAX_REQUIREMENTS` | `1000` | да | max требований в анализ |
| `PARSER_FALLBACK_TO_LLM` | `true` | нет | fallback из fast в LLM |
| `MAX_REQUIREMENTS_PER_BATCH` | `8` | да | требований в LLM batch |
| `ANALYSIS_RAG_MODE` | `per_requirement` | да | grouped/per_requirement |
| `ANALYSIS_BATCH_CONCURRENCY` | `4` | да | параллельные batch-анализы |
| `ANALYSIS_DUPLICATE_REASONING_THRESHOLD` | `3` | нет | порог дубликатов reasoning |
| `ANALYSIS_DISCRETE_CONFIDENCE` | `true` | нет | дискретизация confidence |
| `ANALYSIS_URL_OVERUSE_THRESHOLD` | `5` | нет | штраф за переиспользование URL |
| `ANALYSIS_STRICT_MODE` | `false` | нет | дополнительные strict checks |

### 15.4. LLM cache для разработки

| Env | Default | Описание |
|---|---|---|
| `LLM_CACHE_ENABLED` | `false` | включает файловый кэш LLM-ответов |
| `LLM_CACHE_DIR` | `<project>/llm_cache` | каталог кэша |

Кэш нужен для детерминированных dev/eval-прогонов. В production по умолчанию
выключен.

## 16. Хранилища и volume management

Docker Compose монтирует:

```yaml
volumes:
  - ./uploads:/app/uploads
  - ./reports:/app/reports
  - ./runs:/app/runs
  - ./prompt_versions:/app/prompt_versions
  - ./rag_cache:/app/rag_cache
  - ./settings:/app/settings
  - ./local_rag:/app/local_rag
```

Назначение:

| Каталог | Назначение | Можно очищать |
|---|---|---|
| `uploads/` | оригиналы загруженных файлов | да, если не нужны старые run files |
| `reports/` | экспортированные файлы отчетов | да |
| `runs/` | состояние запусков и готовые отчеты JSON | осторожно, это история UI |
| `prompt_versions/` | версии промптов | только с backup |
| `rag_cache/` | кэш Managed RAG | да |
| `settings/` | сохраненные UI-настройки | только с backup |
| `local_rag/` | локальные docs для BM25 | не очищать без замены индекса |
| `llm_cache/` | dev/eval LLM cache | да |

Для промстенда обязательно вынести эти каталоги на persistent volume или
сетевое хранилище, иначе история запусков, промпты и настройки потеряются при
пересоздании контейнеров.

## 17. Docker Compose

Сервис `backend`:

- build: `Dockerfile.backend`;
- container: `cloudru-tz-backend`;
- env_file: `.env`;
- port: `8000:8000`;
- command: `uvicorn backend_api:app --host 0.0.0.0 --port 8000`;
- restart: `unless-stopped`.

Сервис `ui`:

- build: `Dockerfile.ui`;
- container: `cloudru-tz-ui`;
- port: `8501:8501`;
- command: `streamlit run app.py`;
- depends_on: `backend`;
- получает backend URL как `http://backend:8000`.

Backend image копирует весь проект. UI image копирует только `app.py` и
UI requirements, потому что бизнес-логика находится в backend.

## 18. Деплой на промышленный стенд

Ниже не единственный возможный вариант, но это базовый production checklist.

### 18.1. Инфраструктура

Рекомендуемый минимум:

- Linux VM Ubuntu 22.04+;
- Docker Engine;
- Docker Compose plugin;
- 2 vCPU и 4-8 GB RAM для небольших документов;
- 4+ vCPU и 8-16 GB RAM для больших ТЗ и параллельных пользователей;
- persistent disk для `runs`, `reports`, `uploads`, `prompt_versions`,
  `settings`, `rag_cache`, `local_rag`;
- outbound HTTPS к Cloud.ru Foundation Models и Managed RAG;
- reverse proxy с TLS.

### 18.2. Установка Docker

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

После добавления пользователя в группу Docker перелогиньтесь.

### 18.3. Развертывание проекта

```bash
git clone <repo-url> project_diploma
cd project_diploma
mkdir -p uploads reports runs prompt_versions rag_cache settings
```

Создайте `.env`, задайте реальные ключи и endpoint-ы.

```bash
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8000/health
```

### 18.4. Reverse proxy

Для промышленного стенда не рекомендуется открывать Streamlit напрямую в
интернет. Поставьте nginx или другой reverse proxy.

Пример nginx:

```nginx
server {
    listen 80;
    server_name tz-analyzer.example.ru;

    client_max_body_size 200M;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

В текущем UI backend URL задается отдельно. Если хотите ходить через `/api`,
поставьте в UI `Backend URL` значение `https://tz-analyzer.example.ru/api`
или доработайте `BACKEND_API_URL` в окружении UI.

### 18.5. TLS и доступ

Для production:

- включить HTTPS;
- ограничить доступ по VPN, корпоративной сети или SSO;
- не хранить API-ключи в UI settings;
- использовать секреты платформы или защищенный `.env`;
- ограничить доступ к backend API извне, если backend не нужен внешним
  интеграциям;
- настроить backup persistent volumes.

### 18.6. Обновление

```bash
cd project_diploma
git pull
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8000/health
```

Перед обновлением на production рекомендуется сохранить:

```bash
tar -czf backup_$(date +%Y%m%d_%H%M%S).tgz runs reports prompt_versions settings
```

## 19. Операционная эксплуатация

### 19.1. Логи

```bash
docker compose logs -f backend
docker compose logs -f ui
```

### 19.2. Проверка backend

```bash
curl http://127.0.0.1:8000/health
```

### 19.3. Проверка истории запусков

```bash
curl http://127.0.0.1:8000/runs
```

### 19.4. Проверка local_rag

```bash
docker exec cloudru-tz-backend python scripts/check_local_rag.py
```

### 19.5. Очистка временных данных

Безопасно очищать:

- `rag_cache/`;
- `llm_cache/`;
- старые файлы в `reports/`;
- старые файлы в `uploads/`, если не нужно восстановление исходных документов.

Осторожно очищать:

- `runs/` - история запусков и отчеты в UI;
- `prompt_versions/` - версии промптов;
- `settings/` - сохраненные настройки.

## 20. Тюнинг качества и производительности

### 20.1. Быстрее

Для ускорения:

- `PARSER_MODE=fast`;
- `ANALYSIS_RAG_MODE=grouped`;
- увеличить `MAX_REQUIREMENTS_PER_BATCH`;
- увеличить `ANALYSIS_BATCH_CONCURRENCY`;
- увеличить `MANAGED_RAG_CONCURRENCY`;
- включить RAG cache для фиксированной KB version.

Риски:

- меньше recall RAG;
- выше вероятность обрыва или неполного JSON;
- больше rate limit pressure;
- меньше точность источников.

### 20.2. Точнее

Для качества:

- `ANALYSIS_RAG_MODE=per_requirement`;
- `MAX_REQUIREMENTS_PER_BATCH=5..8`;
- `MANAGED_RAG_RESULTS=6..10`;
- `OPENAI_TEMPERATURE=0.2`;
- `MANAGED_RAG_KB_VERSION=latest`;
- не использовать RAG cache с `latest`;
- обновить Managed RAG KB с чистой metadata.

Риски:

- дольше обработка;
- больше запросов к RAG и LLM;
- выше стоимость.

### 20.3. Большие ТЗ

Для документов на сотни требований:

- оставьте `PARSER_FAST_MAX_REQUIREMENTS` осознанным лимитом;
- проверьте блок `Покрытие извлечения`;
- смотрите `category_counts_omitted` и `cap_omitted_examples`;
- при необходимости увеличьте лимит до 1000;
- уменьшите batch size, если появляются placeholder-строки
  `Не удалось получить оценку от LLM`;
- оставьте `per_requirement`, если важна точность.

## 21. Типовые проблемы

### 21.1. Backend недоступен в UI

Проверьте:

```bash
docker compose ps
docker compose logs backend
curl http://127.0.0.1:8000/health
```

В UI проверьте `Backend URL`.

В Docker Compose для UI должен быть:

```env
BACKEND_API_URL=http://backend:8000
```

### 21.2. Ошибка API key

Проверьте:

- `OPENAI_API_KEY`;
- `MANAGED_RAG_API_KEY`;
- доступ ключа к нужному сервису;
- не вставлен ли ключ только в UI, но запуск был создан до этого;
- не был ли ключ потерян при перезапуске контейнера, если он не задан в `.env`.

### 21.3. RAG через прямой запрос отвечает, а через UI нет

Проверьте:

- тот ли `MANAGED_RAG_URL`;
- `/retrieve` или `/retrieve_generate`;
- `MANAGED_RAG_KB_VERSION`;
- `OPENAI_MODEL`;
- `MANAGED_RAG_API_KEY`;
- `MANAGED_RAG_MAX_TOKENS` и `MANAGED_RAG_TEMPERATURE` для
  `/retrieve_generate`;
- не используется ли устаревший RAG cache для фиксированной версии;
- что UI сохранил настройки backend-у.

По умолчанию backend использует `/retrieve`, а не `/retrieve_generate`.
Это значит, что генерация ответа делается прямым LLM-анализатором на чанках,
а не Managed RAG answer. Если нужно воспроизвести ручной RAG-запрос через
Managed RAG generation, укажите URL с `/retrieve_generate`.

### 21.4. В отчете есть `Не удалось получить оценку от LLM`

Это placeholder, когда LLM не вернула verdict даже после retry.

Что делать:

- уменьшить `MAX_REQUIREMENTS_PER_BATCH`;
- оставить `ANALYSIS_RAG_MODE=per_requirement`;
- проверить логи backend;
- проверить max tokens и выбранную модель;
- проверить, не слишком ли большой контекст из RAG/local_rag;
- повторить анализ.

### 21.5. Много `needs_clarification`

Возможные причины:

- Managed RAG KB не содержит нужных страниц;
- KB проиндексирована без metadata;
- выбран grouped RAG;
- слишком мало `MANAGED_RAG_RESULTS`;
- требование слишком дробное и требует ручной интерпретации;
- модель не нашла прямого подтверждения.

Что делать:

- использовать `per_requirement`;
- увеличить `MANAGED_RAG_RESULTS`;
- проверить `Трассировка RAG`;
- обновить KB;
- добавить нужные страницы в Managed RAG;
- при необходимости добавить временный curated fact.

### 21.6. Таблицы в DOCX/PDF выглядят плохо

Текущая версия уже конвертирует Markdown-таблицы в DOCX-таблицы и очищает
Markdown-синтаксис для DOCX/PDF. Если таблица все еще уезжает:

- проверьте ширину таблицы и число колонок;
- для DOCX уменьшите font size в `src/report/generator.py`;
- для PDF сложные таблицы лучше выносить в XLSX;
- проверьте, не пришел ли summary от LLM как сложный Markdown с вложенными
  таблицами.

### 21.7. Старый DOC не парсится

Для `.doc` нужен `soffice` или `libreoffice` внутри окружения backend.
В текущем `Dockerfile.backend` LibreOffice не установлен. Для production,
где ожидаются `.doc`, добавьте пакет LibreOffice в backend image или требуйте
от пользователей пересохранять документы в `.docx`.

## 22. Точки доработки

### 22.1. Авторизация

Сейчас Streamlit UI не содержит собственной авторизации. Для production нужно:

- закрыть доступ reverse proxy;
- добавить SSO/OIDC на proxy или в приложение;
- ограничить backend API по сети;
- добавить audit log пользователей, если важно.

### 22.2. Очередь задач

Сейчас используются FastAPI `BackgroundTasks`. Для одного процесса и умеренной
нагрузки этого достаточно.

Для production с несколькими пользователями лучше вынести задачи в:

- Celery/RQ/Dramatiq;
- Redis/RabbitMQ;
- отдельные worker-контейнеры;
- хранилище статусов в PostgreSQL.

### 22.3. База данных

Сейчас все хранится в JSON-файлах. Это просто и удобно для PoC, но для
промышленного стенда лучше использовать:

- PostgreSQL для runs, settings, prompt versions;
- S3-compatible object storage для uploads/reports;
- миграции схемы;
- retention policy.

### 22.4. Managed RAG KB

Самое важное улучшение качества:

- переиндексировать KB с нормальными metadata columns;
- хранить title, URL, platform, service, source_type отдельно;
- убрать buried JSON inside content;
- добавить все страницы по специфичным фичам;
- завести версионирование KB и release notes;
- для production использовать `latest` только если процесс публикации KB
  управляемый и проверенный.

### 22.5. Набор regression/eval

Рекомендуется завести:

- набор эталонных ТЗ;
- ожидаемые verdict-ы по ключевым пунктам;
- метрики качества;
- сравнение до/после изменения промптов;
- smoke-тест export форматов.

В проекте уже есть `src/metrics/`, но его нужно довести до регулярного CI.

### 22.6. Observability

Для production полезно добавить:

- request id;
- structured logs;
- Prometheus metrics;
- latency по этапам parse/RAG/LLM/report;
- счетчики ошибок RAG/LLM;
- размер документов;
- число требований;
- стоимость или токены, если API возвращает usage.

## 23. Разработка

### 23.1. Локальный backend без Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.backend.txt
uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload
```

### 23.2. Локальный UI без Docker

```bash
python -m venv .venv-ui
source .venv-ui/bin/activate
pip install -r requirements.ui.txt
BACKEND_API_URL=http://127.0.0.1:8000 streamlit run app.py
```

### 23.3. Проверка синтаксиса

```bash
python -m py_compile app.py backend_api.py config.py src/settings_store.py src/managed_rag/client.py src/parser/requirement_extractor.py src/analysis/analyzer.py src/analysis/prompts.py src/report/generator.py
```

### 23.4. Проверка diff

```bash
git diff --check
```

## 24. Безопасность

Важные моменты:

- API-ключи не сохраняются в `settings/ui_settings.json`;
- если пользователь вводит ключ в UI, он живет в session state и передается
  backend-у в runtime settings текущего запроса;
- для постоянной конфигурации используйте `.env` или secret manager;
- backend CORS сейчас разрешает `*`, для production стоит ограничить origins;
- Streamlit UI не имеет встроенной auth;
- загруженные ТЗ сохраняются на диск, значит диск должен быть защищен;
- отчеты могут содержать коммерчески чувствительную информацию;
- backup и retention должны соответствовать внутренним правилам.

## 25. Production checklist

Перед переносом на промстенд проверьте:

- [ ] Есть актуальные API-ключи Foundation Models и Managed RAG.
- [ ] `MANAGED_RAG_KB_VERSION=latest` или зафиксирована нужная версия KB.
- [ ] Managed RAG KB содержит актуальную документацию Cloud.ru.
- [ ] Persistent volumes настроены для `runs`, `reports`, `uploads`,
      `prompt_versions`, `settings`, `local_rag`.
- [ ] Настроен HTTPS reverse proxy.
- [ ] Доступ ограничен VPN/SSO/security group.
- [ ] `client_max_body_size` достаточен для больших ТЗ.
- [ ] Проверен `/health`.
- [ ] Проверен тестовый анализ на реальном ТЗ.
- [ ] Проверен export DOCX/PDF/XLSX.
- [ ] Проверена вкладка `Промпты` и сохранение версий.
- [ ] Проверен `scripts/check_local_rag.py`.
- [ ] Настроен backup.
- [ ] Описан регламент обновления KB и промптов.
- [ ] Зафиксированы рекомендуемые параметры скорости для прома.

## 26. Короткая схема данных запуска

Файл `runs/<run_id>.json` примерно выглядит так:

```json
{
  "id": "run-id",
  "document_name": "tz.docx",
  "status": "completed",
  "stage": "analysis_completed",
  "progress_done": 220,
  "progress_total": 220,
  "error": "",
  "created_at": "2026-05-16 12:00:00",
  "updated_at": "2026-05-16 12:30:00",
  "files": [],
  "parsed_files": [],
  "requirements": [],
  "report": {},
  "settings": {}
}
```

Это основной объект, который UI подхватывает при обновлении страницы или при
открытии прошлых результатов из истории.

## 27. Главное, что нужно знать новой команде

- Основной источник знаний - Cloud.ru Managed RAG.
- `latest` для KB version используется по умолчанию.
- Для `latest` локальный RAG-кэш отключается автоматически.
- Локальный `local_rag` - только вспомогательный BM25-подмес по сохраненным
  страницам `cloud.ru/docs`.
- UI не анализирует документы сам, а только управляет backend API.
- Состояние запусков хранится в `runs/`.
- Промпты версионируются и могут меняться без деплоя кода.
- API-ключи не сохраняются в UI settings.
- Ограничитель требований управляет временем, стоимостью и стабильностью,
  но его применение прозрачно отображается в отчете.
- DOCX/PDF export строится из `AnalysisReport`, а не из UI.
- Для промышленного использования нужны auth, persistent storage, backup,
  ограниченный CORS и, желательно, отдельная очередь задач.
