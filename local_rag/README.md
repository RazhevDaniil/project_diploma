# local_rag — локальный BM25-индекс по cloud.ru/docs

## Что это
PoC структурного решения для обогащения LLM-контекста: вместо костылей
с curated_facts и enforce_verdict — реальные тексты страниц cloud.ru/docs.

- `raw/<sha1>.json` — crawled страницы (url, title, text, fetched_at).
  225 страниц, ~555 КБ текста.
- `sitemap_filtered.txt` — список ключевых URL'ов для crawl'a (218 шт.).

Индекс строится в памяти при первом обращении (`get_default_search()`).
Размер: 1225 чанков по ~600 символов, 6834 уникальных терминов. Build ~0.3 сек.

## Как обновлять
Crawl делается через Playwright (`/tmp/full_crawl.py`):

```bash
python3 /tmp/full_crawl.py 0 80   # партия 1
python3 /tmp/full_crawl.py 80 80  # партия 2
python3 /tmp/full_crawl.py 160 80 # партия 3
```

Файлы пропускаются, если уже есть в `raw/`. TTL — на ваш выбор;
сейчас просто полная перепарка по запросу.

## Покрытие
**18/20 ключевых тем из эталонов:**
WORM, SFTP (CyberDuck), PFS, Object Lock (Compliance/Governance/Legal Hold),
Versioning, Lifecycle, Cold class, Tier III, Multi-AZ, К1 ФСТЭК, ISO 27001,
лимиты бакетов, vCPU/RAM, GPU H100/A100, SLA, реестр РПО, поминутный
биллинг (через waf/pricing), API мониторинг.

**Не покрыто** (нужно добавить URL):
- Метрики качества Rэксплуатации (специфика заказчика, нет в cloud.ru/docs)
- Astra Linux / РЕД ОС (BYO-образы, нет отдельной страницы)
