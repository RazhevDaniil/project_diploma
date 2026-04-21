# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cloud.ru TZ Analyzer — an AI-powered bot for reviewing technical specifications (ТЗ / тендерная документация) against Cloud.ru's service capabilities. It crawls official documentation from cloud.ru/docs, extracts requirements from uploaded documents, checks each against the FAISS-backed knowledge base using RAG, and produces compliance reports with verdicts and links to documentation.

The UI language is Russian. All LLM prompts, labels, and reports are in Russian.

## Commands

```bash
# Activate virtualenv (Python 3.13)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Crawl cloud.ru/docs and build the knowledge base (all ~7000 pages)
python seed_knowledge_base.py

# Crawl only first 500 pages (faster, for testing)
python seed_knowledge_base.py --max-pages 500

# Crawl with custom concurrency
python seed_knowledge_base.py --max-pages 500 --concurrency 20

# Clear index before re-crawling
python seed_knowledge_base.py --clear --max-pages 500
```

No automated tests, linter, or CI/CD pipeline exist in this project.

## Architecture

The app is a Streamlit single-page application (`app.py`) with three tabs: document analysis, knowledge base management, and report viewing. LLM provider is selectable in the sidebar at runtime.

### Pipeline flow
1. **Crawl** — `src/crawler/spider.py` fetches sitemap from cloud.ru/docs/sitemap.xml, crawls pages with ThreadPoolExecutor, extracts text content with BeautifulSoup (including Next.js RSC payloads), caches results as JSON in `crawl_cache/` (keyed by URL MD5 hash)
2. **Index** — crawled pages are chunked (500 chars, 80 overlap — constrained by GigaChat Embeddings 514-token limit) and embedded into FAISS via `src/knowledge_base/store.py` with metadata including URL, title, section_path
3. **Parse** — `src/parser/document_parser.py` extracts text + tables from uploaded TZ (PDF/DOCX/XLSX/TXT) into `ParsedDocument`
4. **Extract** — `src/parser/requirement_extractor.py` sends document text to LLM, gets back structured `Requirement` objects (id, section, text, category, tables). Categories: technical, sla, legal, commercial, security, other
5. **Analyze** — `src/analysis/analyzer.py` batches requirements (up to 10 per batch), retrieves relevant chunks from FAISS with source URLs, asks LLM to produce `RequirementVerdict` per requirement (verdict: match/partial/mismatch/needs_clarification, with confidence, reasoning, evidence, recommendation, source_urls)
6. **Report** — `src/report/generator.py` renders Markdown, DOCX, and Excel exports with clickable links to cloud.ru/docs pages

### Two analysis modes
- **RAG mode** — vector similarity search against local FAISS index (default)
- **Live mode** — `src/search/live_search.py` routes by category: technical requirements search cloud.ru/docs via sitemap URL matching, legal/commercial use DuckDuckGo web search

### Key modules
- `src/crawler/spider.py` — threaded crawler: sitemap parsing, page fetching (httpx), content extraction (BeautifulSoup), caching, indexing into FAISS
- `src/llm/client.py` — LLM abstraction over GigaChat and OpenAI-compatible APIs. `call_llm()` for text, `call_llm_json()` for structured JSON responses with 4 fallback extraction strategies and retry logic (3 attempts, exponential backoff)
- `src/knowledge_base/store.py` — FAISS vector store singleton using GigaChat Embeddings API. Batch embedding in 50-chunk batches. Each chunk stores `url`, `title`, `section_path` metadata
- `src/knowledge_base/indexer.py` — loads documents into the vector store with `RecursiveCharacterTextSplitter` chunking
- `src/analysis/prompts.py` — all LLM system/user prompts for classification, analysis, and summary (all in Russian)
- `src/analysis/analyzer.py` — RAG analysis engine producing `AnalysisReport` with verdicts and compliance metrics
- `src/search/live_search.py` — live web/docs search with LLM-based keyword generation and category-aware routing
- `config.py` — all settings (paths, LLM provider, embedding model, RAG params, crawler settings). Most are overridable via environment variables

### LLM providers
Configured via `LLM_PROVIDER` env var (or sidebar in UI):
- `gigachat` (default) — requires `GIGACHAT_CREDENTIALS`
- `openai_compatible` — any OpenAI-API-compatible endpoint (`OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_MODEL`)

### Data directories
- `crawl_cache/` — cached crawled pages (JSON, keyed by URL hash)
- `faiss_index/` — persisted FAISS index
- `uploads/` — temporary uploaded TZ files
- `reports/` — generated report files (MD, DOCX, XLSX)
- `knowledge_base_data/` — additional manually uploaded docs
