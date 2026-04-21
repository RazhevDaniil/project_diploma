"""Configuration for the TZ Analysis Bot."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base_data"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
REPORTS_DIR = PROJECT_ROOT / "reports"

for d in [KNOWLEDGE_BASE_DIR, FAISS_INDEX_DIR, UPLOAD_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)

# LLM Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat")  # "gigachat" or "openai_compatible"
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

# OpenAI-compatible fallback
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "local-model")

# Embedding Settings (GigaChat Embeddings API)
GIGACHAT_EMBEDDING_MODEL = os.getenv("GIGACHAT_EMBEDDING_MODEL", "Embeddings")

# RAG Settings
# GigaChat Embeddings API limit: 514 tokens per text.
# For Russian text ~1.5-2 chars/token, so 500 chars ≈ 250-330 tokens — safe margin.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Analysis Settings
MAX_REQUIREMENTS_PER_BATCH = 10
SIMILARITY_THRESHOLD = 0.3

# Crawler Settings
CRAWL_BASE_URL = os.getenv("CRAWL_BASE_URL", "https://cloud.ru/docs")
CRAWL_SITEMAP_URL = os.getenv("CRAWL_SITEMAP_URL", "https://cloud.ru/docs/sitemap.xml")
# Additional sources: cloud.ru/documents contains legal/compliance docs (SLA, certificates, licenses)
CRAWL_DOCUMENTS_URL = os.getenv("CRAWL_DOCUMENTS_URL", "https://cloud.ru/documents")
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "0"))  # 0 = all pages
CRAWL_CONCURRENCY = int(os.getenv("CRAWL_CONCURRENCY", "10"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "0.2"))  # seconds between requests
CRAWL_CACHE_DIR = PROJECT_ROOT / "crawl_cache"
CRAWL_CACHE_DIR.mkdir(exist_ok=True)
