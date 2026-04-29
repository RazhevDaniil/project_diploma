"""Configuration for the TZ Analysis Bot."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"
PROMPT_VERSIONS_DIR = PROJECT_ROOT / "prompt_versions"
PROMPT_STORE_PATH = PROMPT_VERSIONS_DIR / "prompts.json"
MANAGED_RAG_CACHE_DIR = PROJECT_ROOT / "rag_cache"

for d in [UPLOAD_DIR, REPORTS_DIR, RUNS_DIR, PROMPT_VERSIONS_DIR, MANAGED_RAG_CACHE_DIR]:
    d.mkdir(exist_ok=True)

# Foundation Models API settings (OpenAI-compatible)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://foundation-models.api.cloud.ru/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.05"))

# Cloud.ru Managed RAG settings
MANAGED_RAG_URL = os.getenv(
    "MANAGED_RAG_URL",
    "https://e424a162-618c-4862-b789-b089abd81b46.managed-rag.inference.cloud.ru/api/v2/retrieve_generate",
)
MANAGED_RAG_KB_VERSION = os.getenv("MANAGED_RAG_KB_VERSION", "eb73eb63-ec91-47c9-851e-1c14949b7a14")
MANAGED_RAG_API_KEY = os.getenv("MANAGED_RAG_API_KEY", OPENAI_API_KEY)
MANAGED_RAG_RESULTS = int(os.getenv("MANAGED_RAG_RESULTS", "2"))
MANAGED_RAG_CONTEXT_CHUNKS = int(os.getenv("MANAGED_RAG_CONTEXT_CHUNKS", "3"))
MANAGED_RAG_MAX_TOKENS = int(os.getenv("MANAGED_RAG_MAX_TOKENS", "256"))
MANAGED_RAG_TEMPERATURE = float(os.getenv("MANAGED_RAG_TEMPERATURE", "0.01"))
MANAGED_RAG_RETRIEVAL_TYPE = os.getenv("MANAGED_RAG_RETRIEVAL_TYPE", "SEMANTIC")
MANAGED_RAG_CONCURRENCY = int(os.getenv("MANAGED_RAG_CONCURRENCY", "4"))
MANAGED_RAG_CACHE_ENABLED = os.getenv("MANAGED_RAG_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

# Analysis Settings
MAX_REQUIREMENTS_PER_BATCH = 10
