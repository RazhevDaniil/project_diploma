"""Configuration for the TZ Analysis Bot."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROMPT_VERSIONS_DIR = PROJECT_ROOT / "prompt_versions"
PROMPT_STORE_PATH = PROMPT_VERSIONS_DIR / "prompts.json"

for d in [UPLOAD_DIR, REPORTS_DIR, PROMPT_VERSIONS_DIR]:
    d.mkdir(exist_ok=True)

# Foundation Models API settings (OpenAI-compatible)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://foundation-models.api.cloud.ru/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")

# Cloud.ru Managed RAG settings
MANAGED_RAG_URL = os.getenv(
    "MANAGED_RAG_URL",
    "https://e424a162-618c-4862-b789-b089abd81b46.managed-rag.inference.cloud.ru/api/v2/retrieve_generate",
)
MANAGED_RAG_KB_VERSION = os.getenv("MANAGED_RAG_KB_VERSION", "eb73eb63-ec91-47c9-851e-1c14949b7a14")
MANAGED_RAG_API_KEY = os.getenv("MANAGED_RAG_API_KEY", OPENAI_API_KEY)
MANAGED_RAG_RESULTS = int(os.getenv("MANAGED_RAG_RESULTS", "3"))
MANAGED_RAG_CONTEXT_CHUNKS = int(os.getenv("MANAGED_RAG_CONTEXT_CHUNKS", "5"))
MANAGED_RAG_MAX_TOKENS = int(os.getenv("MANAGED_RAG_MAX_TOKENS", "512"))
MANAGED_RAG_TEMPERATURE = float(os.getenv("MANAGED_RAG_TEMPERATURE", "0.01"))
MANAGED_RAG_RETRIEVAL_TYPE = os.getenv("MANAGED_RAG_RETRIEVAL_TYPE", "SEMANTIC")

# Analysis retrieval settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Analysis Settings
MAX_REQUIREMENTS_PER_BATCH = 10
