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
SETTINGS_DIR = PROJECT_ROOT / "settings"
UI_SETTINGS_PATH = SETTINGS_DIR / "ui_settings.json"

for d in [UPLOAD_DIR, REPORTS_DIR, RUNS_DIR, PROMPT_VERSIONS_DIR, MANAGED_RAG_CACHE_DIR, SETTINGS_DIR]:
    d.mkdir(exist_ok=True)

# Foundation Models API settings (OpenAI-compatible)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://foundation-models.api.cloud.ru/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.05"))
LLM_REQUEST_DELAY = float(os.getenv("LLM_REQUEST_DELAY", "0"))

# Cloud.ru Managed RAG settings
MANAGED_RAG_URL = os.getenv(
    "MANAGED_RAG_URL",
    "https://e424a162-618c-4862-b789-b089abd81b46.managed-rag.inference.cloud.ru/api/v2/retrieve",
)
MANAGED_RAG_KB_VERSION = os.getenv("MANAGED_RAG_KB_VERSION", "eb73eb63-ec91-47c9-851e-1c14949b7a14")
MANAGED_RAG_API_KEY = os.getenv("MANAGED_RAG_API_KEY", OPENAI_API_KEY)
MANAGED_RAG_RESULTS = int(os.getenv("MANAGED_RAG_RESULTS", "6"))
MANAGED_RAG_CONTEXT_CHUNKS = int(os.getenv("MANAGED_RAG_CONTEXT_CHUNKS", "6"))
MANAGED_RAG_MAX_TOKENS = int(os.getenv("MANAGED_RAG_MAX_TOKENS", "2048"))
MANAGED_RAG_TEMPERATURE = float(os.getenv("MANAGED_RAG_TEMPERATURE", "0.01"))
MANAGED_RAG_RETRIEVAL_TYPE = os.getenv("MANAGED_RAG_RETRIEVAL_TYPE", "SEMANTIC")
MANAGED_RAG_CONCURRENCY = int(os.getenv("MANAGED_RAG_CONCURRENCY", "4"))
MANAGED_RAG_CACHE_ENABLED = os.getenv("MANAGED_RAG_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

# Analysis Settings
MAX_REQUIREMENTS_PER_BATCH = int(os.getenv("MAX_REQUIREMENTS_PER_BATCH", "8"))
PARSER_MODE = os.getenv("PARSER_MODE", "fast").lower()
PARSER_CHUNK_SIZE = int(os.getenv("PARSER_CHUNK_SIZE", "6000"))
PARSER_CONCURRENCY = int(os.getenv("PARSER_CONCURRENCY", "4"))
PARSER_FAST_MIN_REQUIREMENTS = int(os.getenv("PARSER_FAST_MIN_REQUIREMENTS", "20"))
PARSER_FAST_MAX_REQUIREMENTS = int(os.getenv("PARSER_FAST_MAX_REQUIREMENTS", "1000"))
# Патч 1 (ZK10): при PARSER_MODE=fast и слишком малом числе извлечённых
# требований уходим в LLM-ветку вместо тихого возврата 0. Дефолт включён,
# чтобы парсер всегда «громко» сообщал о провале structured_fast.
PARSER_FALLBACK_TO_LLM = os.getenv("PARSER_FALLBACK_TO_LLM", "true").lower() in {"1", "true", "yes", "on"}
ANALYSIS_RAG_MODE = os.getenv("ANALYSIS_RAG_MODE", "per_requirement").lower()
ANALYSIS_BATCH_CONCURRENCY = int(os.getenv("ANALYSIS_BATCH_CONCURRENCY", "4"))
# Патч 7 (ZK10): сколько раз должен встретиться одинаковый reasoning между
# разными requirement_id, чтобы пометить дубликаты как «template
# hallucination» и понизить им confidence.
ANALYSIS_DUPLICATE_REASONING_THRESHOLD = int(os.getenv("ANALYSIS_DUPLICATE_REASONING_THRESHOLD", "3"))
# Патч 10 (ZK10): дискретизация confidence в фиксированные уровни
# {0.95, 0.75, 0.55, 0.4, 0.25}. По умолчанию включена.
ANALYSIS_DISCRETE_CONFIDENCE = os.getenv("ANALYSIS_DISCRETE_CONFIDENCE", "true").lower() in {"1", "true", "yes", "on"}
# Патч 13 (ZK10): сколько раз должен встретиться один URL в источниках
# одного отчёта, чтобы пометить остальные verdict'ы с этим URL как
# «перерайз» (понижение confidence у всех, кроме топ-3 по релевантности).
ANALYSIS_URL_OVERUSE_THRESHOLD = int(os.getenv("ANALYSIS_URL_OVERUSE_THRESHOLD", "5"))
# Патч 15 (ZK10): строгий режим. Включает дополнительные проверки
# (faithfulness, mismatch-floor warning, audit-log per-requirement).
ANALYSIS_STRICT_MODE = os.getenv("ANALYSIS_STRICT_MODE", "false").lower() in {"1", "true", "yes", "on"}
