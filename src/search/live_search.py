"""Live search — finds relevant documentation for each requirement.

Technical/IT requirements → search cloud.ru/docs by keyword matching against sitemap + page fetch.
Legal/security requirements → search cloud.ru/documents + cloud.ru/docs + web (scoped to cloud.ru).
Commercial/other → web search scoped to cloud.ru, then RAG fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.crawler.spider import (
    CrawledPage,
    _fetch_page,
    fetch_sitemap_urls,
    filter_docs_urls,
)
from src.llm.client import call_llm

import config as cfg

logger = logging.getLogger(__name__)

# ── Trusted domains for URL filtering ────────────────────────────────────────

# Only these domains are considered relevant for Cloud.ru TZ analysis.
# Everything else from web search is noise.
TRUSTED_DOMAINS = {
    "cloud.ru",
    "cloudru.tech",
    "sbercloud.ru",
    # Government/regulatory — relevant for legal/compliance
    "fstec.ru",
    "rkn.gov.ru",
    "roskomnadzor.gov.ru",
    "consultant.ru",
    "garant.ru",
    "publication.pravo.gov.ru",
}


def _is_trusted_url(url: str) -> bool:
    """Check if a URL belongs to a trusted domain."""
    try:
        hostname = urlparse(url).hostname or ""
        # Match domain and subdomains (e.g., docs.cloud.ru, www.cloud.ru)
        for domain in TRUSTED_DOMAINS:
            if hostname == domain or hostname.endswith("." + domain):
                return True
    except Exception:
        pass
    return False


def _filter_trusted_results(results: list[SearchResult]) -> list[SearchResult]:
    """Keep only results from trusted domains."""
    trusted = [r for r in results if _is_trusted_url(r.url)]
    removed = len(results) - len(trusted)
    if removed > 0:
        logger.info("Filtered out %d results from untrusted domains", removed)
    return trusted


# ── Cached sitemap data ──────────────────────────────────────────────────────

_sitemap_urls: list[str] | None = None
_url_index: dict[str, str] | None = None  # lowercase path segment → full URL


def _get_sitemap_urls() -> list[str]:
    """Lazily fetch and cache sitemap URLs."""
    global _sitemap_urls
    if _sitemap_urls is None:
        all_urls = fetch_sitemap_urls()
        # Include both /docs/ and /documents/ URLs
        _sitemap_urls = [u for u in all_urls
                         if u.startswith("https://cloud.ru/docs")
                         or u.startswith("https://cloud.ru/documents")]
    return _sitemap_urls


def _get_url_index() -> dict[str, str]:
    """Build an index of URL path segments for keyword matching."""
    global _url_index
    if _url_index is None:
        _url_index = {}
        for url in _get_sitemap_urls():
            # Extract path after cloud.ru/
            path = url.replace("https://cloud.ru/", "").lower()
            _url_index[path] = url
    return _url_index


# ── Search result model ──────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single search result with extracted content."""
    url: str
    title: str
    snippet: str  # short preview
    content: str  # full extracted text
    source_type: str  # "cloud_docs", "cloud_documents", "web", "rag_fallback"


# ── Cloud.ru docs search ─────────────────────────────────────────────────────


def _generate_search_keywords(requirement_text: str) -> list[str]:
    """Use LLM to generate search keywords for a requirement."""
    prompt = f"""Из следующего требования из технического задания извлеки 3-5 ключевых технических терминов
для поиска в документации облачного провайдера (Cloud.ru).
Термины должны быть на русском и/или английском, через запятую. Только термины, без пояснений.

Требование: {requirement_text[:500]}

Ключевые термины:"""

    try:
        result = call_llm(prompt, max_tokens=200, temperature=0.0)
        keywords = [k.strip().lower() for k in result.split(",") if k.strip()]
        return keywords[:5]
    except Exception as e:
        logger.warning("Failed to generate keywords: %s", e)
        # Fallback: extract significant words
        words = re.findall(r'[а-яёa-z]{4,}', requirement_text.lower())
        return list(set(words))[:5]


def _match_urls_by_keywords(keywords: list[str], max_results: int = 10,
                            url_prefix: str | None = None) -> list[str]:
    """Find cloud.ru URLs whose path matches the keywords.

    Args:
        url_prefix: if set, only match URLs starting with this prefix
                    (e.g. "documents/" for cloud.ru/documents)
    """
    url_index = _get_url_index()
    scored: dict[str, int] = {}

    for keyword in keywords:
        kw = keyword.lower().replace(" ", "").replace("-", "")
        for path, url in url_index.items():
            if url_prefix and not path.startswith(url_prefix):
                continue
            path_normalized = path.replace("-", "").replace("_", "").replace("/", " ")
            if kw in path_normalized:
                scored[url] = scored.get(url, 0) + 1

    # Sort by score (most keyword matches first), take top N
    ranked = sorted(scored.items(), key=lambda x: -x[1])
    return [url for url, _ in ranked[:max_results]]


def search_cloud_docs(requirement_text: str, max_pages: int = 5) -> list[SearchResult]:
    """Search cloud.ru/docs for content relevant to a requirement.

    1. LLM generates keywords from requirement
    2. Match keywords against sitemap URLs
    3. Fetch matched pages (from cache or web)
    4. Return results with full content
    """
    keywords = _generate_search_keywords(requirement_text)
    logger.info("Cloud docs search keywords: %s", keywords)

    matched_urls = _match_urls_by_keywords(keywords, max_results=max_pages * 2,
                                           url_prefix="docs/")
    if not matched_urls:
        logger.warning("No cloud.ru/docs URLs matched keywords: %s", keywords)
        return []

    results: list[SearchResult] = []
    for url in matched_urls[:max_pages]:
        page = _fetch_page(url)
        if page is None or len(page.content) < 50:
            continue
        results.append(SearchResult(
            url=page.url,
            title=page.title,
            snippet=page.content[:300],
            content=page.content,
            source_type="cloud_docs",
        ))

    logger.info("Found %d cloud.ru/docs pages for requirement", len(results))
    return results


def search_cloud_documents(requirement_text: str, max_pages: int = 5) -> list[SearchResult]:
    """Search cloud.ru/documents for legal/compliance docs (SLA, certificates, licenses).

    Same approach as search_cloud_docs but scoped to /documents/ path.
    """
    keywords = _generate_search_keywords(requirement_text)
    logger.info("Cloud documents search keywords: %s", keywords)

    matched_urls = _match_urls_by_keywords(keywords, max_results=max_pages * 2,
                                           url_prefix="documents/")

    # Also try direct page fetch for cloud.ru/documents root
    if not matched_urls:
        # Try broader search across all cloud.ru URLs
        matched_urls = _match_urls_by_keywords(keywords, max_results=max_pages * 2)
        matched_urls = [u for u in matched_urls if "/documents/" in u]

    if not matched_urls:
        logger.warning("No cloud.ru/documents URLs matched keywords: %s", keywords)
        return []

    results: list[SearchResult] = []
    for url in matched_urls[:max_pages]:
        page = _fetch_page(url)
        if page is None or len(page.content) < 50:
            continue
        results.append(SearchResult(
            url=page.url,
            title=page.title,
            snippet=page.content[:300],
            content=page.content,
            source_type="cloud_documents",
        ))

    logger.info("Found %d cloud.ru/documents pages", len(results))
    return results


# ── Web search (DuckDuckGo) ──────────────────────────────────────────────────


def search_web(query: str, max_results: int = 5,
               site_scope: str | None = "cloud.ru") -> list[SearchResult]:
    """Search the web via DuckDuckGo.

    Args:
        site_scope: if set, prepend "site:<domain>" to scope results.
                    Set to None for unrestricted search.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []

    # Scope search to trusted domain when possible
    search_query = f"site:{site_scope} {query}" if site_scope else query

    results: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(search_query, region="ru-ru", max_results=max_results))

        for r in search_results:
            url = r.get("href", "")
            results.append(SearchResult(
                url=url,
                title=r.get("title", ""),
                snippet=r.get("body", ""),
                content=r.get("body", ""),  # DuckDuckGo gives snippets only
                source_type="web",
            ))
    except Exception as e:
        logger.warning("Web search failed: %s", e)

    # Filter to trusted domains
    results = _filter_trusted_results(results)

    logger.info("Found %d trusted web results for query: %s", len(results), search_query[:80])
    return results


# ── RAG fallback ─────────────────────────────────────────────────────────────


def _fallback_to_rag(requirement_text: str, max_results: int = 5) -> list[SearchResult]:
    """Fallback: search the local FAISS RAG index when live search yields no results."""
    from src.knowledge_base.store import get_vectorstore, search as rag_search

    vs = get_vectorstore()
    if not vs:
        return []

    docs = rag_search(requirement_text, k=max_results)
    results: list[SearchResult] = []
    for doc in docs:
        url = doc.metadata.get("url", doc.metadata.get("source", ""))
        title = doc.metadata.get("title", "")
        results.append(SearchResult(
            url=url,
            title=title,
            snippet=doc.page_content[:300],
            content=doc.page_content,
            source_type="rag_fallback",
        ))
    if results:
        logger.info("RAG fallback returned %d results", len(results))
    return results


# ── Unified search dispatcher ────────────────────────────────────────────────

# Categories that should search cloud.ru/docs first
CLOUD_DOCS_CATEGORIES = {"technical", "sla"}

# Categories that should also search cloud.ru/documents (legal/compliance)
CLOUD_DOCUMENTS_CATEGORIES = {"security", "legal"}

# Categories that search web + RAG
WEB_SEARCH_CATEGORIES = {"commercial", "other"}


def search_for_requirement(
    requirement_text: str,
    category: str,
    max_results: int = 5,
) -> list[SearchResult]:
    """Route search based on requirement category.

    Search chain (results are accumulated, not replaced):
    - technical/sla → cloud.ru/docs → RAG fallback
    - security/legal → cloud.ru/documents + cloud.ru/docs → web (site:cloud.ru) → RAG fallback
    - commercial/other → web (site:cloud.ru) → RAG fallback
    """
    results: list[SearchResult] = []

    if category in CLOUD_DOCS_CATEGORIES:
        # Technical/SLA: search docs, then RAG
        results = search_cloud_docs(requirement_text, max_pages=max_results)
        if not results:
            logger.info("No cloud.ru/docs results, trying web site:cloud.ru for: %s",
                        requirement_text[:80])
            results = search_web(requirement_text[:200], max_results=max_results,
                                 site_scope="cloud.ru")
        if not results:
            results = _fallback_to_rag(requirement_text, max_results=max_results)

    elif category in CLOUD_DOCUMENTS_CATEGORIES:
        # Security/Legal: search documents + docs + scoped web
        results = search_cloud_documents(requirement_text, max_pages=max_results)
        docs_results = search_cloud_docs(requirement_text, max_pages=max_results)
        results.extend(docs_results)

        if not results:
            logger.info("No cloud.ru results, trying web site:cloud.ru for: %s",
                        requirement_text[:80])
            results = search_web(requirement_text[:200], max_results=max_results,
                                 site_scope="cloud.ru")
        if not results:
            results = _fallback_to_rag(requirement_text, max_results=max_results)

    else:
        # Commercial/Other: web scoped to cloud.ru, then RAG
        results = search_web(requirement_text[:200], max_results=max_results,
                             site_scope="cloud.ru")
        if not results:
            # Try broader web search but still filter
            results = search_web(
                f"Cloud.ru {requirement_text[:200]}", max_results=max_results,
                site_scope=None,
            )
        if not results:
            results = _fallback_to_rag(requirement_text, max_results=max_results)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            unique.append(r)
    return unique[:max_results]
