"""Spider — crawls cloud.ru/docs via sitemap and extracts page content."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup, Tag

import config as cfg

logger = logging.getLogger(__name__)

# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class CrawledPage:
    """A single crawled documentation page."""

    url: str
    title: str
    breadcrumbs: list[str] = field(default_factory=list)
    content: str = ""  # cleaned plain text
    section_path: str = ""  # e.g. "Evolution / Object Storage / Начало работы"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "breadcrumbs": self.breadcrumbs,
            "content": self.content,
            "section_path": self.section_path,
        }

    @staticmethod
    def from_dict(d: dict) -> "CrawledPage":
        return CrawledPage(**d)


# ── Sitemap parsing ──────────────────────────────────────────────────────────

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def fetch_sitemap_urls(sitemap_url: str = cfg.CRAWL_SITEMAP_URL) -> list[str]:
    """Fetch all <loc> URLs from the sitemap XML."""
    logger.info("Fetching sitemap: %s", sitemap_url)
    resp = httpx.get(sitemap_url, timeout=30, follow_redirects=True)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)

    # Handle both sitemap index and urlset
    urls: list[str] = []

    # Check for sitemap index (contains other sitemaps)
    for sitemap in root.findall("sm:sitemap", SITEMAP_NS):
        loc = sitemap.find("sm:loc", SITEMAP_NS)
        if loc is not None and loc.text:
            urls.extend(fetch_sitemap_urls(loc.text.strip()))

    # Regular URL entries
    for url_el in root.findall("sm:url", SITEMAP_NS):
        loc = url_el.find("sm:loc", SITEMAP_NS)
        if loc is not None and loc.text:
            urls.append(loc.text.strip())

    logger.info("Found %d URLs in sitemap", len(urls))
    return urls


def filter_docs_urls(urls: list[str], base_url: str = cfg.CRAWL_BASE_URL) -> list[str]:
    """Keep only URLs under the docs and documents base paths."""
    documents_url = cfg.CRAWL_DOCUMENTS_URL
    filtered = [u for u in urls if u.startswith(base_url) or u.startswith(documents_url)]
    logger.info("Filtered to %d docs/documents URLs", len(filtered))
    return filtered


# ── Page content extraction ──────────────────────────────────────────────────


def _extract_content_from_html(html: str, url: str) -> CrawledPage:
    """Extract clean text content from a cloud.ru/docs HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    # Title: try h1 first, then <title>
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    # Breadcrumbs
    breadcrumbs: list[str] = []
    bc_nav = soup.find("nav", class_=re.compile(r"breadcrumb", re.I))
    if bc_nav:
        for a in bc_nav.find_all("a"):
            text = a.get_text(strip=True)
            if text:
                breadcrumbs.append(text)
    # Add title as last breadcrumb if not already there
    if title and (not breadcrumbs or breadcrumbs[-1] != title):
        breadcrumbs.append(title)

    section_path = " / ".join(breadcrumbs) if breadcrumbs else title

    # Main content: look for <section> with id (article body)
    content_parts: list[str] = []

    # Try the article section first
    article_section = soup.find("section", id=True)
    if article_section:
        content_parts.append(_extract_text_from_element(article_section))
    else:
        # Fallback: look for main content area
        main = soup.find("main") or soup.find("article")
        if main:
            content_parts.append(_extract_text_from_element(main))

    # Also try to extract content from Next.js RSC payload (pageData.body)
    if not content_parts or len("".join(content_parts).strip()) < 100:
        rsc_content = _extract_from_rsc_payload(soup)
        if rsc_content:
            content_parts.append(rsc_content)

    content = "\n\n".join(content_parts).strip()

    # Clean up excessive whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r" {2,}", " ", content)

    return CrawledPage(
        url=url,
        title=title,
        breadcrumbs=breadcrumbs,
        content=content,
        section_path=section_path,
    )


def _extract_text_from_element(el: Tag) -> str:
    """Extract readable text from an HTML element, preserving structure."""
    # Remove nav, footer, script, style elements
    for tag_name in ("nav", "footer", "script", "style", "noscript", "header"):
        for tag in el.find_all(tag_name):
            tag.decompose()

    lines: list[str] = []
    for child in el.descendants:
        if isinstance(child, str):
            text = child.strip()
            if text:
                lines.append(text)
        elif hasattr(child, "name"):
            if child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(child.name[1])
                prefix = "#" * level
                text = child.get_text(strip=True)
                if text:
                    lines.append(f"\n{prefix} {text}\n")
            elif child.name == "br":
                lines.append("\n")
            elif child.name in ("p", "div", "section", "article"):
                lines.append("\n")
            elif child.name == "li":
                lines.append("\n- ")
            elif child.name == "tr":
                lines.append("\n")
            elif child.name in ("td", "th"):
                lines.append(" | ")

    text = "".join(lines)
    # Clean up
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_from_rsc_payload(soup: BeautifulSoup) -> str:
    """Try to extract page body from Next.js RSC script payload."""
    for script in soup.find_all("script"):
        text = script.string or ""
        if "pageData" not in text and "body" not in text:
            continue
        # Look for HTML body content in the RSC payload
        body_match = re.search(r'"body"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if body_match:
            try:
                body_html = json.loads(f'"{body_match.group(1)}"')
                body_soup = BeautifulSoup(body_html, "html.parser")
                return _extract_text_from_element(body_soup)
            except (json.JSONDecodeError, Exception):
                continue
    return ""


# ── Caching ──────────────────────────────────────────────────────────────────


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _load_cached_page(url: str) -> Optional[CrawledPage]:
    cache_file = cfg.CRAWL_CACHE_DIR / f"{_cache_key(url)}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return CrawledPage.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def _save_cached_page(page: CrawledPage):
    cache_file = cfg.CRAWL_CACHE_DIR / f"{_cache_key(page.url)}.json"
    cache_file.write_text(json.dumps(page.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# ── Threaded crawler ─────────────────────────────────────────────────────────

_HTTP_HEADERS = {
    "User-Agent": "CloudRu-TZ-Analyzer/1.0 (internal tool)",
    "Accept": "text/html",
    "Accept-Language": "ru-RU,ru;q=0.9",
}


def _fetch_page(url: str) -> Optional[CrawledPage]:
    """Fetch and parse a single page (sync, thread-safe)."""
    # Check cache first
    cached = _load_cached_page(url)
    if cached is not None and cached.content:
        return cached

    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True, headers=_HTTP_HEADERS)
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None

    time.sleep(cfg.CRAWL_DELAY)

    page = _extract_content_from_html(resp.text, url)

    # Skip pages with no meaningful content
    if len(page.content) < 50:
        logger.debug("Skipping %s — too little content (%d chars)", url, len(page.content))
        return None

    _save_cached_page(page)
    return page


def crawl_docs_sync(
    urls: Optional[list[str]] = None,
    max_pages: int = cfg.CRAWL_MAX_PAGES,
    concurrency: int = cfg.CRAWL_CONCURRENCY,
    progress_callback=None,
) -> list[CrawledPage]:
    """Crawl documentation pages using a thread pool.

    Args:
        urls: List of URLs to crawl. If None, fetches from sitemap.
        max_pages: Max pages to crawl (0 = unlimited).
        concurrency: Max concurrent threads.
        progress_callback: Optional callable(done, total) for progress updates.
    """
    if urls is None:
        all_urls = fetch_sitemap_urls()
        urls = filter_docs_urls(all_urls)

    if max_pages > 0:
        urls = urls[:max_pages]

    total = len(urls)
    logger.info("Starting crawl of %d pages (concurrency=%d)", total, concurrency)

    pages: list[CrawledPage] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_fetch_page, url): url for url in urls}

        for future in as_completed(futures):
            done_count += 1
            try:
                page = future.result()
            except Exception as e:
                logger.warning("Error crawling %s: %s", futures[future], e)
                continue
            if page is not None:
                pages.append(page)
            if progress_callback and done_count % 50 == 0:
                progress_callback(done_count, total)

    if progress_callback:
        progress_callback(total, total)

    logger.info("Crawled %d pages with content out of %d total", len(pages), total)
    return pages


# ── Index crawled pages into knowledge base ──────────────────────────────────


def index_crawled_pages(pages: list[CrawledPage]) -> int:
    """Index crawled pages into the FAISS vector store."""
    from langchain_core.documents import Document
    from src.knowledge_base.store import create_or_update_vectorstore

    docs = []
    for page in pages:
        if not page.content.strip():
            continue
        docs.append(Document(
            page_content=page.content,
            metadata={
                "source": page.url,
                "title": page.title,
                "section_path": page.section_path,
                "url": page.url,
            },
        ))

    if not docs:
        logger.warning("No documents to index")
        return 0

    vs = create_or_update_vectorstore(docs)
    return vs.index.ntotal
