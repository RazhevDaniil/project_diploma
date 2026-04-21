"""Seed the knowledge base by crawling cloud.ru/docs and cloud.ru/documents.

This script:
1. Fetches the sitemap from cloud.ru/docs/sitemap.xml
2. Also tries cloud.ru/documents/sitemap.xml for legal/compliance docs
3. Crawls all documentation pages (or a subset via --max-pages)
4. Extracts text content from each page
5. Indexes everything into the FAISS vector store

Usage:
    python seed_knowledge_base.py                  # crawl all pages
    python seed_knowledge_base.py --max-pages 500  # crawl first 500 pages
    python seed_knowledge_base.py --clear           # clear index before crawling
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.knowledge_base.store import reset_vectorstore
from src.crawler.spider import (
    fetch_sitemap_urls,
    filter_docs_urls,
    crawl_docs_sync,
    index_crawled_pages,
)


def main():
    parser = argparse.ArgumentParser(description="Crawl cloud.ru/docs and index into FAISS")
    parser.add_argument("--max-pages", type=int, default=cfg.CRAWL_MAX_PAGES,
                        help="Max pages to crawl (0 = all)")
    parser.add_argument("--concurrency", type=int, default=cfg.CRAWL_CONCURRENCY,
                        help="Number of concurrent requests")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing index before crawling")
    args = parser.parse_args()

    if args.clear:
        print("Clearing existing index...")
        reset_vectorstore()

    print("=== Crawling cloud.ru/docs + cloud.ru/documents ===\n")

    print("Fetching sitemap from cloud.ru/docs...")
    all_urls = fetch_sitemap_urls()

    # Also try cloud.ru/documents sitemap
    documents_sitemap = cfg.CRAWL_DOCUMENTS_URL.rstrip("/") + "/sitemap.xml"
    try:
        print(f"Fetching sitemap from {documents_sitemap}...")
        documents_urls = fetch_sitemap_urls(documents_sitemap)
        all_urls.extend(documents_urls)
        print(f"Found {len(documents_urls)} additional URLs from cloud.ru/documents")
    except Exception as e:
        print(f"No sitemap at {documents_sitemap} ({e}), will crawl /documents/ pages directly")
        # Add known cloud.ru/documents URLs for direct crawling
        all_urls.append(cfg.CRAWL_DOCUMENTS_URL)

    doc_urls = filter_docs_urls(all_urls)
    print(f"Found {len(doc_urls)} total documentation URLs")

    if args.max_pages > 0:
        doc_urls = doc_urls[:args.max_pages]
        print(f"Limiting to {args.max_pages} pages")

    def progress(done, total):
        print(f"  Progress: {done}/{total} pages fetched ({done * 100 // total}%)")

    print(f"\nCrawling {len(doc_urls)} pages (concurrency={args.concurrency})...")
    pages = crawl_docs_sync(
        urls=doc_urls,
        max_pages=args.max_pages,
        concurrency=args.concurrency,
        progress_callback=progress,
    )
    print(f"\nExtracted content from {len(pages)} pages")

    print("\nIndexing into FAISS...")
    total_vectors = index_crawled_pages(pages)
    print(f"Done! Total vectors in index: {total_vectors}")


if __name__ == "__main__":
    main()
