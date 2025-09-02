#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crescent Crawler (fixed, lint-clean, live logs)
- Sitemap-aware, robots-aware, zh-first filter, shard JSONL writer
- Replaces deprecated tldextract.registered_domain with top_domain_under_public_suffix
"""

import gzip
import logging
import os
import queue
import re
import sys
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from typing import Iterator, List, Optional, Set, Tuple

import requests
import tldextract
from bs4 import BeautifulSoup


# ----------------------------
# Logging (immediate/line-flushed)
# ----------------------------
class LineFlushHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        stream = self.stream
        stream.write(msg + "\n")
        try:
            stream.flush()
        except Exception:
            pass


logger = logging.getLogger("crescent.crawl")
logger.setLevel(logging.INFO)
_handler = LineFlushHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [_handler]
logger.propagate = False


# ----------------------------
# Helpers
# ----------------------------
def top_domain_under_public_suffix(url: str) -> str:
    ext = tldextract.extract(url)
    # New property name (tldextract >= 5)
    if hasattr(ext, "top_domain_under_public_suffix"):
        return ext.top_domain_under_public_suffix  # type: ignore[attr-defined]
    # Fallback for older tldextract
    if hasattr(ext, "registered_domain"):
        return ext.registered_domain  # type: ignore[attr-defined]
    # Last resort
    parts = [p for p in (ext.domain, ext.suffix) if p]
    return ".".join(parts)


def hostname_of(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or ""


def is_cjk_text(s: str) -> bool:
    # Heuristic: contains CJK Unified Ideographs or common Chinese punctuations
    return bool(re.search(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F]", s))


def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith("#") or href.lower().startswith("javascript:"):
        return None
    return urllib.parse.urljoin(base, href)


def extract_links(url: str, html: str) -> List[str]:
    out: List[str] = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href")
        u = normalize_url(url, href) if href else None
        if u:
            out.append(u)
    return out


def extract_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    # remove noise
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    # Wikipedia: drop navigation elements if obvious (best-effort)
    main = soup.find(id="mw-content-text") or soup.body
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    # normalize whitespace
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    body = "\n".join(lines)
    return title, body


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_sitemaps_from_robots(
    session: requests.Session, base_root: str, timeout: float
) -> List[str]:
    robots_url = urllib.parse.urljoin(base_root, "/robots.txt")
    logger.info(f"[Crescent][crawl] robots: GET {robots_url}")
    try:
        r = session.get(robots_url, timeout=timeout)
        if r.status_code != 200:
            logger.info(
                f"[Crescent][crawl] robots: status={r.status_code}, fallback to /sitemap.xml"
            )
            return [urllib.parse.urljoin(base_root, "/sitemap.xml")]
        sitemaps: List[str] = []
        for line in r.text.splitlines():
            line = line.strip()
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemaps.append(sm)
        if not sitemaps:
            sitemaps = [urllib.parse.urljoin(base_root, "/sitemap.xml")]
        return sitemaps
    except Exception as e:
        logger.info(f"[Crescent][crawl] robots error: {e}, fallback to /sitemap.xml")
        return [urllib.parse.urljoin(base_root, "/sitemap.xml")]


def parse_sitemap_xml(content: bytes) -> List[str]:
    # Accept both urlset and sitemapindex; tolerate missing namespaces
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    urls: List[str] = []

    # With namespace
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    for u in root.findall(".//sm:url/sm:loc", ns):
        if u.text:
            urls.append(u.text.strip())
    for u in root.findall(".//sm:sitemap/sm:loc", ns):
        if u.text:
            urls.append(u.text.strip())

    # Fallback: no namespace
    if not urls:
        for u in root.findall(".//url/loc"):
            if u.text:
                urls.append(u.text.strip())
        for u in root.findall(".//sitemap/loc"):
            if u.text:
                urls.append(u.text.strip())

    return urls


def fetch_url(session: requests.Session, url: str, timeout: float) -> Optional[requests.Response]:
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        return r
    except Exception as e:
        logger.info(f"[Crescent][crawl] GET fail: {url} ({e})")
        return None


def load_sitemap_urls(
    session: requests.Session,
    sitemaps: List[str],
    timeout: float,
    max_pages: int,
) -> Iterator[str]:
    seen_sm: Set[str] = set()
    q: "queue.Queue[str]" = queue.Queue()
    for sm in sitemaps:
        q.put(sm)

    collected = 0
    while not q.empty() and collected < max_pages:
        sm = q.get()
        if sm in seen_sm:
            continue
        seen_sm.add(sm)
        logger.info(f"[Crescent][crawl] sitemap: GET {sm}")
        r = fetch_url(session, sm, timeout)
        if not r or r.status_code != 200:
            logger.info(f"[Crescent][crawl] sitemap: status={getattr(r, 'status_code', None)} skip")
            continue
        data = r.content
        if sm.endswith(".gz"):
            try:
                data = gzip.decompress(data)
            except OSError:
                pass
        links = parse_sitemap_xml(data)
        if not links:
            continue
        # If it's a sitemap index, links may be sitemaps; enqueue them.
        is_index = all(
            link.endswith(".xml") or link.endswith(".xml.gz") for link in links[:10]
        )  # heuristic
        if is_index:
            for nxt in links:
                if nxt not in seen_sm:
                    q.put(nxt)
