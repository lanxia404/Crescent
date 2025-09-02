#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crescent - Build Corpus (clean, fixed)
 - 讀取已抓取的網站資料 (HTML/JSON/JSONL/txt)
 - 抽正文、過濾標題、行級/文件級全域去重（不限網站/語言）
 - 即時日誌輸出
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from html import unescape
from typing import Generator, List, Set, Tuple


# ---------- Logging ----------
class LineFlushHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        stream = self.stream
        stream.write(msg + "\n")
        try:
            stream.flush()
        except Exception:
            pass


def setup_logging():
    fmt = "[Crescent][build_corpus] %(message)s"
    logger = logging.getLogger()
    logger.handlers.clear()
    handler = LineFlushHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------- Unicode helpers ----------
# Python re 不支援 \p{...}；改用 unicodedata 來判斷類別：
# P* = 標點, S* = 符號
def strip_punct_and_symbols(s: str) -> str:
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if not (cat.startswith("P") or cat.startswith("S")):
            out.append(ch)
    return "".join(out)


_SEP_RE = re.compile(r"[|\-—–•·]+")


def normalize_for_compare(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = _SEP_RE.sub(" ", s)
    s = strip_punct_and_symbols(s)
    s = " ".join(s.lower().split())
    return s


def normalize_for_hash(s: str) -> str:
    return normalize_for_compare(s)


# ---------- Hashing ----------
def h64(data: str) -> int:
    d = hashlib.blake2b(data.encode("utf-8"), digest_size=8)
    return int.from_bytes(d.digest(), "big", signed=False)


def hdoc(data: str) -> bytes:
    d = hashlib.blake2b(data.encode("utf-8"), digest_size=16)
    return d.digest()


# ---------- HTML 解析 ----------
def _bs_extract(html: str) -> Tuple[str, List[str]]:
    """
    優先使用 BeautifulSoup（若可用），抽文本與標題候選。
    回傳 (text, title_candidates)
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return "", []

    parser = "lxml"
    try:
        import lxml  # noqa: F401
    except Exception:
        parser = "html.parser"

    soup = BeautifulSoup(html, parser)

    # 移除噪音節點
    for tag in soup(["script", "style", "noscript", "template", "nav", "footer", "aside", "form"]):
        tag.decompose()

    titles: List[str] = []
    if soup.title and soup.title.string:
        titles.append(soup.title.string.strip())
    for sel in [
        ("meta", {"property": "og:title"}),
        ("meta", {"name": "og:title"}),
        ("meta", {"name": "twitter:title"}),
        ("meta", {"property": "twitter:title"}),
    ]:
        m = soup.find(*sel)
        if m and m.get("content"):
            titles.append(m.get("content").strip())
    for tagname in ["h1", "h2"]:
        h = soup.find(tagname)
        if h:
            t = h.get_text(" ", strip=True)
            if t:
                titles.append(t)
                break

    text = soup.get_text("\n", strip=True)
    return text, titles


_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style|noscript|template).*?</\1>")
_TAG_RE = re.compile(r"(?is)<[^>]+>")


def _regex_extract(html: str) -> Tuple[str, List[str]]:
    titles: List[str] = []
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if m:
        titles.append(unescape(m.group(1)).strip())
    html = _SCRIPT_STYLE_RE.sub("\n", html)
    text = _TAG_RE.sub("\n", html)
    text = unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return text, titles


def extract_text_and_titles_from_html(html: str) -> Tuple[str, List[str]]:
    if not html:
        return "", []
    try:
        text, titles = _bs_extract(html)
        if text:
            return text, titles
    except Exception:
        pass
    return _regex_extract(html)


# ---------- 讀取輸入 ----------
def iter_paths(in_path: str) -> Generator[str, None, None]:
    if os.path.isdir(in_path):
        for root, _, files in os.walk(in_path):
            for fn in files:
                yield os.path.join(root, fn)
    else:
        yield in_path


def is_gz(p: str) -> bool:
    return p.endswith(".gz")


def open_auto(p: str):
    if is_gz(p):
        return gzip.open(p, "rt", encoding="utf-8", errors="ignore")
    return open(p, "r", encoding="utf-8", errors="ignore")


def process_jsonl(path: str) -> Generator[Tuple[str, List[str]], None, None]:
    with open_auto(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            html = obj.get("html") or ""
            text = obj.get("text") or ""
            titles: List[str] = []
            if obj.get("title"):
                titles.append(str(obj.get("title")))
            if html:
                txt, ts = extract_text_and_titles_from_html(html)
                titles.extend(ts)
                yield (txt, titles)
            elif text:
                yield (str(text), titles)


def process_json(path: str) -> Generator[Tuple[str, List[str]], None, None]:
    try:
        with open_auto(path) as f:
            data = json.load(f)
    except Exception:
        return
    records = data if isinstance(data, list) else [data]
    for obj in records:
        if not isinstance(obj, dict):
            continue
        html = obj.get("html") or ""
        text = obj.get("text") or ""
        titles: List[str] = []
        if obj.get("title"):
            titles.append(str(obj.get("title")))
        if html:
            txt, ts = extract_text_and_titles_from_html(html)
            titles.extend(ts)
            yield (txt, titles)
        elif text:
            yield (str(text), titles)


def process_html(path: str) -> Generator[Tuple[str, List[str]], None, None]:
    try:
        with open_auto(path) as f:
            html = f.read()
    except Exception:
        return
    txt, titles = extract_text_and_titles_from_html(html)
    yield (txt, titles)


def process_txt(path: str) -> Generator[Tuple[str, List[str]], None, None]:
    try:
        with open_auto(path) as f:
            txt = f.read()
    except Exception:
        return
    yield (txt, [])


def load_docs(in_path: str) -> Generator[Tuple[str, List[str], str], None, None]:
    for p in iter_paths(in_path):
        lp = p.lower()
        if lp.endswith(".jsonl") or lp.endswith(".jsonl.gz"):
            for txt, titles in process_jsonl(p):
                yield (txt, titles, p)
        elif lp.endswith(".json") or lp.endswith(".json.gz"):
            for txt, titles in process_json(p):
                yield (txt, titles, p)
        elif (
            lp.endswith(".html")
            or lp.endswith(".htm")
            or lp.endswith(".html.gz")
            or lp.endswith(".htm.gz")
        ):
            for txt, titles in process_html(p):
                yield (txt, titles, p)
        elif lp.endswith(".txt"):
            for txt, titles in process_txt(p):
                yield (txt, titles, p)
        else:
            continue


# ---------- 淨化與過濾 ----------
def should_drop_line(line_norm: str, title_norms: Set[str], min_len: int) -> bool:
    if not line_norm:
        return True
    if len(line_norm) < min_len:
        return True
    if line_norm in title_norms:
        return True
    for t in title_norms:
        if t and (line_norm.startswith(t) and (len(line_norm) <= len(t) + 10)):
            return True
    return False


def clean_doc(
    text: str,
    titles: List[str],
    min_line_chars: int,
    filter_titles: bool,
    global_line_fps: Set[int],
) -> Tuple[List[str], int, int]:
    title_norms: Set[str] = set()
    if filter_titles:
        for t in titles:
            tn = normalize_for_compare(t)
            if tn:
                title_norms.add(tn)

    kept: List[str] = []
    seen_local: Set[int] = set()
    dropped = 0
    dup = 0

    raw_lines = [ln.strip() for ln in text.splitlines()]

    for ln in raw_lines:
        ln_unesc = unescape(ln).strip()
        if not ln_unesc:
            continue

        ln_norm = normalize_for_compare(ln_unesc)

        if should_drop_line(ln_norm, title_norms, min_line_chars):
            dropped += 1
            continue

        fp = h64(ln_norm)
        if (fp in global_line_fps) or (fp in seen_local):
            dup += 1
            continue

        seen_local.add(fp)
        global_line_fps.add(fp)
        kept.append(ln_unesc)

    return kept, dropped, dup


# ---------- 主程式 ----------
def main():
    parser = argparse.ArgumentParser(description="Build a cleaned corpus from crawled data.")
    parser.add_argument(
        "--in-path", required=True, help="資料來源（檔案或資料夾），可為 HTML/JSONL/JSON/txt"
    )
    parser.add_argument("--out-file", required=True, help="輸出語料檔路徑（txt）")
    parser.add_argument(
        "--min-line-chars", type=int, default=12, help="最短保留行字元數（規範化後）"
    )
    parser.add_argument("--no-filter-titles", action="store_true", help="不要根據標題候選過濾行")
    parser.add_argument("--no-dedup-lines", action="store_true", help="關閉行級去重（不建議）")
    parser.add_argument("--no-dedup-docs", action="store_true", help="關閉文件級去重（不建議）")
    parser.add_argument("--log-interval", type=int, default=500, help="每處理多少頁輸出一次進度")
    args = parser.parse_args()

    setup_logging()

    filter_titles = not args.no_filter_titles
    dedup_lines = not args.no_dedup_lines
    dedup_docs = not args.no_dedup_docs

    global_line_fps: Set[int] = set() if dedup_lines else set()
    doc_fps: Set[bytes] = set() if dedup_docs else set()

    total_files = 0
    total_pages = 0
    total_kept_lines = 0
    total_dropped_lines = 0
    total_dup_lines = 0
    skipped_empty = 0
    skipped_dup_doc = 0

    for _ in iter_paths(args.in_path):
        total_files += 1
    logging.info(f"scan inputs: files={total_files}, from={args.in_path}")

    with open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
        for i, (text, titles, src) in enumerate(load_docs(args.in_path), start=1):
            total_pages += 1
            if not text or not text.strip():
                skipped_empty += 1
                if total_pages % args.log_interval == 0:
                    logging.info(f"pages={total_pages} (empty+{skipped_empty}) ...")
                continue

            kept_lines, dropped, dup = clean_doc(
                text=text,
                titles=titles,
                min_line_chars=args.min_line_chars,
                filter_titles=filter_titles,
                global_line_fps=global_line_fps,
            )

            total_dropped_lines += dropped
            total_dup_lines += dup

            if not kept_lines:
                skipped_empty += 1
                if total_pages % args.log_interval == 0:
                    logging.info(f"pages={total_pages} (empty+{skipped_empty}) ...")
                continue

            page_text_norm = normalize_for_hash("\n".join(kept_lines))
            if dedup_docs:
                fp = hdoc(page_text_norm)
                if fp in doc_fps:
                    skipped_dup_doc += 1
                    if total_pages % args.log_interval == 0:
                        logging.info(
                            f"pages={total_pages} (dupdoc+{skipped_dup_doc}) kept_lines={total_kept_lines}"
                        )
                    continue
                doc_fps.add(fp)

            out.write("\n".join(kept_lines).rstrip() + "\n\n")
            total_kept_lines += len(kept_lines)

            if total_pages % args.log_interval == 0:
                logging.info(
                    "pages=%d kept_lines=%d dropped=%d dup_lines=%d dup_docs=%d empty=%d -> %s"
                    % (
                        total_pages,
                        total_kept_lines,
                        total_dropped_lines,
                        total_dup_lines,
                        skipped_dup_doc,
                        skipped_empty,
                        os.path.basename(args.out_file),
                    )
                )

    logging.info(
        "done pages=%d kept_lines=%d dropped=%d dup_lines=%d dup_docs=%d empty=%d out=%s"
        % (
            total_pages,
            total_kept_lines,
            total_dropped_lines,
            total_dup_lines,
            skipped_dup_doc,
            skipped_empty,
            args.out_file,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("interrupted by user")
        sys.exit(130)
