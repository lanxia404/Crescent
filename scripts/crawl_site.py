#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bs4 import BeautifulSoup
from langdetect import LangDetectException, detect
from tldextract import extract

from crescent.crawl.dedup import Deduper
from crescent.crawl.extract import extract_text
from crescent.crawl.fetch import Fetcher
from crescent.crawl.normalize import join_url, normalize_url
from crescent.crawl.robots import RobotsCache
from crescent.crawl.sitemap import discover_sitemaps, expand_sitemap


def same_site(u: str, base_host: str) -> bool:
    h = extract(u).registered_domain
    return h == base_host


async def crawl(args):
    seeds = []
    for s in args.seeds:
        if s.startswith("http"):
            seeds.append(normalize_url(s))
        else:
            seeds.append("https://" + s.strip().strip("/"))
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "pages.jsonl")
    txt_path = os.path.join(out_dir, "corpus.txt")

    # 準備
    base_host = extract(seeds[0]).registered_domain
    robots = RobotsCache(ua=args.user_agent)
    fetcher = Fetcher(ua=args.user_agent, max_concurrency=args.concurrency, rate_per_sec=args.rps)
    deduper = Deduper(threshold=args.dedup_thresh)
    seen_urls = set()
    queue = asyncio.Queue()

    for s in seeds:
        await queue.put(s)

    if args.use_sitemap:
        maps = []
        for s in seeds:
            maps += await discover_sitemaps(s)
        for m in maps:
            for u in await expand_sitemap(m):
                if same_site(u, base_host):
                    await queue.put(normalize_url(u))

    pages = 0

    async def worker():
        nonlocal pages
        while not queue.empty() and pages < args.max_pages:
            url = await queue.get()
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if not same_site(url, base_host):
                continue
            if not await robots.allowed(url):
                continue
            try:
                r = await fetcher.get(url)
                ct = r.headers.get("content-type", "")
                if "text/html" not in ct:
                    continue
                html = r.text
                text = extract_text(html, url)
                if not text:
                    continue
                # 語言過濾
                try:
                    lang = detect(text[:1000])
                except LangDetectException:
                    lang = "unk"
                if args.lang and lang != args.lang:
                    continue
                if deduper.seen(text):
                    continue

                # 寫出
                rec = {"url": url, "lang": lang, "n_chars": len(text)}
                with (
                    open(jsonl_path, "a", encoding="utf-8") as fj,
                    open(txt_path, "a", encoding="utf-8") as ft,
                ):
                    fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ft.write(text + "\n\n")
                pages += 1

                # 擴展鏈結
                if pages < args.max_pages:
                    soup = BeautifulSoup(html, "lxml")
                    for a in soup.find_all("a", href=True):
                        href = a.get("href")
                        if href.startswith("mailto:") or href.startswith("javascript:"):
                            continue
                        nxt = join_url(url, href)
                        if nxt not in seen_urls:
                            await queue.put(nxt)
            except Exception:
                continue

    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    await asyncio.gather(*workers)
    print(f"[Crescent][crawl] done pages={pages}, out={out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True, help="起點 URL 或 domain（可多個）")
    ap.add_argument("--out-dir", default="data/crawl", help="輸出資料夾")
    ap.add_argument(
        "--lang", default="", help="只保留此語言（如 zh-cn/zh-tw/ja/en），用 langdetect 簡單判斷"
    )
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rps", type=float, default=2.0, help="每秒請求上限")
    ap.add_argument("--user-agent", default="CrescentCrawler/0.1")
    ap.add_argument("--use-sitemap", action="store_true", help="啟用 sitemap 擴展 URL")
    ap.add_argument("--dedup-thresh", type=float, default=0.9, help="近似去重閾值（MinHashLSH）")
    args = ap.parse_args()
    asyncio.run(crawl(args))


if __name__ == "__main__":
    main()
