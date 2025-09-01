import re
from typing import List

import httpx
import tldextract

SITEMAP_HINTS = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap.txt"]


async def discover_sitemaps(base_url: str) -> List[str]:
    # 嘗試常見 sitemap 路徑
    host = tldextract.extract(base_url).registered_domain
    if not host:
        return []
    roots = [f"https://{host}"] if base_url.startswith("http") else [f"https://{base_url}"]
    cands = [r + h for r in roots for h in SITEMAP_HINTS]
    out = []
    async with httpx.AsyncClient(timeout=10) as cli:
        for u in cands:
            try:
                r = await cli.get(u)
                if r.status_code == 200 and r.text:
                    out.append(u)
            except Exception:
                pass
    return list(dict.fromkeys(out))


async def expand_sitemap(url: str) -> List[str]:
    # 粗略解析 xml/txt 中的 URL
    urls: List[str] = []
    try:
        async with httpx.AsyncClient(timeout=20) as cli:
            r = await cli.get(url)
            if r.status_code != 200:
                return []
            text = r.text
    except Exception:
        return []

    # 簡單匹配 <loc>...</loc> 或純文本行
    urls += re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text, re.I)
    if not urls:
        urls += [ln.strip() for ln in text.splitlines() if ln.strip().startswith("http")]
    return list(dict.fromkeys(urls))
