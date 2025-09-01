from urllib.parse import urldefrag, urljoin

from url_normalize import url_normalize


def normalize_url(url: str) -> str:
    # 去除 fragment、規一化大小寫/編碼
    u = urldefrag(url)[0]
    return url_normalize(u)


def join_url(base: str, href: str) -> str:
    return normalize_url(urljoin(base, href))
