import asyncio

import httpx
from aiolimiter import AsyncLimiter


class Fetcher:
    def __init__(self, ua="CrescentCrawler/0.1", max_concurrency=8, rate_per_sec=2):
        self.ua = ua
        self.sem = asyncio.Semaphore(max_concurrency)
        self.limiter = AsyncLimiter(rate_per_sec, time_period=1)

    async def get(self, url: str, timeout=20):
        async with self.sem, self.limiter:
            async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": self.ua}) as cli:
                r = await cli.get(url, follow_redirects=True)
                r.raise_for_status()
                return r
