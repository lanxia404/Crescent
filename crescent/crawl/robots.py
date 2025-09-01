import httpx
import tldextract


class RobotsCache:
    def __init__(self, ua: str = "CrescentCrawler/0.1"):
        self.ua = ua
        self._rules = {}  # host -> httpx.Robots

    async def allowed(self, url: str) -> bool:
        host = tldextract.extract(url).registered_domain
        if not host:
            return False
        if host not in self._rules:
            robots_url = f"https://{host}/robots.txt"
            try:
                async with httpx.AsyncClient(timeout=10) as cli:
                    r = await cli.get(robots_url, headers={"User-Agent": self.ua})
                    text = r.text if r.status_code == 200 else ""
            except Exception:
                text = ""
            # 極簡 parser：httpx 內建沒有 robots；用 python 的 urllib 也可，但同步。
            # 這裡做最小允許：若無 robots 或下載失敗，視為允許。
            self._rules[host] = text
        txt = self._rules[host]
        if not txt:
            return True
        # 粗略檢查：當 * Disallow: /
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("user-agent:"):
                ua = line.split(":", 1)[1].strip()
                allow_all = ua == "*" or self.ua.lower().startswith(ua.lower())
            if line.lower().startswith("disallow:") and allow_all:
                path = line.split(":", 1)[1].strip()
                if path == "/":
                    return False
        return True
