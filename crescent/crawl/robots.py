import httpx
import tldextract


class RobotsCache:
    def __init__(self, ua: str = "CrescentCrawler/0.1"):
        self.ua = ua
        self._rules_text = {}  # host -> robots.txt content (raw)

    async def allowed(self, url: str) -> bool:
        td = tldextract.extract(url)
        host = td.top_domain_under_public_suffix or td.registered_domain
        if not host:
            return False
        if host not in self._rules_text:
            robots_url = f"https://{host}/robots.txt"
            try:
                async with httpx.AsyncClient(timeout=10) as cli:
                    r = await cli.get(robots_url, headers={"User-Agent": self.ua})
                    text = r.text if r.status_code == 200 else ""
            except Exception:
                text = ""
            self._rules_text[host] = text

        txt = self._rules_text[host]
        if not txt:
            return True  # 無 robots 視為允許（保守可改 False）

        # 極簡判斷：若存在通用 UA 的 Disallow: /
        in_star = False
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if lower.startswith("user-agent:"):
                ua = lower.split(":", 1)[1].strip()
                in_star = ua == "*" or self.ua.lower().startswith(ua)
            elif lower.startswith("disallow:") and in_star:
                path = lower.split(":", 1)[1].strip()
                if path == "/":
                    return False
        return True
