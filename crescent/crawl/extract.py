from typing import Optional

import trafilatura


def extract_text(html: str, url: str) -> Optional[str]:
    # trafilatura 會做 boilerplate removal、main content 抽取
    res = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    if not res:
        return None
    # 清理掉過長空白
    text = "\n".join(line.strip() for line in res.splitlines() if line.strip())
    return text if text else None
