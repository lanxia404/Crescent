from datasketch import MinHash, MinHashLSH


def text_to_minhash(text: str, n_gram: int = 5) -> MinHash:
    mh = MinHash(num_perm=128)
    tokens = [text[i : i + n_gram] for i in range(max(1, len(text) - n_gram + 1))]
    for t in tokens:
        mh.update(t.encode("utf-8"))
    return mh


class Deduper:
    def __init__(self, threshold=0.9):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self._idx = 0

    def seen(self, text: str) -> bool:
        mh = text_to_minhash(text)
        # 查是否撞到近似文
        matches = self.lsh.query(mh)
        if matches:
            return True
        key = f"doc-{self._idx}"
        self._idx += 1
        self.lsh.insert(key, mh)
        return False
