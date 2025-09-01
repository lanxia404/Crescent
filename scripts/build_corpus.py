#!/usr/bin/env python3
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/crawl", help="爬取輸出資料夾")
    ap.add_argument("--out-file", default="data/crawl_corpus.txt")
    args = ap.parse_args()

    parts = []
    for p in sorted(glob.glob(os.path.join(args.in_dir, "**", "corpus.txt"), recursive=True)):
        with open(p, "r", encoding="utf-8") as f:
            parts.append(f.read())
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    print(
        f"[Crescent][build_corpus] merged => {
            args.out_file} (from {
            len(parts)} shards)"
    )


if __name__ == "__main__":
    main()
