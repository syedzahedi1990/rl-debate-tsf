"""Download ETT-small datasets (ETTh1/h2/m1/m2) into ./dataset/ETT-small/.

Source: https://github.com/zhouhaoyi/ETDataset (the canonical Informer-suite host).
"""

from __future__ import annotations

import os
import sys
import urllib.request


BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]


def main(target_dir: str = "dataset/ETT-small") -> int:
    os.makedirs(target_dir, exist_ok=True)
    for name in FILES:
        dest = os.path.join(target_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[ok] {dest} already present ({os.path.getsize(dest)} B)")
            continue
        url = f"{BASE}/{name}"
        print(f"[get] {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"[ok] {dest} ({os.path.getsize(dest)} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
