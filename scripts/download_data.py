"""Download the UCI SECOM dataset into data/raw/ and check it against pinned hashes.

Source: McCann, M. & Johnston, A. (2008). SECOM [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C54305
License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The raw files are not committed to this repo. Run:
    python scripts/download_data.py

If a file does not match the SHA-256 below, the script exits with status 1
and the numbers in reports/ should not be expected to reproduce.
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
EXPECTED_SHA256 = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
}
FILES = tuple(EXPECTED_SHA256)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    if all((RAW / f).exists() for f in FILES):
        print("already present:", ", ".join(FILES))
    else:
        print("downloading", URL)
        with urllib.request.urlopen(URL, timeout=120) as resp:
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in FILES:
                member = next(n for n in zf.namelist() if n.endswith(name))
                (RAW / name).write_bytes(zf.read(member))
    bad = []
    for f in FILES:
        digest = hashlib.sha256((RAW / f).read_bytes()).hexdigest()
        ok = digest == EXPECTED_SHA256[f]
        print(f"{f}  sha256={digest}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f)
    if bad:
        print(f"hash mismatch: {', '.join(bad)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
