#!/usr/bin/env python3
"""Reproduce the embedded sha256 in RESULTS/brief3/selection.json.

Hash definition (documented in RESULTS/brief3.md):
  SHA-256 of UTF-8 bytes of
    json.dumps(body_without_sha256_field, indent=2, sort_keys=True) + "\\n"
  i.e. canonical key order, pretty-printed JSON of the sealed body *excluding*
  the ``sha256`` field itself (but including ``utc_timestamp`` and all other
  fields). This is *not* ``sha256sum selection.json`` of the on-disk file
  (that hashes the file *with* the embedded digest, hence a different value).

Exit 0 if embedded digest matches recomputation; else exit 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def selection_body_digest(obj: dict) -> str:
    body = dict(obj)
    body.pop("sha256", None)
    blob = json.dumps(body, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_bytes_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path",
        nargs="?",
        default="RESULTS/brief3/selection.json",
        type=Path,
    )
    args = ap.parse_args()
    path: Path = args.path
    obj = json.loads(path.read_text())
    embedded = obj.get("sha256")
    recomputed = selection_body_digest(obj)
    whole = file_bytes_digest(path)
    print(f"path:              {path}")
    print(f"embedded sha256:   {embedded}")
    print(f"recomputed (body): {recomputed}")
    print(f"sha256sum(file):   {whole}")
    print(
        "definition: SHA-256 of json.dumps(obj without 'sha256', "
        "indent=2, sort_keys=True) + '\\n'"
    )
    if embedded != recomputed:
        print("FAIL: embedded digest does not match recomputed body hash", file=sys.stderr)
        return 1
    print("OK: embedded digest matches body hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
