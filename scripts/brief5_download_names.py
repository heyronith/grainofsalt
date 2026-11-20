#!/usr/bin/env python3
"""Download Step-2 names_final.csv from Modal volume after b5_kl finishes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "step2" / "names"


def main() -> None:
    import modal

    vol = modal.Volume.from_name("grainofsalt-vol")
    OUT.mkdir(parents=True, exist_ok=True)
    files = [
        "data/step2/names/names_final.csv",
        "data/step2/names/names_meta.json",
        "data/step2/names/familiarity_kl_summary.csv",
    ]
    with vol.batch_download() as batch:
        for rel in files:
            local = OUT / Path(rel).name
            batch.download_file(rel, str(local))
    # Fallback: read via open if batch_download API differs
    print("downloaded to", OUT)
    for f in OUT.iterdir():
        print(" ", f.name, f.stat().st_size)


if __name__ == "__main__":
    # Prefer read via modal volume read API used elsewhere
    try:
        main()
    except Exception as e:
        print("batch_download failed:", e)
        print("Trying modal.Volume.read…")
        import modal

        vol = modal.Volume.from_name("grainofsalt-vol")
        OUT.mkdir(parents=True, exist_ok=True)
        for name in ("names_final.csv", "names_meta.json", "familiarity_kl_summary.csv"):
            remote = f"/data/step2/names/{name}"
            # Volume paths are relative to volume root without leading slash in some APIs
            try:
                data = b"".join(vol.read_file(f"data/step2/names/{name}"))
                (OUT / name).write_bytes(data)
                print("got", name, len(data))
            except Exception as e2:
                print("miss", name, e2)
