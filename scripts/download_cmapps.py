from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
FILES = [
    "train_FD001.txt",
    "train_FD002.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
]


def download_cmapps(dest_dir: Path | None = None) -> None:
    target_dir = dest_dir or (Path(__file__).resolve().parent.parent / "data" / "reference")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading NASA C-MAPSS reference datasets into {target_dir}...")
    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        dest_path = target_dir / filename
        if dest_path.exists() and dest_path.stat().st_size > 1000:
            print(f" - {filename} already exists ({dest_path.stat().st_size:,} bytes).")
            continue

        print(f" - Fetching {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"   Saved {filename} ({dest_path.stat().st_size:,} bytes)")
        except Exception as exc:  # noqa: BLE001
            print(f"   Failed to download {filename}: {exc}", file=sys.stderr)

    print("Download complete. You can now run `make benchmark` or `pytest`.")


if __name__ == "__main__":
    download_cmapps()
