"""Download NASA C-MAPSS files into data/reference/ and verify them against MANIFEST.json.

Usage:
    python scripts/download_cmapps.py            # download missing files, verify all
    python scripts/download_cmapps.py --verify   # verify only, no network
    python scripts/download_cmapps.py --write-manifest  # (maintainers) record new checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# NASA's Open Data portal does not serve stable per-file URLs, so files come from a public mirror
# and are pinned by SHA-256 in MANIFEST.json; a mismatch is a hard failure.
BASE_URL = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
FILES = tuple(
    f"{kind}_{domain}.txt" for domain in DOMAINS for kind in ("train", "test", "RUL")
)
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
MANIFEST = REFERENCE_DIR / "MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = target_dir / filename
        if destination.exists():
            continue
        print(f" - fetching {filename}")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", destination)


def verify(target_dir: Path, manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    problems = []
    for filename, expected in manifest.items():
        path = target_dir / filename
        if not path.exists():
            problems.append(f"{filename}: missing")
        elif sha256(path) != expected["sha256"]:
            problems.append(f"{filename}: checksum mismatch")
    return problems


def write_manifest(target_dir: Path, manifest_path: Path) -> None:
    files = {}
    for filename in FILES:
        path = target_dir / filename
        lines = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        files[filename] = {"sha256": sha256(path), "bytes": path.stat().st_size, "rows": lines}
    payload = {"source": BASE_URL, "dataset": "NASA C-MAPSS turbofan degradation", "files": files}
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--verify", action="store_true", help="verify only, do not download")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        download(args.dest)
    if args.write_manifest:
        write_manifest(args.dest, MANIFEST)
        print(f"wrote {MANIFEST}")
        return 0
    problems = verify(args.dest, MANIFEST)
    for problem in problems:
        print(f"   {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"verified {len(FILES)} C-MAPSS files in {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
