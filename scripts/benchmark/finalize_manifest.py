#!/usr/bin/env python3
"""Create a checksummed benchmark artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--status", choices=("passed", "failed"), required=True)
    parser.add_argument("--error")
    parser.add_argument("--metadata", action="append", default=[])
    arguments = parser.parse_args()

    results_dir = arguments.results_dir.resolve(strict=True)
    metadata: dict[str, str] = {}
    for value in arguments.metadata:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid --metadata value: {value!r}")
        metadata[key] = item
    files = []
    for path in sorted(results_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(results_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "schema_version": 1,
        "platform": arguments.platform,
        "system": arguments.system,
        "architecture": arguments.architecture,
        "commit": arguments.commit,
        "status": arguments.status,
        "error": arguments.error,
        "metadata": metadata,
        "files": files,
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
