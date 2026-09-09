#!/usr/bin/env python3
"""Combine platform benchmark artifacts into one auditable report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path, payload: dict[str, object]) -> None:
    for item in payload.get("files", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise RuntimeError(f"invalid file entry in {root / 'manifest.json'}")
        path = (root / item["path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(f"manifest path escapes artifact: {path}") from error
        if not path.is_file():
            raise RuntimeError(f"manifest file is missing: {path}")
        if path.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"manifest size mismatch: {path}")
        if sha256(path) != item.get("sha256"):
            raise RuntimeError(f"manifest checksum mismatch: {path}")


def main() -> int:
    arguments = parse_arguments()
    input_dir = arguments.input_dir.resolve(strict=True)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    for path in sorted(input_dir.rglob("manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        verify_manifest(path.parent, payload)
        payload["artifact_path"] = path.parent.relative_to(input_dir).as_posix()
        manifests.append(payload)
    if not manifests:
        raise RuntimeError(f"no manifest.json files found below {input_dir}")

    commits = {item.get("commit") for item in manifests}
    if len(commits) != 1:
        raise RuntimeError(f"artifacts contain multiple commits: {sorted(commits)}")

    combined_rows: list[dict[str, str]] = []
    for manifest in manifests:
        root = input_dir / manifest["artifact_path"]
        for summary in sorted(root.rglob("*_summary.csv")):
            with summary.open(encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    combined_rows.append(
                        {
                            "platform": str(manifest.get("platform", "unknown")),
                            "suite": summary.stem.removesuffix("_summary"),
                            **row,
                        }
                    )
    if combined_rows:
        fieldnames = list(combined_rows[0])
        with (output_dir / "cpu_all_platforms.csv").open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    lines = [
        "# InspireCV full benchmark",
        "",
        f"Commit: `{next(iter(commits))}`",
        "",
        "| Platform | System | Architecture | Status | Result files |",
        "|---|---|---|---|---:|",
    ]
    for item in manifests:
        lines.append(
            f"| {item.get('platform', 'unknown')} | {item.get('system', 'unknown')} | "
            f"{item.get('architecture', 'unknown')} | {item.get('status', 'unknown')} | "
            f"{len(item.get('files', []))} |"
        )

    if combined_rows:
        lines.extend(
            [
                "",
                "## CPU comparison overview",
                "",
                "| Platform | Suite | InspireCV wins | OpenCV wins | Ties | Cases |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in combined_rows:
            groups.setdefault((row["platform"], row["suite"]), []).append(row)
        for (platform_name, suite), rows in sorted(groups.items()):
            winners = [row.get("winner_vs_opencv", "") for row in rows]
            lines.append(
                f"| {platform_name} | {suite} | {winners.count('inspirecv')} | "
                f"{winners.count('opencv')} | {winners.count('tie')} | {len(rows)} |"
            )
    lines.extend(
        [
            "",
            "Every platform artifact contains raw logs, environment metadata, "
            "per-case reports, and SHA256 checksums. Hosted-runner values are intended "
            "for trend analysis because the underlying machines are shared.",
            "",
        ]
    )
    report = "\n".join(lines)
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    (output_dir / "manifests.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
