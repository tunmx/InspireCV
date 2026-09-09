#!/usr/bin/env python3
"""Aggregate repeated InspireCV CPU benchmark reports."""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


IDENTITY_FIELDS = (
    "family",
    "operation",
    "type",
    "channels",
    "source_width",
    "source_height",
    "destination_width",
    "destination_height",
    "timing_scope",
    "contract",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--family", help="only emit rows from this benchmark family")
    parser.add_argument(
        "--minimum-p50-speedup",
        type=float,
        help="fail if any gated row has baseline/candidate P50 below this value",
    )
    parser.add_argument(
        "--minimum-p95-speedup",
        type=float,
        help="fail if any gated row has baseline/candidate P95 below this value",
    )
    parser.add_argument(
        "--minimum-geomean-speedup",
        type=float,
        help="fail if the gated rows' geometric-mean P50 speedup is below this value",
    )
    parser.add_argument(
        "--gate-exclude-operation",
        action="append",
        default=[],
        help="operation to keep in the report but exclude from performance gates",
    )
    parser.add_argument(
        "--gate-operation",
        action="append",
        default=[],
        help="when provided, apply performance gates only to these operations",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    for name in (
        "minimum_p50_speedup",
        "minimum_p95_speedup",
        "minimum_geomean_speedup",
    ):
        value = getattr(arguments, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return arguments


def read_report(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    with path.resolve(strict=True).open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("# "):
                key, separator, value = line[2:].strip().partition("=")
                if separator:
                    metadata[key] = value
            elif line.strip():
                data_lines.append(line)
    rows = list(csv.DictReader(data_lines))
    if not rows:
        raise RuntimeError(f"no benchmark rows in {path}")
    failed = [row for row in rows if row["accuracy"] == "failed"]
    if failed:
        names = ", ".join(row["operation"] for row in failed)
        raise RuntimeError(f"accuracy gate failed in {path}: {names}")
    return metadata, rows


def identity(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in IDENTITY_FIELDS)


def validate_group(paths: list[Path]) -> list[list[dict[str, str]]]:
    reports = [read_report(path) for path in paths]
    reference = [identity(row) for row in reports[0][1]]
    reference_settings = {
        key: reports[0][0].get(key)
        for key in (
            "opencv",
            "suite",
            "samples",
            "warmups",
            "opencv_threads_requested",
        )
    }
    for path, (metadata, rows) in zip(paths[1:], reports[1:]):
        if [identity(row) for row in rows] != reference:
            raise RuntimeError(f"benchmark cases differ in {path}")
        settings = {key: metadata.get(key) for key in reference_settings}
        if settings != reference_settings:
            raise RuntimeError(
                f"benchmark settings differ in {path}: "
                f"{settings} != {reference_settings}"
            )
    return [rows for _, rows in reports]


def median(rows: tuple[dict[str, str], ...], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def case_name(row: dict[str, str]) -> str:
    return (
        f"{row['operation']} "
        f"{row['source_width']}x{row['source_height']}->"
        f"{row['destination_width']}x{row['destination_height']}"
    )


def main() -> int:
    arguments = parse_arguments()
    baseline_reports = validate_group(arguments.baseline)
    candidate_reports = validate_group(arguments.candidate)
    if len(baseline_reports) != len(candidate_reports):
        raise RuntimeError("baseline and candidate repeat counts differ")

    baseline_cases = list(zip(*baseline_reports))
    candidate_cases = list(zip(*candidate_reports))
    if [identity(rows[0]) for rows in baseline_cases] != [
        identity(rows[0]) for rows in candidate_cases
    ]:
        raise RuntimeError("baseline and candidate benchmark cases differ")

    destination = (
        arguments.output.open("w", encoding="utf-8", newline="")
        if arguments.output
        else sys.stdout
    )
    fieldnames = list(IDENTITY_FIELDS) + [
        f"{arguments.baseline_label}_p50_us",
        f"{arguments.baseline_label}_p95_us",
        f"{arguments.candidate_label}_p50_us",
        f"{arguments.candidate_label}_p95_us",
        f"{arguments.candidate_label}_speedup",
        "opencv_p50_us",
        "opencv_p95_us",
        f"{arguments.candidate_label}_vs_opencv",
        "winner_vs_opencv",
        "accuracy",
        "max_abs_error",
    ]
    writer = csv.DictWriter(destination, fieldnames=fieldnames)
    writer.writeheader()
    gate_rows: list[tuple[str, float, float]] = []
    for baseline_rows, candidate_rows in zip(baseline_cases, candidate_cases):
        if arguments.family and candidate_rows[0]["family"] != arguments.family:
            continue
        baseline_p50 = median(baseline_rows, "inspirecv_p50_us")
        baseline_p95 = median(baseline_rows, "inspirecv_p95_us")
        candidate_p50 = median(candidate_rows, "inspirecv_p50_us")
        candidate_p95 = median(candidate_rows, "inspirecv_p95_us")
        opencv_p50 = median(candidate_rows, "opencv_p50_us")
        candidate_vs_opencv = opencv_p50 / candidate_p50
        winner_vs_opencv = (
            arguments.candidate_label
            if candidate_vs_opencv > 1.02
            else "opencv" if candidate_vs_opencv < 0.98 else "tie"
        )
        row = {field: candidate_rows[0][field] for field in IDENTITY_FIELDS}
        row.update(
            {
                f"{arguments.baseline_label}_p50_us": f"{baseline_p50:.3f}",
                f"{arguments.baseline_label}_p95_us":
                    f"{baseline_p95:.3f}",
                f"{arguments.candidate_label}_p50_us": f"{candidate_p50:.3f}",
                f"{arguments.candidate_label}_p95_us":
                    f"{candidate_p95:.3f}",
                f"{arguments.candidate_label}_speedup":
                    f"{baseline_p50 / candidate_p50:.4f}",
                "opencv_p50_us": f"{opencv_p50:.3f}",
                "opencv_p95_us":
                    f"{median(candidate_rows, 'opencv_p95_us'):.3f}",
                f"{arguments.candidate_label}_vs_opencv":
                    f"{candidate_vs_opencv:.4f}",
                "winner_vs_opencv": winner_vs_opencv,
                "accuracy": candidate_rows[0]["accuracy"],
                "max_abs_error":
                    f"{max(float(item['max_abs_error']) for item in candidate_rows):.7f}",
            }
        )
        writer.writerow(row)
        operation = candidate_rows[0]["operation"]
        included_by_gate_filter = (
            not arguments.gate_operation or operation in arguments.gate_operation
        )
        if (
            included_by_gate_filter
            and operation not in arguments.gate_exclude_operation
        ):
            gate_rows.append(
                (
                    case_name(candidate_rows[0]),
                    baseline_p50 / candidate_p50,
                    baseline_p95 / candidate_p95,
                )
            )
    if arguments.output:
        destination.close()

    failures: list[str] = []
    if arguments.minimum_p50_speedup is not None:
        failures.extend(
            f"{name}: P50 {p50:.4f}x < {arguments.minimum_p50_speedup:.4f}x"
            for name, p50, _ in gate_rows
            if p50 < arguments.minimum_p50_speedup
        )
    if arguments.minimum_p95_speedup is not None:
        failures.extend(
            f"{name}: P95 {p95:.4f}x < {arguments.minimum_p95_speedup:.4f}x"
            for name, _, p95 in gate_rows
            if p95 < arguments.minimum_p95_speedup
        )
    geometric_mean = (
        math.exp(sum(math.log(p50) for _, p50, _ in gate_rows) / len(gate_rows))
        if gate_rows
        else None
    )
    if arguments.minimum_geomean_speedup is not None:
        if geometric_mean is None:
            failures.append("geometric mean is unavailable because no rows were gated")
        elif geometric_mean < arguments.minimum_geomean_speedup:
            failures.append(
                f"P50 geometric mean {geometric_mean:.4f}x < "
                f"{arguments.minimum_geomean_speedup:.4f}x"
            )
    if failures:
        raise RuntimeError("performance gate failed:\n  " + "\n  ".join(failures))
    if any(
        value is not None
        for value in (
            arguments.minimum_p50_speedup,
            arguments.minimum_p95_speedup,
            arguments.minimum_geomean_speedup,
        )
    ):
        geometric_mean_text = (
            f"{geometric_mean:.4f}x" if geometric_mean is not None else "n/a"
        )
        print(
            f"performance gate passed: rows={len(gate_rows)}, "
            f"P50 geometric mean={geometric_mean_text}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
