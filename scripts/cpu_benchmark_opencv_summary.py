#!/usr/bin/env python3
"""Aggregate repeated InspireCV-versus-OpenCV CPU benchmark reports."""

import argparse
import csv
import statistics
import sys
from pathlib import Path

from cpu_benchmark_summary import IDENTITY_FIELDS, validate_group


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-label", default="inspirecv")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def median(rows: tuple[dict[str, str], ...], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def spread_percent(values: list[float]) -> float:
    return (max(values) / min(values) - 1.0) * 100.0


def main() -> int:
    arguments = parse_arguments()
    reports = validate_group(arguments.reports)
    cases = list(zip(*reports))
    destination = (
        arguments.output.open("w", encoding="utf-8", newline="")
        if arguments.output
        else sys.stdout
    )
    fieldnames = list(IDENTITY_FIELDS) + [
        f"{arguments.candidate_label}_p50_us",
        f"{arguments.candidate_label}_p95_us",
        f"{arguments.candidate_label}_p50_run_spread_pct",
        "opencv_p50_us",
        "opencv_p95_us",
        "opencv_p50_run_spread_pct",
        f"{arguments.candidate_label}_vs_opencv",
        "winner_vs_opencv",
        "accuracy",
        "max_abs_error",
        "mismatches",
    ]
    writer = csv.DictWriter(destination, fieldnames=fieldnames)
    writer.writeheader()
    for rows in cases:
        candidate_p50 = median(rows, "inspirecv_p50_us")
        opencv_p50 = median(rows, "opencv_p50_us")
        candidate_p50_runs = [float(row["inspirecv_p50_us"]) for row in rows]
        opencv_p50_runs = [float(row["opencv_p50_us"]) for row in rows]
        speedup = opencv_p50 / candidate_p50
        winner = (
            arguments.candidate_label
            if speedup > 1.02
            else "opencv" if speedup < 0.98 else "tie"
        )
        row = {field: rows[0][field] for field in IDENTITY_FIELDS}
        row.update(
            {
                f"{arguments.candidate_label}_p50_us": f"{candidate_p50:.3f}",
                f"{arguments.candidate_label}_p95_us":
                    f"{median(rows, 'inspirecv_p95_us'):.3f}",
                f"{arguments.candidate_label}_p50_run_spread_pct":
                    f"{spread_percent(candidate_p50_runs):.2f}",
                "opencv_p50_us": f"{opencv_p50:.3f}",
                "opencv_p95_us": f"{median(rows, 'opencv_p95_us'):.3f}",
                "opencv_p50_run_spread_pct":
                    f"{spread_percent(opencv_p50_runs):.2f}",
                f"{arguments.candidate_label}_vs_opencv": f"{speedup:.4f}",
                "winner_vs_opencv": winner,
                "accuracy": rows[0]["accuracy"],
                "max_abs_error":
                    f"{max(float(item['max_abs_error']) for item in rows):.7f}",
                "mismatches": str(
                    max(int(item["mismatches"]) for item in rows)
                ),
            }
        )
        writer.writerow(row)
    if arguments.output:
        destination.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
