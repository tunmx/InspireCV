#!/usr/bin/env python3
"""Run two affine_latency_probe binaries in alternating order and retain CSVs."""

import argparse
import csv
import math
from pathlib import Path
import statistics
import subprocess


def read_report(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["case"]: row for row in rows}
    if not rows or len(rows) != len(result):
        raise ValueError(f"empty report or duplicate cases: {path}")
    for row in rows:
        for field in ("median_us", "p95_us"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid latency: {path}, {row['case']}, {field}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=9)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--constant-border", action="store_true")
    mode.add_argument("--task-only", action="store_true")
    mode.add_argument("--unaffected", action="store_true")
    parser.add_argument("--max-case-regression", type=float, default=0.05)
    parser.add_argument("--max-overall-regression", type=float, default=0.02)
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    binaries = {"baseline": args.baseline.resolve(strict=True),
                "candidate": args.candidate.resolve(strict=True)}
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {key: [] for key in binaries}
    for run in range(args.runs):
        order = ("baseline", "candidate") if run % 2 == 0 else ("candidate", "baseline")
        for name in order:
            path = args.output / f"{name}-{run:02d}.csv"
            # Refuse to overwrite prior measurements; each experiment gets its
            # own output directory, including failed performance gates.
            with path.open("x", encoding="utf-8") as output:
                command = [str(binaries[name])]
                if args.constant_border:
                    command.append("--constant-border")
                if args.task_only:
                    command.append("--task-only")
                if args.unaffected:
                    command.append("--unaffected")
                subprocess.run(command, stdout=output, check=True)
            reports[name].append(read_report(path))
        print(f"round {run + 1}/{args.runs} complete", flush=True)
    keys = reports["baseline"][0].keys()
    for collection in reports.values():
        if any(report.keys() != keys for report in collection):
            raise ValueError("baseline/candidate cases differ")
    ratios, failures = [], []
    with (args.output / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "baseline_median_us", "candidate_median_us",
                         "baseline_p95_us", "candidate_p95_us", "paired_latency_delta_percent"))
        print("case | baseline us | candidate us | paired latency delta")
        for key in sorted(keys):
            before = [float(row[key]["median_us"]) for row in reports["baseline"]]
            after = [float(row[key]["median_us"]) for row in reports["candidate"]]
            ratio = statistics.median(b / a for a, b in zip(before, after))
            ratios.append(ratio)
            medians = (statistics.median(before), statistics.median(after))
            tails = tuple(statistics.median(float(row[key]["p95_us"]) for row in reports[name])
                          for name in ("baseline", "candidate"))
            writer.writerow((key, *medians, *tails, (ratio - 1) * 100))
            print(f"{key} | {medians[0]:.6f} | {medians[1]:.6f} | {ratio - 1:+.2%}")
            if ratio - 1 > args.max_case_regression:
                failures.append(key)
    overall = math.exp(statistics.mean(map(math.log, ratios))) - 1
    print(f"overall geometric-mean latency delta: {overall:+.2%}")
    if failures or overall > args.max_overall_regression:
        print(f"FAIL: per-case violations: {failures}")
        return 1
    print("PASS: latency is within configured limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
