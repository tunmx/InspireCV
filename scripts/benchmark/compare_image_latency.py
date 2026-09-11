#!/usr/bin/env python3
"""Allocation-inclusive, isolated Image latency comparison; standard library only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(source.read()).hexdigest()


def validate(record, case, sample_count):
    if record.get("case") != case:
        raise ValueError("probe returned wrong case")
    samples = record.get("samples_us", [])
    if len(samples) != sample_count or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("invalid timing samples")
    if not isinstance(record.get("loops"), int) or record["loops"] <= 0:
        raise ValueError("invalid loop count")
    if not math.isclose(record.get("median_us", -1), statistics.median(samples), rel_tol=1e-7):
        raise ValueError("inconsistent median")
    if not re.fullmatch(r"[0-9a-f]{16}", record.get("hash", "")):
        raise ValueError("invalid output hash")
    return record


def summarize(case, pairs, expected_change=False, control=False, case_limit=5):
    before = [a["median_us"] for a, _ in pairs]
    after = [b["median_us"] for _, b in pairs]
    ratios = [b / a for a, b in zip(before, after)]
    ratio = statistics.median(ratios)
    before_hashes = {a["hash"] for a, _ in pairs}
    after_hashes = {b["hash"] for _, b in pairs}
    deterministic = len(before_hashes) == len(after_hashes) == 1
    same_output = before_hashes == after_hashes
    limit = 1 + case_limit / 100
    perf_pass = ratio <= limit and (not control or ratio >= 1 / limit)
    output_pass = deterministic and (same_output or (expected_change and not control))
    return {"case": case, "baseline_us": statistics.median(before),
            "candidate_us": statistics.median(after), "ratio": ratio,
            "change_pct": (ratio - 1) * 100, "paired_ratios": ratios,
            "baseline_hashes": sorted(before_hashes), "candidate_hashes": sorted(after_hashes),
            "output_status": "MATCH" if deterministic and same_output else "EXPECTED_CHANGE" if output_pass else "FAIL",
            "perf_status": "PASS" if perf_pass else "UNSTABLE" if control else "REGRESSION",
            "pass": perf_pass and output_pass}


def environment():
    return {"platform": platform.platform(), "machine": platform.machine(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "variables": {key: os.environ[key] for key in sorted(os.environ)
                          if key.startswith(("MALLOC_", "OMP_", "INSPIRECV_", "LD_", "DYLD_"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--filter", default=".", help="case ID regular expression")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--batch-us", type=float, default=5000)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--timeout", type=float, default=60, help="seconds per process")
    parser.add_argument("--control", action="store_true", help="A/A: run the baseline on both sides")
    parser.add_argument("--aa-control", type=Path, help="summary.json from matching A/A run")
    parser.add_argument("--expected-output-change", default="(?!)", help="explicitly acknowledge known correctness fixes for matching case IDs")
    args = parser.parse_args()
    if args.runs < 3 or args.samples < 3 or args.samples > 101 or args.samples % 2 == 0 or not 100 <= args.batch_us <= 1000000 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("invalid timing parameters; use >=3 runs, odd samples in [3,101], batch-us in [100,1000000]")
    args.baseline = args.baseline.resolve(strict=True)
    args.candidate = args.baseline if args.control else args.candidate
    if args.candidate is None:
        parser.error("--candidate is required unless --control is used")
    args.candidate = args.candidate.resolve(strict=True)
    select, expected = re.compile(args.filter), re.compile(args.expected_output_change)
    def cases(binary):
        result = subprocess.run([str(binary), "--list"], text=True, capture_output=True, check=True, timeout=args.timeout)
        result = result.stdout.splitlines()
        if not result or len(set(result)) != len(result):
            raise ValueError("empty or duplicate case list")
        return sorted(result)
    before_cases, after_cases = cases(args.baseline), cases(args.candidate)
    if before_cases != after_cases:
        raise ValueError("baseline/candidate case coverage differs")
    selected = [case for case in before_cases if select.search(case)]
    if not selected:
        raise ValueError("filter selects no cases")
    # Never overwrite previous evidence, including an incomplete run.
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {"runs": args.runs, "batch_us": args.batch_us, "samples": args.samples,
                "case_limit_pct": 5, "overall_limit_pct": 2}
    metadata = {"schema": 1, "control": args.control, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "baseline": str(args.baseline), "candidate": str(args.candidate),
                "baseline_sha256": digest(args.baseline), "candidate_sha256": digest(args.candidate),
                "environment": environment(), "settings": settings, "cases": selected,
                "expected_output_change": args.expected_output_change,
                "timing": "warm-cache batch means, includes output allocation/destruction; drawing mutates a reused canvas"}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    control_valid = args.control
    if args.aa_control:
        control = json.loads(args.aa_control.read_text())
        control_valid = (control["status"] == "PASS" and control["metadata"]["control"]
                         and control["metadata"]["baseline_sha256"] == metadata["baseline_sha256"]
                         and control["metadata"]["environment"] == metadata["environment"]
                         and control["metadata"]["settings"] == settings
                         and set(selected) <= set(control["metadata"]["cases"]))
    rows = []
    for index, case in enumerate(selected):
        pairs = []
        for run in range(args.runs):
            pair = {}
            order = ("baseline", "candidate") if (index + run) % 2 == 0 else ("candidate", "baseline")
            for side in order:
                binary = args.baseline if side == "baseline" else args.candidate
                command = [str(binary), "--case", case, str(args.batch_us), str(args.samples)]
                stem = args.output / f"{index:03d}-{run:02d}-{side}"
                # File streams preserve partial output even on a crash/timeout.
                with stem.with_suffix(".stdout").open("w") as stdout, stem.with_suffix(".stderr").open("w") as stderr:
                    subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=args.timeout)
                pair[side] = validate(json.loads(stem.with_suffix(".stdout").read_text()), case, args.samples)
            pairs.append((pair["baseline"], pair["candidate"]))
        row = summarize(case, pairs, bool(expected.search(case)), args.control)
        rows.append(row)
        with (args.output / "cases.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")
        if (index + 1) % 10 == 0 or index + 1 == len(selected):
            print(f"{index + 1}/{len(selected)} cases; {sum(not r['pass'] for r in rows)} outside gates", flush=True)
    geometric_ratio = math.exp(statistics.mean(math.log(row["ratio"]) for row in rows))
    overall_pass = geometric_ratio <= 1.02 and (not args.control or geometric_ratio >= 1 / 1.02)
    passed = all(row["pass"] for row in rows) and overall_pass
    status = ("PASS" if passed else "UNSTABLE") if args.control else "FAIL" if not passed else "PASS" if control_valid else "UNQUALIFIED"
    report = {"status": status, "control_valid": control_valid, "overall_change_pct": (geometric_ratio - 1) * 100,
              "metadata": metadata, "results": rows}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Isolated Image latency comparison", "", f"Status: {status}; matching A/A control: {control_valid}.", "",
             f"Geometric mean latency change: {(geometric_ratio - 1) * 100:+.2f}% (not application throughput).",
             "", "Baseline and candidate are InspireCV revisions, not OpenCV. Lower latency is better.", "",
             "| Case | InspireCV baseline µs | InspireCV candidate µs | Change | Performance | Output |",
             "|---|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['baseline_us']:.3f} | {row['candidate_us']:.3f} | {row['change_pct']:+.2f}% | {row['perf_status']} | {row['output_status']} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"{status}: {len(rows)} cases, geometric mean {(geometric_ratio - 1) * 100:+.2f}%; {args.output / 'summary.md'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
