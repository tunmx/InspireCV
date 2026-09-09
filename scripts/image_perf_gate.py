#!/usr/bin/env python3
"""Compare Image benchmark throughput using alternating repeated runs."""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


ROW_PATTERN = re.compile(
    r"^(u8|f32)\s+(\S+)\s+(\d+x\d+)\s+(\d+x\d+)\s+"
    r"(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$",
    re.MULTILINE,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--filter", default="[bench][image]")
    parser.add_argument("--output-dir", type=Path,
                        help="retain each raw report and its parsed throughput values")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--max-case-regression", type=float, default=0.05)
    parser.add_argument("--max-overall-regression", type=float, default=0.02)
    return parser.parse_args()


def run_benchmark(binary: Path, test_filter: str,
                  images_dir: Path, report_prefix: Optional[Path] = None) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="inspirecv-image-perf-") as directory:
        environment = os.environ.copy()
        environment["INSPIRECV_IMAGES_DIR"] = str(images_dir)
        # PNG encoding between cases changes allocator/cache/thermal state;
        # visual reports belong in a separate correctness run, not this gate.
        environment["INSPIRECV_IMAGE_BENCHMARK_SAVE_IMAGES"] = "0"
        subprocess.run(
            [str(binary), test_filter], cwd=directory, env=environment,
            check=True, text=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        report = Path(directory) / "image_benchmark.txt"
        if not report.is_file():
            raise RuntimeError(f"image benchmark report missing for {binary}")
        measurements = {}
        contents = report.read_text(encoding="utf-8")
        for dtype, operation, input_size, output_size, loops, _, _, throughput in \
                ROW_PATTERN.findall(contents):
            key = "/".join((dtype, operation, input_size, output_size, loops))
            measurements[key] = float(throughput)
        if report_prefix is not None:
            with report_prefix.with_suffix(".txt").open("x", encoding="utf-8") as stream:
                stream.write(contents)
            with report_prefix.with_suffix(".json").open("x", encoding="utf-8") as stream:
                json.dump(measurements, stream, indent=2, sort_keys=True)
    if not measurements:
        raise RuntimeError(f"no image benchmark rows found for {binary}")
    return measurements


def main() -> int:
    arguments = parse_arguments()
    if arguments.runs < 3:
        raise SystemExit("--runs must be at least 3")
    baseline = arguments.baseline.resolve(strict=True)
    candidate = arguments.candidate.resolve(strict=True)
    images_dir = arguments.images_dir.resolve(strict=True)
    if arguments.output_dir is not None:
        arguments.output_dir.mkdir(parents=True, exist_ok=False)
    samples = ({}, {})

    def append(target: dict[str, list[float]], values: dict[str, float]) -> None:
        for key, value in values.items():
            target.setdefault(key, []).append(value)

    for run_index in range(arguments.runs):
        order = ((baseline, samples[0], "baseline"), (candidate, samples[1], "candidate"))
        if run_index % 2:
            order = tuple(reversed(order))
        for binary, target, label in order:
            prefix = (arguments.output_dir / f"{label}-{run_index:02d}"
                      if arguments.output_dir is not None else None)
            append(target, run_benchmark(binary, arguments.filter, images_dir, prefix))
        print(f"round {run_index + 1}/{arguments.runs} complete", flush=True)

    if samples[0].keys() != samples[1].keys():
        raise RuntimeError("baseline and candidate benchmark rows differ")

    ratios = []
    failed = []
    print("case  baseline median  candidate median  paired median delta")
    for key in sorted(samples[0]):
        baseline_median = statistics.median(samples[0][key])
        candidate_median = statistics.median(samples[1][key])
        # Preserve the alternating run pairs. Comparing independent medians can
        # turn a frequency or thermal shift between rounds into a false binary
        # regression, especially for sub-millisecond operations.
        paired_ratios = [candidate / baseline for baseline, candidate in zip(
            samples[0][key], samples[1][key])]
        ratio = statistics.median(paired_ratios)
        ratios.append(ratio)
        delta = ratio - 1.0
        print(f"{key}  {baseline_median:.3f} MPix/s  "
              f"{candidate_median:.3f} MPix/s  {delta:+.2%}")
        if delta < -arguments.max_case_regression:
            failed.append(key)

    overall_ratio = math.exp(statistics.mean(math.log(value) for value in ratios))
    overall_delta = overall_ratio - 1.0
    print(f"overall geometric-mean throughput delta: {overall_delta:+.2%}")
    if failed:
        print(f"FAIL: per-case regression exceeded limit: {failed}")
        return 1
    if overall_delta < -arguments.max_overall_regression:
        print("FAIL: overall regression exceeded limit")
        return 1
    print("PASS: Image performance is within configured limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
