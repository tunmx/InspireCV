#!/usr/bin/env python3
"""Compare Task TensorView performance between two test binaries.

The binaries are run in alternating order to reduce thermal and scheduler
bias. A gate is evaluated from per-size medians instead of one noisy sample.
"""

import argparse
import math
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


RESULT_PATTERN = re.compile(
    r"\[TaskTensorViewBench\]\s+size=(\d+).*?average_us=([0-9.]+)"
)
BENCHMARK_FILTER = "[benchmark][task_tensor_view][nchw]"
FORMAT_BENCHMARK_FILTER = "[bench][no_check]"
FORMAT_RESULT_PATTERN = re.compile(r"^(.*?)\s+([0-9.]+) ms$", re.MULTILINE)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="working directory used by both Catch2 binaries",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="image fixture directory exported as INSPIRECV_IMAGES_DIR",
    )
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--format-runs", type=int, default=5)
    parser.add_argument(
        "--skip-formats",
        action="store_true",
        help="run only the synthetic TensorView benchmark",
    )
    parser.add_argument(
        "--max-case-regression",
        type=float,
        default=0.08,
        help="maximum allowed median regression for any size (default: 0.08)",
    )
    parser.add_argument(
        "--max-overall-regression",
        type=float,
        default=0.03,
        help="maximum geometric-mean regression (default: 0.03)",
    )
    parser.add_argument(
        "--max-format-case-regression",
        type=float,
        default=0.15,
        help="maximum median regression for format cases >= 0.05 ms",
    )
    parser.add_argument(
        "--max-format-overall-regression",
        type=float,
        default=0.05,
        help="maximum geometric-mean regression across format cases",
    )
    return parser.parse_args()


def run_benchmark(binary: Path, workdir: Path) -> dict[int, float]:
    result = subprocess.run(
        [str(binary), BENCHMARK_FILTER],
        cwd=workdir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    measurements = {
        int(size): float(average_us)
        for size, average_us in RESULT_PATTERN.findall(result.stdout)
    }
    if not measurements:
        raise RuntimeError(
            f"no Task benchmark results found in output from {binary}"
        )
    return measurements


def collect_samples(
    baseline: Path, candidate: Path, workdir: Path, runs: int
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    baseline_samples: dict[int, list[float]] = {}
    candidate_samples: dict[int, list[float]] = {}

    def append(target: dict[int, list[float]], values: dict[int, float]) -> None:
        for size, duration in values.items():
            target.setdefault(size, []).append(duration)

    for run_index in range(runs):
        if run_index % 2 == 0:
            append(baseline_samples, run_benchmark(baseline, workdir))
            append(candidate_samples, run_benchmark(candidate, workdir))
        else:
            append(candidate_samples, run_benchmark(candidate, workdir))
            append(baseline_samples, run_benchmark(baseline, workdir))
    return baseline_samples, candidate_samples


def run_format_benchmark(
    binary: Path, workdir: Path, images_dir: Optional[Path]
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="inspirecv-task-perf-") as temporary:
        report = Path(temporary) / "formats.txt"
        environment = os.environ.copy()
        environment["INSPIRECV_BENCH_OUTPUT"] = str(report)
        if images_dir is not None:
            environment["INSPIRECV_IMAGES_DIR"] = str(images_dir)
        subprocess.run(
            [str(binary), FORMAT_BENCHMARK_FILTER],
            cwd=workdir,
            check=True,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        measurements = {
            label.strip(): float(duration_ms)
            for label, duration_ms in FORMAT_RESULT_PATTERN.findall(
                report.read_text(encoding="utf-8")
            )
        }
    if not measurements:
        raise RuntimeError(f"no format benchmark results found for {binary}")
    return measurements


def collect_format_samples(
    baseline: Path,
    candidate: Path,
    workdir: Path,
    images_dir: Optional[Path],
    runs: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    baseline_samples: dict[str, list[float]] = {}
    candidate_samples: dict[str, list[float]] = {}

    def append(target: dict[str, list[float]], values: dict[str, float]) -> None:
        for name, duration in values.items():
            target.setdefault(name, []).append(duration)

    for run_index in range(runs):
        if run_index % 2 == 0:
            append(baseline_samples, run_format_benchmark(baseline, workdir, images_dir))
            append(candidate_samples, run_format_benchmark(candidate, workdir, images_dir))
        else:
            append(candidate_samples, run_format_benchmark(candidate, workdir, images_dir))
            append(baseline_samples, run_format_benchmark(baseline, workdir, images_dir))
    return baseline_samples, candidate_samples


def main() -> int:
    arguments = parse_arguments()
    if arguments.runs < 3:
        raise SystemExit("--runs must be at least 3")
    if not arguments.skip_formats and arguments.format_runs < 3:
        raise SystemExit("--format-runs must be at least 3")

    baseline = arguments.baseline.resolve(strict=True)
    candidate = arguments.candidate.resolve(strict=True)
    workdir = arguments.workdir.resolve(strict=True)
    images_dir = (
        arguments.images_dir.resolve(strict=True)
        if arguments.images_dir is not None
        else None
    )
    baseline_samples, candidate_samples = collect_samples(
        baseline, candidate, workdir, arguments.runs
    )

    if baseline_samples.keys() != candidate_samples.keys():
        raise RuntimeError("baseline and candidate benchmark sizes differ")

    ratios: list[float] = []
    failed_sizes: list[int] = []
    print("size  baseline median  candidate median  delta")
    for size in sorted(baseline_samples):
        baseline_median = statistics.median(baseline_samples[size])
        candidate_median = statistics.median(candidate_samples[size])
        ratio = candidate_median / baseline_median
        ratios.append(ratio)
        delta = ratio - 1.0
        print(
            f"{size:>4}  {baseline_median:>13.5f} us"
            f"  {candidate_median:>14.5f} us  {delta:+7.2%}"
        )
        if delta > arguments.max_case_regression:
            failed_sizes.append(size)

    overall_ratio = math.exp(statistics.mean(math.log(value) for value in ratios))
    overall_delta = overall_ratio - 1.0
    print(f"overall geometric-mean delta: {overall_delta:+.2%}")

    failed = False
    if failed_sizes:
        print(f"FAIL: per-size regression exceeded limit: {failed_sizes}")
        failed = True
    if overall_delta > arguments.max_overall_regression:
        print("FAIL: overall regression exceeded limit")
        failed = True

    if not arguments.skip_formats:
        baseline_formats, candidate_formats = collect_format_samples(
            baseline, candidate, workdir, images_dir, arguments.format_runs
        )
        if baseline_formats.keys() != candidate_formats.keys():
            raise RuntimeError("baseline and candidate format benchmark cases differ")

        format_rows: list[tuple[float, str, float, float]] = []
        for name in baseline_formats:
            baseline_median = statistics.median(baseline_formats[name])
            candidate_median = statistics.median(candidate_formats[name])
            if baseline_median >= 0.05:
                format_rows.append(
                    (
                        candidate_median / baseline_median,
                        name,
                        baseline_median,
                        candidate_median,
                    )
                )

        format_overall_ratio = math.exp(
            statistics.mean(math.log(row[0]) for row in format_rows)
        )
        format_overall_delta = format_overall_ratio - 1.0
        print(
            f"format cases >= 0.05 ms: {len(format_rows)}, "
            f"geometric-mean delta: {format_overall_delta:+.2%}"
        )
        print("largest format deltas:")
        for ratio, name, baseline_median, candidate_median in sorted(
            format_rows, reverse=True
        )[:5]:
            print(
                f"  {ratio - 1.0:+7.2%}  {name}: "
                f"{baseline_median:.3f} -> {candidate_median:.3f} ms"
            )
        failed_format_cases = [
            name
            for ratio, name, _, _ in format_rows
            if ratio - 1.0 > arguments.max_format_case_regression
        ]
        if failed_format_cases:
            print(
                "FAIL: format-case regression exceeded limit: "
                f"{failed_format_cases}"
            )
            failed = True
        if format_overall_delta > arguments.max_format_overall_regression:
            print("FAIL: overall format regression exceeded limit")
            failed = True

    if failed:
        return 1
    print("PASS: Task performance is within configured limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
