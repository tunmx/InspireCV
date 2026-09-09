#!/usr/bin/env python3
"""Build, verify, and benchmark InspireCV on a native host."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--opencv-prefix", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        return f"unavailable: {error}"
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"


def cpu_description() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.partition(":")[2].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def find_opencv_dir(prefix: Path) -> Path:
    candidates = sorted(prefix.resolve().rglob("OpenCVConfig.cmake"))
    if not candidates:
        raise RuntimeError(f"OpenCVConfig.cmake is missing below {prefix}")
    return candidates[0].parent


def opencv_runtime_environment(prefix: Path) -> dict[str, str]:
    resolved = prefix.resolve()
    directories = {resolved / "bin", resolved / "lib"}
    for pattern in ("opencv_core*.dll", "libopencv_core*.so*", "libopencv_core*.dylib"):
        directories.update(path.parent for path in resolved.rglob(pattern))
    existing = [str(path) for path in sorted(directories) if path.is_dir()]
    environment: dict[str, str] = {}
    if existing:
        environment["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])
        if sys.platform.startswith("linux"):
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                existing + [os.environ.get("LD_LIBRARY_PATH", "")]
            )
        elif sys.platform == "darwin":
            environment["DYLD_LIBRARY_PATH"] = os.pathsep.join(
                existing + [os.environ.get("DYLD_LIBRARY_PATH", "")]
            )
    return environment


def executable(build_dir: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [build_dir / f"{name}{suffix}", build_dir / "Release" / f"{name}{suffix}"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"cannot find {name} below {build_dir}")


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_environment = os.environ.copy()
    if environment:
        merged_environment.update(environment)
    print("+", subprocess.list2cmdline(command), flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + subprocess.list2cmdline(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=merged_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\nelapsed_seconds={time.monotonic() - started:.3f}\n")
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        print("\n".join(tail), file=sys.stderr)
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(source_dir: Path, arguments: list[str], fallback: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() or fallback


def write_summary(results_dir: Path, platform_label: str) -> None:
    lines = [
        f"# InspireCV benchmark: {platform_label}",
        "",
        "Correctness tests and all benchmark commands completed successfully.",
        "",
        "| CPU suite | InspireCV wins | OpenCV wins | Ties | Accuracy failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in sorted((results_dir / "cpu").glob("*_summary.csv")):
        rows = list(csv.DictReader(summary.open(encoding="utf-8")))
        winners = [row.get("winner_vs_opencv", "") for row in rows]
        accuracy_failures = sum(row.get("accuracy") == "failed" for row in rows)
        lines.append(
            f"| {summary.stem.removesuffix('_summary')} | "
            f"{winners.count('inspirecv')} | {winners.count('opencv')} | "
            f"{winners.count('tie')} | {accuracy_failures} |"
        )
    lines.extend(
        [
            "",
            "Raw logs and per-case CSV files are included in this artifact. "
            "Hosted-runner timings are trend data, not release gates.",
            "",
        ]
    )
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    results_dir: Path,
    metadata: dict[str, object],
    status: str,
    error: str | None = None,
) -> None:
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
    payload = dict(metadata)
    payload.update({"status": status, "error": error, "files": files})
    (results_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    arguments = parse_arguments()
    if arguments.repeats < 1 or arguments.jobs < 1:
        raise ValueError("--repeats and --jobs must be positive")
    if arguments.samples < 3 or arguments.samples % 2 == 0:
        raise ValueError("--samples must be an odd integer of at least 3")
    if arguments.warmups < 1:
        raise ValueError("--warmups must be positive")

    source_dir = arguments.source_dir.resolve()
    build_dir = arguments.build_dir.resolve()
    results_dir = arguments.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "platform": arguments.platform_label,
        "system": platform.system(),
        "architecture": platform.machine(),
        "cpu": cpu_description(),
        "python": platform.python_version(),
        "cmake": command_version(["cmake", "--version"]),
        "compiler": command_version([os.environ.get("CXX", "c++"), "--version"]),
        "commit": os.environ.get("GITHUB_SHA")
        or git_value(source_dir, ["rev-parse", "HEAD"], "unknown"),
        "ref": os.environ.get("GITHUB_REF")
        or git_value(source_dir, ["symbolic-ref", "--short", "HEAD"], "detached"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        "repeats": arguments.repeats,
        "samples": arguments.samples,
        "warmups": arguments.warmups,
        "opencv_version": "4.5.5",
    }
    try:
        opencv_dir = find_opencv_dir(arguments.opencv_prefix)
        configure = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DOpenCV_DIR={opencv_dir}",
            "-DINSPIRECV_BUILD_OBJECT_LIBS=OFF",
            "-DINSPIRECV_BUILD_SHARED_LIBS=OFF",
            "-DINSPIRECV_BUILD_TESTS=ON",
            "-DINSPIRECV_BUILD_CPU_BENCHMARKS=ON",
            "-DINSPIRECV_BUILD_SAMPLE=OFF",
            "-DINSPIRECV_BUILD_EXAMPLES=OFF",
            "-DINSPIRECV_INSTALL=OFF",
        ]
        if os.name != "nt" and shutil.which("ninja"):
            configure.extend(["-G", "Ninja"])
        run_logged(configure, results_dir / "logs" / "configure.log", cwd=source_dir)
        run_logged(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--parallel",
                str(arguments.jobs),
            ],
            results_dir / "logs" / "build.log",
            cwd=source_dir,
        )
        test_environment = {
            **opencv_runtime_environment(arguments.opencv_prefix),
            "INSPIRECV_IMAGES_DIR": str(source_dir / "images"),
            "INSPIRECV_GT_DIR": str(source_dir / "images" / "task_gt"),
            "INSPIRECV_IMAGE_BENCHMARK_SAVE_IMAGES": "0",
        }
        run_logged(
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-C",
                "Release",
                "--output-on-failure",
            ],
            results_dir / "correctness" / "ctest.log",
            cwd=source_dir,
            environment=test_environment,
        )

        tests = executable(build_dir, "inspirecv_tests")
        benchmark = executable(build_dir, "inspirecv_cpu_benchmark")
        for repeat in range(1, arguments.repeats + 1):
            repeat_name = f"run_{repeat:02d}"
            image_dir = results_dir / "image" / repeat_name
            image_dir.mkdir(parents=True, exist_ok=True)
            run_logged(
                [str(tests), "[bench][image]"],
                image_dir / "console.log",
                cwd=image_dir,
                environment=test_environment,
            )
            task_dir = results_dir / "task" / repeat_name
            task_dir.mkdir(parents=True, exist_ok=True)
            task_environment = dict(test_environment)
            task_environment["INSPIRECV_BENCH_OUTPUT"] = str(
                task_dir / "format_benchmark.txt"
            )
            run_logged(
                [str(tests), "[bench][no_check]"],
                task_dir / "format_console.log",
                cwd=task_dir,
                environment=task_environment,
            )
            run_logged(
                [str(tests), "[benchmark][task_tensor_view][nchw]"],
                task_dir / "tensor_view_console.log",
                cwd=task_dir,
                environment=test_environment,
            )

        cpu_dir = results_dir / "cpu"
        cpu_dir.mkdir(parents=True, exist_ok=True)
        for suite in ("full", "u8c3", "matrix"):
            reports = []
            for repeat in range(1, arguments.repeats + 1):
                report = cpu_dir / f"{suite}_run_{repeat:02d}.csv"
                reports.append(report)
                run_logged(
                    [
                        str(benchmark),
                        "--suite",
                        suite,
                        "--samples",
                        str(arguments.samples),
                        "--warmups",
                        str(arguments.warmups),
                        "--opencv-threads",
                        "0",
                        "--machine",
                        arguments.platform_label,
                        "--report",
                        str(report),
                    ],
                    cpu_dir / f"{suite}_run_{repeat:02d}.log",
                    cwd=source_dir,
                    environment=test_environment,
                )
            run_logged(
                [
                    sys.executable,
                    str(source_dir / "scripts" / "cpu_benchmark_opencv_summary.py"),
                    "--reports",
                    *[str(path) for path in reports],
                    "--candidate-label",
                    "inspirecv",
                    "--output",
                    str(cpu_dir / f"{suite}_summary.csv"),
                ],
                cpu_dir / f"{suite}_summary.log",
                cwd=source_dir,
            )
        write_summary(results_dir, arguments.platform_label)
        write_manifest(results_dir, metadata, "passed")
        return 0
    except Exception as error:
        write_manifest(results_dir, metadata, "failed", str(error))
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
