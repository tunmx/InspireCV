#!/usr/bin/env python3
"""Build a small, pinned OpenCV installation for CPU comparisons."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="4.5.5")
    parser.add_argument(
        "--commit",
        default="dad26339a975b49cfb6c7dbe4bd5276c9dcb36e2",
        help="peeled commit for the version tag",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def find_package_directory(prefix: Path) -> Path:
    candidates = sorted(prefix.rglob("OpenCVConfig.cmake"))
    if not candidates:
        raise RuntimeError(f"OpenCVConfig.cmake is missing below {prefix}")
    return candidates[0].parent


def main() -> int:
    arguments = parse_arguments()
    if arguments.jobs < 1:
        raise ValueError("--jobs must be positive")

    install_dir = arguments.install_dir.resolve()
    work_dir = arguments.work_dir.resolve()
    marker = install_dir / (
        f".inspirecv-opencv-{arguments.version}-{arguments.commit}-shared.complete"
    )
    if marker.is_file():
        package_dir = find_package_directory(install_dir)
        print(f"OpenCV {arguments.version} cache: {package_dir}")
        return 0

    source_dir = work_dir / f"opencv-{arguments.version}-src"
    build_dir = work_dir / f"opencv-{arguments.version}-build"
    work_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.is_dir():
        run(
            [
                "git",
                "clone",
                "--branch",
                arguments.version,
                "--depth",
                "1",
                "https://github.com/opencv/opencv.git",
                str(source_dir),
            ]
        )
    actual_commit = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if actual_commit != arguments.commit:
        raise RuntimeError(
            f"OpenCV {arguments.version} resolved to {actual_commit}, "
            f"expected {arguments.commit}"
        )

    configure = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DBUILD_LIST=core,imgproc",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_JAVA=OFF",
        "-DBUILD_opencv_java=OFF",
        "-DBUILD_opencv_python2=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DBUILD_ZLIB=OFF",
        "-DWITH_1394=OFF",
        "-DWITH_CUDA=OFF",
        "-DWITH_FFMPEG=OFF",
        "-DWITH_GSTREAMER=OFF",
        "-DWITH_GTK=OFF",
        "-DWITH_IPP=OFF",
        "-DWITH_ITT=OFF",
        "-DWITH_JASPER=OFF",
        "-DWITH_JPEG=OFF",
        "-DWITH_OPENCL=OFF",
        "-DWITH_OPENEXR=OFF",
        "-DWITH_OPENJPEG=OFF",
        "-DWITH_OPENGL=OFF",
        "-DWITH_PNG=OFF",
        "-DWITH_QT=OFF",
        "-DWITH_TBB=OFF",
        "-DWITH_TIFF=OFF",
        "-DWITH_V4L=OFF",
        "-DWITH_WEBP=OFF",
    ]
    if os.name != "nt" and shutil.which("ninja"):
        configure.extend(["-G", "Ninja"])
    run(configure)
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "install",
            "--parallel",
            str(arguments.jobs),
        ]
    )
    package_dir = find_package_directory(install_dir)
    marker.write_text(str(package_dir) + "\n", encoding="utf-8")
    print(f"OpenCV {arguments.version}: {package_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
