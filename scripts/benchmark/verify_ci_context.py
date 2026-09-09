#!/usr/bin/env python3
"""Refuse to run benchmark CI outside the dedicated repository and branch."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="tunmx/inspirecv-benchmark")
    parser.add_argument("--ref", default="refs/heads/benchmark")
    arguments = parser.parse_args()
    actual_repository = os.environ.get("GITHUB_REPOSITORY", "")
    actual_ref = os.environ.get("GITHUB_REF", "")
    if (actual_repository, actual_ref) != (arguments.repository, arguments.ref):
        print(
            "benchmark context rejected: "
            f"repository={actual_repository!r}, ref={actual_ref!r}",
            file=sys.stderr,
        )
        return 2
    print(f"benchmark context accepted: {actual_repository}@{actual_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
