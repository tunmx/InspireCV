#!/usr/bin/env python3
import sys
import os
import re
from typing import Dict, Tuple, List

# Platform mapping consistent with image bench script
PLATFORM_NAMES = {
    ("Linux", "x86_64"): "Linux-x86_64",
    ("Linux", "aarch64"): "Linux-Arm64",
    ("Linux", "arm64"): "Linux-Arm64",
    ("Darwin", "x86_64"): "MacOS-x86_64",
    ("Darwin", "arm64"): "MacOS-Arm64",
}

def platform_from_filename(path: str) -> str:
    name = os.path.basename(path).lower()
    plat = ""
    if "ubuntu" in name or "linux" in name:
        if "arm64" in name or "aarch64" in name:
            plat = "Linux-Arm64"
        elif "x86_64" in name or "amd64" in name or "x86" in name:
            plat = "Linux-x86_64"
    elif "macos" in name or "darwin" in name or "osx" in name:
        if "arm64" in name or "m1" in name or "m2" in name or "aarch64" in name:
            plat = "MacOS-Arm64"
        elif "x86_64" in name or "amd64" in name or "x86" in name or "intel" in name:
            plat = "MacOS-x86_64"
    return plat

def parse_ext_from_filename(path: str) -> str:
    """Extract extension token between OS and arch from task benchmark filename."""
    name = os.path.basename(path)
    # Expect like task_benchmark_<os>-<ext>_<arch>.txt
    if name.startswith("task_benchmark_"):
        core = name[len("task_benchmark_"):]
    else:
        core = os.path.splitext(name)[0]
    if core.endswith(".txt"):
        core = core[:-4]
    # Split off arch by last underscore
    if "_" not in core:
        return ""
    before_arch, arch = core.rsplit("_", 1)
    lower = before_arch.lower()
    os_prefixes = ["ubuntu", "linux", "macos", "darwin"]
    os_prefix = next((p for p in os_prefixes if lower.startswith(p)), "")
    if lower.startswith(os_prefix):
        ext = before_arch[len(os_prefix):]
        while ext.startswith("-") or ext.startswith("_"):
            ext = ext[1:]
    else:
        parts = before_arch.split("_")
        ext = "_".join(parts[1:]) if len(parts) > 1 else ""
    if ext.lower() in ("m1", "m2"):
        ext = ext.upper()
    return ext

def parse_task_bench_file(path: str) -> Tuple[str, Dict[str, float], str]:
    """Return platform, op->avg_ms, device_ext."""
    plat = platform_from_filename(path)
    ext = parse_ext_from_filename(path)
    rows: Dict[str, float] = {}
    if not os.path.isfile(path):
        return plat, rows, ext
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("loops=") or line.startswith("--"):
                continue
            # Match "<op>  <value> ms"
            m = re.match(r"^(.*\S)\s+([\d.]+)\s*ms$", line)
            if not m:
                continue
            op = m.group(1)
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            rows[op] = val
    return plat, rows, ext

def op_sort_key(op: str) -> Tuple[int, str]:
    # Put non-affine ops first, then affine ones (contain ' + ')
    return (0 if " + " not in op else 1, op)

def format_table(all_rows: Dict[str, Dict[str, float]], platforms: List[str], device_notes: Dict[str, str]) -> str:
    out: List[str] = []
    out.append("# StreamTask Benchmark Comparison")
    out.append("")
    out.append("Platforms: " + ", ".join(platforms))
    out.append("")
    out.append("Values: average time (ms)")
    out.append("")
    out.append("Devices:")
    for plat in platforms:
        ext = device_notes.get(plat, "")
        if ext:
            label = plat.replace("_", " ")
            out.append(f"- **{label}**: {ext}")
    out.append("")
    # Header
    out.append("| op | " + " | ".join(platforms) + " |")
    out.append("|:---|" + "|".join([":---------:" for _ in platforms]) + "|")
    # Sorted keys
    keys = sorted(all_rows.keys(), key=op_sort_key)
    for k in keys:
        plat_vals = all_rows[k]
        cells = []
        for p in platforms:
            if p in plat_vals:
                cells.append(f"{plat_vals[p]:.3f} ms")
            else:
                cells.append("-")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(out)

def main():
    if len(sys.argv) < 2:
        print("Usage: gen_task_bench_table.py <task_benchmark.txt> [<task_benchmark.txt> ...]")
        print("Example: gen_task_bench_table.py benchmark/task_benchmark_*.txt > benchmark/task_benchmark_compare.md")
        sys.exit(1)
    platform_to_rows: Dict[str, Dict[str, float]] = {}
    device_notes: Dict[str, str] = {}
    for p in sys.argv[1:]:
        plat, rows, ext = parse_task_bench_file(p)
        if not plat or not rows:
            continue
        platform_to_rows[plat] = rows
        if ext:
            device_notes[plat] = ext
    platforms = ["Linux-x86_64", "Linux-Arm64", "MacOS-x86_64", "MacOS-Arm64"]
    # Merge all ops
    all_rows: Dict[str, Dict[str, float]] = {}
    for plat, rows in platform_to_rows.items():
        for op, v in rows.items():
            if op not in all_rows:
                all_rows[op] = {}
            all_rows[op][plat] = v
    print(format_table(all_rows, platforms, device_notes))

if __name__ == "__main__":
    main()


