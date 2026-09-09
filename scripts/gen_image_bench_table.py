#!/usr/bin/env python3
import sys
import os
from typing import Dict, Tuple, List

RowKey = Tuple[str, str, str, str]  # (dtype, op, inWxH, outWxH)

PLATFORM_NAMES = {
    ("Linux", "x86_64"): "Linux-x86_64",
    ("Linux", "aarch64"): "Linux-Arm64",
    ("Linux", "arm64"): "Linux-Arm64",
    ("Darwin", "x86_64"): "MacOS-x86_64",
    ("Darwin", "arm64"): "MacOS-Arm64",
}

def platform_from_filename(path: str) -> str:
    """Best-effort platform mapping from filename when header is unreliable."""
    name = os.path.basename(path).lower()
    plat = ""
    if "ubuntu" in name or "linux" in name:
        if "arm64" in name or "aarch64" in name:
            plat = "Linux-Arm64"
        elif "x86_64" in name or "amd64" in name or "x86" in name:
            plat = "Linux-x86_64"
    elif "macos" in name or "darwin" in name or "osx" in name:
        # m1/m2 imply arm64
        if "arm64" in name or "m1" in name or "m2" in name or "aarch64" in name:
            plat = "MacOS-Arm64"
        elif "x86_64" in name or "amd64" in name or "x86" in name:
            plat = "MacOS-x86_64"
    return plat

def parse_ext_from_filename(path: str) -> str:
    """Extract extension token between OS and arch from benchmark filename."""
    name = os.path.basename(path)
    if name.startswith("image_benchmark_"):
        core = name[len("image_benchmark_"):]
    else:
        core = os.path.splitext(name)[0]
    if core.endswith(".txt"):
        core = core[:-4]
    # Split off arch by last underscore
    if "_" not in core:
        return ""
    before_arch, arch = core.rsplit("_", 1)
    # Identify OS prefix (ubuntu/linux/macos/darwin)
    lower = before_arch.lower()
    os_prefixes = ["ubuntu", "linux", "macos", "darwin"]
    os_prefix = next((p for p in os_prefixes if lower.startswith(p)), "")
    if not os_prefix:
        # Try split by first token
        parts = before_arch.split("_")
        if parts:
            os_prefix = parts[0].lower()
    if lower.startswith(os_prefix):
        ext = before_arch[len(os_prefix):]
        # Trim leading separators
        while ext.startswith("-") or ext.startswith("_"):
            ext = ext[1:]
    else:
        # Fallback: everything before arch minus first token as OS
        parts = before_arch.split("_")
        ext = "_".join(parts[1:]) if len(parts) > 1 else ""
    # Normalize common tokens
    if ext.lower() in ("m1", "m2"):
        ext = ext.upper()
    return ext

def detect_platform(header_lines: List[str]) -> str:
    system = ""
    arch = ""
    for ln in header_lines:
        if ln.startswith("System:"):
            # Example: "System: Linux (x86_64)"
            try:
                parts = ln.split(":", 1)[1].strip()
                if "(" in parts and parts.endswith(")"):
                    system = parts.split("(", 1)[0].strip()
                    arch = parts.split("(", 1)[1].strip(")")
                else:
                    system = parts.strip()
            except Exception:
                pass
            break
    plat = PLATFORM_NAMES.get((system, arch))
    if not plat:
        # Fallback to raw
        plat = f"{system or 'Unknown'}-{arch or 'Unknown'}"
    return plat

def parse_bench_file(path: str) -> Tuple[str, Dict[RowKey, float]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # Detect platform from header
    plat = platform_from_filename(path) or detect_platform(lines[:12])
    # Find header row index
    try:
        hdr_idx = lines.index("dtype  op  inWxH  outWxH  loops  total_ms  avg_ms  mpix_per_s")
    except ValueError:
        hdr_idx = -1
    rows: Dict[RowKey, float] = {}
    if hdr_idx >= 0:
        for ln in lines[hdr_idx+1:]:
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 8:
                continue
            dtype = parts[0]
            op = parts[1]
            in_wh = parts[2]
            out_wh = parts[3]
            # parts[4] loops, parts[5] total_ms, parts[6] avg_ms, parts[7] mpix/s
            try:
                avg_ms = float(parts[6])
            except Exception:
                continue
            key: RowKey = (dtype, op, in_wh, out_wh)
            rows[key] = avg_ms
    # Build tags: ext + enabled features
    # Parse features from header
    features = []
    def has_flag(key: str) -> bool:
        for ln in lines[:16]:
            if ln.startswith(key + ":"):
                val = ln.split(":", 1)[1].strip().lower()
                return val in ("true", "on", "yes", "1")
        return False
    if has_flag("LTO"):
        features.append("LTO")
    if has_flag("SSE"):
        features.append("SSE")
    if has_flag("AVX2"):
        features.append("AVX2")
    if has_flag("NEON"):
        features.append("NEON")
    ext = parse_ext_from_filename(path)
    tag = ""
    if ext:
        tag = ext
    if features:
        tag = (tag + "/" if tag else "") + "/".join(features)
    return plat, rows, tag

def size_key(wh: str) -> Tuple[int, int]:
    try:
        w, h = wh.lower().split('x')
        return (int(w), int(h))
    except Exception:
        return (10**9, 10**9)

def dtype_rank(dt: str) -> int:
    # Interleave by fp32 first then u8
    return 0 if dt == "f32" else 1

def base_op(op: str) -> str:
    # Remove dtype tokens 'f32'/'u8' from op to group same algorithms
    toks = [t for t in op.split('_') if t not in ("f32", "u8")]
    # Collapse consecutive empties if any
    return "_".join([t for t in toks if t])

def format_table(all_rows: Dict[RowKey, Dict[str, float]], platforms: List[str]) -> str:
    # Header
    out = []
    out.append("| dtype | op | inWxH | outWxH | " + " | ".join(platforms) + " |")
    out.append("|:-----:|:---|:-----:|:------:|" + "|".join([":---------:" for _ in platforms]) + "|")
    # Sort keys
    keys = sorted(
        all_rows.keys(),
        key=lambda k: (base_op(k[1]), size_key(k[2]), size_key(k[3]), dtype_rank(k[0]))
    )
    for k in keys:
        plat_vals = all_rows[k]
        cells = []
        for p in platforms:
            if p in plat_vals:
                cells.append(f"{plat_vals[p]:.3f} ms")
            else:
                cells.append("-")
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(cells) + " |")
    return "\n".join(out)

def main():
    if len(sys.argv) < 2:
        print("Usage: gen_image_bench_table.py <bench_file> [<bench_file> ...]")
        print("Example: gen_image_bench_table.py benchmark/image_benchmark_*.txt > benchmark/image_benchmark_compare.md")
        sys.exit(1)
    # Aggregate rows from all inputs
    platform_to_rows: Dict[str, Dict[RowKey, float]] = {}
    platform_tags: Dict[str, str] = {}
    for p in sys.argv[1:]:
        if not os.path.isfile(p):
            continue
        plat, rows, tag = parse_bench_file(p)
        if not rows:
            continue
        platform_to_rows[plat] = rows
        if tag:
            platform_tags[plat] = tag
    # Canonical platform order
    platforms = ["Linux-x86_64", "Linux-Arm64", "MacOS-x86_64", "MacOS-Arm64"]
    # Merge rows
    all_rows: Dict[RowKey, Dict[str, float]] = {}
    for plat, rows in platform_to_rows.items():
        for key, v in rows.items():
            if key not in all_rows:
                all_rows[key] = {}
            all_rows[key][plat] = v
    # Emit
    print("# Image API Benchmark Comparison")
    print()
    print("Platforms: " + ", ".join(platforms))
    print()
    print("Values: average time (ms)")
    print()
    # Device notes (extensions parsed from filenames)
    print("Devices:")
    for plat in platforms:
        ext = platform_tags.get(plat, "")
        if ext:
            # Normalize plat label
            label = plat.replace("_", " ")
            print(f"- **{label}**: {ext}")
    print()
    print(format_table(all_rows, platforms))

if __name__ == "__main__":
    main()


