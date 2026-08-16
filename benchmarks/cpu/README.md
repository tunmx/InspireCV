# CPU comparison and optimization plan

The current CPU implementation is competitive on focused paths, but it does not
generally beat OpenCV. On Ryzen, the optimized u8 C3 rotate90, horizontal flip
and SwapRB routes now meet or beat OpenCV across the measured dimensions. Erode,
small Task tensor writers and several sampling paths still have substantial
headroom.

OpenCV's own imgproc performance suite covers operations such as
[`resize`](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/perf/perf_resize.cpp),
but it is an OpenCV regression suite rather than a comparison with another
library. This directory contains a direct comparison using the public APIs of
both libraries.

## Method

The benchmark uses deterministic synthetic pixels and does no file I/O. Source
creation is outside the timed region. Image rows include destination allocation
through the public API; Task rows use a preallocated destination and measure the
complete reused Pipeline execution. Each row has 10 warm-ups followed by 101
timed samples. InspireCV and OpenCV run in alternating order, and the tables use
the median of at least three complete runs. The final untargeted regression gate
uses seven runs. Raw reports retain P50 and P95.

| Item | Apple ARM64 | Linux x86-64 |
|---|---|---|
| CPU | Apple M4, 10 cores | AMD Ryzen 5 5600, 6 cores / 12 threads |
| OS | macOS 15.6.1 | Ubuntu 22.04, Linux 6.8 |
| Compiler | AppleClang 17 | GCC 11.4 |
| InspireCV | 1.0.1, Release `-O3` | 1.0.1, Release `-O3` |
| OpenCV | 5.0.0, NEON | 5.0.0, SSE4.1/AVX/AVX2 runtime dispatch |
| OpenCV options | core+imgproc, IPP/OpenCL/TBB off | core+imgproc, IPP/OpenCL/TBB off |
| Scheduling | public API with Apple's GCD | process pinned to logical CPU 2; OpenCV reports one thread |

OpenCV uses GCD automatically on macOS when TBB is absent. Its API documents
that positive `setNumThreads` values are unsupported for GCD, and the reports
correctly show 10 available threads after requesting one. The M4 table is
therefore a public-API comparison, not a strict one-core comparison. The M4
NEON isolation below is still valid because it compares two InspireCV Task
builds directly and those measured paths do not launch worker threads.

Accuracy is evaluated before timing:

- `exact` means every byte or float bit pattern matched.
- `within_tolerance` is used only for BGR-to-gray, with maximum u8 error 1.
- `different_contract` records the error but does not pretend the two
  algorithms are equivalent. This applies to the current linear-coordinate
  convention, non-integral nearest mapping, affine sampling and sigma-zero
  Gaussian kernel policy.
- Any aligned row outside its tolerance makes the executable return a failure.

All formal reports completed with no accuracy-gate failures.

## OpenCV 5.0.0 comparison

`Speedup` is `OpenCV P50 / InspireCV P50`; values above 1 mean InspireCV is
faster. Times are microseconds. The full 30-row matrix is in the raw CSV files.

### Apple M4, NEON build

OpenCV uses its default GCD scheduling in this table.

| Operation | InspireCV P50 / P95 | OpenCV P50 / P95 | Speedup | Accuracy |
|---|---:|---:|---:|---|
| Image nearest 512x512 -> 256x256 | 85.5 / 112.5 | 41.5 / 75.5 | 0.49x | exact |
| Image linear 512x512 -> 256x256 | 207.1 / 299.0 | 27.2 / 45.5 | 0.13x | different contract |
| Image linear 1920x1080 -> 224x224 | 167.8 / 280.4 | 105.5 / 146.4 | 0.63x | different contract |
| Image linear 1920x1080 -> 3840x2160 | 28388.0 / 31332.2 | 8856.8 / 9925.3 | 0.31x | different contract |
| Image rotate90 1920x1080 | 1125.0 / 1792.0 | 1660.1 / 2099.5 | 1.48x | exact |
| Image horizontal flip 1920x1080 | 511.7 / 803.7 | 586.4 / 940.8 | 1.15x | exact |
| Image SwapRB 1920x1080 | 507.7 / 718.5 | 517.8 / 721.8 | 1.02x | exact |
| Image BGR to gray 1920x1080 | 220.8 / 383.2 | 319.0 / 408.2 | 1.44x | tolerance 1 |
| Image threshold 1920x1080 | 132.7 / 198.2 | 133.0 / 211.1 | 1.00x | exact |
| Image Gaussian5 1920x1080 | 1746.2 / 2433.4 | 358.4 / 549.1 | 0.21x | different contract |
| Image erode3 1920x1080 | 813.5 / 1827.8 | 271.7 / 391.2 | 0.33x | exact |
| Image affine 1920x1080 | 11925.0 / 13228.1 | 4140.7 / 4671.5 | 0.35x | different contract |
| Task CHW identity 224x224 | 15.1 / 19.9 | 33.2 / 43.6 | 2.20x | exact |
| Task CHW identity 640x640 | 126.1 / 181.5 | 296.1 / 442.9 | 2.35x | exact |
| Task HWC identity 224x224 | 17.3 / 39.8 | 12.2 / 33.6 | 0.71x | exact |
| Task resize to CHW 1920x1080 -> 224x224 | 297.8 / 382.5 | 147.3 / 205.4 | 0.49x | different contract |

### Ryzen 5 5600, one logical CPU

`AVX uplift` is the median SSE4.1 build P50 divided by the median full-project
AVX2 build P50. OpenCV P50/P95 is from the AVX2 run; OpenCV itself uses runtime
CPU dispatch in both builds.

| Operation | SSE4.1 P50 | AVX2 P50 / P95 | AVX uplift | OpenCV P50 / P95 | Speedup | Accuracy |
|---|---:|---:|---:|---:|---:|---|
| Image nearest 512x512 -> 256x256 | 46.9 | 38.4 / 38.5 | 1.22x | 55.9 / 58.9 | 1.46x | exact |
| Image linear 512x512 -> 256x256 | 446.2 | 393.7 / 396.7 | 1.13x | 58.7 / 61.6 | 0.15x | different contract |
| Image linear 1920x1080 -> 224x224 | 342.1 | 305.2 / 308.6 | 1.12x | 93.9 / 97.3 | 0.31x | different contract |
| Image linear 1920x1080 -> 3840x2160 | 56490.9 | 49824.8 / 49866.9 | 1.13x | 6883.7 / 6908.6 | 0.14x | different contract |
| Image rotate90 1920x1080 | 1255.9 | 1264.5 / 1274.3 | 0.99x | 3132.1 / 3147.8 | 2.48x | exact |
| Image horizontal flip 1920x1080 | 247.5 | 252.8 / 256.0 | 0.98x | 2162.6 / 2173.2 | 8.56x | exact |
| Image SwapRB 1920x1080 | 144.0 | 143.3 / 146.5 | 1.01x | 170.8 / 174.2 | 1.19x | exact |
| Image BGR to gray 1920x1080 | 4283.6 | 574.0 / 575.7 | 7.46x | 563.0 / 564.2 | 0.98x | tolerance 1 |
| Image threshold 1920x1080 | 37.9 | 34.9 / 38.1 | 1.09x | 59.4 / 62.3 | 1.70x | exact |
| Image Gaussian5 1920x1080 | 5578.5 | 4799.1 / 4891.2 | 1.16x | 767.1 / 816.1 | 0.16x | different contract |
| Image erode3 1920x1080 | 9213.5 | 397.2 / 400.3 | 23.20x | 126.7 / 130.2 | 0.32x | exact |
| Image affine 1920x1080 | 35763.6 | 33640.7 / 33679.6 | 1.06x | 10416.1 / 10455.7 | 0.31x | different contract |
| Task CHW identity 224x224 | 76.0 | 77.3 / 80.2 | 0.98x | 95.4 / 98.3 | 1.24x | exact |
| Task CHW identity 640x640 | 628.6 | 637.9 / 639.5 | 0.99x | 565.2 / 571.4 | 0.89x | exact |
| Task HWC identity 224x224 | 24.9 | 28.6 / 28.7 | 0.87x | 14.4 / 14.5 | 0.50x | exact |
| Task resize to CHW 1920x1080 -> 224x224 | 685.5 | 655.4 / 657.0 | 1.05x | 164.7 / 168.0 | 0.25x | different contract |

The AVX2 result is concentrated rather than general. Grayscale and erode gain
7.5x and 23.2x, while Task identity regresses slightly and most
sampling/geometry paths gain only 5% to 13%. The three u8 C3 rows now use a
dedicated SSSE3 path in both x86 builds, so their gain does not require the
full-project AVX2 option.

### Expanded algorithm-by-scale matrix

The focused table above is not the complete comparison. A second Ryzen run
covers 91 cases across resize, geometry, color conversion, thresholding,
filtering, morphology and Task preprocessing. It uses seven separate process
runs, each with 10 warm-ups and 101 samples per row. The values below are the
median P50 across those seven runs. `Speedup` remains OpenCV / InspireCV.

Of the 91 rows, 65 have aligned contracts: 57 are byte- or bit-exact and eight
BGR-to-gray rows have maximum u8 error 1. There were no accuracy failures. The
remaining 26 rows are retained as raw timing references but are not counted as
correctness comparisons. Across the aligned rows InspireCV wins 40, ties eight
and loses 17 by the 2% tie band; their unweighted geometric-mean speedup is
1.411x.

| Aligned operation | Scale rows | Geomean speedup | Range | InspireCV / tie / OpenCV | Accuracy |
|---|---:|---:|---:|---:|---|
| Rotate90 | 8 | 4.862x | 2.631-6.149x | 8 / 0 / 0 | exact |
| Horizontal flip | 8 | 7.192x | 4.400-8.586x | 8 / 0 / 0 | exact |
| SwapRB | 8 | 1.358x | 1.024-2.168x | 8 / 0 / 0 | exact |
| BGR to gray | 8 | 1.060x | 0.982-1.379x | 2 / 6 / 0 | maximum error 1 |
| Threshold | 8 | 1.829x | 1.399-3.315x | 8 / 0 / 0 | exact |
| Erode3 | 8 | 0.232x | 0.169-0.461x | 0 / 0 / 8 | exact |
| Nearest resize | 7 | 1.068x | 0.552-1.497x | 3 / 2 / 2 | exact |
| Task identity CHW | 5 | 1.073x | 0.698-1.392x | 3 / 0 / 2 | exact |
| Task identity HWC | 5 | 0.557x | 0.508-0.641x | 0 / 0 / 5 | exact |

The common Image sweep uses 64x64, 112x112, 224x224, 512x512,
640x480, 1280x720, 1920x1080 and 3840x2160. Resize needs source and
destination dimensions together, so its aligned rows are shown separately.
Times are microseconds.

| Nearest resize | InspireCV P50 | OpenCV P50 | Speedup | Result |
|---|---:|---:|---:|---|
| 64x64 -> 128x128 | 13.706 | 13.806 | 1.007x | tie |
| 128x128 -> 256x256 | 136.476 | 75.302 | 0.552x | OpenCV |
| 224x224 -> 112x112 | 7.203 | 10.780 | 1.497x | InspireCV |
| 512x512 -> 256x256 | 38.342 | 55.935 | 1.459x | InspireCV |
| 640x480 -> 320x240 | 44.403 | 65.012 | 1.464x | InspireCV |
| 1920x1080 -> 960x540 | 871.289 | 871.768 | 1.000x | tie |
| 1920x1080 -> 3840x2160 | 16310.775 | 14589.868 | 0.894x | OpenCV |

Task identity behavior changes with both layout and scale. CHW wins at 64,
224 and 320, loses at 112, and is within 2.2% at 640. HWC remains consistently
slower because the current route still performs channel conversion and float
normalization as separate work.

| Task identity size | CHW InspireCV / OpenCV | CHW speedup | HWC InspireCV / OpenCV | HWC speedup |
|---|---:|---:|---:|---:|
| 64x64 | 6.422 / 8.937 | 1.392x | 2.705 / 1.733 | 0.641x |
| 112x112 | 19.045 / 13.295 | 0.698x | 7.394 / 4.679 | 0.633x |
| 224x224 | 77.375 / 94.978 | 1.228x | 28.564 / 14.517 | 0.508x |
| 320x320 | 159.219 / 194.346 | 1.221x | 59.441 / 30.277 | 0.509x |
| 640x640 | 787.290 / 769.807 | 0.978x | 232.718 / 118.714 | 0.510x |

CHW at 320 and 640 is bimodal between process runs: its P50 max/min spread is
44.2% and 65.2%, while OpenCV is 5.2% and 9.3%. The seven-run median is reported
instead of hiding that variation, but these two rows need an alignment/cache
aliasing isolation benchmark before treating their winner as stable.

The following contracts differ, so these ratios show optimization headroom
only. OpenCV output is not the correctness oracle for these rows.

| Different-contract operation | Rows | Raw geomean speedup | Measured range |
|---|---:|---:|---:|
| Linear resize | 8 | 0.165x | 0.138-0.171x |
| Gaussian5 with sigma zero | 8 | 0.076x | 0.018-0.214x |
| Affine linear | 5 | 0.315x | 0.304-0.332x |
| Task resize to CHW | 4 | 0.277x | 0.242-0.297x |
| Non-integral nearest 1920x1080 -> 224x224 | 1 | 1.377x | 1.377x |

The expanded matrix confirms that the next aligned priorities are Erode3,
Task HWC, 2x nearest upsampling and Task CHW at 112. Linear resize, Gaussian,
affine and Task resize have larger raw gaps, but each must be optimized against
the current InspireCV output rather than changed to OpenCV's sampling policy.

### x86 u8 C3 optimization gate

The already optimized 512x512 and 1920x1080 cases were checked first, so work
on additional dimensions could not trade away earlier gains.

| Size | Operation | Current vs prior optimized InspireCV | Current vs OpenCV 5.0.0 | Result |
|---|---|---:|---:|---|
| 512x512 | Rotate90 | 1.96x | 5.15x | faster than both |
| 512x512 | Horizontal flip | 1.01x | 7.76x | prior held; faster than OpenCV |
| 512x512 | SwapRB | 1.01x | 1.08x | prior held; faster than OpenCV |
| 1920x1080 | Rotate90 | 1.99x | 4.98x | faster than both |
| 1920x1080 | Horizontal flip | 1.00x | 8.74x | prior held; faster than OpenCV |
| 1920x1080 | SwapRB | 1.00x | 1.18x | prior held; faster than OpenCV |

All six P50 and P95 rows are within the 3% release limit. Rotate90 is about 2x
faster; the other four rows preserve the prior optimized implementation within
0.31% and still lead OpenCV. The complete same-ISA comparison is in
`ryzen5_5600_prior_vs_current_sse41_opencv.csv`.

The next table is a sustained multi-resolution sweep. It uses one AVX2 binary;
an internal environment switch selects the previous scalar routes, so function
placement and the rest of the binary remain identical. Old and current paths
alternate on logical CPU 2. Each cell is the median P50 of three runs with 10
warm-ups and 101 samples per operation. Times are microseconds; P95 values
remain in the summary CSV. Absolute times should be compared within this table,
because the longer sweep places different sustained load on the CPU than the
short full-suite run above.

| Size | Operation | Old InspireCV P50 | Current InspireCV P50 | OpenCV 5.0.0 P50 | Current vs old | Current vs OpenCV | Result |
|---|---|---:|---:|---:|---:|---:|---|
| 16x16 | Rotate90 | 1.032 | 0.160 | 0.531 | 6.45x | 3.32x | InspireCV faster |
| 16x16 | Horizontal flip | 0.952 | 0.110 | 0.330 | 8.65x | 3.00x | InspireCV faster |
| 16x16 | SwapRB | 0.281 | 0.090 | 0.421 | 3.12x | 4.68x | InspireCV faster |
| 112x112 | Rotate90 | 45.926 | 3.066 | 16.923 | 14.98x | 5.52x | InspireCV faster |
| 112x112 | Horizontal flip | 43.130 | 1.704 | 12.234 | 25.31x | 7.18x | InspireCV faster |
| 112x112 | SwapRB | 11.072 | 1.012 | 2.204 | 10.94x | 2.18x | InspireCV faster |
| 224x224 | Rotate90 | 232.495 | 73.241 | 78.331 | 3.17x | 1.07x | InspireCV faster |
| 224x224 | Horizontal flip | 159.938 | 6.373 | 51.830 | 25.10x | 8.13x | InspireCV faster |
| 224x224 | SwapRB | 44.373 | 3.938 | 4.779 | 11.27x | 1.21x | InspireCV faster |
| 512x512 | Rotate90 | 1205.226 | 381.413 | 598.236 | 3.16x | 1.57x | InspireCV faster |
| 512x512 | Horizontal flip | 833.945 | 37.392 | 273.214 | 22.30x | 7.31x | InspireCV faster |
| 512x512 | SwapRB | 231.652 | 18.857 | 22.363 | 12.29x | 1.19x | InspireCV faster |
| 1920x1080 | Rotate90 | 9586.516 | 2961.913 | 5128.306 | 3.24x | 1.73x | InspireCV faster |
| 1920x1080 | Horizontal flip | 6576.201 | 254.781 | 2164.596 | 25.81x | 8.50x | InspireCV faster |
| 1920x1080 | SwapRB | 1836.905 | 149.438 | 179.644 | 12.29x | 1.20x | InspireCV faster |
| 3840x2160 | Rotate90 | 43235.039 | 17496.488 | 25750.196 | 2.47x | 1.47x | InspireCV faster |
| 3840x2160 | Horizontal flip | 26428.689 | 2072.200 | 9209.344 | 12.75x | 4.44x | InspireCV faster |
| 3840x2160 | SwapRB | 7349.738 | 1567.271 | 1541.000 | 4.69x | 0.98x | tie (within 2%) |

All 30 operation/size rows from 16x16 through 3840x2160 are byte-exact, and
both current P50 and P95 are lower than the old route in every row. The 24
untargeted rows have no P50 or P95 regression above 3% across the final seven-run
gate; their P50 geometric-mean speedup is 1.011x. The summary tool enforces those
thresholds and exits with an error when a gate fails.

The x86 boundary test covers widths 1-65, odd dimensions, scalar tails, four
input alignments and input guard bytes. It passed AddressSanitizer and
UndefinedBehaviorSanitizer with 1,364,524 assertions across the image contract
group. The x86 source is excluded from ARM builds; a three-run M4 check reported
no row above the 3% regression gate.

### Remaining slower paths on Ryzen

These rows have aligned behavior and can be compared directly. `OpenCV lead` is
`Current InspireCV P50 / OpenCV P50`; a larger value is a larger remaining gap.

| Operation | Size | Current InspireCV P50 | OpenCV 5.0.0 P50 | OpenCV lead | Accuracy |
|---|---|---:|---:|---:|---|
| Erode3 | 512x512 | 94.411 | 26.672 | 3.54x | exact |
| Erode3 | 1920x1080 | 398.182 | 129.889 | 3.07x | exact |
| Task BGR-to-RGB f32 HWC | 224x224 | 28.425 | 14.387 | 1.98x | exact |
| Nearest resize | 128x128 -> 256x256 | 132.422 | 73.941 | 1.79x | exact |
| Task BGR-to-RGB f32 CHW | 112x112 | 19.066 | 13.746 | 1.39x | exact |
| Task BGR-to-RGB f32 CHW | 640x640 | 640.159 | 573.543 | 1.12x | exact |
| Nearest resize | 1920x1080 -> 3840x2160 | 16296.986 | 14616.576 | 1.11x | exact |

The priority order is Erode3, Task HWC conversion, 2x nearest upsampling, then
the remaining Task CHW writers. Grayscale is within 2% of OpenCV, while threshold
and nearest downsampling already lead OpenCV on this machine.

The implementation points to concrete causes:

- Erode3's AVX2 loop loads and reduces all nine neighboring vectors for every
  output row. A rolling horizontal-minimum buffer can reuse two of the three
  rows and reduce repeated source traffic.
- The exact u8 C3 2x-nearest route has a NEON block implementation but no x86
  C3 block implementation; x86 currently performs a three-byte copy per output
  pixel. This is the smallest, lowest-risk next kernel.
- Task HWC performs BGR-to-RGB into a temporary tile and then normalizes it in a
  second SSE pass. A fused channel-reverse/u8-to-f32 writer removes that extra
  pass and temporary traffic.
- Task CHW has a NEON planar writer, but the x86 planar writer falls through to
  the scalar loop. An isolated SSSE3/AVX2 deinterleave-and-normalize kernel
  should improve 112x112 and 640x640; 224x224 leads OpenCV in the final run and
  remains a mandatory no-regression case.

The following raw timing gaps are not correctness comparisons because the
contracts differ. InspireCV must keep its current result as the oracle while
optimizing them.

| Operation | Representative size | Current InspireCV P50 | OpenCV 5.0.0 P50 | Raw timing gap |
|---|---|---:|---:|---:|
| Gaussian5 | 512x512 | 1151.813 | 125.571 | 9.17x |
| Linear resize | 1920x1080 -> 3840x2160 | 49879.932 | 6877.681 | 7.25x |
| Task resize to CHW | 640x480 -> 112x112 | 165.575 | 39.937 | 4.15x |
| Affine linear | 1920x1080 | 34373.872 | 10411.416 | 3.30x |

These numbers identify optimization opportunity only; matching OpenCV's output
would change the current interpolation or kernel policy and is not acceptable.

## NEON isolation

`INSPIRECV_TASK_ENABLE_ARM_NEON=OFF` removes explicit Task NEON branches, but an
ARM64 compiler may still auto-vectorize portable code and NEON remains part of
the architecture baseline. The switch does not disable Image backend NEON.

| Task operation | Explicit NEON off P50 | Explicit NEON on P50 | NEON uplift | Accuracy |
|---|---:|---:|---:|---|
| CHW identity 112x112 | 3.750 | 4.000 | 0.94x | exact |
| CHW identity 224x224 | 14.041 | 15.333 | 0.92x | exact |
| CHW identity 640x640 | 117.625 | 126.541 | 0.93x | exact |
| HWC identity 224x224 | 15.542 | 17.292 | 0.90x | exact |
| Resize 640x480 -> CHW 112x112 | 71.834 | 71.583 | 1.00x | different contract vs OpenCV |
| Resize 1920x1080 -> CHW 224x224 | 288.458 | 292.500 | 0.99x | different contract vs OpenCV |

The current explicit Task NEON identity path is 6% to 10% slower than the
portable, compiler-vectorized path on M4. It should not be selected merely
because NEON is available. Other format families such as YUV need their own
isolation rows before changing the global default.

## Implementation findings

- The x86 baseline compiles the project with SSE4.1. The isolated Task AVX2
  source list currently contains only `platform/avx2_memcpy.cc`.
- `INSPIRECV_ENABLE_AVX2=ON` applies `-mavx2 -mfma` to all C++ sources. This can
  make the binary unusable on older CPUs, even though only a subset of kernels
  benefits materially.
- x86 u8 C3 horizontal flip, rotate90 and SwapRB use a dedicated SSSE3 source.
  ARM and non-u8/C3 routes are unchanged, and scalar tails preserve odd-width
  behavior.
- Bilinear u8 C3 still does scalar per-pixel interpolation. Precomputed
  coordinates help, but the algorithm lacks a separable fixed-point row kernel.
- The AVX2 grayscale and erode paths prove that focused kernels can remove most
  of the architecture gap. They should become isolated runtime-dispatched
  translation units rather than requiring a global AVX2 build.
- Task CHW identity is already worth protecting on M4. Task HWC and resize are
  the higher-value Task targets.

## Improvement plan

### 1. Replace the global ISA model

Create a small CPU kernel dispatch table selected once from runtime feature
detection. Build portable, SSE4.1, AVX2 and NEON implementations in separate
translation units. Keep the public API unchanged and retain a baseline binary
that starts on non-AVX2 x86 CPUs. Add an internal test override so the same
binary can run portable and ISA-specific implementations back-to-back.

The next dispatch candidates are BGR-to-gray, erode3 and the Task tensor
writers. Once these are migrated, the global AVX2 option can become a
build-time minimum-CPU choice rather than the only way to access optimized
kernels.

### 2. Common x86 C3 operations: completed

- Add SSSE3 packed shuffles for u8 C3 SwapRB.
- Add reversed C3 block loads/stores for horizontal flip.
- Add a cache-tiled C3 transpose plus reversal for rotate90.
- Retain the current scalar tail for widths not divisible by the SIMD block.

These operations are byte-exact and have no algorithm-policy ambiguity. Their
formal results and regression gate are recorded above.

### 3. Redesign sampling and filters

- Split bilinear resize into coordinate-table preparation, horizontal row
  filtering and vertical combination. Use fixed-point u8 weights and row reuse,
  with NEON and AVX2 C1/C3/C4 kernels. Preserve InspireCV's current coordinate
  convention bit-for-bit.
- Add specialized axis-aligned resize routes before the general affine sampler.
- Keep the current Gaussian sigma heuristic. Specialize common 3x3 and 5x5
  coefficients and vectorize the separable passes; do not silently adopt
  OpenCV's different sigma-zero binomial policy.
- Extend the current erode3 idea to isolated SSE4.1/AVX2/NEON row and column
  kernels, then profile whether a sliding-window or van Herk route wins for
  larger kernels.
- Precompute affine x increments and use block interpolation for the general
  fallback, with exact scalar tails and border handling.

### 4. Refine Task routes

- Route identity conversion on M4 to the measured portable writer until a new
  NEON implementation beats it.
- Fuse u8 C3 channel swap, normalization and HWC store. The current HWC route is
  1.4x slower than OpenCV on M4 and 2.0x slower on Ryzen.
- Reuse resize coordinate tables across Pipeline executions and fuse the final
  interpolation with CHW normalization.
- Add YUV, C4, NC4HW4, misaligned stride and tail cases to the ISA isolation
  matrix before changing dispatch for those families.

### 5. Add threading only after single-core kernels

Keep the SIMD gate single-core on x86. Add a separate production-throughput
suite for default schedulers and introduce row parallelism only above measured
pixel thresholds. An optional new `ResizeInto`/workspace API can separate
allocation cost without changing existing Image entry points.

## Release gates for CPU optimization

An optimized kernel is ready only when all of these pass:

1. The candidate matches the current InspireCV implementation byte-for-byte or
   bit-for-bit for its declared contract. OpenCV is not the oracle when the
   coordinate or kernel policy differs.
2. Deterministic randomized tests cover widths around every vector boundary,
   1-row/1-column inputs, odd dimensions, C1/C3/C4, padded and misaligned
   strides, border pixels, invalid inputs and scalar tails.
3. Portable and every ISA route produce identical output under AddressSanitizer
   and UndefinedBehaviorSanitizer; x86 dispatch is also tested on an AVX2-disabled
   runtime path.
4. Performance uses at least three alternating runs, 10 warm-ups and 101
   samples per row. A target kernel should improve P50 by at least 10% with no
   P95 regression. Untargeted rows may not regress more than 3% individually or
   2% by geometric mean.
5. The existing public API, object-library dependency mode and non-AVX2 x86
   compatibility remain intact.

## Reproduce

Build the comparison runner against an OpenCV installation:

```bash
cmake -S . -B build-cpu-pk \
  -DCMAKE_BUILD_TYPE=Release \
  -DINSPIRECV_BUILD_CPU_BENCHMARKS=ON \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv5
cmake --build build-cpu-pk --target inspirecv_cpu_benchmark -j

./build-cpu-pk/inspirecv_cpu_benchmark \
  --suite full --samples 101 --warmups 10 --opencv-threads 1 \
  --machine local_cpu --report cpu_run.csv
```

On Linux, prefix the runner with `taskset -c 2` for the one-logical-CPU suite.
Use `--suite u8c3` for the 16x16-to-4K geometry sweep. Build the x86 candidate
with `-DINSPIRECV_ENABLE_AVX2=ON`. Use `--suite matrix` for the 91-row
algorithm-by-scale comparison. Summarize repeated candidate-versus-OpenCV
reports with:

```bash
python3 scripts/cpu_benchmark_opencv_summary.py \
  --reports matrix_run1.csv matrix_run2.csv matrix_run3.csv \
  --candidate-label inspirecv --output matrix_summary.csv
```

Compare two InspireCV builds and enforce a performance gate with:

```bash
python3 scripts/cpu_benchmark_summary.py \
  --baseline sse_repeat1.csv sse_repeat2.csv sse_repeat3.csv \
  --candidate avx_repeat1.csv avx_repeat2.csv avx_repeat3.csv \
  --baseline-label sse41 --candidate-label avx2 \
  --minimum-p50-speedup 0.97 \
  --minimum-p95-speedup 0.97 \
  --minimum-geomean-speedup 0.98 \
  --output sse_vs_avx2_summary.csv
```

Primary raw reports are:

- `apple_m4_neon_opencv5_0_0_gcd_default_repeat1.csv` and repeats 2-3.
- `ryzen5_5600_sse41_opencv5_0_0_repeat1.csv` and repeats 2-3.
- `ryzen5_5600_avx2_opencv5_0_0_repeat1.csv` and repeats 2-3.
- `ryzen5_5600_u8c3_final_summary.csv` plus the six `final2_u8c3` input reports.
- `ryzen5_5600_full_final_summary.csv` plus the fourteen `final2_full` input reports.
- `ryzen5_5600_prior_vs_current_sse41_opencv.csv`.
- `ryzen5_5600_algorithm_matrix_summary.csv` plus
  `ryzen5_5600_algorithm_matrix_run1.csv` through `run7.csv`.
- `apple_m4_task_no_neon_vs_neon_summary.csv`.
- `ryzen5_5600_sse41_vs_avx2_opencv5_0_0_summary.csv`.

The OpenCV 4.5.5/4.6.0 reports remain in this directory as supplementary
version-pinned history; the OpenCV 5.0.0 reports are the current comparison.
