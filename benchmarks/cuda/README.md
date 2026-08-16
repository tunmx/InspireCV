# CUDA auto-dispatch profiles

CUDA host round-trip thresholds are benchmark-derived. The shipped `Auto`
policy uses the RTX 3060 results below as a conservative universal default on
every compatible CUDA device; GPU and CPU names are recorded as benchmark
provenance, not used as runtime gates. Callers can still select `kCpu` or
`kCuda` explicitly.

## Task fused preprocessing: RTX 3060

Measured on 2026-08-16 with the following configuration:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GiB, compute capability 8.6 |
| GPU multiprocessors | 28 |
| Driver | 550.144.03; CUDA driver API 12.4 |
| CUDA runtime/toolkit | 12.2 |
| CPU | AMD Ryzen 5 5600 6-Core Processor |
| OS | Linux 6.8.0-124-generic x86_64 |
| Compiler | GCC 11.4.0, Release |
| Samples | 101 per CPU/CUDA path after 10 warm-ups |
| Accuracy | CPU and CUDA float output compared bit-for-bit |
| Recommendation gate | CUDA p50 at least 15% faster; CUDA p95 no slower |

The current CUDA operation is fused packed BGR/RGB `uint8` input to normalized
HWC/CHW `float32` output with nearest or bilinear affine sampling. The host
round-trip profile includes pageable upload and download. The default Auto
profile covers only the measured BGR-to-RGB, bilinear, CHW and square-output
configuration. Other color routes, nearest, HWC and non-square output stay on
CPU until they have their own threshold data.

| Output workload | Largest measured input recommended for CUDA | Source bytes |
|---:|---:|---:|
| 112x112 | 640x480x3 | 921,600 |
| 224x224 | 1920x1080x3 | 6,220,800 |
| 640x640 | 3840x2160x3 | 24,883,200 |

For output sizes between rows, `Auto` linearly interpolates the maximum source
storage. It does not extrapolate above the largest measured 4K input. Outputs
smaller than 112x112 and identity transforms stay on CPU. Identity preprocessing
was slower on CUDA for every measured square size from 64x64 through 1024x1024.

The implemented Auto path was then measured independently with the same 101
sample method. Its p50 must stay within 10% plus 2 us of the backend selected
by the table.

| Input to output | CPU p50 | CUDA p50 | Auto p50 | Auto selected |
|---:|---:|---:|---:|---:|
| 112x112 to 112x112 | 19.186 us | 51.146 us | 19.036 us | CPU |
| 640x480 to 112x112 | 173.628 us | 132.891 us | 133.152 us | CUDA |
| 960x540 to 112x112 | 174.771 us | 168.118 us | 174.079 us | CPU |
| 1920x1080 to 224x224 | 687.649 us | 514.573 us | 514.412 us | CUDA |
| 2560x1440 to 224x224 | 689.373 us | 811.514 us | 688.251 us | CPU |
| 3840x2160 to 640x640 | 6145.216 us | 2237.208 us | 2222.992 us | CUDA |

Raw results are stored in
[`task_host_rtx3060_cuda12_2.csv`](task_host_rtx3060_cuda12_2.csv).
Regenerate them from a CUDA Release build with:

```bash
INSPIRECV_CUDA_THRESHOLD_SAMPLES=101 \
INSPIRECV_CUDA_THRESHOLD_REPORT=task_host_rtx3060_cuda12_2.csv \
  ./build-cuda/inspirecv_tests task_cuda_host_auto_threshold_benchmark

./build-cuda/inspirecv_tests task_cuda_auto_dispatch_performance
```

## Image geometry: RTX 3060

Image geometry used the same machine, Release configuration, 10 warm-ups and
101 timed samples per backend. CUDA timing includes pageable host upload,
kernel execution and host download. The matrix contains 143 u8/f32 operation
and resolution combinations; every CPU/CUDA output comparison was bit-exact.

| Operation | Type | Workload | CPU p50 | CUDA p50 | Speedup | Gate result |
|---|---:|---:|---:|---:|---:|---:|
| bilinear resize | u8 c3 | 64x64 to 48x48 | 15.404 us | 18.168 us | 0.848x | CPU |
| bilinear resize | u8 c3 | 128x128 to 96x96 | 61.543 us | 22.354 us | 2.753x | CUDA |
| general WarpAffine | u8 c3 | 64x64 | 68.655 us | 18.288 us | 3.754x | CUDA |
| rotate90 | u8 c3 | 96x96 | 31.760 us | 21.093 us | 1.506x | CUDA |
| bilinear resize | f32 c3 | 512x512 to 384x384 | 598.761 us | 434.758 us | 1.377x | CUDA |
| rotate90 | f32 c3 | 512x512 | 843.585 us | 576.973 us | 1.462x | CUDA |
| bilinear resize | f32 c3 | 3840x2160 to 2880x1620 | 39253.181 us | 36646.517 us | 1.071x | CPU |

Auto uses the following 3060-derived default ranges on every compatible CUDA
device, with a safety margin beyond the first passing measurement:

| Profiled Image path | CUDA Auto range |
|---|---|
| u8 c3 bilinear resize, 3/4 in each dimension | source pixels from 128x128 through 3840x2160 |
| u8 c3 general affine, same-size output | source/output pixels from 64x64 through 3840x2160 |
| u8 c3 90/180/270-degree rotation | source pixels from 96x96 through 3840x2160 |
| f32 c3 bilinear resize, 3/4 in each dimension | source pixels from 512x512 through 2560x1440 |
| f32 c3 90/180/270-degree rotation | source pixels from 512x512 through 3840x2160 |

Nearest resize stays on CPU in Auto. Three formal 101-sample runs agreed on
the large-image measurements, but a shorter whole-suite run showed a materially
different host allocation crossover. Explicit CUDA remains available; Auto
will not depend on that unstable boundary. Axis-aligned one- and three-channel
x86 affine requests and x86 f32 single-channel rotate270 also retain the CPU
implementation because its frozen SIMD behavior is not equivalent to the
portable scalar mapping.

The full report is
[`image_host_rtx3060_cuda12_2.csv`](image_host_rtx3060_cuda12_2.csv). The two
additional nearest-resize repetitions are
[`image_nearest_repeat2_rtx3060_cuda12_2.csv`](image_nearest_repeat2_rtx3060_cuda12_2.csv)
and
[`image_nearest_repeat3_rtx3060_cuda12_2.csv`](image_nearest_repeat3_rtx3060_cuda12_2.csv).
Regenerate the full matrix with:

```bash
INSPIRECV_IMAGE_CUDA_BENCHMARK_SAMPLES=101 \
INSPIRECV_IMAGE_CUDA_BENCHMARK_REPORT=image_host_rtx3060_cuda12_2.csv \
  ./build-cuda/inspirecv_tests image_cuda_full_geometry_benchmark
```

Add separate raw reports for materially different GPUs so the universal
baseline can be reviewed with evidence. Device-specific data does not change
runtime selection unless the default table itself is deliberately revised.

## Device-resident Image chain

`cuda::DeviceImage` was measured with a bilinear resize to 3/4 size, a general
same-size affine transform and rotate90. CPU and the existing host Image CUDA
path execute the same three public operations. The resident round-trip uploads
once and downloads once; device-only timing excludes both transfers. Each row
below is the median across three independent runs of 51 samples after warm-up.

| Source | CPU chain | CUDA per-operation round-trip | Resident round-trip | Resident device |
|---:|---:|---:|---:|---:|
| 640x480 | 5452.580 us | 444.813 us | 194.594 us | 47.459 us |
| 1280x720 | 15809.800 us | 3167.980 us | 497.021 us | 123.131 us |
| 1920x1080 | 37121.200 us | 6979.820 us | 1035.810 us | 262.962 us |

Every downloaded result was bit-exact against the CPU chain. The raw runs are
in
[`device_image_chain_rtx3060_cuda12_2.csv`](device_image_chain_rtx3060_cuda12_2.csv).
Regenerate them with:

```bash
INSPIRECV_DEVICE_IMAGE_BENCHMARK_SAMPLES=51 \
  ./build-cuda/inspirecv_tests '[benchmark][image][cuda][device]'
```

## Device Task batching

The batch benchmark uses 320x240 BGR device images and writes normalized
112x112 CHW float tensors. The serialized path synchronizes after every
image; `RunBatch` validates all descriptors, enqueues the full batch on one
stream and synchronizes once.

| Batch | Serialized | `RunBatch` | Per image | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 10.670 us | 10.680 us | 10.680 us | 0.999x |
| 4 | 42.820 us | 32.411 us | 8.103 us | 1.321x |
| 8 | 85.961 us | 61.936 us | 7.742 us | 1.388x |
| 16 | 172.052 us | 120.025 us | 7.502 us | 1.433x |

The three raw repetitions are in
[`task_batch_rtx3060_cuda12_2.csv`](task_batch_rtx3060_cuda12_2.csv). The
correctness test compares every batch item bit-for-bit with an independent CPU
execution. Run the benchmark with:

```bash
./build-cuda/inspirecv_tests '[benchmark][task][cuda][device][batch]'
```

## Fused YUV420 Task preprocessing

The explicit CUDA Task path accepts even-sized NV12, NV21 and I420 device
buffers and fuses YUV conversion, affine/resize sampling and float
normalization. These NV12 measurements use nearest sampling; CPU and CUDA
outputs were compared bit-for-bit for NV12/NV21/I420, RGB/BGR and CHW/HWC
combinations.

| NV12 input to output | CPU | CUDA host round-trip | CUDA device |
|---:|---:|---:|---:|
| 640x480 to 224x224 | 212.308 us | 133.179 us | 9.368 us |
| 1920x1080 to 224x224 | 217.006 us | 312.395 us | 10.089 us |
| 3840x2160 to 640x640 | 2045.260 us | 1372.820 us | 56.807 us |

The three raw repetitions are in
[`task_yuv_rtx3060_cuda12_2.csv`](task_yuv_rtx3060_cuda12_2.csv). Pageable
host transfer makes the 1920x1080 host round-trip slower than CPU, so YUV420
is deliberately excluded from the default Auto table. Run the benchmark with:

```bash
./build-cuda/inspirecv_tests '[benchmark][task][cuda][yuv420]'
```
