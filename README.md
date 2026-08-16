# InspireCV

InspireCV is a compact C++14 library for image operations and model
preprocessing. The current version is **1.0.2**.

- `Image` covers image I/O, geometry, filters, drawing and pixel operations.
- `task::Pipeline` converts camera frames or images directly to another image
  or to model-ready tensor memory.
- The default OKCV backend has no OpenCV dependency; an OpenCV backend can be
  selected when a project already uses OpenCV.

## Quick start

Requirements: CMake 3.15 or newer, a C++14 compiler and the bundled Eigen
headers. OpenCV is optional.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DINSPIRECV_BUILD_EXAMPLES=ON
cmake --build build -j
```

Run an example:

```bash
./build/example_image_io input.jpg
```

The convenience script builds and installs a default Release configuration to
`build/install`:

```bash
./command/build.sh build
```

### CMake integration

For most source-tree integrations, use the namespaced target. This works with
the default object build and also when static or shared mode is selected:

```cmake
add_subdirectory(third_party/InspireCV)
target_link_libraries(my_target PRIVATE InspireCV::inspirecv)
```

The object target named `inspirecv` is retained for projects such as
InspireFace that embed its objects directly. Existing integration does not
need to change:

```cmake
set(INSPIRECV_BUILD_OBJECT_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/InspireCV)

target_sources(my_target PRIVATE $<TARGET_OBJECTS:inspirecv>)
target_include_directories(my_target PRIVATE third_party/InspireCV/include)
```

Select a direct static or shared build only when the host project needs that
specific library type:

```cmake
set(INSPIRECV_BUILD_OBJECT_LIBS OFF CACHE BOOL "" FORCE)
set(INSPIRECV_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE) # ON for shared
add_subdirectory(third_party/InspireCV)
target_link_libraries(my_target PRIVATE InspireCV::inspirecv)
```

An installed package exports the same target:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/inspirecv
cmake --build build --parallel
cmake --install build
```

```cmake
find_package(InspireCV 1 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE InspireCV::inspirecv)
```

### Build options

| Option | Default | Purpose |
|---|---:|---|
| `INSPIRECV_BACKEND_OPENCV` | `OFF` | Use OpenCV as the Image backend. |
| `INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO` | `OFF` | Use OpenCV codecs with OKCV operations. |
| `INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI` | `OFF` | Use OpenCV window functions with OKCV operations. |
| `INSPIRECV_BUILD_OBJECT_LIBS` | `ON` | Build the compatible object-library target. |
| `INSPIRECV_BUILD_SHARED_LIBS` | `OFF` | Build shared instead of static when object mode is off. |
| `INSPIRECV_BUILD_TESTS` | `OFF` | Build the `inspirecv_tests` executable. |
| `INSPIRECV_BUILD_CPU_BENCHMARKS` | `OFF` | Build the direct CPU comparison with OpenCV core/imgproc. |
| `INSPIRECV_BUILD_SAMPLE` | `OFF` | Build internal sample programs. |
| `INSPIRECV_BUILD_EXAMPLES` | `OFF` | Build the programs under `example/`. |
| `INSPIRECV_INSTALL` | `ON` | Generate install rules and the `find_package` package. |
| `INSPIRECV_TASK_ENABLE_ARM_NEON` | `ON` | Enable Task NEON kernels on ARM. |
| `INSPIRECV_TASK_DISABLE_TILING` | `OFF` | Disable Task row tiling. |
| `INSPIRECV_ENABLE_CUDA` | `OFF` | Build the optional Image and Task CUDA backends (CMake 3.18+). |
| `INSPIRECV_TASK_ENABLE_CUDA` | `OFF` | Compatibility alias for `INSPIRECV_ENABLE_CUDA`. |
| `INSPIRECV_CUDA_ARCHITECTURES` | `86` | CUDA architectures passed to CMake; `86` targets RTX 30-series. |
| `INSPIRECV_ENABLE_AVX2` | `OFF` | Enable AVX2 globally on x86; Task still runtime-dispatches isolated AVX2 helpers. |

OpenCV **4.5.5** is the recommended and CI-validated version for the optional
backend. Use it with:

```bash
cmake -S . -B build-opencv \
  -DCMAKE_BUILD_TYPE=Release \
  -DINSPIRECV_BACKEND_OPENCV=ON
cmake --build build-opencv -j
```

If OpenCV is installed outside the default search path, add
`-DOpenCV_DIR=/path/to/opencv-4.5.5/lib/cmake/opencv4`. Newer OpenCV releases
may also work, but 4.5.5 is the release baseline.

## Image API

Include `<inspirecv/inspirecv.h>`. `Image` stores unsigned 8-bit pixels;
`ImageT<float>` exposes the same operation family for float data.

```cpp
#include <inspirecv/inspirecv.h>

int main() {
    auto image = inspirecv::Image::Create("input.jpg", 3);
    if (image.Empty()) return 1;

    auto gray = image.ToGray();
    auto resized = gray.Resize(320, 240);
    auto blurred = resized.GaussianBlur(5, 0.0);
    auto mask = blurred.Threshold(120.0, 255.0, 0);

    return mask.Write("mask.png") ? 0 : 2;
}
```

The main operation groups are:

| Group | Operations |
|---|---|
| Storage and I/O | `Create`, `Read`, `Write`, `Clone`, `Reset`, `Data` |
| Geometry | `Resize`, `Crop`, `WarpAffine`, rotate, flip, `Pad` |
| Color and pixels | `ToGray`, `SwapRB`, `MeanChannels`, `Fill`, `Add`, `Mul`, `Threshold`, `AbsDiff` |
| Filters | `GaussianBlur`, `Erode`, `Dilate`, `Blend` |
| Drawing | `DrawLine`, `DrawRect`, `DrawCircle`, rectangle fill |
| Shared geometry | `Point`, `Rect`, `Size`, `TransformMatrix` |

`Image` uses the library's BGR convention for color-sensitive operations and
file I/O. An image created with external storage can avoid an input copy; the
caller must keep that storage alive for the image's lifetime.

## Task preprocessing API

Include `<inspirecv/task/pipeline.h>` for the focused API or
`<inspirecv/task/task.h>` for the compatibility umbrella. A `Pipeline` is
movable, not copyable, and retains reusable planning and scratch state.

### Camera buffer to Image

The following pipeline converts NV21 to a tightly packed BGR image. `Run`
allocates the destination once and writes into it directly, without an
intermediate HWC tensor.

```cpp
#include <inspirecv/inspirecv.h>
#include <inspirecv/task/pipeline.h>

inspirecv::task::PipelineOptions options;
options.input_format = inspirecv::task::PixelFormat::kNv21;
options.output_format = inspirecv::task::PixelFormat::kBgr;
options.sampling = inspirecv::task::SamplingMode::kLinear;

inspirecv::task::Pipeline pipeline(options);
if (pipeline.ConfigurationStatus() != inspirecv::task::Status::kOk) {
    return;
}
if (pipeline.SetTransform(inspirecv::TransformMatrix::Identity()) !=
    inspirecv::task::Status::kOk) {
    return;
}

inspirecv::task::RawImageView source;
source.data = nv21_data;
source.width = input_width;
source.height = input_height;
source.row_stride_bytes = 0; // infer the natural stride

inspirecv::Image output;
auto status = pipeline.Run(source, output_width, output_height, &output);
```

For a frame loop, allocate the image once. `RunInto` writes directly to the
same pixel address on every call:

```cpp
auto output = inspirecv::Image::Create(output_width, output_height, 3);

for (;;) {
    source.data = next_frame();
    if (pipeline.RunInto(source, &output) != inspirecv::task::Status::kOk) {
        break;
    }
    consume(output);
}
```

Source and destination storage must not overlap. `RunInto` does not replace or
allocate the destination pixel buffer, though the first execution may create
internal planning or scratch storage. `Run` is transactional; a failed
`RunInto` may leave part of the caller-provided destination modified.

### Camera buffer or Image to tensor

`TensorBuffer` writes into caller-owned HWC, CHW or channel-packed memory and
accepts explicit row/channel strides:

```cpp
std::vector<float> output(3 * output_width * output_height);

inspirecv::task::TensorBuffer tensor;
tensor.data = output.data();
tensor.width = output_width;
tensor.height = output_height;
tensor.channels = 3;
tensor.element_type = inspirecv::task::ElementType::kFloat32;
tensor.order = inspirecv::task::TensorOrder::kChw;

const auto status = pipeline.Run(source, tensor);
if (status != inspirecv::task::Status::kOk) {
    std::cerr << inspirecv::task::StatusMessage(status) << '\n';
}
```

Use `PipelineOptions::mean` and `scale` for float normalization. Inputs include
RGB, BGR, RGBA, BGRA, gray, NV12, NV21 and I420. The conversion registry also
covers supported YUV, YCrCb, XYZ, HSV, BGR555 and BGR565 pairs. Cubic sampling
is represented by the API but currently returns `kUnsupportedSampling`.

`ConfigurationStatus()` reports invalid option enum values immediately after
construction. `SetTransform()` accepts only invertible matrices and leaves the
previous transform unchanged on failure. These checks keep setup errors out of
the frame-processing loop while preserving the reusable pipeline state.

`Image` itself does not store RGB/BGR metadata. Keep `Pipeline::OutputFormat()`
with the result when producing RGB. The former `StreamTask` facade remains in
`<inspirecv/task/legacy.h>` for migration; new code should use `Pipeline`.

### Optional CUDA acceleration

CUDA is optional and disabled by default. Build a CUDA-enabled library first;
`86` targets the RTX 3060 and other Ampere SM 8.6 devices. Use a semicolon-
separated architecture list when distributing one binary to several GPU
generations.

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DINSPIRECV_BUILD_TESTS=ON \
  -DINSPIRECV_ENABLE_CUDA=ON \
  -DINSPIRECV_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j

# Example multi-architecture build:
# -DINSPIRECV_CUDA_ARCHITECTURES="75;86;89"
```

Image and Task share the same process-wide CUDA switch and default preference.
`kAuto` is the recommended setting: it uses the conservative RTX 3060-derived
threshold table and keeps smaller or unsupported operations on CPU.

```cpp
#include <inspirecv/inspirecv.h>

// Returns false on a CPU-only build or when no usable CUDA device is present.
if (!inspirecv::SetCudaAccelerationEnabled(true)) {
    // Continue normally: Image and Task retain their CPU implementations.
}
inspirecv::SetAccelerationPreference(
  inspirecv::AccelerationPreference::kAuto);

auto resized = image.Resize(384, 384);
const auto image_backend = inspirecv::GetLastImageExecutionBackend();

const auto status = pipeline.Run(source, tensor);
const auto backend = pipeline.LastExecutionBackend();

inspirecv::SetCudaAccelerationEnabled(false);
```

The global preference can also force a backend for testing or deployment
policy. A forced CUDA request still falls back safely when that operation is
not supported.

```cpp
inspirecv::SetAccelerationPreference(
  inspirecv::AccelerationPreference::kCpu);   // Always use CPU.
inspirecv::SetAccelerationPreference(
  inspirecv::AccelerationPreference::kCuda);  // Try CUDA, otherwise CPU.
inspirecv::SetAccelerationPreference(
  inspirecv::AccelerationPreference::kAuto);  // Recommended default.
```

Task pipelines may override the global preference individually. `kDefault`
inherits the process-wide setting and is the default value.

```cpp
#include <inspirecv/task/task.h>

inspirecv::task::PipelineOptions options;
options.input_format = inspirecv::task::PixelFormat::kBgr;
options.output_format = inspirecv::task::PixelFormat::kRgb;
options.backend_preference =
  inspirecv::task::BackendPreference::kCuda;  // Only this pipeline.

inspirecv::task::Pipeline cuda_pipeline(options);
```

Image CUDA currently covers nearest/bilinear resize, 90/180/270-degree rotate,
and the validated WarpAffine combinations for contiguous `uint8`/`float32`
images with one to four channels. The Task path fuses affine/resize sampling,
RGB/BGR conversion and float normalization. Unsupported or unprofiled requests
continue through their original CPU path. CUDA remains off by default, so
existing applications keep their current execution behavior.

`kAuto` applies a conservative host round-trip baseline measured on an RTX 3060.
The same thresholds are used on every compatible CUDA device; the GPU and CPU
model are not used as an eligibility check.
`PipelineOptions::backend_preference` can override it with `kCpu` or `kCuda`
for one Task pipeline. Image uses the process-wide
`AccelerationPreference`. Operations outside the maintained table stay on CPU;
explicit CUDA remains available for validated combinations. The existing
`task::SetCudaEnabled` and Task preference functions forward to the same global
control for source compatibility. Profiles, versions and raw measurements are
in [benchmarks/cuda](benchmarks/cuda/README.md).

For frames already resident in GPU memory, `<inspirecv/task/cuda.h>` provides
`task::cuda::Pipeline`, `DeviceImageView` and `DeviceTensorBuffer`. Its `Run`
method is asynchronous on a caller-supplied stream and avoids host/device
copies. CUDA types are not exposed in the public header. The build links the
CUDA runtime statically; deployment therefore needs a compatible NVIDIA
driver, but not a separate CUDA runtime library. Building still requires the
CUDA Toolkit, and other GPU generations should set
`INSPIRECV_CUDA_ARCHITECTURES` accordingly.

For a multi-stage GPU path, `<inspirecv/cuda/image.h>` adds an owning
`cuda::DeviceImage`. Upload once, enqueue Image geometry and Task
preprocessing on the same stream, then keep the tensor on the GPU for
inference:

```cpp
#include <inspirecv/inspirecv.h>
#include <inspirecv/task/cuda.h>

inspirecv::SetCudaAccelerationEnabled(true);

inspirecv::cuda::DeviceImage device_frame;
inspirecv::cuda::DeviceImage resized;
if (inspirecv::cuda::DeviceImage::Upload(host_frame, &device_frame) !=
    inspirecv::cuda::Status::kOk) {
    return;
}

void* stream = inference_stream;  // cudaStream_t passed without CUDA headers
device_frame.Resize(320, 240, true, &resized, stream);

inspirecv::task::cuda::DeviceTensorBuffer tensor;
tensor.data = device_tensor_address;
tensor.width = 112;
tensor.height = 112;
tensor.channels = 3;
tensor.order = inspirecv::task::TensorOrder::kChw;

inspirecv::task::cuda::Pipeline gpu_pipeline(options);
gpu_pipeline.SetTransform(transform_320x240_to_112x112);
gpu_pipeline.Run(resized, tensor, stream);
inspirecv::cuda::Synchronize(stream);  // or let the inference runtime wait
```

`Upload` and `Download` are synchronization boundaries. `Resize`,
`WarpAffine`, rotate and the device Task pipeline enqueue work asynchronously;
their reusable device buffers avoid an allocation on every frame. For several
same-shaped frames, `task::cuda::Pipeline::RunBatch` validates the complete
descriptor array first, reuses its geometry cache and synchronizes only when
the caller chooses. NV12, NV21 and I420 can enter this device Task path
directly; odd-sized 4:2:0 buffers stay on the established CPU path.

These resident and batch interfaces are explicit. They do not change existing
`Image` or host `task::Pipeline` calls, and they do not broaden the default
Auto table.

## Architecture and platforms

```text
public API
  Image ------------------------ task::Pipeline
    |                                  |
    v                                  v
Image backend adapter          raw-memory ConversionSession
    |                           validate -> compile -> execute
    +-- OKCV                           |
    +-- OpenCV adapter                 +-- portable kernels
    +-- CUDA geometry (optional)       +-- NEON / SSE4.1 / AVX2
                                      +-- CUDA fused path (optional)
```

Only the Task API adapter depends on `Image`. Task planning, execution and
kernels operate on raw views and do not depend on either CV backend. The Image
facade likewise does not depend on Task. CMake checks these include boundaries
during configuration.

| Target | Acceleration | Validation level |
|---|---|---|
| macOS ARM64 | NEON | Build, unit tests, image comparison and benchmark |
| macOS x86_64 | SSE4.1/AVX2 dispatch | Build and unit tests |
| Linux x86_64 | SSE4.1/AVX2 dispatch | GCC build and unit tests |
| Linux x86_64 + NVIDIA | CUDA SM 86 backend | RTX 3060 exact-output tests, memory checks and benchmark |
| Android arm64-v8a | NEON | NDK cross-build |
| Android armeabi-v7a | NEON | NDK cross-build |
| iOS device/simulator ARM64 | NEON | Cross-build |

Other C++14 targets use portable kernels where no specialized path is enabled,
but are not part of the current release validation matrix.

The implementation layout mirrors these boundaries:

```text
include/inspirecv/           public headers
src/inspirecv/core/          Image facade, shared acceleration and CUDA geometry
src/inspirecv/backends/      OKCV and OpenCV implementations
src/inspirecv/task/api/      Image-aware Task adapter
src/inspirecv/task/planning/ validation and compiled conversion plans
src/inspirecv/task/execution/ routes, scheduling and workspace reuse
src/inspirecv/task/kernels/  portable and ISA-specific kernels
src/inspirecv/task/cuda/     optional CUDA control, dispatch and fused kernel
src/test/                    correctness, contract and performance tests
example/                     small user-facing programs
```

## Benchmark

Reference numbers below are from a 2026-08-16 verification run of the 1.0.1
Release build on macOS ARM64 with AppleClang 17 and NEON. The tensor rows are
timed-loop averages; the Image bridge rows are medians of nine alternating
rounds. They are a regression reference, not a guarantee for other machines.

The direct CPU comparison against OpenCV 5.0.0, raw reports, accuracy policy,
NEON/AVX2 isolation results and optimization plan are in
[benchmarks/cpu](benchmarks/cpu/README.md).

Identity BGR `uint8` to normalized NCHW `float32`:

| Input/output size | Pipeline latency |
|---:|---:|
| 112 x 112 | 3.00 us |
| 160 x 160 | 6.17 us |
| 320 x 320 | 24.41 us |
| 640 x 640 | 102.47 us |

The direct Image bridge used BGR 640 x 480 to BGR 112 x 112 bilinear resize:

| Backend | HWC tensor | Preallocated `Image` | Difference |
|---|---:|---:|---:|
| OKCV | 67.92 us | 67.99 us | +0.10% |
| OpenCV | 68.55 us | 67.76 us | -1.16% |

Run the synthetic Task benchmarks with:

```bash
./build/inspirecv_tests '[benchmark][task_tensor_view]'
./build/inspirecv_tests '[benchmark][task][api][image_bridge]'
./build-cuda/inspirecv_tests '[benchmark][task][cuda]'
./build-cuda/inspirecv_tests '[benchmark][task][cuda][threshold]'
```

The CUDA backend was measured on an RTX 3060 with CUDA 12.2 and driver
550.144.03. Values are medians of five Release runs. `CUDA round-trip` includes
pageable host-to-device and device-to-host copies; `CUDA device` measures the
kernel path when both buffers already reside in GPU memory.

| Source to tensor | CPU | CUDA round-trip | Round-trip speedup | CUDA device | Device speedup |
|---:|---:|---:|---:|---:|---:|
| 112x112 to 112x112 | 19.194 us | 51.458 us | 0.373x | 5.908 us | 3.249x |
| 320x240 to 112x112 | 170.826 us | 63.267 us | 2.700x | 5.920 us | 28.856x |
| 640x480 to 112x112 | 172.967 us | 127.783 us | 1.354x | 5.931 us | 29.163x |
| 1280x720 to 224x224 | 683.927 us | 301.133 us | 2.271x | 18.668 us | 36.636x |
| 1920x1080 to 224x224 | 685.836 us | 514.281 us | 1.334x | 21.111 us | 32.487x |
| 3840x2160 to 640x640 | 5590.077 us | 2289.323 us | 2.442x | 122.334 us | 45.695x |

The default `Auto` profile uses the following conservative host round-trip
boundaries, benchmarked on an RTX 3060 + Ryzen 5 5600. They apply to every
compatible CUDA device for the BGR-to-RGB, bilinear, CHW, square-output path.
Unprofiled operation combinations stay on CPU.

| Output workload | Largest input selected for CUDA | Source storage |
|---:|---:|---:|
| 112x112 | 640x480x3 | 921,600 bytes |
| 224x224 | 1920x1080x3 | 6,220,800 bytes |
| 640x640 | 3840x2160x3 | 24,883,200 bytes |

The boundary gate requires CUDA p50 to be at least 15% faster and CUDA p95 to
be no slower than CPU. Auto-path verification and all 101-sample measurements
are recorded in [benchmarks/cuda](benchmarks/cuda/README.md).

Image geometry uses the same gate and includes pageable upload/download in the
CUDA time. Representative 101-sample p50 results are:

| Image operation | Type | Size | CPU | CUDA round-trip | Speedup | Auto |
|---|---:|---:|---:|---:|---:|---:|
| bilinear resize | u8 c3 | 64x64 to 48x48 | 15.404 us | 18.168 us | 0.848x | CPU |
| bilinear resize | u8 c3 | 128x128 to 96x96 | 61.543 us | 22.354 us | 2.753x | CUDA |
| general WarpAffine | u8 c3 | 64x64 | 68.655 us | 18.288 us | 3.754x | CUDA |
| rotate90 | u8 c3 | 96x96 | 31.760 us | 21.093 us | 1.506x | CUDA |
| bilinear resize | f32 c3 | 512x512 to 384x384 | 598.761 us | 434.758 us | 1.377x | CUDA |
| rotate90 | f32 c3 | 512x512 | 843.585 us | 576.973 us | 1.462x | CUDA |
| bilinear resize | f32 c3 | 3840x2160 to 2880x1620 | 39253.181 us | 36646.517 us | 1.071x | CPU |

The shipped Image Auto table is deliberately narrower than the raw crossover:

| Image path | Default CUDA Auto range, using the RTX 3060 baseline |
|---|---|
| u8 c3 bilinear resize at 3/4 scale | 128x128 through 3840x2160 source |
| u8 c3 general WarpAffine, same-size output | 64x64 through 3840x2160 |
| u8 c3 rotate90/180/270 | 96x96 through 3840x2160 |
| f32 c3 bilinear resize at 3/4 scale | 512x512 through 2560x1440 source |
| f32 c3 rotate90/180/270 | 512x512 through 3840x2160 |
| nearest resize and other type/channel/shape combinations | CPU |

The 112x112 host round-trip is slower than CPU because transfer and launch
overhead dominate. CUDA is useful when the input is larger, the tensor stays
on the GPU for inference, or several preprocessing operations can remain in
the same GPU pipeline.

A three-operation Image chain (bilinear resize, general affine, rotate90)
shows the effect of retaining the intermediate images on the GPU. Values are
medians across three 51-sample runs:

| Source | CPU chain | CUDA per-operation round-trip | Resident upload + download once | Resident device only |
|---:|---:|---:|---:|---:|
| 640x480 | 5452.580 us | 444.813 us | 194.594 us | 47.459 us |
| 1280x720 | 15809.800 us | 3167.980 us | 497.021 us | 123.131 us |
| 1920x1080 | 37121.200 us | 6979.820 us | 1035.810 us | 262.962 us |

Fused NV12-to-normalized-RGB preprocessing is also useful when the camera
frame or inference tensor is device-resident. Pageable host transfer can still
make CUDA slower for some resize ratios, so this path is not part of Auto:

| NV12 source to tensor | CPU | CUDA host round-trip | CUDA device |
|---:|---:|---:|---:|
| 640x480 to 224x224 | 212.308 us | 133.179 us | 9.368 us |
| 1920x1080 to 224x224 | 217.006 us | 312.395 us | 10.089 us |
| 3840x2160 to 640x640 | 2045.260 us | 1372.820 us | 56.807 us |

For 320x240 BGR frames producing 112x112 CHW tensors, one-stream batching
reduced synchronization overhead as the batch grew:

| Batch | Serialized device calls | `RunBatch` | Batch per image | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 10.670 us | 10.680 us | 10.680 us | 0.999x |
| 4 | 42.820 us | 32.411 us | 8.103 us | 1.321x |
| 8 | 85.961 us | 61.936 us | 7.742 us | 1.388x |
| 16 | 172.052 us | 120.025 us | 7.502 us | 1.433x |

Raw three-run results and the exact test commands are kept in
[benchmarks/cuda](benchmarks/cuda/README.md).

Generate CPU, CUDA, absolute-difference and side-by-side PNGs for center crop,
rotation, translation and a combined affine transform from a real input image:

```bash
mkdir -p /tmp/inspirecv-cuda-visual
INSPIRECV_CUDA_VISUAL_SOURCE=/path/to/input.png \
INSPIRECV_CUDA_VISUAL_DIR=/tmp/inspirecv-cuda-visual \
  ./build-cuda/inspirecv_tests \
  task_cuda_host_pipeline_matches_cpu_fused_preprocessing
```

Generate the equivalent synthetic Image geometry report for bilinear resize,
general affine warp and 90/180/270-degree rotation:

```bash
mkdir -p /tmp/inspirecv-image-cuda-visual
INSPIRECV_IMAGE_CUDA_VISUAL_DIR=/tmp/inspirecv-image-cuda-visual \
  ./build-cuda/inspirecv_tests image_cuda_visual_report
```

Image benchmarks use `[bench][image]` and may require a fixture directory:

```bash
INSPIRECV_IMAGES_DIR=/path/to/images ./build/inspirecv_tests '[bench][image]'
INSPIRECV_IMAGE_CUDA_BENCHMARK_SAMPLES=101 \
  ./build-cuda/inspirecv_tests image_cuda_full_geometry_benchmark
```

For release comparisons, use independent binaries. The gate alternates their
execution order and checks per-case and geometric-mean regression:

```bash
python3 scripts/task_perf_gate.py \
  --baseline /path/to/baseline/inspirecv_tests \
  --candidate /path/to/candidate/inspirecv_tests \
  --workdir "$PWD" \
  --images-dir /path/to/images
```

## Tests

```bash
cmake -S . -B build-test \
  -DCMAKE_BUILD_TYPE=Release \
  -DINSPIRECV_BUILD_TESTS=ON
cmake --build build-test -j
cmake -E chdir build-test ctest --output-on-failure
```

CTest supplies the repository fixture paths and also verifies that a separate
consumer can install and load InspireCV through `find_package`. Run the binary
directly only when selecting a focused Catch2 filter.

The default GitHub Actions workflow checks object, static and shared builds on
Linux, macOS and Windows, builds the public examples, tests installed/source
consumers, and runs ASan/UBSan on Linux. OpenCV is intentionally separated into
the manual-only `OpenCV 4.5.5 backend (manual)` workflow: it downloads OpenCV
4.5.5 from GitHub, compiles it from source, then validates all three InspireCV
library modes. It has no push, pull-request or scheduled trigger. To run it,
open the repository's **Actions** page, select that workflow and choose
**Run workflow**. Automatic execution can later be enabled by adding the
desired `push` or `pull_request` event to
`.github/workflows/opencv_backend_manual.yaml`.

Useful focused filters:

```bash
./build-test/inspirecv_tests '[image]~[bench]'
./build-test/inspirecv_tests '[task]~[benchmark]'
./build-test/inspirecv_tests '[task][api][pipeline]'
./build-test/inspirecv_tests '[build][contract]'
```

Release checks cover format families, custom strides, vector tails, invalid
input, backend contracts and byte-/bit-exact CPU/CUDA output comparisons. CUDA
runs are followed by Compute Sanitizer memcheck before release.

## Version and build information

Image and Task share one library identity:

```cpp
#include <inspirecv/version.h>

const auto& info = inspirecv::GetLibraryInfo();
std::cout << info.version_string << " backend=" << info.cv_backend << '\n';
```

`LibraryInfo` reports semantic version `1.0.2`, backend, compiler, target,
build type, LTO, SSE, AVX2, NEON and Task tiling settings. `GetVersion()` and
`GetCVBackend()` remain as compatibility helpers.

## Changelog

- **1.0.2:** completed device-resident and batch CUDA paths, fused YUV preprocessing, cross-backend conformance, portable package checks and release CI.
- **1.0.1:** added shared Image/Task CUDA control, Image geometry acceleration, universal Auto thresholds and reproducible CUDA benchmarks.
- **1.0.0:** unified Image/Task integration, redesigned Pipeline API, direct Image output, modular internals, shared build identity and stricter cross-backend regression gates.
- **0.9.0:** introduced Task preprocessing, tensor layouts, YUV conversion and architecture-specific execution paths.
- **0.6.0:** added color constants, filters, pixel arithmetic, SIMD acceleration and float Image support over the 0.6 development line.
- **0.5.0:** improved external-memory handling and added embedded ARM platform adaptation.
- **0.4.2:** established the initial Image, geometry, drawing and selectable-backend API.

Third-party components bundled or optionally used by the project include
Eigen, Catch2, stb and OpenCV.
