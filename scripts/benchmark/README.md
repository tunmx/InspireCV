# Targeted CPU regression probes

These probes compare two revisions without an OpenCV or CUDA dependency. Build
both revisions with the same compiler, optimization flags and library mode.
Use separate build directories and retain the baseline binary unchanged.

For example, with static archives in `build-before` and `build-after`:

```sh
c++ -std=c++14 -O2 -Iinclude -Isrc -I3rdparty/Eigen-3.4.0-Headers \
  scripts/benchmark/affine_latency_probe.cpp build-before/libinspirecv.a -o /tmp/affine-before
c++ -std=c++14 -O2 -Iinclude -Isrc -I3rdparty/Eigen-3.4.0-Headers \
  scripts/benchmark/affine_latency_probe.cpp build-after/libinspirecv.a -o /tmp/affine-after
python3 scripts/benchmark/compare_latency_probes.py \
  --baseline /tmp/affine-before --candidate /tmp/affine-after \
  --output artifacts/affine-comparison --runs 9
```

Use matching deployment targets when linking on macOS. On Linux, prefix the
Python invocation with `taskset -c 2` to keep both binaries on one logical CPU.
Do not compile or run other benchmarks during measurement.

The latency probe covers uint8/float32, one/three channels, 17/128/512/1024 square
images, identity, integer/fractional translation, and scaling across a replicated
border. It also measures matrix composition, inversion and an odd-length point
mapping. Source construction is excluded; image output allocation is included.
Every case has five warm-ups, calibrated batches and eleven timed batches.
The reported P50/P95 are percentiles of **batch-mean per-call latency**, not
individual-request tail latency. Nanosecond-scale matrix timings should also
be checked against end-to-end Task benchmarks.

Pass `--constant-border` to the comparison script for a separate run with
constant-value padding. Use a different output directory for that run.
`--task-only` measures the public Task NCHW pipeline at 112/160/320/640 pixels
and three controlled destination page offsets. This isolates allocation-layout
effects from code changes; input/output buffers are reused throughout timing.
`--unaffected` isolates threshold, arithmetic, rotation and general affine
operations through the public Image API, with deterministic external input
buffers and no image decoding/encoding between cases.

The comparison alternates binary order, retains every batch in CSV, and uses
the median of paired run ratios. Default gates are 5% per case and 2% overall;
they allow measurement noise, not a guarantee of identical wall-clock time.
An existing report is never overwritten. Failed runs remain available for
inspection; use a new output directory for the next experiment.

For exact Task output comparisons, compile `task_numeric_snapshot.cpp` against
each archive using the same include paths, run both binaries and compare stdout
with `cmp`. It emits 2,816 deterministic records: matrix coefficients, inverse
coefficients, point mappings of different lengths, and hashes of complete image
outputs. Compare like-for-like builds: historical LTO and platform/compiler
contracts are not universally bit-identical.
Use `--full-pixels` on both snapshot binaries to retain every output byte as
hexadecimal in addition to the hashes, and compare the complete reports with
`cmp`.

The regression tests themselves run in the normal CTest suite. A focused run is:

```sh
./build-after/inspirecv_tests '[regression]'
```

For broader performance gates, use `scripts/image_perf_gate.py` and
`scripts/task_perf_gate.py`; both accept `--baseline`, `--candidate` and
`--images-dir` (see `--help`).

The Image gate disables debug PNG encoding between cases. Visual validation
should run separately; encoding changes allocator, cache and thermal state even
when it is outside the timed region. Pass `--output-dir` to retain every raw
Image report and its parsed measurements. Use a new directory for each run.

When comparing revisions that register different unit tests, link the **same
benchmark-only harness objects** against both libraries. For the Image gate,
these are `src/test/test.cpp.o` and
`src/test/cases/image/test_image_benchmark.cpp.o` from one test build. Linking
entire test executables with different registration/initialization code can
change heap layout before timing starts. Retain the normal, complete test
executables for correctness and consumer tests.

Also run an A/A control (the same baseline binary supplied for both arguments)
before interpreting borderline regressions. A failing A/A control means the
environment/harness cannot certify a per-case performance guarantee; it is not
permission to raise the regression limits or mark a failing A/B run as passed.
On glibc, a fixed `MALLOC_MMAP_THRESHOLD_=131072` can make allocation policy
repeatable across invocations; apply it to **both** sides and retain unmodified
environment results too. The controlled-buffer Task probe is a separate check
of kernel behavior, not a substitute for allocation-inclusive measurements.
