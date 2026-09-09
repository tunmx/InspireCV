# Benchmark CI

The full benchmark workflow runs only in `tunmx/inspirecv-benchmark` when its
`benchmark` branch receives a push. Merging a pull request into that branch also
produces a push, so there is no separate pull-request trigger. Every job repeats
the repository and branch check before doing any work.

## Coverage

| Artifact label | Runner | What is measured |
|---|---|---|
| `linux-x86_64` | GitHub `ubuntu-24.04` | Unit tests; full, U8C3, and matrix CPU comparisons; Image and Task benchmarks |
| `linux-arm64` | GitHub `ubuntu-24.04-arm` | Same native CPU suite with ARM64/NEON |
| `macos-arm64` | GitHub `macos-15` | Same native CPU suite on Apple Silicon |
| `macos-x86_64` | GitHub `macos-15-intel` | Same native CPU suite on Intel macOS |
| `windows-x86_64` | GitHub `windows-2025` | Same native CPU suite on Windows x64 |
| `android-emulator-x86_64` | GitHub Ubuntu + accelerated emulator | Android unit tests and full Image/Task benchmarks; ARM64 APK-native test binary compile check |
| `cuda-rtx3060` | Self-hosted controller + SSH | CUDA unit tests and every `[benchmark][cuda]` case on `192.168.0.116` |

OpenCV 4.5.5 is built from its exact peeled tag commit
`dad26339a975b49cfb6c7dbe4bd5276c9dcb36e2`, with only `core` and `imgproc`,
then cached per OS and architecture. CPU comparisons use three independent
runs, 101 samples, 10 warmups, and OpenCV's internal threading disabled. The raw
CSV files also enforce each operation's accuracy contract.

Android ARM64 is compile-checked but not timed on an emulated ARM CPU. Useful
ARM64 Android performance numbers require a physical device attached to a
self-hosted runner. Linux ARM64 is measured natively by GitHub Actions.

## RTX 3060 controller setup

GitHub-hosted runners cannot route to a private `192.168.x.x` address. Register
a runner in the benchmark repository on a machine that can reach the GPU host,
and add the custom label `inspirecv-benchmark-controller`. The controller needs
`bash`, `git`, `tar`, OpenSSH, and Python 3; CUDA is required only on the remote
3060 host.

Create these Actions settings in `tunmx/inspirecv-benchmark`:

| Kind | Name | Value |
|---|---|---|
| Variable | `BMK_CUDA_SSH_ENABLED` | `true` after the self-hosted runner is online |
| Variable | `BMK_CUDA_SSH_HOST` | `192.168.0.116` (optional; this is the default) |
| Variable | `BMK_CUDA_SSH_USER` | `tunm` (optional; this is the default) |
| Secret | `BMK_CUDA_SSH_PRIVATE_KEY` | A dedicated, unencrypted CI private key |
| Secret | `BMK_CUDA_SSH_KNOWN_HOSTS` | The verified SSH host-key line for the GPU machine |

Verify the 3060 host-key fingerprint through a trusted channel before storing
the `ssh-keyscan` output. The workflow keeps strict host-key checking enabled.
The remote executor uploads the exact workflow commit with `git archive`, builds
in a unique `/tmp/inspirecv-benchmark.*` directory, retrieves the reports, and
removes only that validated temporary directory. It does not pull from or push
to any Git remote on the GPU host.

The CUDA job refuses to benchmark while `nvidia-smi` reports another compute
process. It records the process list and fails without terminating anyone's
work; rerun after the GPU is idle. This prevents an occupied device from
producing a misleading threshold or performance regression.

## Results

Each platform uploads an immutable artifact named with the tested commit SHA.
Artifacts contain correctness logs, raw benchmark logs, per-case reports,
machine/toolchain metadata, and a `manifest.json` with SHA256 checksums. The
aggregate job produces `cpu_all_platforms.csv`, a Markdown run summary, and all
platform manifests. Retention is 90 days, and CI never commits generated results
back to a branch.

GitHub-hosted hardware is shared and can change between runs. Those timings are
suited to trend analysis and comparisons within one run; release performance
gates should use a stable self-hosted machine.
