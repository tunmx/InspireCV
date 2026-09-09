#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <source-dir> <build-dir> <results-dir> <commit>" >&2
    exit 64
fi

source_dir=$(cd "$1" && pwd)
build_dir=$2
results_dir=$3
commit=$4
mkdir -p "$build_dir" "$results_dir/logs" "$results_dir/correctness" "$results_dir/cuda"
build_dir=$(cd "$build_dir" && pwd)
results_dir=$(cd "$results_dir" && pwd)
repeats=${INSPIRECV_CUDA_BENCHMARK_REPEATS:-3}
samples=${INSPIRECV_CUDA_BENCHMARK_SAMPLES:-101}

# Non-interactive SSH sessions commonly omit /usr/local/cuda/bin even when the
# standard CUDA symlink is installed.
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

if ! [[ "$repeats" =~ ^[1-9][0-9]*$ && "$samples" =~ ^[1-9][0-9]*$ ]]; then
    echo "CUDA benchmark repeats and samples must be positive integers" >&2
    exit 64
fi

run_logged() {
    local log_path=$1
    shift
    echo "+ $*"
    if ! "$@" >"$log_path" 2>&1; then
        tail -n 120 "$log_path" >&2 || true
        return 1
    fi
}

finalize() {
    local exit_code=$?
    trap - EXIT
    local status=passed
    local finalize_arguments=()
    if [[ "$exit_code" -ne 0 ]]; then
        status=failed
        finalize_arguments+=(--error "remote CUDA suite exited with status $exit_code")
    fi
    python3 "$source_dir/scripts/benchmark/finalize_manifest.py" \
        --results-dir "$results_dir" \
        --platform cuda-rtx3060 \
        --system Linux \
        --architecture "$(uname -m)" \
        --commit "$commit" \
        --status "$status" \
        --metadata "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" \
        --metadata "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)" \
        --metadata "repeats=$repeats" \
        --metadata "samples=$samples" \
        "${finalize_arguments[@]}" || true
    exit "$exit_code"
}
trap finalize EXIT

nvidia-smi -q >"$results_dir/nvidia-smi.txt"
nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader \
    >"$results_dir/gpu_compute_processes.csv"
nvcc --version >"$results_dir/nvcc.txt"
cmake --version >"$results_dir/cmake.txt"
uname -a >"$results_dir/uname.txt"
cat /proc/loadavg >"$results_dir/loadavg.txt"
if [[ -s "$results_dir/gpu_compute_processes.csv" ]]; then
    echo "the benchmark GPU is busy; active compute processes:" >&2
    cat "$results_dir/gpu_compute_processes.csv" >&2
    echo "refusing to record contended GPU timings" >&2
    exit 1
fi

run_logged "$results_dir/logs/configure.log" \
    cmake -S "$source_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DINSPIRECV_ENABLE_CUDA=ON \
        -DINSPIRECV_CUDA_ARCHITECTURES=86 \
        -DINSPIRECV_BUILD_OBJECT_LIBS=OFF \
        -DINSPIRECV_BUILD_SHARED_LIBS=OFF \
        -DINSPIRECV_BUILD_TESTS=ON \
        -DINSPIRECV_BUILD_SAMPLE=OFF \
        -DINSPIRECV_BUILD_EXAMPLES=OFF \
        -DINSPIRECV_INSTALL=OFF
run_logged "$results_dir/logs/build.log" \
    cmake --build "$build_dir" --config Release --parallel "$(nproc)"

test_environment=(
    env
    "INSPIRECV_IMAGES_DIR=$source_dir/images"
    "INSPIRECV_GT_DIR=$source_dir/images/task_gt"
    "INSPIRECV_IMAGE_BENCHMARK_SAVE_IMAGES=0"
)
run_logged "$results_dir/correctness/ctest.log" \
    "${test_environment[@]}" ctest --test-dir "$build_dir" \
        -C Release --output-on-failure

tests="$build_dir/inspirecv_tests"
if [[ ! -x "$tests" ]]; then
    tests="$build_dir/Release/inspirecv_tests"
fi
if [[ ! -x "$tests" ]]; then
    echo "inspirecv_tests is missing below $build_dir" >&2
    exit 1
fi

mapfile -t cuda_cases < <(
    "$tests" --list-test-names-only "[benchmark][cuda]" \
        | sed -e '/^[[:space:]]*$/d' -e '/^InspireCV v[0-9]/d'
)
if [[ "${#cuda_cases[@]}" -eq 0 ]]; then
    echo "no [benchmark][cuda] test cases were discovered" >&2
    exit 1
fi
printf '%s\n' "${cuda_cases[@]}" >"$results_dir/cuda/test_cases.txt"

# Each case gets a fresh process and CUDA context. This keeps one benchmark's
# allocator/stream state from affecting the next case while preserving every
# performance assertion in the tests themselves.
for repeat in $(seq 1 "$repeats"); do
    run_name=$(printf 'run_%02d' "$repeat")
    for case_name in "${cuda_cases[@]}"; do
        case_slug=$(printf '%s' "$case_name" | tr -cs 'A-Za-z0-9._-' '_')
        run_logged "$results_dir/cuda/${run_name}_${case_slug}.log" \
            "${test_environment[@]}" \
            "INSPIRECV_CUDA_THRESHOLD_SAMPLES=$samples" \
            "INSPIRECV_CUDA_THRESHOLD_REPORT=$results_dir/cuda/task_threshold_${run_name}.csv" \
            "INSPIRECV_IMAGE_CUDA_BENCHMARK_SAMPLES=$samples" \
            "INSPIRECV_IMAGE_CUDA_BENCHMARK_REPORT=$results_dir/cuda/image_geometry_${run_name}.csv" \
            "INSPIRECV_DEVICE_IMAGE_BENCHMARK_SAMPLES=$samples" \
            "INSPIRECV_DEVICE_IMAGE_BENCHMARK_REPORT=$results_dir/cuda/image_device_chain_${run_name}.csv" \
            "$tests" "$case_name"
    done
done

cat >"$results_dir/summary.md" <<EOF
# InspireCV benchmark: NVIDIA RTX 3060

CUDA correctness tests and every discovered CUDA-tagged benchmark completed successfully.
Raw per-run reports include Task host/device/batch/YUV measurements and Image
host/device geometry measurements. Repeats: $repeats; samples where configurable: $samples.
EOF
