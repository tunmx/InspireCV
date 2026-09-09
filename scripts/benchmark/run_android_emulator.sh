#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 <source-dir> <build-root> <results-dir>" >&2
}

if [[ $# -ne 3 ]]; then
    usage
    exit 64
fi

source_dir=$(cd "$1" && pwd)
build_root=$2
results_dir=$3
mkdir -p "$build_root" "$results_dir"
build_root=$(cd "$build_root" && pwd)
results_dir=$(cd "$results_dir" && pwd)

: "${ANDROID_HOME:?ANDROID_HOME must point to the Android SDK}"
android_ndk_version=${INSPIRECV_ANDROID_NDK_VERSION:-27.2.12479018}
android_api=${INSPIRECV_ANDROID_API:-35}
benchmark_repeats=${INSPIRECV_ANDROID_BENCHMARK_REPEATS:-3}
commit=${GITHUB_SHA:-unknown}
avd_name="inspirecv-benchmark-${GITHUB_RUN_ID:-local}"
device_root=/data/local/tmp/inspirecv-benchmark
emulator_pid=

if ! [[ "$benchmark_repeats" =~ ^[1-9][0-9]*$ ]]; then
    echo "INSPIRECV_ANDROID_BENCHMARK_REPEATS must be positive" >&2
    exit 64
fi

cleanup() {
    if [[ -n "$emulator_pid" ]]; then
        adb -s emulator-5554 emu kill >/dev/null 2>&1 || true
        wait "$emulator_pid" 2>/dev/null || true
    fi
}

finish() {
    local exit_code=$?
    trap - EXIT
    cleanup
    if [[ ! -f "$results_dir/manifest.json" ]]; then
        python3 "$source_dir/scripts/benchmark/finalize_manifest.py" \
            --results-dir "$results_dir" \
            --platform android-emulator-x86_64 \
            --system Android \
            --architecture x86_64 \
            --commit "$commit" \
            --status failed \
            --error "Android benchmark suite exited with status $exit_code" \
            --metadata "api=$android_api" \
            --metadata "ndk=$android_ndk_version" \
            --metadata "repeats=$benchmark_repeats" || true
    fi
    exit "$exit_code"
}
trap finish EXIT

run_logged() {
    local log_path=$1
    shift
    mkdir -p "$(dirname "$log_path")"
    echo "+ $*"
    if ! "$@" >"$log_path" 2>&1; then
        tail -n 100 "$log_path" >&2 || true
        return 1
    fi
}

run_logged "$results_dir/logs/sdkmanager.log" \
    sdkmanager --install \
        "platform-tools" \
        "emulator" \
        "platforms;android-${android_api}" \
        "system-images;android-${android_api};google_apis;x86_64" \
        "ndk;${android_ndk_version}"

ndk_root="$ANDROID_HOME/ndk/$android_ndk_version"
toolchain="$ndk_root/build/cmake/android.toolchain.cmake"
if [[ ! -f "$toolchain" ]]; then
    echo "Android NDK toolchain is missing: $toolchain" >&2
    exit 1
fi

configure_android() {
    local abi=$1
    local build_dir="$build_root/$abi"
    run_logged "$results_dir/logs/configure-${abi}.log" \
        cmake -S "$source_dir" -B "$build_dir" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
            -DANDROID_ABI="$abi" \
            -DANDROID_PLATFORM="android-24" \
            -DANDROID_STL=c++_static \
            -DINSPIRECV_BUILD_OBJECT_LIBS=OFF \
            -DINSPIRECV_BUILD_SHARED_LIBS=OFF \
            -DINSPIRECV_BUILD_TESTS=ON \
            -DINSPIRECV_BUILD_SAMPLE=OFF \
            -DINSPIRECV_BUILD_EXAMPLES=OFF \
            -DINSPIRECV_INSTALL=OFF
    run_logged "$results_dir/logs/build-${abi}.log" \
        cmake --build "$build_dir" --config Release \
            --target inspirecv_tests --parallel
}

# ARM64 is compiled with the same flags even though a GitHub x64 runner cannot
# produce meaningful ARM device timing data.
configure_android arm64-v8a
configure_android x86_64

export ANDROID_AVD_HOME="${RUNNER_TEMP:-$build_root}/avd"
mkdir -p "$ANDROID_AVD_HOME"
printf 'no\n' | avdmanager create avd --force \
    --name "$avd_name" \
    --package "system-images;android-${android_api};google_apis;x86_64"

"$ANDROID_HOME/emulator/emulator" \
    -avd "$avd_name" \
    -port 5554 \
    -no-window \
    -no-audio \
    -no-boot-anim \
    -no-snapshot \
    -gpu swiftshader_indirect \
    >"$results_dir/logs/emulator.log" 2>&1 &
emulator_pid=$!

adb -s emulator-5554 wait-for-device
booted=0
for _ in $(seq 1 180); do
    if [[ $(adb -s emulator-5554 shell getprop sys.boot_completed 2>/dev/null | tr -d '\r') == 1 ]]; then
        booted=1
        break
    fi
    sleep 2
done
if [[ "$booted" -ne 1 ]]; then
    tail -n 100 "$results_dir/logs/emulator.log" >&2 || true
    echo "Android emulator did not boot within 360 seconds" >&2
    exit 1
fi

adb -s emulator-5554 shell settings put global window_animation_scale 0
adb -s emulator-5554 shell settings put global transition_animation_scale 0
adb -s emulator-5554 shell settings put global animator_duration_scale 0
adb -s emulator-5554 shell "rm -rf '$device_root' && mkdir -p '$device_root/images/task_gt'"
adb -s emulator-5554 push "$build_root/x86_64/inspirecv_tests" "$device_root/inspirecv_tests"
adb -s emulator-5554 push "$source_dir/images/kun.jpg" "$device_root/images/kun.jpg"
adb -s emulator-5554 push "$source_dir/images/task_gt/." "$device_root/images/task_gt/"
adb -s emulator-5554 shell chmod 755 "$device_root/inspirecv_tests"

device_environment="INSPIRECV_IMAGES_DIR=$device_root/images INSPIRECV_GT_DIR=$device_root/images/task_gt INSPIRECV_IMAGE_BENCHMARK_SAVE_IMAGES=0"
run_logged "$results_dir/correctness/unit.log" \
    adb -s emulator-5554 shell \
        "cd '$device_root' && $device_environment ./inspirecv_tests '~[benchmark]~[bench]'"

for repeat in $(seq 1 "$benchmark_repeats"); do
    run_name=$(printf 'run_%02d' "$repeat")
    mkdir -p "$results_dir/image/$run_name" "$results_dir/task/$run_name"
    run_logged "$results_dir/image/$run_name/console.log" \
        adb -s emulator-5554 shell \
            "cd '$device_root' && $device_environment ./inspirecv_tests '[bench][image]'"
    adb -s emulator-5554 pull "$device_root/image_benchmark.txt" \
        "$results_dir/image/$run_name/image_benchmark.txt" >/dev/null

    task_report="$device_root/task_format_${run_name}.txt"
    run_logged "$results_dir/task/$run_name/format_console.log" \
        adb -s emulator-5554 shell \
            "cd '$device_root' && $device_environment INSPIRECV_BENCH_OUTPUT='$task_report' ./inspirecv_tests '[bench][no_check]'"
    adb -s emulator-5554 pull "$task_report" \
        "$results_dir/task/$run_name/format_benchmark.txt" >/dev/null
    run_logged "$results_dir/task/$run_name/tensor_view_console.log" \
        adb -s emulator-5554 shell \
            "cd '$device_root' && $device_environment ./inspirecv_tests '[benchmark][task_tensor_view][nchw]'"
done

adb -s emulator-5554 shell getprop >"$results_dir/android_properties.txt"
cat >"$results_dir/summary.md" <<EOF
# InspireCV benchmark: Android emulator x86_64

Correctness and all Image/Task benchmark cases passed for $benchmark_repeats runs.
The arm64-v8a target was compiled, but was not timed because emulated ARM timing
is not representative of an Android ARM device.
EOF

python3 "$source_dir/scripts/benchmark/finalize_manifest.py" \
    --results-dir "$results_dir" \
    --platform android-emulator-x86_64 \
    --system Android \
    --architecture x86_64 \
    --commit "$commit" \
    --status passed \
    --metadata "api=$android_api" \
    --metadata "ndk=$android_ndk_version" \
    --metadata "arm64_compile=passed" \
    --metadata "repeats=$benchmark_repeats"
