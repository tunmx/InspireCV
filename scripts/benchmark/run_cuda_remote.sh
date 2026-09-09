#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <source-dir> <results-dir>" >&2
    exit 64
fi

source_dir=$(cd "$1" && pwd)
results_dir=$2
mkdir -p "$results_dir"
results_dir=$(cd "$results_dir" && pwd)

ssh_host=${BMK_CUDA_SSH_HOST:-192.168.0.116}
ssh_user=${BMK_CUDA_SSH_USER:-tunm}
ssh_key=${BMK_CUDA_SSH_KEY_PATH:-}
known_hosts=${BMK_CUDA_KNOWN_HOSTS_PATH:-}
commit=${GITHUB_SHA:-$(git -C "$source_dir" rev-parse HEAD)}
remote_root=
local_archive=

if ! [[ "$ssh_host" =~ ^[A-Za-z0-9._:-]+$ && "$ssh_user" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid CUDA SSH host or user" >&2
    exit 64
fi
if [[ -z "$known_hosts" || ! -f "$known_hosts" ]]; then
    echo "BMK_CUDA_KNOWN_HOSTS_PATH must name a verified known_hosts file" >&2
    exit 64
fi

ssh_options=(
    -o BatchMode=yes
    -o ConnectTimeout=15
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=3
    -o StrictHostKeyChecking=yes
    -o "UserKnownHostsFile=$known_hosts"
)
if [[ -n "$ssh_key" ]]; then
    if [[ ! -f "$ssh_key" ]]; then
        echo "BMK_CUDA_SSH_KEY_PATH does not exist" >&2
        exit 64
    fi
    ssh_options+=(-i "$ssh_key" -o IdentitiesOnly=yes)
fi
target="$ssh_user@$ssh_host"

ssh_retry() {
    local attempt status
    for attempt in 1 2 3; do
        if ssh "${ssh_options[@]}" "$@"; then
            return 0
        else
            status=$?
        fi
        if [[ "$status" -ne 255 || "$attempt" -eq 3 ]]; then
            return "$status"
        fi
        echo "SSH transport failed; retrying ($attempt/3)" >&2
        sleep 2
    done
}

scp_retry() {
    local attempt status
    for attempt in 1 2 3; do
        if scp "${ssh_options[@]}" "$@"; then
            return 0
        else
            status=$?
        fi
        if [[ "$status" -ne 255 || "$attempt" -eq 3 ]]; then
            return "$status"
        fi
        echo "SCP transport failed; retrying ($attempt/3)" >&2
        sleep 2
    done
}

valid_remote_root() {
    [[ "$remote_root" =~ ^/tmp/inspirecv-benchmark\.[A-Za-z0-9]+$ ]]
}

valid_local_archive() {
    [[ "$local_archive" =~ ^/tmp/inspirecv-source\.[A-Za-z0-9]+$ ]]
}

cleanup() {
    if valid_remote_root; then
        ssh_retry "$target" bash -s -- "$remote_root" <<'REMOTE_CLEANUP' || true
set -euo pipefail
cleanup_root=$1
if [[ "$cleanup_root" =~ ^/tmp/inspirecv-benchmark\.[A-Za-z0-9]+$ ]]; then
    rm -rf -- "$cleanup_root"
else
    echo "refusing unsafe cleanup path: $cleanup_root" >&2
    exit 64
fi
REMOTE_CLEANUP
    fi
    if valid_local_archive && [[ -f "$local_archive" && ! -L "$local_archive" ]]; then
        rm -f -- "$local_archive"
    fi
}
trap cleanup EXIT

local_archive=$(mktemp /tmp/inspirecv-source.XXXXXXXX)
if ! valid_local_archive || [[ ! -f "$local_archive" || -L "$local_archive" ]]; then
    echo "mktemp returned an unsafe local archive path: $local_archive" >&2
    exit 1
fi
git -C "$source_dir" archive --format=tar --output="$local_archive" "$commit"

for attempt in 1 2 3; do
    if remote_root=$(ssh "${ssh_options[@]}" "$target" \
        'mktemp -d /tmp/inspirecv-benchmark.XXXXXXXX'); then
        if valid_remote_root; then
            break
        fi
    fi
    remote_root=
    if [[ "$attempt" -lt 3 ]]; then
        echo "SSH did not return a valid temporary path; retrying ($attempt/3)" >&2
        sleep 2
    fi
done
if ! valid_remote_root; then
    echo "remote mktemp returned an unsafe path: $remote_root" >&2
    exit 1
fi

ssh_retry "$target" \
    "mkdir -p '$remote_root/source' '$remote_root/build' '$remote_root/results'"
scp_retry "$local_archive" "$target:$remote_root/source.tar"
ssh_retry "$target" \
    "tar -xf '$remote_root/source.tar' -C '$remote_root/source' && rm -f -- '$remote_root/source.tar'"

remote_status=0
ssh "${ssh_options[@]}" "$target" \
    "INSPIRECV_CUDA_BENCHMARK_REPEATS='${INSPIRECV_CUDA_BENCHMARK_REPEATS:-3}' INSPIRECV_CUDA_BENCHMARK_SAMPLES='${INSPIRECV_CUDA_BENCHMARK_SAMPLES:-101}' bash '$remote_root/source/scripts/benchmark/remote_cuda_suite.sh' '$remote_root/source' '$remote_root/build' '$remote_root/results' '$commit'" \
    || remote_status=$?

ssh_retry "$target" \
    "tar -C '$remote_root/results' -czf '$remote_root/results.tar.gz' ."
scp_retry "$target:$remote_root/results.tar.gz" \
    "$results_dir/results.tar.gz"
tar -xzf "$results_dir/results.tar.gz" -C "$results_dir"

if [[ "$remote_status" -ne 0 ]]; then
    echo "remote CUDA benchmark failed with status $remote_status" >&2
    exit "$remote_status"
fi
