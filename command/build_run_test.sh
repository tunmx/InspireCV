#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_DIR}/build}"
mkdir -p "${BUILD_DIR}"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"

# Detect platform and set standard optimization flags
OS="$(uname -s || echo unknown)"
ARCH="$(uname -m || echo unknown)"
echo "Detected OS=${OS}, ARCH=${ARCH}"

# Defaults (can be overridden by env)
: "${INSPIRECV_ENABLE_LTO:=ON}"
: "${INSPIRECV_ENABLE_AVX2:=OFF}"
echo "LTO=${INSPIRECV_ENABLE_LTO}"
echo "AVX2=${INSPIRECV_ENABLE_AVX2}"

# Configure CMake in Release mode with LTO if supported
echo "Configuring CMake..."
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
  -DINSPIRECV_BUILD_TESTS=ON \
  -DINSPIRECV_ENABLE_AVX2="${INSPIRECV_ENABLE_AVX2}" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION="${INSPIRECV_ENABLE_LTO}" \
  || { echo "CMake configuration failed"; exit 1; }

# Build using all available cores
echo "Building..."
# Determine parallel jobs
JOBS=4
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
elif [[ "${OS}" == "Darwin" ]]; then
  JOBS="$(sysctl -n hw.ncpu || echo 4)"
fi
cmake --build "${BUILD_DIR}" --config Release --parallel "${JOBS}" \
  || { echo "Build failed"; exit 1; }

# Install
echo "Installing..."
cmake --install "${BUILD_DIR}" --config Release \
  || { echo "Installation failed"; exit 1; }

export INSPIRECV_IMAGES_DIR="${PROJECT_DIR}/images"
export INSPIRECV_GT_DIR="${INSPIRECV_IMAGES_DIR}/task_gt"
echo "INSPIRECV_IMAGES_DIR=${INSPIRECV_IMAGES_DIR}"
echo "INSPIRECV_GT_DIR=${INSPIRECV_GT_DIR}"

# Run tests
echo "Running tests..."
cmake -E chdir "${BUILD_DIR}" \
  ctest --config Release --output-on-failure \
  || { echo "Tests failed"; exit 1; }

echo "Build and test process completed successfully in directory: $BUILD_DIR"
