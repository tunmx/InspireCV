#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_DIR}/build}"
mkdir -p "${BUILD_DIR}"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"

echo "Configuring InspireCV in ${BUILD_DIR}"
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install"

echo "Building InspireCV"
cmake --build "${BUILD_DIR}" --config Release --parallel

echo "Installing InspireCV to ${BUILD_DIR}/install"
cmake --install "${BUILD_DIR}" --config Release

echo "Build completed successfully"
