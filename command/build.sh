#!/bin/bash

# Get build directory from first argument, default to "build" if not provided
BUILD_DIR="${1:-build}"

# Create build directory if it doesn't exist
echo "Creating directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Go to build directory
echo "Changing to directory: $BUILD_DIR"
cd "$BUILD_DIR" || { echo "Failed to change to directory: $BUILD_DIR"; exit 1; }

# Configure CMake in Release mode
echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release .. || { echo "CMake configuration failed"; exit 1; }

# Build using all available cores
echo "Building..."
make -j"$(nproc)" || { echo "Build failed"; exit 1; }

# Install
echo "Installing..."
make install || { echo "Installation failed"; exit 1; }

echo "Build process completed successfully in directory: $BUILD_DIR"