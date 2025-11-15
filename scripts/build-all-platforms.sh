#!/bin/bash
set -e

echo "====================================="
echo "Building All Platform-Specific Libraries"
echo "====================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
LIB_DIR="$PROJECT_ROOT/knapsack-library/lib"

# Create library directories
echo ""
echo "Creating library directories..."
mkdir -p "$LIB_DIR/linux-cpu"
mkdir -p "$LIB_DIR/linux-cuda"
mkdir -p "$LIB_DIR/macos-metal"
mkdir -p "$LIB_DIR/macos-cpu"

# Clean old libraries
echo "Cleaning old libraries..."
rm -f "$LIB_DIR"/linux-cpu/*.a "$LIB_DIR"/linux-cpu/*.h
rm -f "$LIB_DIR"/linux-cuda/*.a "$LIB_DIR"/linux-cuda/*.h
rm -f "$LIB_DIR"/macos-metal/*.a "$LIB_DIR"/macos-metal/*.h
rm -f "$LIB_DIR"/macos-cpu/*.a "$LIB_DIR"/macos-cpu/*.h

echo ""
echo "====================================="
echo "1. Building Linux CPU Library"
echo "====================================="
cd "$PROJECT_ROOT"
docker build --platform linux/amd64 -f docker/Dockerfile.linux-cpu --target builder -t knapsack-linux-cpu-builder .

echo "Extracting Linux CPU library..."
CONTAINER_ID=$(docker create knapsack-linux-cpu-builder)
docker cp "$CONTAINER_ID:/usr/local/lib/libknapsack_cpu.a" "$LIB_DIR/linux-cpu/"
docker cp "$CONTAINER_ID:/usr/local/include/knapsack_cpu.h" "$LIB_DIR/linux-cpu/knapsack_cpu.h"
docker rm "$CONTAINER_ID"

echo "✅ Linux CPU library: $(ls -lh $LIB_DIR/linux-cpu/libknapsack_cpu.a | awk '{print $5}')"

echo ""
echo "====================================="
echo "2. Building Linux CUDA Library"
echo "====================================="
docker build --platform linux/amd64 -f docker/Dockerfile.linux-cuda --target builder -t knapsack-linux-cuda-builder .

echo "Extracting Linux CUDA library..."
CONTAINER_ID=$(docker create knapsack-linux-cuda-builder)
docker cp "$CONTAINER_ID:/usr/local/lib/libknapsack_cuda.a" "$LIB_DIR/linux-cuda/"
docker cp "$CONTAINER_ID:/usr/local/include/knapsack_cuda.h" "$LIB_DIR/linux-cuda/knapsack_cuda.h"
docker rm "$CONTAINER_ID"

echo "✅ Linux CUDA library: $(ls -lh $LIB_DIR/linux-cuda/libknapsack_cuda.a | awk '{print $5}')"

echo ""
echo "====================================="
echo "3. Building macOS Metal Library"
echo "====================================="
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Building natively on macOS..."
    cd "$PROJECT_ROOT/knapsack-library"
    rm -rf build-metal
    mkdir -p build-metal
    cd build-metal
    cmake .. -DUSE_METAL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target knapsack -j$(sysctl -n hw.ncpu)
    
    echo "Copying macOS Metal library..."
    cp libknapsack_metal.a "$LIB_DIR/macos-metal/"
    cp ../include/knapsack_c.h "$LIB_DIR/macos-metal/knapsack_macos_metal.h"
    
    echo "✅ macOS Metal library: $(ls -lh $LIB_DIR/macos-metal/libknapsack_metal.a | awk '{print $5}')"
else
    echo "⚠️  Skipping macOS Metal build (not on macOS)"
    echo "   To build Metal library, run this script on a Mac"
fi

echo ""
echo "====================================="
echo "4. Building macOS CPU-only Library"
echo "====================================="
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Building CPU-only natively on macOS..."
    cd "$PROJECT_ROOT/knapsack-library"
    rm -rf build-macos-cpu
    mkdir -p build-macos-cpu
    cd build-macos-cpu
    cmake .. -DBUILD_CPU_ONLY=ON -DUSE_METAL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target knapsack -j$(sysctl -n hw.ncpu)

    echo "Copying macOS CPU library..."
    # CMake outputs libknapsack_macos_cpu.a on macOS CPU-only builds
    cp libknapsack_macos_cpu.a "$LIB_DIR/macos-cpu/"
    cp ../include/knapsack_c.h "$LIB_DIR/macos-cpu/knapsack_macos_cpu.h"

    echo "✅ macOS CPU library: $(ls -lh $LIB_DIR/macos-cpu/libknapsack_macos_cpu.a | awk '{print $5}')"
else
    echo "⚠️  Skipping macOS CPU build (not on macOS)"
fi

echo ""
echo "====================================="
echo "Build Summary"
echo "====================================="
echo ""
echo "Libraries built and copied to knapsack-library/lib/:"
echo ""

if [ -f "$LIB_DIR/linux-cpu/libknapsack_cpu.a" ]; then
    SIZE=$(ls -lh "$LIB_DIR/linux-cpu/libknapsack_cpu.a" | awk '{print $5}')
    echo "✅ linux-cpu/libknapsack_cpu.a ($SIZE)"
else
    echo "❌ linux-cpu/libknapsack_cpu.a (MISSING)"
fi

if [ -f "$LIB_DIR/linux-cuda/libknapsack_cuda.a" ]; then
    SIZE=$(ls -lh "$LIB_DIR/linux-cuda/libknapsack_cuda.a" | awk '{print $5}')
    echo "✅ linux-cuda/libknapsack_cuda.a ($SIZE)"
else
    echo "❌ linux-cuda/libknapsack_cuda.a (MISSING)"
fi

if [ -f "$LIB_DIR/macos-metal/libknapsack_metal.a" ]; then
    SIZE=$(ls -lh "$LIB_DIR/macos-metal/libknapsack_metal.a" | awk '{print $5}')
    echo "✅ macos-metal/libknapsack_metal.a ($SIZE)"
else
    echo "⚠️  macos-metal/libknapsack_metal.a (not built - run on macOS)"
fi

if [ -f "$LIB_DIR/macos-cpu/libknapsack_macos_cpu.a" ]; then
    SIZE=$(ls -lh "$LIB_DIR/macos-cpu/libknapsack_macos_cpu.a" | awk '{print $5}')
    echo "✅ macos-cpu/libknapsack_macos_cpu.a ($SIZE)"
else
    echo "⚠️  macos-cpu/libknapsack_macos_cpu.a (not built - run on macOS)"
fi

echo ""
echo "====================================="
echo "Verification"
echo "====================================="
echo ""

# Verify no Metal symbols in CPU library
if [ -f "$LIB_DIR/linux-cpu/libknapsack_cpu.a" ]; then
    if nm "$LIB_DIR/linux-cpu/libknapsack_cpu.a" 2>/dev/null | grep -i metal > /dev/null; then
        echo "❌ WARNING: CPU library contains Metal symbols!"
    else
        echo "✅ CPU library verified: No Metal symbols"
    fi
fi

# Verify CUDA symbols in CUDA library
if [ -f "$LIB_DIR/linux-cuda/libknapsack_cuda.a" ]; then
    if nm "$LIB_DIR/linux-cuda/libknapsack_cuda.a" 2>/dev/null | grep -i cuda > /dev/null; then
        echo "✅ CUDA library verified: Contains CUDA symbols"
    else
        echo "⚠️  WARNING: CUDA library may not contain CUDA symbols"
    fi
fi

# Verify Metal symbols in Metal library
if [ -f "$LIB_DIR/macos-metal/libknapsack_metal.a" ]; then
    if nm "$LIB_DIR/macos-metal/libknapsack_metal.a" 2>/dev/null | grep -i metal > /dev/null; then
        echo "✅ Metal library verified: Contains Metal symbols"
    else
        echo "⚠️  WARNING: Metal library may not contain Metal symbols"
    fi
fi

# Verify no Metal symbols in macOS CPU-only library
if [ -f "$LIB_DIR/macos-cpu/libknapsack_macos_cpu.a" ]; then
    if nm "$LIB_DIR/macos-cpu/libknapsack_macos_cpu.a" 2>/dev/null | grep -qi "metal"; then
        echo "❌ WARNING: macOS CPU library contains Metal symbols!"
    else
        echo "✅ macOS CPU library verified: No Metal symbols"
    fi
fi

echo ""
echo "====================================="
echo "✅ All builds complete!"
echo "====================================="
echo ""
echo "Libraries are ready in: knapsack-library/lib/"
echo ""
echo "Next steps:"
echo "1. Commit the libraries: git add knapsack-library/lib/"
echo "2. Update go-chariot to reference these libraries"
echo "3. See docs/GO_CHARIOT_INTEGRATION.md for integration guide"
echo ""
