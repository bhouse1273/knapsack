#!/bin/bash
# Quick validation script for chariot CGO integration debugging

set -e

echo "=================================================="
echo "CHARIOT CGO VALIDATION SUITE"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if library exists
echo "Step 1: Checking for knapsack library..."
LIB_PATHS=(
    "knapsack-library/build/libknapsack.dylib"
    "knapsack-library/build/libknapsack.so"
    "build/libknapsack.dylib"
    "build/libknapsack.so"
    "/usr/local/lib/libknapsack_metal.dylib"
    "/usr/local/lib/libknapsack_cpu.dylib"
)

LIB_FOUND=0
for lib in "${LIB_PATHS[@]}"; do
    if [ -f "$lib" ]; then
        echo -e "${GREEN}✓${NC} Found library: $lib"
        LIB_FOUND=1
        break
    fi
done

if [ $LIB_FOUND -eq 0 ]; then
    echo -e "${YELLOW}⚠${NC} Library not found in expected locations."
    echo "Checking if library was installed via publish-libs.sh..."
fi

echo ""
echo "Step 2: Running Python validation tests..."
echo "------------------------------------------------"

if ! python3 -c "import numpy" 2>/dev/null; then
    echo -e "${YELLOW}⚠${NC} numpy not installed. Installing..."
    pip3 install numpy
fi

if python3 tests/python/test_knapsack_c_api.py; then
    echo -e "${GREEN}✓${NC} All Python tests passed"
    PYTHON_STATUS="PASSED"
else
    echo -e "${RED}✗${NC} Python tests failed"
    PYTHON_STATUS="FAILED"
    exit 1
fi

echo ""
echo "=================================================="
echo "VALIDATION COMPLETE"
echo "=================================================="
echo ""
echo "Summary:"
echo "  - Python tests: ${PYTHON_STATUS}"
echo ""
echo "These tests validate the correct C API behavior."
echo "Share the output with chariot-ecosystem to help debug CGO integration."
echo ""
echo "Next steps for chariot team:"
echo "  1. Review the generated Go/CGO code examples above"
echo "  2. Verify pointer conversion matches examples"
echo "  3. Check array types ([]C.int vs []int)"
echo "  4. Validate library linkage paths"
echo ""
