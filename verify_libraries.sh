#!/bin/bash
# Verify platform-specific libraries are correctly built
# This script checks that each library has the correct platform-specific name
# and contains the expected symbols

set -e

echo "======================================"
echo "Platform-Specific Library Verification"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

found_libs=0
issues_found=0

# Function to check a library
check_library() {
    local lib_path=$1
    local expected_symbols=$2
    local forbidden_symbols=$3
    local lib_name=$(basename "$lib_path")
    
    echo "Checking: $lib_name"
    echo "  Path: $lib_path"
    
    if [ ! -f "$lib_path" ]; then
        echo -e "  ${RED}✗ Library not found${NC}"
        ((issues_found++))
        return
    fi
    
    found_libs=$((found_libs + 1))
    
    # Check file size
    local size=$(du -h "$lib_path" | cut -f1)
    echo "  Size: $size"
    
    # Check for expected symbols
    if [ -n "$expected_symbols" ]; then
        if nm "$lib_path" 2>/dev/null | grep -q "$expected_symbols"; then
            echo -e "  ${GREEN}✓ Expected symbols found ($expected_symbols)${NC}"
        else
            echo -e "  ${YELLOW}⚠ Expected symbols not found ($expected_symbols)${NC}"
        fi
    fi
    
    # Check for forbidden symbols
    if [ -n "$forbidden_symbols" ]; then
        if nm "$lib_path" 2>/dev/null | grep -qi "$forbidden_symbols"; then
            echo -e "  ${RED}✗ Forbidden symbols found ($forbidden_symbols)${NC}"
            ((issues_found++))
        else
            echo -e "  ${GREEN}✓ No forbidden symbols ($forbidden_symbols)${NC}"
        fi
    fi
    
    echo ""
}

# Function to check for legacy libraries
check_legacy() {
    echo "Checking for legacy libraries (libknapsack.a without platform suffix)..."
    
    # Find any libknapsack.a files (excluding third_party)
    legacy_libs=$(find . -name "libknapsack.a" -type f ! -path "*/third_party/*" 2>/dev/null || true)
    
    if [ -n "$legacy_libs" ]; then
        echo -e "${RED}✗ Found legacy libraries:${NC}"
        echo "$legacy_libs"
        echo ""
        echo "These should be renamed to platform-specific names:"
        echo "  - libknapsack_cpu.a (CPU-only)"
        echo "  - libknapsack_cuda.a (NVIDIA GPU)"
        echo "  - libknapsack_metal.a (Apple Metal GPU)"
        echo ""
        echo "Run 'make clean-legacy' to remove them."
        ((issues_found++))
    else
        echo -e "${GREEN}✓ No legacy libraries found${NC}"
    fi
    echo ""
}

# Check for legacy libraries first
check_legacy

# Check CPU library
echo "1. CPU-Only Library (libknapsack_cpu.a)"
echo "----------------------------------------"
cpu_paths=(
    "knapsack-library/build-cpu/libknapsack_cpu.a"
    "knapsack-library/build/libknapsack_cpu.a"
    "build-lib/libknapsack_cpu.a"
    "/usr/local/lib/libknapsack_cpu.a"
)

cpu_found=false
for path in "${cpu_paths[@]}"; do
    if [ -f "$path" ]; then
        check_library "$path" "" "metal|Metal|cuda|CUDA"
        cpu_found=true
        break
    fi
done

if [ "$cpu_found" = false ]; then
    echo -e "${YELLOW}⚠ CPU library not found in common locations${NC}"
    echo "  Build with: make build-cpu"
    echo ""
fi

# Check CUDA library
echo "2. CUDA Library (libknapsack_cuda.a)"
echo "-------------------------------------"
cuda_paths=(
    "knapsack-library/build-cuda/libknapsack_cuda.a"
    "knapsack-library/build/libknapsack_cuda.a"
    "build-lib/libknapsack_cuda.a"
    "/usr/local/lib/libknapsack_cuda.a"
)

cuda_found=false
for path in "${cuda_paths[@]}"; do
    if [ -f "$path" ]; then
        check_library "$path" "cuda" "metal|Metal"
        cuda_found=true
        break
    fi
done

if [ "$cuda_found" = false ]; then
    echo -e "${YELLOW}⚠ CUDA library not found (expected if CUDA not installed)${NC}"
    echo "  Build with: make build-cuda"
    echo ""
fi

# Check Metal library (macOS only)
echo "3. Metal Library (libknapsack_metal.a)"
echo "---------------------------------------"
if [ "$(uname)" = "Darwin" ]; then
    metal_paths=(
        "knapsack-library/build-metal/libknapsack_metal.a"
        "knapsack-library/build/libknapsack_metal.a"
        "build-lib/libknapsack_metal.a"
        "/usr/local/lib/libknapsack_metal.a"
    )
    
    metal_found=false
    for path in "${metal_paths[@]}"; do
        if [ -f "$path" ]; then
            check_library "$path" "Metal" "cuda|CUDA"
            metal_found=true
            break
        fi
    done
    
    if [ "$metal_found" = false ]; then
        echo -e "${YELLOW}⚠ Metal library not found${NC}"
        echo "  Build with: make build-metal"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠ Skipping Metal check (not on macOS)${NC}"
    echo ""
fi

# Summary
echo "======================================"
echo "Summary"
echo "======================================"
echo "Libraries found: $found_libs"
echo "Issues found: $issues_found"
echo ""

if [ $issues_found -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Issues found. Please review and fix.${NC}"
    echo ""
    echo "Recommended actions:"
    echo "  1. Run 'make clean-legacy' to remove old libraries"
    echo "  2. Run 'make build-all' to rebuild platform-specific libraries"
    echo "  3. Run this script again to verify"
    exit 1
fi
