#!/bin/bash
# publish-libs.sh - Copy knapsack and RL libraries to /usr/local/lib

set -e

echo "Publishing knapsack and RL libraries to /usr/local/lib..."

# Determine platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
else
    PLATFORM="linux"
fi

# Copy platform-specific libraries
cp -r lib /usr/local/

echo "Libraries published:"
ls -lh /usr/local/lib/*/lib*.a /usr/local/lib/*/lib*.dylib /usr/local/lib/*/lib*.so 2>/dev/null || true
echo ""
echo "Headers published:"
ls -lh /usr/local/lib/*/*.h 2>/dev/null || true

echo ""
echo "✅ Knapsack and RL libraries successfully published to /usr/local/lib"
