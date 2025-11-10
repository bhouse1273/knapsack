# Git Commit Message

```
feat: implement platform-specific library builds

Replace cross-platform approach with separate optimized libraries
for each platform, eliminating Metal dependencies on Linux and
simplifying go-chariot integration.

## Changes

### CMakeLists.txt
- Add BUILD_CPU_ONLY option to force CPU-only builds
- Add KNAPSACK_CPU_ONLY compile definition
- Conditionally include Metal headers only when USE_METAL=ON
- Automatic CPU-only on non-Apple platforms

### Source Code
- Update Metal guards from `#ifdef __APPLE__` to
  `#if defined(__APPLE__) && !defined(KNAPSACK_CPU_ONLY)`
- Modified files:
  - knapsack-library/src/knapsack_solve.cpp (2 locations)
  - src/v2/BeamSearch.cpp (4 locations)

### Docker
- Create docker/Dockerfile.linux-cpu for CPU-only Linux builds
- Produces libknapsack_cpu.a (274KB) without Metal dependencies
- Simplified single-stage build (no CMakeLists.txt patching)

### Documentation
- docs/PLATFORM_SPECIFIC_LIBS.md - Implementation details
- docs/GO_CHARIOT_INTEGRATION.md - Updated with build tags
- SUCCESS_SUMMARY.md - Complete verification results
- QUICK_REFERENCE.md - Quick build commands

## Benefits

✅ Cleaner Linux builds (no Metal headers required)
✅ Platform-specific optimization (CPU-only 274KB vs 1.7MB)
✅ Simpler Go integration (build tags vs runtime detection)
✅ Better testing isolation (platform-specific failures)
✅ Easier maintenance (clear separation of concerns)

## Verification

- Linux Docker: ✅ 274KB CPU-only library, no Metal symbols
- macOS Metal: ✅ 1.7MB library with Metal GPU acceleration
- macOS CPU: ✅ 1.7MB library without Metal (testing mode)

## Breaking Changes

None - existing APIs unchanged, backward compatible

## Migration

For go-chariot integration:
1. Use docker/Dockerfile.linux-cpu to build library
2. Create platform-specific Go files with build tags
3. Link against libknapsack_cpu.a on Linux
4. Use CGO_LDFLAGS="-lknapsack_cpu -lstdc++ -lm"
```
