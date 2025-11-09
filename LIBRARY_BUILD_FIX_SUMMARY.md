# Library Build Fix Summary

## Issues Addressed

### Issue 1: Missing Config_validate.cpp

**Problem**: The `Config_validate.cpp` file existed in `src/v2/` but was not included in the library build, causing linker errors in go-chariot when it tried to call `ValidateConfig()`.

**Fix**: Added `Config_validate.cpp` to the CMakeLists.txt source list.

**Location**: `knapsack-library/CMakeLists.txt` line 39

```cmake
list(APPEND LIB_SOURCES
  "${PROJ_ROOT}/src/v2/Data.cpp"
  "${PROJ_ROOT}/src/v2/EvalCPU.cpp"
  "${PROJ_ROOT}/src/v2/BeamSearch.cpp"
  "${PROJ_ROOT}/src/v2/Preprocess.cpp"
  "${PROJ_ROOT}/src/v2/Config_validate.cpp"  # <-- ADDED THIS LINE
)
```

### Issue 2: Outdated Validation Code

**Problem**: The original `Config_validate.cpp` code was written for an old version of the `Config` struct that had simple fields like `capacity` and a vector of items. The actual v2 Config uses a more sophisticated structure with `ItemsSpec`, `KnapsackSpec`, and Structure-of-Arrays for attributes.

**Fix**: Completely rewrote `Config_validate.cpp` to match the actual v2::Config structure:

**Old validation** (incorrect):
- Checked `cfg.capacity` (doesn't exist)
- Checked `cfg.items.empty()` (ItemsSpec isn't a vector)
- Checked `cfg.items[i].weight` (ItemsSpec doesn't have operator[])

**New validation** (correct):
- Checks `cfg.items.count > 0`
- Validates all attribute arrays match item count
- For assign mode: validates knapsack specs (K, capacities, capacity_attr)
- Validates all constraints reference valid attributes
- Validates all objective terms reference valid attributes

**Result**: Proper validation for the Structure-of-Arrays (SoA) design used in v2.

### Issue 3: Platform Specification for Docker Builds

**Problem**: Building on M1 Mac without explicit platform specification could cause issues with architecture mismatches.

**Fix**: Added `--platform linux/amd64` to all Docker build commands in `scripts/build-all-platforms.sh`.

**Changes**:
```bash
# Before:
docker build -f docker/Dockerfile.linux-cpu --target builder -t knapsack-linux-cpu-builder .

# After:
docker build --platform linux/amd64 -f docker/Dockerfile.linux-cpu --target builder -t knapsack-linux-cpu-builder .
```

Applied to both CPU and CUDA builds.

## Files Modified

### 1. knapsack-library/CMakeLists.txt
- **Change**: Added `Config_validate.cpp` to LIB_SOURCES
- **Line**: 39
- **Impact**: Library now includes validation functionality

### 2. src/v2/Config_validate.cpp
- **Change**: Complete rewrite to match v2::Config structure
- **Lines**: Entire file (125 lines)
- **Impact**: Validation now works with SoA design
- **Key validations**:
  - Items count > 0
  - All attribute arrays match item count
  - Knapsack specs valid for assign mode
  - All constraints/objectives reference valid attributes

### 3. scripts/build-all-platforms.sh
- **Change**: Added `--platform linux/amd64` to Docker builds
- **Lines**: 31, 46
- **Impact**: Explicit architecture specification for cross-platform builds

## Build Results

### Before Fixes
```
Error: undefined reference to `v2::ValidateConfig(v2::Config const&, std::string*)'
```

### After Fixes
```
✅ Linux CPU library: 312K
Successfully extracted to: knapsack-library/lib/linux-cpu/libknapsack_cpu.a
```

**Library size increased** from 274KB to 312KB due to the addition of comprehensive validation logic.

## Verification

### Library Created Successfully
```bash
$ ls -lh knapsack-library/lib/linux-cpu/
total 624
-rw-r--r--  1 user  staff   3.5K Nov  7 13:35 knapsack_cpu.h
-rw-r--r--  1 user  staff   312K Nov  7 13:35 libknapsack_cpu.a
```

### Symbols Verification
```bash
$ nm knapsack-library/lib/linux-cpu/libknapsack_cpu.a | grep ValidateConfig
0000000000000000 T _ZN2v214ValidateConfigERKNS_6ConfigEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

The symbol is now present and exported (T = text section, globally visible).

## Impact on go-chariot

### Previous Issue
When go-chariot tried to link against the knapsack library, it would fail with:
```
undefined reference to `v2::ValidateConfig(v2::Config const&, std::string*)'
```

### Current Status
The knapsack library now includes all necessary symbols for go-chariot to:
1. Parse JSON config files (Config_json.cpp)
2. Validate config structure (Config_validate.cpp) ✅ FIXED
3. Execute solver logic (BeamSearch.cpp, EvalCPU.cpp, etc.)

## Validation Logic Details

The new `ValidateConfig()` function performs these checks:

### Basic Structure Validation
- **Items count**: Must be > 0
- **Attribute arrays**: All must match items.count size
- **Example**: If `items.count = 100`, then `items.attributes["weight"]` must have exactly 100 values

### Assign Mode Validation
When `mode == "assign"`:
- **K**: Number of knapsacks must be > 0
- **Capacities**: Array size must match K, all values must be positive
- **Capacity attribute**: Must be specified and exist in items.attributes

### Constraint Validation
For each constraint:
- **Attribute reference**: Must exist in items.attributes (if specified)
- **Limit**: Must be non-negative

### Objective Validation
For each objective term:
- **Attribute reference**: Must exist in items.attributes

## Next Steps

### 1. Build Remaining Libraries

The CPU library is now fixed. To build CUDA and Metal libraries:

```bash
# Build all three libraries
make build-all-platforms

# Or build individually:
# CPU (already done)
docker build --platform linux/amd64 -f docker/Dockerfile.linux-cpu --target builder -t knapsack-linux-cpu-builder .

# CUDA
docker build --platform linux/amd64 -f docker/Dockerfile.linux-cuda --target builder -t knapsack-linux-cuda-builder .

# Metal (on macOS)
cd knapsack-library && rm -rf build-metal && mkdir build-metal
cd build-metal && cmake .. -DUSE_METAL=ON && make -j
```

### 2. Verify All Libraries

```bash
make verify-libs
```

Should show:
```
✅ All libraries present
-rw-r--r--  312K  knapsack-library/lib/linux-cpu/libknapsack_cpu.a
-rw-r--r--  ###K  knapsack-library/lib/linux-cuda/libknapsack_cuda.a
-rw-r--r--  ###K  knapsack-library/lib/macos-metal/libknapsack_metal.a
```

### 3. Commit Changes

```bash
git add knapsack-library/CMakeLists.txt
git add src/v2/Config_validate.cpp
git add scripts/build-all-platforms.sh
git add knapsack-library/lib/linux-cpu/

git commit -m "Fix library build: add Config_validate.cpp with v2-compatible validation

PROBLEM:
- go-chariot builds failing with undefined reference to ValidateConfig
- Config_validate.cpp existed but wasn't included in CMakeLists.txt
- Original validation code incompatible with v2::Config SoA structure

SOLUTION:
- Added Config_validate.cpp to library sources (CMakeLists.txt line 39)
- Rewrote validation to work with Structure-of-Arrays design
- Added platform specification to Docker builds (--platform linux/amd64)

VALIDATION:
- Items count > 0
- All attribute arrays match item count
- Assign mode: validates K, capacities, capacity_attr
- All constraints/objectives reference valid attributes

RESULT:
- CPU library built successfully (312KB, up from 274KB)
- ValidateConfig symbol now exported
- Ready for go-chariot integration

FILES:
- Modified: knapsack-library/CMakeLists.txt (added Config_validate.cpp)
- Modified: src/v2/Config_validate.cpp (complete rewrite for v2)
- Modified: scripts/build-all-platforms.sh (added platform flag)
- Updated: knapsack-library/lib/linux-cpu/ (rebuilt library)"
```

### 4. Test in go-chariot

```bash
cd chariot-ecosystem
make docker-build-knapsack-cpu

# Should now build successfully without linker errors
```

## Technical Notes

### Why the Size Increased

The library grew from 274KB to 312KB (38KB increase) because:
1. Added Config_validate.cpp compilation unit (wasn't there before)
2. New validation logic is more comprehensive:
   - Validates attribute arrays
   - Validates knapsack specs
   - Validates constraint/objective references
   - More error message generation code

This is **expected and correct** - we're adding functionality that was missing.

### Structure-of-Arrays (SoA) Design

The v2 Config uses SoA for performance:

**Traditional Array-of-Structures** (old):
```cpp
struct Item { double weight; double value; };
vector<Item> items;
```

**Structure-of-Arrays** (v2):
```cpp
struct ItemsSpec {
  int count;
  map<string, vector<double>> attributes;  // "weight" -> [w1,w2,...], "value" -> [v1,v2,...]
};
```

Benefits:
- Better cache locality
- SIMD-friendly
- Flexible attributes (not hardcoded)
- GPU-friendly memory layout

The new validation code works with this SoA design.

## Conclusion

✅ **All issues resolved**:
1. Config_validate.cpp now included in build
2. Validation logic rewritten for v2::Config
3. Platform explicitly specified for Docker builds
4. CPU library rebuilt successfully (312KB)

✅ **Ready for**:
1. Building CUDA and Metal libraries
2. Committing changes to repository
3. Testing go-chariot integration

The knapsack library now provides complete functionality including config validation, which was the missing piece causing go-chariot link failures.
