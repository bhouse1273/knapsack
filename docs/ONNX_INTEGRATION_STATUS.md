# Real ONNX Integration Status

## Completed
1. ✅ BUILD_ONNX CMake option with auto-detection (homebrew paths)
2. ✅ Python script to generate test ONNX models (`tools/gen_onnx_model.py`)
3. ✅ Documentation for model generation (`docs/ONNX_MODEL_GEN.md`)
4. ✅ CMake conditional compilation for RL libraries with ONNX support

## Next Steps

### Install ONNX Runtime (prerequisite)
```bash
# macOS (Apple Silicon)
brew install onnxruntime

# OR download pre-built from:
# https://github.com/microsoft/onnxruntime/releases
```

### Generate Test Models (requires Python packages)
```bash
pip3 install onnx numpy
python3 tools/gen_onnx_model.py 8 tests/v2/tiny_linear_8.onnx
python3 tools/gen_onnx_model.py 12 tests/v2/tiny_linear_12.onnx
```

### Implementation Plan
1. **Extend RLContext** (rl/rl_api.cpp):
   - Add `#ifdef RL_ONNX_ENABLED` guards
   - Include ONNX Runtime C++ API headers
   - Add `Ort::Env`, `Ort::SessionOptions`, `Ort::Session` members
   - Parse model config: providers, threads, input/output names

2. **Initialize ONNX session** (rl_init_from_json):
   - Create Ort::Env (once per process)
   - Set SessionOptions (threads, optimization level)
   - Load session from model_path
   - Introspect input/output shapes
   - Validate feat_dim matches model input
   - Set model_loaded flag; fallback to bandit on failure

3. **Inference path** (rl_score_batch_with_features):
   - Check if ONNX session loaded
   - Wrap features buffer as Ort::Value with shape [num_candidates, feat_dim]
   - Run session->Run()
   - Extract output tensor (shape [num_candidates])
   - Copy scores to out_scores
   - On error: log, fallback to bandit, return error code

4. **Add tests** (tests/v2/test_rl_api.cpp):
   - Load tiny_linear_8.onnx model
   - Score batch with known features
   - Validate outputs match expected (±tolerance)
   - Test feat_dim mismatch → graceful fallback
   - Test missing model path → bandit fallback

5. **Update docs**:
   - RL_SUPPORT.md: add ONNX config fields
   - RL_CONFIG_EXAMPLE.json: add model_path, providers, threads

## Build Command
```bash
# Configure with ONNX support
cmake -S . -B build -DBUILD_ONNX=ON

# Build RL support
cmake --build build --target rl_support test_rl_api -j8

# Run tests
./build/tests/v2/test_rl_api
```

## Configuration Example
```json
{
  "w_rl": 1.0,
  "alpha": 0.3,
  "feat_dim": 8,
  "model_path": "tests/v2/tiny_linear_8.onnx",
  "model_input": "input",
  "model_output": "output",
  "providers": ["CPU"],
  "threads": {
    "intra": 1,
    "inter": 1
  }
}
```

## Fallback Behavior
- If BUILD_ONNX=OFF: stub bonus (current behavior)
- If model_path missing/invalid: LinUCB bandit
- If feat_dim mismatch: error + bandit fallback
- If inference fails: error + bandit fallback

## Performance Notes
- ONNX Runtime sessions are thread-safe for concurrent Run() calls
- CPU ExecutionProvider is default and portable
- CoreML EP available on macOS for acceleration (optional)
- Typical latency: <1ms for batch of 100 candidates with small models

---
**Current blocker**: ONNX Runtime not installed on system.
**Resolution**: Run `brew install onnxruntime` then reconfigure CMake with `-DBUILD_ONNX=ON`
