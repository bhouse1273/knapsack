# RL ONNX Model Generation

## Prerequisites
```bash
pip3 install onnx numpy
# Optional for testing:
pip3 install onnxruntime
```

## Generate Test Models

### Linear model (feat_dim=8)
```bash
python3 tools/gen_onnx_model.py 8 tests/v2/tiny_linear_8.onnx
```

### Linear model (feat_dim=12)
```bash
python3 tools/gen_onnx_model.py 12 tests/v2/tiny_linear_12.onnx
```

## Model Contract
- **Input**: `input` tensor, shape `[batch, feat_dim]`, type `float32`
- **Output**: `output` tensor, shape `[batch]`, type `float32`
- **Operation**: Linear transformation `output = W @ features + b`

## Testing
The generated model can be tested with:
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("tests/v2/tiny_linear_8.onnx")
features = np.ones((5, 8), dtype=np.float32)  # batch of 5
result = sess.run(['output'], {'input': features})[0]
print(result)  # shape [5]
```

## Integration
The RL library will:
1. Load the ONNX session in `rl_init_from_json` when `model_path` is provided
2. Wrap feature batches as ONNX input tensors
3. Run inference via `session->Run()`
4. Extract output scores
5. Fallback to LinUCB bandit if model loading/inference fails
