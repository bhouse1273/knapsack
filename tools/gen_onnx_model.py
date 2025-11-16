#!/usr/bin/env python3
"""
Generate a tiny ONNX model for testing RL inference.
Linear model: output = W @ features + b
Input: [batch, feat_dim]
Output: [batch]
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import sys

def make_linear_model(feat_dim=8, output_path="tests/v2/tiny_linear.onnx"):
    """Create a simple linear model W*x + b"""
    # Random weights and bias (deterministic seed for reproducibility)
    np.random.seed(42)
    W = np.random.randn(feat_dim).astype(np.float32) * 0.1
    b = np.array([0.5], dtype=np.float32)
    
    # Define input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', feat_dim])
    
    # Define output
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch'])
    
    # Create initializers for W and b
    W_init = helper.make_tensor('W', TensorProto.FLOAT, [feat_dim], W.flatten().tolist())
    b_init = helper.make_tensor('b', TensorProto.FLOAT, [1], b.tolist())
    
    # MatMul node: tmp = input @ W^T (requires W as [1, feat_dim])
    # Reshape W to [feat_dim, 1] then transpose, or use ReduceSum after elementwise multiply
    # Simpler: use Gemm or manual MatMul + ReduceSum
    # Let's use element-wise multiply + ReduceSum for simplicity
    
    # Node 1: Mul(input, W) -> [batch, feat_dim]
    mul_node = helper.make_node('Mul', ['input', 'W'], ['mul_out'])
    
    # Node 2: ReduceSum(mul_out) over axis 1 -> [batch]
    # For ONNX opset 13+, axes should be an input, not an attribute
    axes_init = helper.make_tensor('axes', TensorProto.INT64, [1], [1])
    reduce_node = helper.make_node('ReduceSum', ['mul_out', 'axes'], ['reduce_out'], keepdims=0)
    
    # Node 3: Add(reduce_out, b) -> [batch]
    add_node = helper.make_node('Add', ['reduce_out', 'b'], ['output'])
    
    # Create graph
    graph_def = helper.make_graph(
        nodes=[mul_node, reduce_node, add_node],
        name='TinyLinearModel',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[W_init, b_init, axes_init]
    )
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='knapsack_rl_test')
    model_def.opset_import[0].version = 13
    model_def.ir_version = 8  # IR version 8 (compatible with ONNX Runtime 1.22)
    
    # Check and save
    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_path)
    print(f"Saved ONNX model to {output_path}")
    print(f"  IR version: {model_def.ir_version}, Opset: {model_def.opset_import[0].version}")
    print(f"  Input: 'input' shape [batch, {feat_dim}]")
    print(f"  Output: 'output' shape [batch]")
    print(f"  Weights W: {W[:5]}... (first 5)")
    print(f"  Bias b: {b[0]}")
    
    # Test inference with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        # Test with batch=2
        test_input = np.ones((2, feat_dim), dtype=np.float32)
        result = sess.run(['output'], {'input': test_input})[0]
        print(f"  Test inference (input=ones): {result}")
    except ImportError:
        print("  (onnxruntime not installed; skipping test inference)")

if __name__ == '__main__':
    feat_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tests/v2/tiny_linear.onnx"
    make_linear_model(feat_dim, output_path)
