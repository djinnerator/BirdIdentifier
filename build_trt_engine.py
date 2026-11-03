"""
build_trt_engine.py

Builds a TensorRT engine from Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx
using the TensorRT Python API on NVIDIA Jetson Orin Nano.
"""

import os
import sys
import tensorrt as trt

def build_engine(
    onnx_path="Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx",
    engine_path="Qwen3-4B-Z-Image-AbliteratedV1-int8.engine",
    min_seq_len=1,
    opt_seq_len=64,
    max_seq_len=256,
    batch_size=1,
    workspace_gb=2
):
    print("=" * 70)
    print("Building TensorRT INT8 Engine from ONNX")
    print(f"ONNX Model:  {onnx_path}")
    print(f"Engine Path: {engine_path}")
    print("=" * 70)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model '{onnx_path}' not found!")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    print("Parsing ONNX model...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for idx in range(parser.num_errors):
                print(f"Parser error {idx}:", parser.get_error(idx))
            raise RuntimeError("Failed to parse ONNX model!")
    print("ONNX model parsed successfully!")

    config = builder.create_builder_config()

    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1024 ** 3))
    except Exception:
        config.max_workspace_size = workspace_gb * (1024 ** 3)

    if hasattr(trt.BuilderFlag, "INT8"):
        config.set_flag(trt.BuilderFlag.INT8)
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for inp_name in ["input_ids", "attention_mask"]:
        min_shape = (batch_size, min_seq_len)
        opt_shape = (batch_size, opt_seq_len)
        max_shape = (batch_size, max_seq_len)
        profile.set_shape(inp_name, min_shape, opt_shape, max_shape)
        print(f"Optimization profile for '{inp_name}': min={min_shape}, opt={opt_shape}, max={max_shape}")

    config.add_optimization_profile(profile)

    print("Building serialized network engine (this will take several minutes)...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Failed to build TensorRT serialized network!")

    with open(engine_path, "wb") as f:
        f.write(plan)

    print(f"Successfully saved engine to '{engine_path}' ({os.path.getsize(engine_path)/(1024**3):.2f} GB)!")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "int8"
    if arg == "int8":
        onnx_file = "Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx"
        out_file = "Qwen3-4B-Z-Image-AbliteratedV1-int8.engine"
    elif arg in ("w4a16", "int4"):
        onnx_file = "Qwen3-4B-Z-Image-AbliteratedV1-w4a16.onnx"
        out_file = "Qwen3-4B-Z-Image-AbliteratedV1-w4a16.engine"
    else:
        onnx_file = arg
        out_file = sys.argv[2] if len(sys.argv) > 2 else onnx_file.replace(".onnx", ".engine")

    build_engine(onnx_file, out_file)
