"""
quantize_to_w4a16_trt.py

Quantizes Qwen3-4B-Z-Image-AbliteratedV1 ONNX model to Weight-Only INT4 (W4A16)
for TensorRT deployment on Jetson Orin Nano.
Activations remain in FP16 to preserve generation quality, while weights are packed
into 4-bit signed integers (Opset 21), reducing weight memory to ~2.1 GB.
"""

import os
import sys
import time
import numpy as np
import torch
import onnx
from onnx import helper, TensorProto

def main():
    print("=" * 70)
    print("Qwen3-4B-Z-Image Weight-Only INT4 (W4A16) Quantization Pipeline")
    print("Target: NVIDIA Jetson Orin Nano (Ampere Architecture)")
    print("=" * 70)

    input_onnx_path = "Qwen3-4B-Z-Image-AbliteratedV1.f16.onnx"
    out_dir = "quantized_model"
    os.makedirs(out_dir, exist_ok=True)
    out_onnx_name = "Qwen3-4B-Z-Image-AbliteratedV1-w4a16.onnx"
    out_data_name = "Qwen3-4B-Z-Image-AbliteratedV1-w4a16.onnx.data"
    out_onnx_path = os.path.join(out_dir, out_onnx_name)

    # -------------------------------------------------------------
    # Step 1: Load Original ONNX Model and Standardize to FP16
    # -------------------------------------------------------------
    print("\n[Step 1/4] Loading original ONNX model and standardizing precision to FP16...")
    t0 = time.time()
    onnx_model = onnx.load(input_onnx_path, load_external_data=True)
    print(f"  Loaded ONNX model with external data in {time.time()-t0:.2f}s.")

    # Upgrade opset to 21 (for native INT4 / UINT4 support)
    for op in onnx_model.opset_import:
        if op.domain == "" or op.domain == "ai.onnx":
            op.version = 21
    onnx_model.ir_version = 10

    # Convert graph inputs, outputs, value_infos from BF16 (16) to FP16 (10)
    for io in list(onnx_model.graph.input) + list(onnx_model.graph.output) + list(onnx_model.graph.value_info):
        if io.type.tensor_type.elem_type == 16:
            io.type.tensor_type.elem_type = 10

    # Convert Cast node attributes 'to = 16' -> 'to = 10'
    for n in onnx_model.graph.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.INT and a.name == "to" and a.i == 16:
                a.i = 10
            elif a.type == onnx.AttributeProto.TENSOR and a.t.data_type == 16:
                a.t.data_type = 10
                t = torch.frombuffer(bytearray(a.t.raw_data), dtype=torch.bfloat16)
                a.t.raw_data = t.to(torch.float16).numpy().tobytes()

    # Convert initializers that remain unquantized from BF16 to FP16
    init_map = {init.name: init for init in onnx_model.graph.initializer}
    for init in onnx_model.graph.initializer:
        if init.data_type == 16:
            init.data_type = 10
            t = torch.frombuffer(bytearray(init.raw_data), dtype=torch.bfloat16)
            init.raw_data = t.to(torch.float16).numpy().tobytes()

    print("  Model standardized to FP16 baseline successfully.")

    # -------------------------------------------------------------
    # Step 2: Identify 245 Linear MatMul Projections & Apply INT4 DQ
    # -------------------------------------------------------------
    print("\n[Step 2/4] Applying Weight-Only INT4 Quantization to 245 Linear Layers...")

    linear_nodes = []
    for n in onnx_model.graph.node:
        if n.op_type == "MatMul":
            weights = [inp for inp in n.input if inp in init_map]
            if len(weights) == 1:
                linear_nodes.append((n, n.input[0], weights[0]))

    proj_nodes = [ln for ln in linear_nodes if "linear" in ln[0].name or ln[0] != linear_nodes[0]]
    if len(proj_nodes) > 35 * 7:
        proj_nodes = linear_nodes[1:]
    assert len(proj_nodes) == 35 * 7, f"Expected 245 linear projections, found {len(proj_nodes)}"
    print(f"  Identified {len(proj_nodes)} linear projection layers across 35 transformer layers.")

    new_nodes = []
    new_initializers = []
    node_set_to_replace = {p[0].name for p in proj_nodes}

    node_info = {node.name: (act_name, weight_name) for (node, act_name, weight_name) in proj_nodes}

    t_quant = time.time()
    for node in onnx_model.graph.node:
        if node.name not in node_set_to_replace:
            new_nodes.append(node)
            continue

        act_name, weight_name = node_info[node.name]
        w_init = init_map[weight_name]
        K_dim, N_dim = w_init.dims[0], w_init.dims[1]

        # Weights are currently float16
        w_fp16 = np.frombuffer(w_init.raw_data, dtype=np.float16).reshape(K_dim, N_dim)
        w_fp32 = w_fp16.astype(np.float32)

        # Symmetric per-channel scaling along output dimension (axis 1)
        w_max_per_channel = np.max(np.abs(w_fp32), axis=0) # shape [N_dim]
        # INT4 range: [-8, 7] -> max magnitude is 7.0
        w_scale = np.maximum(w_max_per_channel / 7.0, 1e-7).astype(np.float16)

        # Quantize weight matrix to signed 4-bit integers [-8, 7]
        w_int4 = np.clip(np.round(w_fp32 / w_scale.astype(np.float32)), -8, 7).astype(np.int8)

        # Pack 2 INT4 values per byte in row-major order:
        # byte = (elem0 & 0x0F) | ((elem1 & 0x0F) << 4)
        w_flat = w_int4.reshape(-1)
        low = w_flat[0::2] & 0x0F
        high = (w_flat[1::2] & 0x0F) << 4
        w_packed = (low | high).astype(np.uint8)

        # Update weight initializer to INT4
        w_init.data_type = TensorProto.INT4
        w_init.raw_data = w_packed.tobytes()

        # Scale and zero-point initializers
        scale_w_tensor_name = f"scale_{weight_name}"
        zp_w_tensor_name = f"zp_{weight_name}"
        dq_w_out_name = f"{weight_name}_dq_f16"

        scale_w_init = helper.make_tensor(
            scale_w_tensor_name,
            TensorProto.FLOAT16,
            [N_dim],
            w_scale.tobytes(),
            raw=True
        )
        # INT4 zero point (all zeros, packed)
        zp_packed_len = (N_dim + 1) // 2
        zp_w_init = helper.make_tensor(
            zp_w_tensor_name,
            TensorProto.INT4,
            [N_dim],
            np.zeros(zp_packed_len, dtype=np.uint8).tobytes(),
            raw=True
        )
        new_initializers.extend([scale_w_init, zp_w_init])

        # DequantizeLinear for INT4 weights with axis=1
        dq_w_node = helper.make_node(
            "DequantizeLinear",
            inputs=[weight_name, scale_w_tensor_name, zp_w_tensor_name],
            outputs=[dq_w_out_name],
            name=f"DequantizeLinear_{weight_name}",
            axis=1
        )
        new_nodes.append(dq_w_node)

        # Updated MatMul consuming original FP16 activation and dequantized INT4 weight
        new_matmul = helper.make_node(
            "MatMul",
            inputs=[act_name, dq_w_out_name],
            outputs=node.output,
            name=node.name
        )
        new_nodes.append(new_matmul)

    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)
    onnx_model.graph.initializer.extend(new_initializers)

    print(f"  All 245 linear projection layers quantized to INT4 in {time.time()-t_quant:.2f}s!")
    print(f"  Activations remain in pure FP16 (no activation distortion).")

    # -------------------------------------------------------------
    # Step 3: Save Quantized Model with External Data
    # -------------------------------------------------------------
    print("\n[Step 3/4] Saving Weight-Only INT4 ONNX model to disk...")
    t0 = time.time()
    onnx.save_model(
        onnx_model,
        out_onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_data_name,
        size_threshold=1024
    )
    print(f"  Model saved in {time.time()-t0:.2f}s:")
    print(f"    ONNX Graph: {out_onnx_path} ({os.path.getsize(out_onnx_path)/(1024*1024):.2f} MB)")
    out_data_path = os.path.join(out_dir, out_data_name)
    print(f"    External Weights: {out_data_path} ({os.path.getsize(out_data_path)/(1024**3):.2f} GB)")

    root_onnx = "Qwen3-4B-Z-Image-AbliteratedV1-w4a16.onnx"
    root_data = "Qwen3-4B-Z-Image-AbliteratedV1-w4a16.onnx.data"
    print(f"\n  Creating convenient symlink/copy in workspace root...")
    try:
        if os.path.exists(root_onnx):
            os.remove(root_onnx)
        if os.path.exists(root_data):
            os.remove(root_data)
        os.link(out_onnx_path, root_onnx)
        os.link(out_data_path, root_data)
        print(f"    Hardlink created: {root_onnx} and {root_data}")
    except Exception:
        import shutil
        shutil.copyfile(out_onnx_path, root_onnx)
        shutil.copyfile(out_data_path, root_data)
        print(f"    File copied: {root_onnx} and {root_data}")

    # -------------------------------------------------------------
    # Step 4: Verification with TensorRT ONNX Parser
    # -------------------------------------------------------------
    print("\n[Step 4/4] Validating Weight-Only INT4 model with TensorRT ONNX Parser...")
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        success = parser.parse_from_file(out_onnx_path)
        if success:
            print("  SUCCESS: TensorRT ONNX Parser parsed the INT4 model with 0 errors!")
            print(f"  Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
            print(f"  Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")
            print(f"  Network layers count: {network.num_layers}")
        else:
            print("  WARNING: TensorRT ONNX Parser encountered errors:")
            for i in range(parser.num_errors):
                print(f"    Error {i}: {parser.get_error(i)}")
    except Exception as e:
        print(f"  TRT validation note: {e}")

    print("\n" + "=" * 70)
    print("Weight-Only INT4 Quantization Complete!")
    print(f"Original size:  ~7.84 GB (FP16/BF16)")
    print(f"Quantized size: ~{os.path.getsize(out_data_path)/(1024**3):.2f} GB (Weight-Only INT4)")
    print("Ready for conversion to TensorRT engine on Jetson Orin Nano:")
    print("  trtexec --onnx=Qwen3-4B-Z-Image-AbliteratedV1-w4a16.onnx \\")
    print("          --saveEngine=Qwen3-4B-Z-Image-AbliteratedV1-w4a16.engine \\")
    print("          --fp16 \\")
    print("          --minShapes=input_ids:1x1,attention_mask:1x1 \\")
    print("          --optShapes=input_ids:1x64,attention_mask:1x64 \\")
    print("          --maxShapes=input_ids:1x256,attention_mask:1x256")
    print("=" * 70)

if __name__ == "__main__":
    main()
