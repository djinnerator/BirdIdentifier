"""
quantize_to_int8_trt.py

Quantizes Qwen3-4B-Z-Image-AbliteratedV1 ONNX model to INT8 QDQ format
for TensorRT deployment on Jetson Orin Nano.
Calibration is specialized for 19th-century vintage ornithology field guide
book illustrations in the style of John James Audubon, featuring single birds
and collections of 2 to 5 birds with their binomial names (Genus species).
"""

import os
import sys
import time
import numpy as np
import torch
import onnx
from onnx import helper, TensorProto
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def build_bird_prompts():
    """Generates calibration prompts with 1, 2, 3, 4, and 5 birds using Genus species."""
    template = (
        "A 19th-century vintage ornithology field guide book illustration plate of the bird(s): '{birds}'. "
        "Detailed scientific illustration in the style of John James Audubon, hand-colored engraving, "
        "fine plumage textures, naturalistic perch on a botanical branch, aged antique parchment paper "
        "with fine grain, botanical watercolor notes, copperplate Latin binomial inscription at bottom, "
        "archival museum plate."
    )

    # 1 Bird (Single bird samples)
    single_birds = [
        "Melozone fusca",
        "Poecile atricapillus",
        "Corvus brachyrhynchos",
        "Zenaida macroura",
        "Cardinalis cardinalis",
        "Paradisaea raggiana",
        "Passerina ciris",
        "Bubo virginianus",
        "Archilochus colubris",
        "Sialia sialis",
    ]

    # 2 Birds
    two_birds = [
        "Poecile atricapillus and Corvus brachyrhynchos",
        "Cardinalis cardinalis and Passerina ciris",
        "Zenaida macroura and Melozone fusca",
        "Archilochus colubris and Sialia sialis",
        "Bubo virginianus and Tyto alba",
    ]

    # 3 Birds
    three_birds = [
        "Zenaida macroura, Cardinalis cardinalis, and Paradisaea raggiana",
        "Passerina ciris, Sialia sialis, and Spinus tristis",
        "Bubo virginianus, Tyto alba, and Megascops asio",
        "Melozone fusca, Poecile atricapillus, and Corvus brachyrhynchos",
    ]

    # 4 Birds
    four_birds = [
        "Melozone fusca, Poecile atricapillus, Corvus brachyrhynchos, and Cardinalis cardinalis",
        "Passerina ciris, Sialia sialis, Spinus tristis, and Piranga rubra",
        "Bubo virginianus, Tyto alba, Megascops asio, and Strix varia",
    ]

    # 5 Birds
    five_birds = [
        "Melozone fusca, Poecile atricapillus, Corvus brachyrhynchos, Zenaida macroura, and Cardinalis cardinalis",
        "Paradisaea raggiana, Passerina ciris, Sialia sialis, Spinus tristis, and Archilochus colubris",
        "Bubo virginianus, Tyto alba, Megascops asio, Strix varia, and Asio otus",
    ]

    all_bird_groups = single_birds + two_birds + three_birds + four_birds + five_birds
    prompts = [template.format(birds=bg) for bg in all_bird_groups]
    return prompts

def main():
    print("=" * 70)
    print("Qwen3-4B-Z-Image INT8 TensorRT Quantization Pipeline")
    print("Workload: 19th Century Vintage Ornithology Illustrations (John James Audubon)")
    print("Target: NVIDIA Jetson Orin Nano (Ampere Architecture)")
    print("=" * 70)

    # Paths
    model_dir = "tokenizer"
    model_weights_dir = "." if os.path.exists("model.safetensors") else "safetensors"
    input_onnx_path = "Qwen3-4B-Z-Image-AbliteratedV1.f16.onnx"
    out_dir = "quantized_model"
    os.makedirs(out_dir, exist_ok=True)
    out_onnx_name = "Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx"
    out_data_name = "Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx.data"
    out_onnx_path = os.path.join(out_dir, out_onnx_name)

    # -------------------------------------------------------------
    # Step 1: Calibration Data Collection using PyTorch on CUDA
    # -------------------------------------------------------------
    print("\n[Step 1/5] Calibrating activation ranges on GPU with vintage ornithology bird prompts...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    cfg = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_prompts = build_bird_prompts()
    print(f"  Generated {len(calibration_prompts)} specialized calibration prompts (1 to 5 birds, 'Genus species').")
    print(f"  Sample prompt 1 (single bird): {calibration_prompts[0][:120]}...")
    print(f"  Sample prompt 2 (two birds):   {calibration_prompts[10][:120]}...")
    print(f"  Sample prompt 3 (five birds):  {calibration_prompts[-1][:120]}...")

    t0 = time.time()
    pt_model = AutoModelForCausalLM.from_pretrained(model_weights_dir, config=cfg, dtype=torch.float16).to(device)
    pt_model.eval()
    print(f"  PyTorch model loaded in {time.time()-t0:.2f}s.")

    act_maxes = {}

    def get_hook(name):
        def hook(module, args, output):
            inp = args[0]
            val = torch.max(torch.abs(inp)).item()
            if name not in act_maxes or val > act_maxes[name]:
                act_maxes[name] = val
        return hook

    # Register forward hooks on all 35 transformer layers
    hooks = []
    num_layers = 35
    for i in range(num_layers):
        l = pt_model.model.layers[i]
        hooks.append(l.self_attn.q_proj.register_forward_hook(get_hook(f"layer_{i}_qkv")))
        hooks.append(l.self_attn.o_proj.register_forward_hook(get_hook(f"layer_{i}_o")))
        hooks.append(l.mlp.gate_proj.register_forward_hook(get_hook(f"layer_{i}_gate_up")))
        hooks.append(l.mlp.down_proj.register_forward_hook(get_hook(f"layer_{i}_down")))

    print("  Running GPU calibration forward passes...")
    t_cal = time.time()
    with torch.no_grad():
        for p in calibration_prompts:
            enc = tokenizer(p, return_tensors="pt", padding="max_length", max_length=128, truncation=True).to(device)
            pt_model(**enc)

    for h in hooks:
        h.remove()

    del pt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Calibration complete in {time.time()-t_cal:.2f}s! Collected dynamic ranges for {len(act_maxes)} activation points.")

    # -------------------------------------------------------------
    # Step 2: Load Original ONNX Model and Standardize to FP16
    # -------------------------------------------------------------
    print("\n[Step 2/5] Loading original ONNX model and standardizing precision to FP16...")
    t0 = time.time()
    onnx_model = onnx.load(input_onnx_path, load_external_data=True)
    print(f"  Loaded ONNX model with external data in {time.time()-t0:.2f}s.")

    # Upgrade opset to 19 (for full FP16 QDQ support)
    for op in onnx_model.opset_import:
        if op.domain == "" or op.domain == "ai.onnx":
            op.version = 19
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
    # Step 3: Identify 245 Linear MatMul Projections & Apply INT8 QDQ
    # -------------------------------------------------------------
    print("\n[Step 3/5] Applying INT8 QDQ Quantization to Linear Layers...")

    linear_nodes = []
    for n in onnx_model.graph.node:
        if n.op_type == "MatMul":
            weights = [inp for inp in n.input if inp in init_map]
            if len(weights) == 1:
                linear_nodes.append((n, n.input[0], weights[0]))

    # Skip RoPE positional encoding matmul if present at start
    proj_nodes = [ln for ln in linear_nodes if "linear" in ln[0].name or ln[0] != linear_nodes[0]]
    if len(proj_nodes) > 35 * 7:
        proj_nodes = linear_nodes[1:]
    assert len(proj_nodes) == 35 * 7, f"Expected 245 linear projections, found {len(proj_nodes)}"
    print(f"  Identified {len(proj_nodes)} linear projection layers across 35 transformer layers.")

    new_nodes = []
    new_initializers = []
    node_set_to_replace = {p[0].name for p in proj_nodes}

    role_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    act_keys = ["qkv", "qkv", "qkv", "o", "gate_up", "gate_up", "down"]

    quantized_linear_info = {}
    for idx, (node, act_name, weight_name) in enumerate(proj_nodes):
        layer_idx = idx // 7
        role_idx = idx % 7
        role = role_names[role_idx]
        act_key = f"layer_{layer_idx}_{act_keys[role_idx]}"
        quantized_linear_info[node.name] = (layer_idx, role, act_key, act_name, weight_name)

    dq_act_cache = {}

    for node in onnx_model.graph.node:
        if node.name not in node_set_to_replace:
            new_nodes.append(node)
            continue

        layer_idx, role, act_key, act_name, weight_name = quantized_linear_info[node.name]

        # 1. Activation QDQ (Per-Tensor Symmetric INT8)
        dq_act_name = dq_act_cache.get(act_name)
        if dq_act_name is None:
            max_val = act_maxes.get(act_key, 1.0)
            act_scale_val = max(max_val / 127.0, 1e-6)

            scale_act_tensor_name = f"scale_{act_name}"
            zp_act_tensor_name = f"zp_{act_name}"
            q_act_out_name = f"{act_name}_q_int8"
            dq_act_name = f"{act_name}_dq_f16"

            scale_act_init = helper.make_tensor(
                scale_act_tensor_name,
                TensorProto.FLOAT16,
                [],
                [np.float16(act_scale_val)]
            )
            zp_act_init = helper.make_tensor(
                zp_act_tensor_name,
                TensorProto.INT8,
                [],
                [0]
            )
            new_initializers.extend([scale_act_init, zp_act_init])

            q_node = helper.make_node(
                "QuantizeLinear",
                inputs=[act_name, scale_act_tensor_name, zp_act_tensor_name],
                outputs=[q_act_out_name],
                name=f"QuantizeLinear_{act_name}"
            )
            dq_node = helper.make_node(
                "DequantizeLinear",
                inputs=[q_act_out_name, scale_act_tensor_name, zp_act_tensor_name],
                outputs=[dq_act_name],
                name=f"DequantizeLinear_{act_name}"
            )
            new_nodes.extend([q_node, dq_node])
            dq_act_cache[act_name] = dq_act_name

        # 2. Weight DQ (Per-Channel Symmetric INT8 along axis 1)
        w_init = init_map[weight_name]
        K_dim, N_dim = w_init.dims[0], w_init.dims[1]

        w_fp16 = np.frombuffer(w_init.raw_data, dtype=np.float16).reshape(K_dim, N_dim)
        w_fp32 = w_fp16.astype(np.float32)

        w_max_per_channel = np.max(np.abs(w_fp32), axis=0)
        w_scale = np.maximum(w_max_per_channel / 127.0, 1e-7).astype(np.float16)
        w_zp = np.zeros(N_dim, dtype=np.int8)

        w_int8 = np.clip(np.round(w_fp32 / w_scale.astype(np.float32)), -128, 127).astype(np.int8)

        w_init.data_type = TensorProto.INT8
        w_init.raw_data = w_int8.tobytes()

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
        zp_w_init = helper.make_tensor(
            zp_w_tensor_name,
            TensorProto.INT8,
            [N_dim],
            w_zp.tobytes(),
            raw=True
        )
        new_initializers.extend([scale_w_init, zp_w_init])

        dq_w_node = helper.make_node(
            "DequantizeLinear",
            inputs=[weight_name, scale_w_tensor_name, zp_w_tensor_name],
            outputs=[dq_w_out_name],
            name=f"DequantizeLinear_{weight_name}",
            axis=1
        )
        new_nodes.append(dq_w_node)

        # 3. Updated MatMul
        new_matmul = helper.make_node(
            "MatMul",
            inputs=[dq_act_name, dq_w_out_name],
            outputs=node.output,
            name=node.name
        )
        new_nodes.append(new_matmul)

    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)
    onnx_model.graph.initializer.extend(new_initializers)

    print(f"  Successfully inserted QDQ nodes.")
    print(f"  All 245 linear projection layers quantized to INT8.")

    # -------------------------------------------------------------
    # Step 4: Save Quantized Model with External Data
    # -------------------------------------------------------------
    print("\n[Step 4/5] Saving INT8 QDQ ONNX model to disk...")
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

    root_onnx = "Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx"
    root_data = "Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx.data"
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
    # Step 5: Verification with TensorRT ONNX Parser
    # -------------------------------------------------------------
    print("\n[Step 5/5] Validating INT8 QDQ model with TensorRT ONNX Parser...")
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        success = parser.parse_from_file(out_onnx_path)
        if success:
            print("  SUCCESS: TensorRT ONNX Parser parsed the model with 0 errors!")
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
    print("Quantization Complete!")
    print(f"Original size:  ~7.84 GB (FP16/BF16)")
    print(f"Quantized size: ~{os.path.getsize(out_data_path)/(1024**3):.2f} GB (INT8 QDQ)")
    print("Ready for conversion to TensorRT engine on Jetson Orin Nano:")
    print("  trtexec --onnx=Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx \\")
    print("          --saveEngine=Qwen3-4B-Z-Image-AbliteratedV1-int8.engine \\")
    print("          --int8 --fp16 \\")
    print("          --minShapes=input_ids:1x1,attention_mask:1x1 \\")
    print("          --optShapes=input_ids:1x64,attention_mask:1x64 \\")
    print("          --maxShapes=input_ids:1x256,attention_mask:1x256")
    print("=" * 70)

if __name__ == "__main__":
    main()
