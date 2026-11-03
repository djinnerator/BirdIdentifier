"""
safetensors_to_onnx.py

Converts Qwen3-4B-Z-Image-AbliteratedV1 safetensors model to an ONNX model
outputting penultimate hidden states for Z-Image text encoding.
"""

import os
import time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def export_safetensors_to_onnx(
    model_dir=".",
    tokenizer_dir="tokenizer",
    output_onnx="Qwen3-4B-Z-Image-AbliteratedV1.f16.onnx"
):
    print("=" * 70)
    print("Converting safetensors to ONNX (Qwen3-4B-Z-Image-AbliteratedV1.f16)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    t0 = time.time()

    cfg = AutoConfig.from_pretrained(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.2f}s")

    # Wrapper to output the exact penultimate hidden states required by Z-Image text encoder
    class Qwen3TextEncoderWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            return outputs.hidden_states[-2]

    wrapped_model = Qwen3TextEncoderWrapper(model)

    dummy_text = "A 19th-century vintage ornithology field guide book illustration plate of the bird: 'Melozone fusca'."
    inputs = tokenizer(dummy_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Exporting to {output_onnx} with external data format...")
    t1 = time.time()
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask),
        output_onnx,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['penultimate_hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'penultimate_hidden_states': {0: 'batch_size', 1: 'sequence_length'}
        },
        external_data=True
    )
    print(f"Export finished in {time.time()-t1:.2f}s!")
    data_file = output_onnx + ".data"
    print(f"  ONNX Graph: {output_onnx} ({os.path.getsize(output_onnx)/(1024*1024):.2f} MB)")
    print(f"  External Weights: {data_file} ({os.path.getsize(data_file)/(1024**3):.2f} GB)")

if __name__ == "__main__":
    export_safetensors_to_onnx()
