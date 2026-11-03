"""
onnx_quantize.py

INT8 QDQ Quantization for Qwen3-4B-Z-Image-AbliteratedV1 ONNX Model
Target: NVIDIA Jetson Orin Nano (TensorRT)
"""

from quantize_to_int8_trt import main

if __name__ == "__main__":
    main()
