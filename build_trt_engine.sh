#!/bin/bash
# ==============================================================================
# Build TensorRT Engine on NVIDIA Jetson Orin Nano
# Supports:
#   1. INT8 QDQ Model: Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx (~4.02 GB)
#   2. Weight-Only INT4 Model: Qwen3-4B-Z-Image-AbliteratedV1-w4a16.onnx (~2.37 GB)
#
# Usage:
#   bash build_trt_engine.sh          (defaults to INT8)
#   bash build_trt_engine.sh int8     (builds INT8 QDQ engine)
#   bash build_trt_engine.sh w4a16    (builds Weight-Only INT4 engine)
#   bash build_trt_engine.sh <custom_model.onnx> <custom_output.engine>
# ==============================================================================

set -e

MODE="${1:-int8}"

if [ "$MODE" = "int8" ]; then
    ONNX_MODEL="Qwen3-4B-Z-Image-AbliteratedV1-int8.onnx"
    ENGINE_OUTPUT="Qwen3-4B-Z-Image-AbliteratedV1-int8.engine"
elif [ "$MODE" = "w4a16" ] || [ "$MODE" = "int4" ]; then
    ONNX_MODEL="Qwen3-4B-Z-Image-AbliteratedV1-w4a16.onnx"
    ENGINE_OUTPUT="Qwen3-4B-Z-Image-AbliteratedV1-w4a16.engine"
elif [[ "$MODE" == *.onnx ]]; then
    ONNX_MODEL="$MODE"
    ENGINE_OUTPUT="${2:-${ONNX_MODEL%.onnx}.engine}"
else
    echo "Unknown mode: $MODE. Choose 'int8', 'w4a16', or specify an .onnx file path."
    exit 1
fi

if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: $ONNX_MODEL not found in current directory!"
    exit 1
fi

echo "======================================================================"
echo "Building TensorRT Engine on Jetson Orin Nano"
echo "Model:  $ONNX_MODEL"
echo "Output: $ENGINE_OUTPUT"
echo "======================================================================"

# Determine trtexec path (standard JetPack location or PATH)
TRTEXEC_BIN="trtexec"
if command -v /usr/src/tensorrt/bin/trtexec &> /dev/null; then
    TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
fi

# Detect TensorRT version on Jetson
TRT_VER=$($TRTEXEC_BIN --version 2>&1 | grep -oE "TensorRT v[0-9]+" | grep -oE "[0-9]+" | head -n 1 || echo "8")

echo "Detected TensorRT major version: $TRT_VER"

if [ "$TRT_VER" -ge 10 ]; then
    # TensorRT 10.x+ (JetPack 6.x) syntax:
    # Strongly-typed ONNX models automatically infer INT8/INT4/FP16 from QDQ nodes.
    $TRTEXEC_BIN \
        --onnx="$ONNX_MODEL" \
        --saveEngine="$ENGINE_OUTPUT" \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:1x64,attention_mask:1x64 \
        --maxShapes=input_ids:1x256,attention_mask:1x256 \
        --memPoolSize=workspace:2048M
else
    # TensorRT 8.5/8.6 (JetPack 5.x / 6.0) syntax:
    $TRTEXEC_BIN \
        --onnx="$ONNX_MODEL" \
        --saveEngine="$ENGINE_OUTPUT" \
        --int8 --fp16 \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:1x64,attention_mask:1x64 \
        --maxShapes=input_ids:1x256,attention_mask:1x256 \
        --workspace=2048
fi

echo "======================================================================"
echo "Successfully generated TensorRT engine: $ENGINE_OUTPUT"
echo "======================================================================"
