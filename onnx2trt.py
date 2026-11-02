import tensorrt as trt

# 1. Initialize Logger and Builder
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)

# 2. Create Network Definition with Explicit Batch
network = builder.create_network()

# 3. Parse ONNX Model
parser = trt.OnnxParser(network, TRT_LOGGER)
onnx_file_path = "BirdID_US.onnx"

with open(onnx_file_path, "rb") as model_file:
    if not parser.parse(model_file.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("Failed to parse the ONNX model.")

# 4. Create Builder Configuration and Optimization Profile
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB workspace

profile = builder.create_optimization_profile()

# Inspect input tensor name and set shapes: (min, opt, max)
# Replace 'input' with your actual ONNX input name, and shapes with your dimensions
input_name = network.get_input(0).name
profile.set_shape(
    input_name,
    min=(1, 1, 64, 256),   # Minimum shape
    opt=(1, 1, 64, 256),   # Optimal shape (best performance)
    max=(128, 1, 64, 256),   # Maximum shape
)
config.add_optimization_profile(profile)

# 5. Build and Serialize Engine
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build the serialized TensorRT engine.")

engine_file_path = "BirdID_US.trt"
with open(engine_file_path, "wb") as f:
    f.write(serialized_engine)

print(f"TensorRT engine successfully saved to {engine_file_path}")
