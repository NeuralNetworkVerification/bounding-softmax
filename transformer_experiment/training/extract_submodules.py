# split the sst model into embedder and

import onnx

input_path = "./self-attention-sst-sim.onnx"
output_path = "./self-attention-sst-sim-embedding.onnx"

model = onnx.load(input_path)

onnx.utils.extract_model(input_path, output_path, input_names=["input_1"],
                         output_names=["model_1/flatten/Reshape:0"])

# need to reshape to 2 x 8

output_path = "./self-attention-sst-sim-post-embedding.onnx"
onnx.utils.extract_model(input_path, output_path,
                         input_names=["model_1/flatten/Reshape:0"],
                         output_names=["model_1/dense_4/BiasAdd:0"])
