import onnx
from onnx import numpy_helper as nh
import numpy as np

model = onnx.load('robust_mnist2x10.onnx')
weights = model.graph.initializer

print(len(weights))

w1 = nh.to_array(weights[6])
b1 = nh.to_array(weights[7])

w2 = nh.to_array(weights[3])
b2 = nh.to_array(weights[4])

w3 = nh.to_array(weights[0])
b3 = nh.to_array(weights[1])

np.savez("robust_mnist2x10.npz", w1=w1,b1=b1,w2=w2,b2=b2,w3=w3,b3=b3)

