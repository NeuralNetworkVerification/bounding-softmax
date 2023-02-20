import onnx
from onnx import numpy_helper as nh
import numpy as np
import sys

model = onnx.load(sys.argv[1])
weights = model.graph.initializer

print(len(weights))



w1 = nh.to_array(weights[9])
b1 = nh.to_array(weights[10])

w2 = nh.to_array(weights[6])
b2 = nh.to_array(weights[7])

w3 = nh.to_array(weights[3])
b3 = nh.to_array(weights[4])

w4 = nh.to_array(weights[0])
b4 = nh.to_array(weights[1])


print(w1.shape,b1.shape,w2.shape,b2.shape,w3.shape,b3.shape,w4.shape,b4.shape)

np.savez(f"{sys.argv[1][:-4]}npz", w1=w1,b1=b1,w2=w2,b2=b2,w3=w3,b3=b3,w4=w4,b4=b4)
