import pickle
import numpy as np
import sys

VAL_SEQ = "./validation_sequences.pickle"
VAL_LAB = "./validation_labels.pickle"
EMBED = "./self-attention-sst-sim-embedding.onnx"
CLASS = "./self-attention-sst-sim-post-embedding.onnx"

with open(VAL_SEQ, 'rb') as fp:
    sequences = pickle.load(fp)

with open(VAL_LAB, 'rb') as fp:
    labels = pickle.load(fp)

import onnxruntime

ind = int(sys.argv[1])

seq = np.array(sequences[ind]).astype(np.float32)
lab = labels[ind]

print("input:", seq)
print("label:", lab)

session = onnxruntime.InferenceSession(EMBED, None)
inputName = session.get_inputs()[0].name
outputName = session.get_outputs()[0].name

embedding = session.run([outputName], {inputName: seq[None,:]})[0][0]

print("Embedding:", embedding)

session = onnxruntime.InferenceSession(CLASS, None)
inputName = session.get_inputs()[0].name
outputName = session.get_outputs()[0].name

embedding = session.run([outputName], {inputName: embedding[None,:]})[0][0]
print("Prediction:", embedding)
