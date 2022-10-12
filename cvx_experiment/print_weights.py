from tensorflow.keras.datasets import mnist
import numpy as np
from scipy.special import softmax
import sys

(X,y), (Xt, yt) = mnist.load_data()

W = np.load(sys.argv[1])
w1 = W["w1"]
b1 = W["b1"]
w2 = W["w2"]
b2 = W["b2"]
w3 = W["w3"]
b3 = W["b3"]

img = np.array(Xt[0]).flatten()/255
temp = np.dot(img, w1) + b1
temp = np.clip(temp, a_min = 0, a_max=None)
temp = np.dot(temp, w2) + b2
temp = np.clip(temp, a_min = 0,a_max=None)
y = np.dot(temp,w3) + b3

print("pre softmax", y)
print("post softmax", softmax(y))
