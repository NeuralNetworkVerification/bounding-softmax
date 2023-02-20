from tensorflow.keras.datasets import mnist
import numpy as np
from scipy.special import softmax
import sys

(X,y), (Xt, yt) = mnist.load_data()

correct = 0
for i in range(len(yt)):
    y = []
    for j in range(5):
        W = np.load(f"networks/robust_mnist-large-{j}.npz")
        w1 = W["w1"]
        b1 = W["b1"]
        w2 = W["w2"]
        b2 = W["b2"]
        w3 = W["w3"]
        b3 = W["b3"]
        w4 = W["w4"]
        b4 = W["b4"]

        img = np.array(Xt[i]).flatten()/255
        temp = np.dot(img, w1) + b1
        temp = np.clip(temp, a_min = 0, a_max=None)
        temp = np.dot(temp, w2) + b2
        temp = np.clip(temp, a_min = 0,a_max=None)
        temp = np.dot(temp, w3) + b3
        temp = np.clip(temp, a_min = 0,a_max=None)
        if len(y) == 0:
            y = np.dot(temp,w4) + b4
        else:
            y += np.dot(temp,w4) + b4
    y = np.argmax(y)
    print(y, yt[i])
    if y == yt[i]:
        correct += 1

print(correct / len(yt))
