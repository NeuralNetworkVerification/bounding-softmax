from tensorflow.keras.datasets import mnist
import numpy as np
from scipy.special import softmax
import sys

import torchvision.datasets as datasets
import torchvision.transforms as transforms

#(X,y), (Xt, yt) = mnist.load_data()
cifar_test = datasets.CIFAR10('/home/haozewu/Projects/addSoISupport/Marabou/resources/data/cifardata/', train=False, download=False, transform=transforms.ToTensor())


W = np.load(sys.argv[1])
w1 = W["w1"]
b1 = W["b1"]
w2 = W["w2"]
b2 = W["b2"]
w3 = W["w3"]
b3 = W["b3"]
w4 = W["w4"]
b4 = W["b4"]

X,y = cifar_test[0]
img = X.unsqueeze(0).numpy().transpose(0,2,3,1).flatten()

temp = np.dot(img, w1) + b1
temp = np.clip(temp, a_min = 0, a_max=None)
temp = np.dot(temp, w2) + b2
temp = np.clip(temp, a_min = 0,a_max=None)
temp = np.dot(temp, w3) + b3
temp = np.clip(temp, a_min = 0,a_max=None)
y = np.dot(temp,w4) + b4

print("pre softmax", y)
print("post softmax", softmax(y))
