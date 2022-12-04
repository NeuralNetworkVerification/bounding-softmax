# -*- coding: utf-8 -*-
"""
Trains a convolutional neural network on the CIFAR-10 dataset, then generated adversarial images using the
DeepFool attack and retrains the network on the training set augmented with the adversarial images.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
tf.compat.v1.disable_eager_execution() #  Disable Eager Execution

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np

from art.attacks.evasion import DeepFool
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset

import sys

netId = sys.argv[1]

# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
x_train, y_train = x_train, y_train
x_test, y_test = x_test, y_test
im_shape = x_train[0].shape

# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32,32,3]), # Layer to flat an input
    keras.layers.Dense(100), # Hidden 1
    keras.layers.ReLU(),
    keras.layers.Dense(100), # Hidden 1
    keras.layers.ReLU(),
    keras.layers.Dense(100), # Hidden 1
    keras.layers.ReLU(),
    keras.layers.Dense(10, activation='softmax')]) # Output layer

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Create classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=20, batch_size=128)

# Craft adversarial samples with DeepFool
logger.info("Create DeepFool attack")
adv_crafter = DeepFool(classifier)
logger.info("Craft attack on training examples")
x_train_adv = adv_crafter.generate(x_train)
logger.info("Craft attack test examples")
x_test_adv = adv_crafter.generate(x_test)

# Evaluate the classifier on the adversarial samples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info("Classifier before adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))

# Data augmentation: expand the training set with the adversarial samples
x_train = np.append(x_train, x_train_adv, axis=0)
y_train = np.append(y_train, y_train, axis=0)

# Retrain the CNN on the extended dataset
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
classifier.fit(x_train, y_train, nb_epochs=30, batch_size=128)

# Evaluate the adversarially trained classifier on the test set
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info("Classifier with adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))

model.save(f'robust_cifar10-large-{netId}.h5')

model = tf.keras.models.load_model(f'robust_cifar10-large-{netId}.h5')

import keras2onnx
onnx_model = keras2onnx.convert_keras(model, f'robust_cifar10-large-{netId}')
keras2onnx.save_model(onnx_model, f'robust_cifar10-large-{netId}.onnx')
