import numpy as np

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
tf.compat.v1.disable_eager_execution() #  Disable Eager Execution

# ART
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

# Load MNIST
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Squeeze it to [0, 1]
X_train, X_test = X_train/255.0, X_test/255.0

robust_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=X_train.shape[1:3]), # Layer to flat an input
    keras.layers.Dense(10), # Hidden 1
    keras.layers.ReLU(),
    keras.layers.Dense(10), # Hidden 1
    keras.layers.ReLU(),
    keras.layers.Dense(10, activation='softmax')]) # Output layer

robust_model.compile(
    optimizer=Adam(learning_rate=0.001), # Use adam to optimize the model
    loss='sparse_categorical_crossentropy', # use cross entropy for loss function
    metrics=['accuracy']) # metrics for evaluation, using accuracy

class AdversarialTraining:
    """ Adversarial Training
    Attributes:
        robust_model: model
        attack : instance of ART to generate adversarial examples
    """

    def __init__(self, robust_model, attack):
        self.robust_model = robust_model
        self.attack = attack

    def train(self, mini_batch_size, num_epochs, X_train, Y_train, X_test, Y_test):
        """
        Performe adversarial training
        Args:
            mini_batch_size (int): mini batch size
            num_epochs　(int) : number of epoch
            X_train (ndarray): training data
            Y_train (ndarray): actual output of training data
            X_test (ndarray): test data
            Y_test (ndarray): actual output of test data
        """

        # Minibatch size for each epoch
        num_mini_batches = (X_train.shape[0] - 1) // mini_batch_size + 1
        
        # Loop for epoch
        for epoch in range(num_epochs):

            # Loop for mini batch
            for b in range(num_mini_batches):
                
                # get mini batch
                start = b * mini_batch_size
                end = start + mini_batch_size

                # generate adversarial examples from mini batch
                X_adv = self.attack.generate(X_train[start:end], batch_size=mini_batch_size)
                
                # Logging and evaluate the model with test data per epoch
                if b == num_mini_batches - 1:
                    # train in the adversarial way
                    self.robust_model.fit(
                        X_adv, # train data （adversarial exzamples）
                        Y_train[start:end], # actual output of train data
                        batch_size=mini_batch_size, # mini batch size
                        epochs=1, # the number of epoch
                        verbose=2, # logging level
                        validation_data=(X_test, Y_test)) # Test data to evaluate per epoch
                
                else:
                    # train in the adversarial way
                    self.robust_model.fit(
                        X_adv, # train data （adversarial exzamples）
                        Y_train[start:end], # actual output of train data
                        batch_size=mini_batch_size, # mini batch size
                        epochs=1, # the number of epoch
                        verbose=0) # logging level

# Generate a classfier for ART
art_classifier_robust = KerasClassifier(model=robust_model, clip_values=(0, 1))

# minibatch size
mini_batch_size = 50

# Generate instance of PGD Attack
attack_train = ProjectedGradientDescent(art_classifier_robust, eps=0.1, eps_step=0.005, max_iter=40, batch_size=mini_batch_size)

# Generate instance of AdversarialTraining
at = AdversarialTraining(robust_model, attack_train)

at.train(mini_batch_size=mini_batch_size, num_epochs=20, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

robust_model.save('robust_mnist2x10.h5')

#robust_model.layers[-1].activation = keras.activations.linear
#robust_model.save("robust_mnist_8x100_linear.h5")
robust_model = tf.keras.models.load_model('robust_mnist2x10.h5')

import keras2onnx
onnx_model = keras2onnx.convert_keras(robust_model, 'robust_mnist2x10')
keras2onnx.save_model(onnx_model, 'robust_mnist2x10.onnx')
