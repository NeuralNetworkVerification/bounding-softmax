import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed
from tensorflow.keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Layer
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.optimizers import Adam
tf.compat.v1.disable_eager_execution() #  Disable Eager Execution

# ART
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

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

def MultiHeadsAttModel(l=8 * 8, d=512, dv=64, dout=512, nv=8):
    inp = Input(shape=(28, 28, 1))
    # x = Conv2D(32, (2, 2), activation='relu', padding='same')(inp)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(64, (2, 2), activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    # x = Conv2D(64 * 3, (2, 2), activation='relu')(x)
    x = Flatten()(inp)

    v1 = Input(shape=(l, d))
    q1 = Input(shape=(l, d))
    k1 = Input(shape=(l, d))

    v2 = Dense(dv * nv, activation="relu")(v1)
    q2 = Dense(dv * nv, activation="relu")(q1)
    k2 = Dense(dv * nv, activation="relu")(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)

    # att = tf.einsum('baik,baij->bakj', q, k) / np.sqrt(dv)
    # att = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att)
    # out = tf.einsum('bajk,baik->baji', att, v)

    att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1, -1]) / np.sqrt(dv),
                 output_shape=(l, nv, nv))([q, k])  # l, nv, nv
    att = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att)
    att = Reshape([l, nv, dv])(att)
    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3, 3]),
                 output_shape=(l, nv, dv))([att, v])

    # out = Reshape([l, d])(out)
    # # out = Add()([out, q1])
    # out = Dense(dout, activation="relu")(out)

    model = Model(inputs=[q1, k1, v1], outputs=out)
    model.summary()

    return model

    x = Dense(16, activation='relu')(x)

    if True:
        # x = Reshape([6 * 6, 64 * 3])(x)
        # att = MultiHeadsAttModel(l=6 * 6, d=64 * 3, dv=8 * 3, dout=32, nv=8)
        # x = Reshape([2 * 2, 49 * 4])(inp)
        # att = MultiHeadsAttModel(l=2 * 2, d=49 * 4, dv=7 * 4, dout=32, nv=7)
        x = Reshape([2, 8])(x)
        att = MultiHeadsAttModel(l=2, d=8, dv=4, dout=16, nv=2)
        x = att([x, x, x])
        # x = Reshape([6, 6, 32])(x)
        # x = Reshape([2, 2, 32])(x)
        # x = NormL()(x)
        # x = BatchNormalization()(x)

    x = Flatten()(x)
    # x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


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

robust_model.save('att_small_pdg.h5')

#robust_model.layers[-1].activation = keras.activations.linear
#robust_model.save("robust_mnist_8x100_linear.h5")
robust_model = tf.keras.models.load_model('att_small_pdg.h5')

import keras2onnx
onnx_model = keras2onnx.convert_keras(robust_model, 'att_small_pdg')
keras2onnx.save_model(onnx_model, 'att_small_pdg.onnx')




# class NormL(Layer):
#
#     def __init__(self, **kwargs):
#         super(NormL, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.a = self.add_weight(name='kernel',
#                                  shape=(1, input_shape[-1]),
#                                  initializer='ones',
#                                  trainable=True)
#         self.b = self.add_weight(name='kernel',
#                                  shape=(1, input_shape[-1]),
#                                  initializer='zeros',
#                                  trainable=True)
#         super(NormL, self).build(input_shape)  # Be sure to call this somewhere!
#
#     def call(self, x):
#         eps = 0.000001
#         mu = K.mean(x, keepdims=True, axis=-1)
#         sigma = K.std(x, keepdims=True, axis=-1)
#         ln_out = (x - mu) / (sigma + eps)
#         return ln_out * self.a + self.b
#
#     def compute_output_shape(self, input_shape):
#         return input_shape


if __name__ == '__main__':

    nb_classes = 10

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)

    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # model = load_model('models/self-attention.h5')
    # model.summary()
    # exit()


    # tbCallBack = TensorBoard(log_dir='./Graph/mhatt1', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(X_train, Y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(X_test, Y_test)
              # callbacks=[tbCallBack]
              )
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    #model.save('self-attention.h5')

    import keras2onnx
    onnx_model = keras2onnx.convert_keras(model, model.name)
    temp_model_file = 'self-attention-small.onnx'
    keras2onnx.save_model(onnx_model, temp_model_file)
