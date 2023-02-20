from keras.datasets import mnist
from keras.utils import np_utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28).astype('float32') / 255
x_test = x_test.reshape(10000, 28 * 28).astype('float32') / 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers import Dense
x = Sequential()
# x.add(Dense(10, activation='relu', input_shape=(784,)))
# x.add(Dense(10, activation='relu'))
x.add(Dense(100, activation='relu', input_shape=(784,)))
x.add(Dense(100, activation='relu'))
x.add(Dense(100, activation='relu'))
x.add(Dense(10, activation='softmax'))
x.summary()

# import keras
# x.compile(loss='categorical_crossentropy',
#               optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])
# x.evaluate(x_test, Y_test)

import tf2onnx
folder = 'ensembles/'
for i in range(5):
    # weights_path = folder + 'mnist-' + str(i) + '.weights'
    weights_path = folder + 'mnist-large-' + str(i) + '.weights'
    x.load_weights(weights_path)
    # onnx_path = folder + 'mnist-' + str(i) + '.onnx'
    onnx_path = folder + 'mnist-large-' + str(i) + '.onnx'
    model_proto, _ = tf2onnx.convert.from_keras(x, output_path=onnx_path)
