import os
import numpy as np
from absl import app
import edward2 as ed
from absl import logging
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from keras.datasets import mnist
from keras.utils import np_utils
import utils


# revise from ''uncertainty-baselines'' on GitHub:
# https://github.com/google/uncertainty-baselines/blob/main/baselines/mnist/deterministic.py


def lenet5(input_shape, num_classes):
    """Builds LeNet5."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    # conv1 = tf.keras.layers.Conv2D(6,
    #                                kernel_size=5,
    #                                padding='SAME',
    #                                activation='relu')(inputs)
    # pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
    #                                      strides=[2, 2],
    #                                      padding='SAME')(conv1)
    # conv2 = tf.keras.layers.Conv2D(16,
    #                                kernel_size=5,
    #                                padding='SAME',
    #                                activation='relu')(pool1)
    # pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
    #                                      strides=[2, 2],
    #                                      padding='SAME')(conv2)
    # conv3 = tf.keras.layers.Conv2D(120,
    #                                kernel_size=5,
    #                                padding='SAME',
    #                                activation=tf.nn.relu)(pool2)
    # flatten = tf.keras.layers.Flatten()(conv3)
    # dense1 = tf.keras.layers.Dense(84, activation=tf.nn.relu)(flatten)
    # logits = tf.keras.layers.Dense(num_classes)(dense1)
    # outputs = tf.keras.layers.Lambda(lambda x: ed.Categorical(logits=x))(logits)

    flatten = tf.keras.layers.Flatten()(inputs)

    # dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)(flatten)
    # dense2 = tf.keras.layers.Dense(10, activation=tf.nn.relu)(dense1)
    # logits = tf.keras.layers.Dense(num_classes)(dense2)

    dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)(flatten)
    dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu)(dense1)
    dense3 = tf.keras.layers.Dense(100, activation=tf.nn.relu)(dense2)
    logits = tf.keras.layers.Dense(num_classes)(dense3)

    outputs = tf.keras.layers.Lambda(lambda x: ed.Categorical(logits=x))(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


np.random.seed(0)
tf.random.set_seed(0)
tf.io.gfile.makedirs('./tmp/det_training')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("X_train original shape", x_train.shape)
print("y_train original shape", y_train.shape)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("Training matrix shape", x_train.shape)
print("Testing matrix shape", x_test.shape)
# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)
num_classes = 10

# tf.executing_eagerly()
# Note that we need to disable v2 behavior after we load the data.
tf1.enable_eager_execution()
tf1.disable_v2_behavior()

ensemble_filenames = []
for i in range(5):
    # TODO(trandustin): We re-build the graph for each ensemble member. This
    # is due to an unknown bug where the variables are otherwise not
    # re-initialized to be random. While this is inefficient in graph mode, I'm
    # keeping this for now as we'd like to move to eager mode anyways.
    model = lenet5(x_train.shape[1:], num_classes)


    def negative_log_likelihood(y, rv_y):
        del rv_y  # unused arg
        return -model.output.distribution.log_prob(tf.squeeze(y))  # pylint: disable=cell-var-from-loop


    def accuracy(y_true, y_sample):
        del y_sample  # unused arg
        return tf.equal(
            tf.argmax(input=model.output.distribution.logits, axis=1),  # pylint: disable=cell-var-from-loop
            tf.cast(tf.squeeze(y_true), tf.int64))


    def log_likelihood(y_true, y_sample):
        del y_sample  # unused arg
        return model.output.distribution.log_prob(tf.squeeze(y_true))  # pylint: disable=cell-var-from-loop


    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss=negative_log_likelihood,
        metrics=[log_likelihood, accuracy])
    member_dir = os.path.join('./tmp/det_training', 'member_' + str(i))
    tensorboard = tf1.keras.callbacks.TensorBoard(
        log_dir=member_dir,
        update_freq=256 * 5)

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=256,
        # epochs=(256 * 5000) // 50000,
        epochs=20,
        validation_data=(x_test, y_test),
        validation_freq=max(
            (5 * 256) // 50000, 1),
        verbose=1,
        callbacks=[tensorboard]
    )

    # member_filename = os.path.join(member_dir, 'mnist-' + str(i) + '.weights')
    member_filename = os.path.join(member_dir, 'mnist-large-' + str(i) + '.weights')
    ensemble_filenames.append(member_filename)
    model.save_weights(member_filename)

labels = tf.keras.layers.Input(shape=y_train.shape[1:])
ll = tf.keras.backend.function([model.input, labels], [
    model.output.distribution.log_prob(tf.squeeze(labels)),
    model.output.distribution.logits,
])

ensemble_metrics_vals = {
    'train': utils.ensemble_metrics(
        x_train, y_train, model, ll, weight_files=ensemble_filenames),
    'test': utils.ensemble_metrics(
        x_test, y_test, model, ll, weight_files=ensemble_filenames),
}

for split, metrics in ensemble_metrics_vals.items():
    logging.info(split)
    for metric_name in metrics:
        logging.info('%s: %s', metric_name, metrics[metric_name])

print('mission complete.')

