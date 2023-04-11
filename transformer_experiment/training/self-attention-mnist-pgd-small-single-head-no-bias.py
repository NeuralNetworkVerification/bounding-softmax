from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Reshape, MultiHeadAttention
from keras.utils import np_utils
from keras.optimizers import Adam
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


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
        Perform adversarial training
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
                        X_adv,  # train data （adversarial example）
                        Y_train[start:end],  # actual output of train data
                        batch_size=mini_batch_size,  # mini batch size
                        epochs=1,  # the number of epoch
                        verbose=0,  # logging level
                        validation_data=(X_test, Y_test))  # Test data to evaluate per epoch

                else:
                    # train in the adversarial way
                    self.robust_model.fit(
                        X_adv,  # train data （adversarial example）
                        Y_train[start:end],  # actual output of train data
                        batch_size=mini_batch_size,  # mini batch size
                        epochs=1,  # the number of epoch
                        verbose=0)  # logging level


if __name__ == '__main__':
    nb_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    inp = Input(shape=(28, 28, 1))
    x = Flatten()(inp)
    x = Dense(16, activation='relu')(x)
    x = Reshape([2, 8])(x)
    attention = MultiHeadAttention(num_heads=1,
                                   key_dim=4,
                                   use_bias=False)
    # x, weights = attention(x, x, return_attention_scores=True)
    x = attention(x, x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    from art.attacks.evasion import ProjectedGradientDescent
    from art.estimators.classification import KerasClassifier
    art_model = KerasClassifier(model=model, clip_values=(0, 1))
    mini_batch_size = 50
    pgd_attack = ProjectedGradientDescent(art_model,
                                          eps=0.1,
                                          eps_step=0.005,
                                          max_iter=40,
                                          batch_size=mini_batch_size)
    adv_train = AdversarialTraining(robust_model=model,
                                    attack=pgd_attack)
    adv_train.train(mini_batch_size=mini_batch_size,
                    num_epochs=20,
                    X_train=x_train,
                    Y_train=y_train,
                    X_test=x_test,
                    Y_test=y_test)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save('self-attention-mnist-pgd-small-single-head-no-bias.h5')

    exit()

    # reload the keras and onnx models to check if equivalent
    model = load_model('self-attention-mnist-pgd-small-single-head-no-bias.h5')
    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print(model.predict(x_test[0:1]))

    # upgrade tf2onnx to convert multi-head-attention
    import tf2onnx
    model_proto, _ = tf2onnx.convert.from_keras(model, output_path='self-attention-mnist-pgd-small-single-head-no-bias.onnx')

    import onnx
    import onnxruntime
    # onnx_model = onnx.load_model('self-attention-mnist-pgd-small.onnx')
    session = onnxruntime.InferenceSession('self-attention-mnist-pgd-small-single-head-no-bias.onnx')
    output = session.run(None, {"input_1": x_test[0:1]})
    print(output[0])

