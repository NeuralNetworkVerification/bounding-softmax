from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Reshape, MultiHeadAttention
from keras.utils import np_utils

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
    x = Dense(256, activation='relu')(x)
    x = Reshape([4, 64])(x)
    attention = MultiHeadAttention(num_heads=4, key_dim=16)
    # x, weights = attention(x, x, return_attention_scores=True)
    x = attention(x, x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model.save('self-attention-mnist-big.h5')

    # upgrade tf2onnx to convert multi-head-attention
    import tf2onnx
    model_proto, _ = tf2onnx.convert.from_keras(model, output_path='self-attention-mnist-big.onnx')

    # exit()

    # reload the keras and onnx models to check if equivalent
    model = load_model('self-attention-mnist-big.h5')
    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print(model.predict(x_test[0:1]))

    import onnx
    import onnxruntime
    # onnx_model = onnx.load_model('self-attention-mnist-big.onnx')
    session = onnxruntime.InferenceSession('self-attention-mnist-big.onnx')
    output = session.run(None, {"input_1": x_test[0:1]})
    print(output[0])
