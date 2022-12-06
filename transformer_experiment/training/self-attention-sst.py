import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten, Embedding, Reshape, Lambda
from keras import backend as K
import tf2onnx


def MultiHeadsAttModel(l=8 * 8, d=512, dv=64, dout=512, nv=8):
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
    # model.summary()

    return model


dataset, info = tfds.load('glue/sst2', with_info=True)
dataset_train, dataset_validation = dataset['train'], dataset['validation']
training_reviews = []
training_labels = []
validation_reviews = []
validation_labels = []
for item in dataset_train.take(-1):
    review, label = item["sentence"], item["label"]
    training_reviews.append(str(review.numpy()))
    training_labels.append(label.numpy())

for item in dataset_validation.take(-1):
    review, label = item["sentence"], item["label"]
    validation_reviews.append(str(review.numpy()))
    validation_labels.append(label.numpy())

vocab_size = 4000
embedding_dim = 16
max_length = 50
trunc_type = 'post'
pad_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_reviews)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_reviews)
training_padded = pad_sequences(training_sequences,
                                maxlen=max_length,
                                truncating=trunc_type,
                                padding=pad_type)
validation_sequences = tokenizer.texts_to_sequences(validation_reviews)
validation_padded = pad_sequences(validation_sequences,
                                  maxlen=max_length,
                                  truncating=trunc_type,
                                  padding=pad_type)
training_labels_final = np.array(training_labels)
validation_labels_final = np.array(validation_labels)

inp = Input(shape=(50,))
x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inp)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
x = Reshape([2, 8])(x)
att = MultiHeadsAttModel(l=2, d=8, dv=4, dout=16, nv=2)
x = att([x, x, x])
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inp, outputs=x)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

num_epochs = 10
model.fit(training_padded,
          training_labels_final,
          epochs=num_epochs,
          validation_data=(validation_padded, validation_labels_final))

model.save('self-attention-sst.h5')
model_proto, _ = tf2onnx.convert.from_keras(model, output_path='self-attention-sst.onnx')
