import numpy as np
import tensorflow_datasets as tfds
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten, Embedding, Reshape, MultiHeadAttention


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

# import pickle
# with open("validation_sequences.pickle", "wb") as fp:
#     pickle.dump(validation_padded, fp)
# with open("validation_labels.pickle", "wb") as fp:
#     pickle.dump(validation_labels_final, fp)
# exit()

inp = Input(shape=(50,))
x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inp)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
x = Reshape([2, 8])(x)
attention = MultiHeadAttention(num_heads=2, key_dim=4)
# x, weights = attention(x, x, return_attention_scores=True)
x = attention(x, x)
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

# upgrade tf2onnx to convert multi-head-attention
import tf2onnx
model_proto, _ = tf2onnx.convert.from_keras(model, output_path='self-attention-sst.onnx')

# exit()


# reload the keras and onnx models to check if equivalent
model = load_model('self-attention-sst.h5')
model.summary()
score = model.evaluate(validation_padded, validation_labels_final, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print(model.predict(validation_padded[0:1]))

import onnx
import onnxruntime
# onnx_model = onnx.load_model('self-attention-sst.onnx')
session = onnxruntime.InferenceSession('self-attention-sst.onnx')
output = session.run(None, {"input_1": validation_padded[0:1].astype('float32')})
print(output[0])




