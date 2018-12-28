# https://www.tensorflow.org/tutorials/keras/basic_text_classification

import tensorflow as tf
from tensorflow import keras
import numpy as np

# load data
imdb = keras.datasets.imdb

# prepare word index
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# function to get text from integers
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# get train data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# pre-process data
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential([
    # This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index.
    # These vectors are learned as the model trains. The vectors add a dimension to the output array.
    # The resulting dimensions are: (batch, sequence, embedding).
    keras.layers.Embedding(vocab_size, 16),
    # Returns a fixed-length output vector for each example by averaging over the sequence dimension.
    # This allows the model to handle input of variable length, in the simplest way possible.
    keras.layers.GlobalAveragePooling1D(),
    # This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
    keras.layers.Dense(16, activation=tf.nn.relu),
    # The last layer is densely connected with a single output node. Using the sigmoid activation function,
    # this value is a float between 0 and 1, representing a probability, or confidence level.
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.summary()

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    # Since this is a binary classification problem and the model outputs a probability (a single-unit layer with
    # a sigmoid activation), we'll use the binary_crossentropy loss function.
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# divide train data to train and validation groups
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

# evaluate
results = model.evaluate(test_data, test_labels)

# This fairly naive approach achieves an accuracy of about 87%.
# With more advanced approaches, the model should get closer to 95%.
print(results)

# draw
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# draw
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
