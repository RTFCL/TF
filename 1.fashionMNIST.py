# https://www.tensorflow.org/tutorials/keras/basic_classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Group labels
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Show one image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale values to range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show 5x5 images with class
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# Configure layers
model = keras.Sequential([
    # transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile network
model.compile(
    # This is how the model is updated based on the data it sees and its loss function.
    optimizer=tf.train.AdamOptimizer(),
    # This measures how accurate the model is during training. We want to minimize this function to
    # "steer" the model in the right direction.
    loss='sparse_categorical_crossentropy',
    # Used to monitor the training and testing steps. The following example uses accuracy,
    # the fraction of the images that are correctly classified.
    metrics=['accuracy']
)

# Train model
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print ('Test accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)

# A prediction is an array of 10 numbers. These describe the "confidence" of the
# model that the image corresponds to each of the 10 different articles of clothing.
print (predictions[0])

# Highest confidence
print (np.argmax(predictions[0]))