# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model. (okkkk)
#
# Desired accuracy AND validation_accuracy > 83%(untuk callback)
# =============================================================================

import tensorflow as tf
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_acc') is not None and logs.get('val_acc') > 0.9) and (
                logs.get('acc') is not None and logs.get('acc') > 0.9):
            self.model.stop_training = True
def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    # load data
    (training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

    training_images = training_images / 255.0
    test_images = test_images / 255.0
    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(3,3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # COMPILE MODEL HERE
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])
    # TRAIN YOUR MODEL HERE


    model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=100, callbacks=[myCallback()])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
