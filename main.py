import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, AveragePooling2D, Dropout, Dense, Flatten, BatchNormalization
from keras.callbacks import LearningRateScheduler


EPOCH = 50
BATCH_SIZE = 256


def fix_random(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=16,
        inter_op_parallelism_threads=16
    )
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    #  0:44:164(3e-4)


fix_random(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
input_shape = (x_train.shape[1], x_train.shape[2], 1)


x_train, x_test = x_train/255., x_test/255.


def lr_decay(epoch):
    return max(0.0001, 0.0030 - epoch//1.5 * 0.0001)


class Inception(keras.layers.Layer):
    """
    From: https://qiita.com/Suguru_Toyohara/items/3b694b41f4adb843cd23
    """
    def __init__(self, output_filter=64, **kwargs):

        self.c1_conv1 = keras.layers.Conv2D(output_filter//4, 1, padding="same", name="c1_conv1")
        self.c1_conv2 = keras.layers.Conv2D(output_filter//4, 3, padding="same", name="c1_conv2")
        self.c1_conv3 = keras.layers.Conv2D(output_filter//4, 3, padding="same", name="c1_conv3")

        self.c2_conv1 = keras.layers.Conv2D(output_filter//4, 1, padding="same", name="c2_conv1")
        self.c2_conv2 = keras.layers.Conv2D(output_filter//4, 3, padding="same", name="c2_conv2")

        self.c3_MaxPool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="same",name="c3_MaxPool")
        self.c3_conv = keras.layers.Conv2D(output_filter//4, 1, padding="same", name="c3_conv")

        self.c4_conv = keras.layers.Conv2D(output_filter//4, 1, padding="same", name="c4_conv")

        self.concat = keras.layers.Concatenate()

        super(Inception, self).__init__(**kwargs)

    def call(self, input_x, training=False):

        x1 = self.c1_conv1(input_x)
        x1 = self.c1_conv2(x1)
        cell1 = self.c1_conv3(x1)

        x2 = self.c2_conv1(input_x)
        cell2 = self.c2_conv2(x2)

        x2 = self.c3_MaxPool(input_x)
        cell3 = self.c3_conv(x2)

        cell4 = self.c4_conv(input_x)

        return self.concat([cell1, cell2, cell3, cell4])


model = keras.models.Sequential([
    Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=input_shape),
    Dropout(0.1),
    Inception(8),
    AveragePooling2D((2, 2), strides=(2, 2)),
    BatchNormalization(),
    Dropout(0.1),
    Conv2D(10, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'),
    AveragePooling2D((2, 2), strides=(2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(20, activation='relu'),
    Dense(10, activation='softmax')
], name='mynet')

model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH,
          validation_data=(x_test, y_test), verbose=1,
          callbacks=[LearningRateScheduler(lr_decay)]
)

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
