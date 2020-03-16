from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os
import sys

def plot_learning_curves(history):

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

#数据归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#x_train: [None, 28, 28] -> [None,784]
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28,1)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)



#使用relu函数
# model = tf.keras.models.Sequential()
# model.add(keras.layers.Conv2D(filter=32,kernel_size=3,
#                               padding='same',activation='relu',
#                               input_shape=(28,28,1)))
#
# model.add(keras.layers.Conv2D(filters=32,kernel_size=3,
#                               padding='same',activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=2))
#
#
# model.add(keras.layers.Conv2D(filter=64, kernel_size=3,
#                               padding='same',activation='relu'))
#
# model.add(keras.layers.Conv2D(filters=64,kernel_size=3,
#                               padding='same',activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=2))
#
# model.add(keras.layers.Conv2D(filter=128,kernel_size=3,
#                               padding='same',activation='relu'))
#
# model.add(keras.layers.Conv2D(filters=128,kernel_size=3,
#                               padding='same',activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=2))
# model.add(keras.layers.Flatten())
#
# model.add(keras.layers.Dense(128,activation='relu'))
# model.add(keras.layers.Dense(10,activation='softmax'))

#使用selu函数
model = tf.keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32,kernel_size=3,
                              padding='same',activation='selu',
                              input_shape=(28,28,1)))

model.add(keras.layers.SeparableConv2D(filters=32,kernel_size=3,
                                        padding='same',activation='selu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))


model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3,
                              padding='same',activation='selu'))

model.add(keras.layers.SeparableConv2D(filters=64,kernel_size=3,
                              padding='same',activation='selu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))

model.add(keras.layers.SeparableConv2D(filters=128,kernel_size=3,
                              padding='same',activation='selu'))

model.add(keras.layers.SeparableConv2D(filters=128,kernel_size=3,
                              padding='same',activation='selu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128,activation='selu'))
#model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='', metrics=['accuracy'])

#使用了三个callback：Tensorboard, earlystopping, ModelCheckpoint
#logdir = './spearable-cnn-selu-callbacks'
logdir = os.path.join("dnn-callbacks")
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")
print("out:",output_model_file)
callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.ModelCheckpoint(output_model_file,save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
history = model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid),callbacks=callbacks)

plot_learning_curves(history)