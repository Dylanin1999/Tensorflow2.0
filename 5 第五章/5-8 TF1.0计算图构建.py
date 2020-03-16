from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os
import sys

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)

print(np.max(x_train_scaled),np.min(x_train_scaled))

hidden_units = [100,100]
class_num = 10
x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.int64,[None])

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,hidden_unit,activation=tf.nn.relu)

logits = tf.layers.dense(input_for_next_layer,class_num)


loss = tf.losses.sparse_categorical_crossentropy(labels=y,logits=logits)

prediction=tf.argmax(logits,1)
correct_prediction = tf.equal(prediction,y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
print(x)
print(logits)