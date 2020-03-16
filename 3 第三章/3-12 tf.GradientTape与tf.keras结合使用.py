import tensorflow as tf
from tensorflow import keras

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras
#print(tf.__verison__)
print(sys.version_info)
#for module in mpl, np, pd, sklearn, tf, keras:
 #   print(module.__name__,module.__version__)

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data,housing.target,random_state=7)
x_train, x_valid, y_train,y_valid = train_test_split(x_train_all,y_train_all,random_state=11)
print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
print(x_test.shape,y_test.shape)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#metric 使用
metric = keras.metrics.MeanSquaredError()
print(metric([5.],[2.]))
print(metric([0.],[1.]))
print(metric.result())

metric.reset_states()
metric([1.],[3.])
print(metric.result())


#1、batch遍历训练集 metric
#  1.1、自动求导
#2、epoch结束验证集metric

epochs = 100
batch_size = 32
steps_pre_epoch = len(x_train_scaled)
optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquaredError()

def random_batch(x,y,batch_size=32):
    idx = np.random.randint(0,len(x),size=batch_size)
    return  x[idx],y[idx]

model = tf.keras.models.Sequential([
    keras.layers.Dense(30,activation='relu',input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)
])

for epoch in range(epochs):
    metric.reset_states()
    for step in range(steps_pre_epoch):
        x_batch,y_batch = random_batch(x_train_scaled,y_train,batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(y_batch, y_pred))
            metric(y_batch,y_pred)
        grads = tape.gradient(loss, model.variables)
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradient(grads_and_vars)
        print("\rEpoch",epoch,"train_mse:",metric.result().numpy(),end="")
    y_valid_pred = model(x_valid_scaled)
    vaild_loss = tf.reduce_mean(keras.losses.mean_squared_error(y_valid_pred,y_valid))
    print("\t","valid mse: ",vaild_loss.numpy())



model.compile(loss = "mean_squared_error",optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]

