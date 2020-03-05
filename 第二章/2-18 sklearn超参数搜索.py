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

# print(tf.__verison__)
print(sys.version_info)
# for module in mpl, np, pd, sklearn, tf, keras:
#   print(module.__name__,module.__version__)

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)


# RandomizeSearchCV
# 1、转化为SKLearn的model
# 2、定义参数集合
# 3、搜索参数

def build_model(hidden_layers=1, layer_size=30, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation='relu', input_shape=x_train.shape[1:]))

    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layer_size, activation='relu'))

    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)
history = sklearn_model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid),
                            callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

from scipy.stats import  reciprocal
plot_learning_curves(history)
param_distribution={
    "hidden_layers":[1,2,3,4],
    "layer_size":np.arange(1,100),
    "learning_rate":reciprocal(1e-4,1e-2)
}

from sklearn.model_selection import  RandomizedSearchCV
random_search_cv = RandomizedSearchCV(sklearn_model, param_distribution,n_iter=10,cv=3, n_jobs=1)
random_search_cv.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled,y_valid),callbacks=callbacks)

#cross_validation
