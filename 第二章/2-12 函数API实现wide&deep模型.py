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


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#函数式API 功能API
input = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30,activation='sigmoid')(input)
hidden2 = keras.layers.Dense(30,activation='sigmoid')(hidden1)

concat = keras.layers.concatenate([input,hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input],outputs=[output])

model.summary()
model.compile(loss = "mean_squared_error",optimizer='adam')
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]

history = model.fit(x_train_scaled,y_train,validation_data=(x_valid_scaled,y_valid),epochs=10,callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)

model.evaluate(x_test_scaled,y_test)