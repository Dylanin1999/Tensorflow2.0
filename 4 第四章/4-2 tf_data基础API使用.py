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


dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)

for item in dataset:
    print(item)

#1、repeat epoch
#2、get batch
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)


dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),
    cycle_length=5,
    block_length=5,
)
for item in dataset2:
    print(item)


x = np.array([[1,2],[3,4],[5,6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x,y))
print(dataset3)
for item_x,item_y in dataset3:
    print(item_x,item_y)

x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)
for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpt())

dataset4 = tf.data.Dataset.from_tensor_slices({"feature":x,"label":y})
for item in dataset4:
    print(item)


