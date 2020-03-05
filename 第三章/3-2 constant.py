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

t = tf.constant([[1.,2.,3.],[4.,5.,6.]])
print(t)
print(t[:, 1:])
print(t[...,1])

#ops
print(t+10)
print(tf.square(t))
print(t@tf.transpose(t))

#numpy conversion
print(t.numpy)
print(np.square(t))
np_t = np.array([[1.,2.,3.],[4.,5.,6.]])
print(tf.constant(np_t))

#Scalar
t=tf.constant(2.718)
print(t.numpy)
print(t.shape)