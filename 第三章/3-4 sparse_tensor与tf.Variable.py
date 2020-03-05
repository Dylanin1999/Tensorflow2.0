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

s = tf.SparseTensor(indices = [[0,1],[1,0],[2,3]],values=[1,2,3,],dense_shape=[3,4])
print(s)
print(tf.sparse.to_dense(s))

s2 = s*2.0
print(s2)

try:
    s3 = s+1
except TypeError as ex:
    print(ex)

s4 = tf.constant([[10.,20.],
                  [30.,40.],
                  [50.,60.],
                  [70.,80.]])
print(tf.sparse.sparse_dense_matmul(s,s4))

s5 = tf.SparseTensor(indices = [[0,1],[1,0],[2,3]],values=[1,2,3,],dense_shape=[3,4])
print(s5)
s6 = tf.sparse.reorder(s5)
print(tf.sparse.to_dense(s6))

v = tf.Variable([[1,2,3],[4,5,6]])
print(v)
print(v.value())
print(v.numpy())

v.assign(2*v)
print(v.numpy())
v[0,1].assign(42)
print(v.numpy())
v[1].assign([7.,8,9])
print(v.numpy())

try:
    v[1] = [7,8,9]
except TypeError as ex:
    print(ex)
