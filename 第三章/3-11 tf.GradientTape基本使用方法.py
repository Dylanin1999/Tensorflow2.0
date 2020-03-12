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

def g(x1,x2):
    return  (x1+5)*(x2**2)

def f(x):
    return 3.*x**2+2.*x-1

x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1,x2)

dz_x1 = tape.gradient(z,x1)
print(dz_x1)
try:
    dz_x2 = tape.gradient(z,x2)
except RuntimeError as ex:
    print(ex)


with tf.GradientTape(persistent = True) as tape:
    z = g(x1,x2)

dz_1 = tape.gradient(z,x1)
dz_2 = tape.gradient(z,x2)
print(dz_1,dz_2)
del tape

with tf.GradientTape() as tape:
    z = g(x1,x2)

dz_x1x2 = tape.gradient(z,[x1,x2])
print(dz_x1x2)


x1_constant = tf.constant(2.0)
x2_constant = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x1)
    tape.watch(x2)
    z = g(x1,x2)

dz_x1x2 = tape.gradient(z,[x1,x2])
print(dz_x1x2)

x = tf.Variable(5.0)
with tf.GradientTape() as  tape:
    z1 = 3*x
    z2 = x**2
tape.gradient([z1,z2],x)


#=====
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z=g(x1,x2)
    inner_grads = inner_tape.gradient(z,[x1,x2])
outer_grads = [outer_tape.gradient(inner_grads,[x1,x2])
               for inner_grad in inner_grads]

print(outer_grads)
del inner_tape
del outer_tape


learning_rate = 0.1
x = tf.Variable(0.0)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z,x)
    x.assign_sub(learning_rate*dz_dx)
print(x)


optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z,x)
    optimizer.apply_gradients([(dz_dx,x)])
print(x)





