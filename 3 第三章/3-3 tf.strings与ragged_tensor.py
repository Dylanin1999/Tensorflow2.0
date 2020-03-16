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

#strings
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t,unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t,"UTF8"))

#string array
t = tf.constant(["cafe","coffee","咖啡"])
print(tf.strings.length(t,unit="UTF8_CHAR"))
r= tf.strings.unicode_decode(t,"UTF8")
print(r)

#ragged tensor
r = tf.ragged.constant([[11,12],[21,22,23],[],[41]])
#index ops
print(r)
print(r[1])
print(r[1:2])

r2 = tf.ragged.constant([[51,52],[],[71]])
print(tf.concat([r,r2],axis=0))

r3 = tf.ragged.constant([[13,14],[15],[],[42,43]])
print(tf.concat([r,r3],axis=1))
