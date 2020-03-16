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

def f(x):
    return 3.*x**2+2.*x-1

def approximae_derivative(f,x,eps=1e-3):
    return (f(x+eps)-f(x-eps))/(2.*eps)

print(approximae_derivative(f,1.))

def g(x1,x2):
    return  (x1+5)*(x2**2)

def approximae_gradient(g,x1,x2,eps=1e-3):
    dg_x1 = approximae_derivative(lambda x:g(x,x2),x1,eps)
    dg_x2 = approximae_derivative(lambda x:g(x1,x),x2,eps)
    return dg_x1,dg_x2

print(approximae_gradient(g,2.,3.))
