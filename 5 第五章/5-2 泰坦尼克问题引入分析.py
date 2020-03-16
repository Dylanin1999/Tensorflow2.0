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

train_file = "./data/Titanic/train.csv"
eval_file = "./data/Titanic/eval_csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)
print(train_df.head())
print(eval_df.head())

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

print(train_df.head())
print(eval_df.head())
print(y_train.head())
print(y_eval.head())

train_df.describe()
print(train_df.shape,eval_df.shape)
train_df.agg.hist(bins=20)
train_df.sex.value_counts().plot(kind='barh')
train_df['class'].value_counts().plot(kind='barh')

pd.concat([train_df,y_train],axis=1).groupby('sex').survived.mean().plot(kind='barh')



