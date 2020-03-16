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


categorial_columns = ['sex','n_siblings_spouses','parch','class',
                      'deck','embark_town','alone']
numeric_columns = ['age','fare']
feature_columns = []

for categorial_column in categorial_columns:
    vocab = train_df[categorial_column].unique()
    print(vocab)
    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                categorial_column,vocab)))

for categorial_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(
            categorial_column, dtype=tf.float32))

def make_dataset(data_df, label_df, epochs=10,shuffle=True,
                 batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(data_df),label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset

train_dataset = make_dataset(train_df, y_train, batch_size=5)
for x,y in train_dataset.take(1):
    print(x,y)


for x,y in train_dataset.take(1):
    age_column = feature_columns[7]
    gender_cloumn = feature_columns[0]
    print(keras.layers.DenseFeatures(age_column)(x).numpy())
    print(keras.layers.DenseFeatures(gender_cloumn)(x).numpy())


for x,y in train_dataset.take(1):
    print(keras.layers.DenseFeatures(feature_columns)(x).numpy())


