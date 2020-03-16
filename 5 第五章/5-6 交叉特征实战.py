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

for numeric_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(
            numeric_column, dtype=tf.float32))
# cross feature: age [1,2,3,4,5],gender:[male,female]
# age_x_gender:[(1,male),(2,male),...，(5,male),...,(5,female)]

feature_columns.append(
    tf.feature_column.indicator_column(
        ['age','sex'],hash_backet_size=100))

def make_dataset(data_df, label_df, epochs=10,shuffle=True,
                 batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(data_df),label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


output_dir = 'baseline_nodel_new_features'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
baseline_estimator = tf.estimator.BaselineClassifier(
    model_dir=output_dir,n_classes=2)
baseline_estimator.train(input_fn=lambda : make_dataset(
    train_df,y_train,epochs=100))

baseline_estimator.evaluate(input_fn=lambda : make_dataset(
    eval_df,y_eval,epochs=1,shuffle=False,batch_size=20))

liner_output_dir = 'linear_model_new_features'
if not os.path.exists(liner_output_dir):
    os.mkdir(liner_output_dir)
line_estimator = tf.estimator.LinearClassifier(
    model_dir=liner_output_dir,n_classes=2,
    feature_columns=feature_columns)
line_estimator.train(input_fn=lambda : make_dataset(train_df,y_train,epochs=100))
line_estimator.evaluate(input_fn=lambda :make_dataset(
    eval_df, y_eval,epochs=1,shuffle=False))


dnn_output_dir = './dnn_model_new_features'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)
dnn_estimator = tf.estimator.DNNClassifier(
    model_dir=dnn_output_dir,n_classes=2,
    feature_columns=feature_columns,hidden_units=[218,128],
    activation_fn=tf.nn.relu, optimizer='Adam')
dnn_estimator.train(input_fn=lambda : make_dataset(train_df,y_train,epochs=100))

dnn_estimator.evaluate(input_fn=lambda : make_dataset(
    eval_df,y_eval,epochs=1, shuffle=False))