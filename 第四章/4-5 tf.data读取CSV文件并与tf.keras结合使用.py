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
print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
print(x_test.shape,y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)
output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def save_to_csv(out_dir,data,name_prefix,header=None,n_parts=10):
    path_format = os.path.join(output_dir,"{}_{:02d}.csv")
    filenames = []

    for file_idx, row_indices in enumerate(
        np.array_split(np.arange(len(data)),n_parts)):
        part_csv = path_format.format(name_prefix,file_idx)
        filenames.append(part_csv)
        with open(part_csv,'wt',encoding="utf-8") as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                f.write(",".join([repr(col) for  col in data[row_index]]))
                f.write('\n')
    return filenames

train_data = np.c_[x_train_scaled,y_train]
valid_data = np.c_[x_valid_scaled,y_valid]
test_data = np.c_[x_test_scaled,y_test]
header_cols = housing.feature_names+["MidianHouseValue"]
header_str = ",".join(header_cols)
train_filename = save_to_csv(output_dir,train_data,"train",header_str,n_parts=20)
valid_filename = save_to_csv(output_dir,valid_data,"valid",header_str,n_parts=10)
test_filename = save_to_csv(output_dir,test_data,"test",header_str,n_parts=10)

import pprint
print("train filenames:")
pprint.pprint(train_filename)
print("valid filenames:")
pprint.pprint(valid_filename)
print("test datanames:")
pprint.pprint(test_filename)


#1、filename->dataset
#2、read file->dataset->datasets->merge
filename_dataset = tf.data.Dataset.list_files(train_filename)
for filename in filename_dataset:
    print(filename)


n_readers = 5
dataset = filename_dataset.interleave(
    lambda filename:tf.data.TextLineDataset(filename),
    cycle_length=n_readers
)
for line in dataset.take(15):
    print(line.numpy())

#tf.io.decode_csv(str,record_defaults)

sample_str = '1,2,3,4,5'
record_defaults = [
    tf.constant(0,dtype=tf.int32),
    0,
    np.nan,
    "hello",
    tf.constant([])
]
parsed_fields = tf.io.decode_csv(sample_str,record_defaults)
print(parsed_fields)

try:
    parsed_fields = tf.io.decode_csv(',,,,',record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

try:
    parsed_fields = tf.io.decode_csv('1,2,3,4,5,6,7', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)


def parse_csv_line(line,n_fields):
    defs = [tf.constant(np.nan)]*n_fields
    parsed_fields = tf.io.decode_csv(line,record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x,y

def parse_csv_line(line,n_fields):
    defs = [tf.constant(np.nan)]*n_fields
    parsed_fields = tf.io.decode_csv(line,record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x,y

parse_csv_line(b'0.6363646332204844,-1.0895425985107923,0.09260902815633619,-0.20538124656801682,1.2025670451003232,-0.03630122549633783,-0.6784101660505877,0.182235342347858,2.429'
,n_fields=9)

#1、filename->dataset
#2、read file -> dataset->dataset->merge
#3、parse csv

def csv_reader_dataset(filenames,n_readers=5,batch_size=32,n_parse_threads=5,shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename:tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

train_set = csv_reader_dataset(train_filename,batch_size=3)
for x_batch,y_batch in train_set.take(2):
    print("x:")
    pprint.pprint(x_batch)
    print("y:")
    pprint.pprint(y_batch)


batch_size = 32
train_set = csv_reader_dataset(train_filename,batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filename,batch_size=batch_size)
test_set = csv_reader_dataset(test_filename,batch_size=batch_size)

model = tf.keras.models.Sequential([
    keras.layers.Dense(30,activation='relu',input_shape=[8]),
    keras.layers.Dense(1)
])
model.compile(loss = "mean_squared_error",optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]
history = model.fit(train_set,validation_data=valid_set,steps_per_epoch = 11160// batch_size,
                    validation_steps = 3870//batch_size, epochs=100,callbacks=callbacks)
model.evaluate(test_set,steps=5160//batch_size)
