import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf


#tf

favorite_books = [name.encode('utf-8') for name in ['machine learning', 'cc150']]
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
print(favorite_books_bytelist)
hours_floatlist = tf.train.FloatList(value=[15.5,9.5,70,80])
print(hours_floatlist)

age_int64list = tf.train.Int64List(value=[42])
print(age_int64list)

features = tf.train.Features(
    feature={
        "favorite_books": tf.train.Feature(bytes_list=favorite_books_bytelist),
        "hours": tf.train.Feature(float_list=hours_floatlist),
        "age": tf.train.Feature(int64_list=age_int64list)
    }
)
print(features)

example = tf.train.Example(features=features)
print(example)
serialized_example = example.SerializeToString()
print(serialized_example)

output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = "test.tfrecords"
filename_fullpath = os.path.join(output_dir,filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)

dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)


expected_features = {
    "favorite_books":tf.io.VarLenFeature(dtype=tf.string),
    "hours":tf.io.VarLenFeature(dtype=tf.float32),
    "age":tf.io.FixedLenFeature([],dtype=tf.int64),
}
dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features)
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))

filename_fullpath_zip = filename_fullpath+'.zip'
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(filename_fullpath_zip,options) as writer:
    for i in range(3):
        writer.write(serialized_example)

dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], compression_type="GZIP")
for serialized_example_tensor in dataset_zip:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features)
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))