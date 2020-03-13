import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

source_dir = "./generate_csv"
print(os.listdir(source_dir))

def get_filename_by_prefix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir,filename))
    return  results

train_filenames = get_filename_by_prefix(source_dir,"train")
valid_filenames = get_filename_by_prefix(source_dir,"valid")
test_filenames = get_filename_by_prefix(source_dir,"test")

import pprint
pprint.pprint(train_filenames)
pprint.pprint(valid_filenames)
pprint.pprint(test_filenames)

def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)]*n_fields
    parsed_fields = tf.io.decode_csv(line,record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x,y

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

batch_size = 32
train_set = csv_reader_dataset(train_filenames,batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames,batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames,batch_size=batch_size)


def serialize_example(x,y):
    #converts x, y to tf.train.Example and Serialize
    input_features = tf.train.FloatList(value=y)
    label = tf.train.FloatList(value=y)
    feature = tf.train.Features(
        features={
            "input_features": tf.train.Feature(
                float_list = input_features),
            "label": tf.train.Feature(float_list=label)
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializerToString()

def csv_dataset_to_tfrecords(base_filename, dataset,
                             n_shards,steps_per_shard,
                             compression_type=None):
    options = tf.io.TFRecordOptions(
        compression_type=compression_type)
    all_filename=[]
    for shard_id in range(n_shards):
        filename_fullpath='{}_{:05d}-of-{:05d}',format(
            base_filename,shard_id,n_shards)
        with tf.io.TFRecordWriter(filename_fullpath,options) as writer:
            for x_batch,y_batch in dataset.take(steps_per_shard):
                for x_example, y_example in zip(x_batch,y_batch):
                    writer.write(
                        serialize_example(x_example,y_example))
        all_filename.append(filename_fullpath)
        return all_filename


n_shards = 20
train_step_per_shard = 11610//batch_size//n_shards
valid_step_per_shard = 3800//batch_size//n_shards
test_step_per_shard = 5170//batch_size//n_shards

output_dir = "generate_tfrecords"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_basename = os.path.join(output_dir,"train")
valid_basename = os.path.join(output_dir,"valid")
test_basename = os.path.join(output_dir,"test")

train_tfrecord_filenames = csv_dataset_to_tfrecords(
    train_filenames,train_set,n_shards,train_step_per_shard,None)
valid_tfrecord_filenames = csv_dataset_to_tfrecords(
    valid_basename,valid_set,n_shards,valid_step_per_shard,None)
test_tfrecord_filenames = csv_dataset_to_tfrecords(
    test_filenames,test_set,n_shards,test_step_per_shard,None)

n_shards = 20
train_step_per_shard = 11610//batch_size//n_shards
valid_step_per_shard = 3800//batch_size//n_shards
test_step_per_shard = 5170//batch_size//n_shards

output_dir = "generate_tfrecords_zip"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_basename = os.path.join(output_dir,"train")
valid_basename = os.path.join(output_dir,"valid")
test_basename = os.path.join(output_dir,"test")

train_tfrecord_filenames = csv_dataset_to_tfrecords(
    train_filenames,train_set,n_shards,train_step_per_shard,compression_type="GZIP")
valid_tfrecord_filenames = csv_dataset_to_tfrecords(
    valid_basename,valid_set,n_shards,valid_step_per_shard,compression_type="GZIP")
test_tfrecord_filenames = csv_dataset_to_tfrecords(
    test_filenames,test_set,n_shards,test_step_per_shard,compression_type="GZIP")