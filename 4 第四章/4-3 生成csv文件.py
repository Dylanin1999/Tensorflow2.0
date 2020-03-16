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
