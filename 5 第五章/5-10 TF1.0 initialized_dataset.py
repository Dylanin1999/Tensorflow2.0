from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os
import sys

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)

y_train = np.asarray(y_train,dtype=np.int64)
y_valid = np.asarray(y_valid,dtype=np.int64)
y_test = np.asarray(y_test,dtype=np.int64)



print(np.max(x_train_scaled),np.min(x_train_scaled))

def make_dataset(images, labels,epochs,batch_size,shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images,labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset

batch_size = 20
epochs = 10
dataset = make_dataset(x_train_scaled,y_train,epochs = epochs,batch_size=batch_size)
dataset_iter = dataset.make_one_host_iterator()

#1、auto initialization
#2、can not be re-initialized. make_initializable_iterator
x,y = dataset_iter.get_next()
with tf.Session() as sess:
    x_val, y_val = sess.run([x,y])
    print(x_val.shape)
    print(y_val.shape)
for data, label in dataset.take(1):
    print(data)
    print(label)


hidden_units = [100,100]
class_num = 10
x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.int64,[None])

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,hidden_unit,activation=tf.nn.relu)

logits = tf.layers.dense(input_for_next_layer,class_num)


loss = tf.losses.sparse_categorical_crossentropy(labels=y,logits=logits)

prediction=tf.argmax(logits,1)
correct_prediction = tf.equal(prediction,y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
print(x)
print(logits)

#session

init = tf.global_variables_initializer()

train_steps_per_epoch = x_train.shape[0]//batch_size


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):

            loss_val,accuracy_val,_ = sess.run([loss, accuracy,train_op])
            print('\r[Train] epoch: %d,step:%d,loss:%3.5f,accuracy:%2.2f'%(
            epoch, step,loss_val,accuracy_val),end='')
