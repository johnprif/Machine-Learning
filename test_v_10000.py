import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels),(test_imgs, test_labels)=fashion_mnist.load_data()

img_x, img_y = 28, 28
train_imgs = train_imgs.reshape(train_imgs.shape[0], img_x, img_y, 1)
test_imgs = test_imgs.reshape(test_imgs.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

print('The train image dataset has shape:', train_imgs.shape)
print('The test image dataset has shape:',test_imgs.shape)

train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

training_size = 3500
test_size = 1750

x_train_filter, y_train_filter = np.empty(shape=(training_size, 28, 28, 1)), []

for label in list(set(train_labels)):
    sample_filter = np.where((train_labels == label))
    x_train_filter = np.append(x_train_filter, np.array(train_imgs[sample_filter][:training_size]), axis=0)
    y_train_filter += [label] * training_size

x_train_filter = x_train_filter[training_size:, :, :]

import matplotlib.pyplot as plt


plt.figure(figsize=(20,20))
for i in range(0,35000,100):
    plt.subplot(10,35,i/100+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train_filter[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(y_train_filter[i])

train_imgs = train_imgs.reshape(training_size*10, 784) #28*28
test_imgs = test_imgs.reshape(test_size*10, 784)
train_labels = np.eye(len(set(train_labels)))[train_labels]

print('The flattened train image dataset has shape:', train_imgs.shape)
print('The flattened test image dataset has shape:',test_imgs.shape)

#create tensorflow session
feature_number = len(train_imgs[0])
label_number = len(train_labels[0])

#knn
k_num = [1]

x_data_train = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
y_data_train = tf.placeholder(shape=[None, label_number], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

#components for distance
subtract = tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))
square = tf.square(subtract)

#euclidean distance
distance = tf.sqrt(tf.reduce_sum(square, axis=2))

config = tf.ConfigProto(device_count = {'GPU': 1})
sess = tf.Session(config=config)

prediction_times = []
total_acc_rate = []

for i in k_num:
    start_time = time.time()
    _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=i)
    top_k_label = tf.gather(y_data_train, top_k_indices)
    sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
    prediction = tf.argmax(sum_up_predictions, axis=1)
    test_size = 17500
    test_batch_size = 5
    prediction_outcome_final = []
    for step in range(test_size // test_batch_size):
        test_offset = step * test_batch_size
        test_batch_data = test_imgs[test_offset:(test_offset + test_batch_size), :]
        prediction_outcome = sess.run(prediction, feed_dict={x_data_train: train_imgs,
                               x_data_test: test_batch_data,
                               y_data_train: train_labels})
        for result in prediction_outcome:
            prediction_outcome_final.append(result)
    accuracy = 0
    for pred, actual in zip(prediction_outcome_final, test_labels):
        if pred == actual:
            accuracy += 1
    elapsed_time = (time.time() - start_time)
    prediction_times.append(elapsed_time)
    acc_rate = accuracy / len(prediction_outcome_final)
    total_acc_rate.append(acc_rate)
    print('When k =', str(i), ', the accuracy is', acc_rate, "| Time Taken:", elapsed_time, 's')