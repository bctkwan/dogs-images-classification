# CSCI3230 Neural Network Project
# Kwan Chun Tat
# 1155033423

import numpy as np
from PIL import Image
import tensorflow as tf
import sys

# read python dictionary and images
if sys.version_info[0] < 3:
	Dictionary = np.load("../testing.npy").item()
else:
	Dictionary = np.load("../testing.npy", encoding="latin1").item()
images = np.array([image.astype(np.uint8) for image in Dictionary["reshaped"]])
images = images.reshape(-1,100, 100, 3)
#images = images.astype(np.float32)

# self defined function
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, k):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, 2, 2, 1], padding='SAME')

# create graph 
x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
x_standardized = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x)

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_standardized, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, 3)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, 3)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool(h_conv3, 3)

W_fc1 = weight_variable([13 * 13 * 128, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 13*13*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 25])
b_fc2 = bias_variable([25])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

saver = tf.train.Saver(var_list=tf.trainable_variables())

prediction = tf.argmax(y_conv, 1)

with tf.Session() as sess:
	saver.restore(sess, "./model.ckpt")
	pred = sess.run(prediction, feed_dict={x: images, keep_prob: 1.0})
	file = open("labels.txt", "w")
	for i in pred:
		file.write("%d\n" % i)