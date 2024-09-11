# CSCI3230 Neural Network Project
# Kwan Chun Tat
# 1155033423

import numpy as np
from PIL import Image
import tensorflow as tf
import sys

# parameters
batch_size = 100
stop_after_last_best_epochs = 50
train_ratio = 0.9

# read list and stratified divide training and validation set
file_label = np.genfromtxt("./train_label.txt", dtype=None, names=["filename","label"], delimiter=",")
train_indices = np.zeros(len(file_label), dtype=bool)
validation_indices = np.zeros(len(file_label), dtype=bool)
for value in np.unique(file_label["label"]):
	value_indices = np.nonzero(file_label["label"]==value)[0]
	np.random.shuffle(value_indices)
	n = int(train_ratio * len(value_indices))
	train_indices[value_indices[:n]] = True
	validation_indices[value_indices[n:]] = True
train_file_label = file_label[train_indices]
validation_file_label = file_label[validation_indices]
np.random.shuffle(train_file_label)
np.random.shuffle(validation_file_label)

# read images
if sys.version_info[0] < 3:
	train_images = np.array([np.array(Image.open("./train_reshaped/" + fname)) for fname in train_file_label["filename"]])
	validation_images = np.array([np.array(Image.open("./train_reshaped/" + fname)) for fname in validation_file_label["filename"]])
else:
	train_images = np.array([np.array(Image.open("./train_reshaped/" + fname.decode("utf-8"))) for fname in train_file_label["filename"]])
	validation_images = np.array([np.array(Image.open("./train_reshaped/" + fname.decode("utf-8"))) for fname in validation_file_label["filename"]])

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

def random_flip(x):
	return tf.map_fn(lambda image: tf.image.random_flip_left_right(image), x)

def random_hue(x):
	return tf.map_fn(lambda image: tf.image.random_hue(image, max_delta=0.5), x)

# create graph 
train = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
x_flip = tf.cond(tf.equal(train, tf.constant(True)), lambda: random_flip(x), lambda: x)
x_hue = tf.cond(tf.equal(train, tf.constant(True)), lambda: random_hue(x_flip), lambda: x_flip)
x_standardized = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x_hue)
y = tf.placeholder(tf.int32, shape=[None])
y_one_hot = tf.one_hot(indices=y, depth=25)

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
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_one_hot, logits = y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(var_list=tf.trainable_variables())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	i = 0
	epoch = 0
	best_epoch = 0
	best_validation_accuracy = 0
	start = 0
	end = min(batch_size, len(train_file_label["label"]))
	while True:
		train_accuracy = accuracy.eval(feed_dict={train:False, x:train_images[start:end], y:train_file_label["label"][start:end], keep_prob:1.0})
		print('Epoch %d, batch %d, training accuracy %g' %(epoch, i, train_accuracy))
		train_step.run(feed_dict={train:True, x:train_images[start:end], y:train_file_label["label"][start:end], keep_prob:0.5})
		i = i + 1
		start = start + batch_size
		end = min(end + batch_size, len(train_file_label["label"]))
		if start >= len(train_file_label["label"]):
			validation_accuracy = accuracy.eval(feed_dict={train:False, x:validation_images, y:validation_file_label["label"], keep_prob:1.0})
			print('\n----------------------------------------------')
			print('Epoch %g done, validation accuracy %g' % (epoch, validation_accuracy))
			if validation_accuracy > best_validation_accuracy:
				print('New best validation accuracy')
				best_epoch = epoch
				best_validation_accuracy = validation_accuracy
				save_path = saver.save(sess, "./model.ckpt")
			print('----------------------------------------------\n')
			if epoch - best_epoch > stop_after_last_best_epochs:
				print('Training stopped')
				break
			i = 0
			epoch = epoch + 1
			start = 0
			end = min(batch_size, len(train_file_label["label"]))
