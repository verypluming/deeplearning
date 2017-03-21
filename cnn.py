# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy
import time

BATCH_SIZE = 50

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolution
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# mini_batch
def mini_batch(data, label, i):
  index = (i * BATCH_SIZE) % data.shape[0]
  return data[index:index+BATCH_SIZE], label[index:index+BATCH_SIZE]

# load input data
def input_data():
  data = numpy.loadtxt('./data.txt')
  label = numpy.loadtxt('./label.txt')
  perm = numpy.arange(label.shape[0])
  numpy.random.shuffle(perm)
  data = data[perm]
  label = label[perm]
  TEST_SIZE = 500

  test_data = data[:TEST_SIZE]
  test_label = label[:TEST_SIZE]

  train_data = data[TEST_SIZE:]
  train_label = label[TEST_SIZE:]

  return train_data, train_label, test_data, test_label

if __name__ == '__main__':
  #print("--- MNIST read start ---")
  #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  #print("--- MNIST read finished ---")

  start_time = time.time()
  print("start time: " + str(start_time))

  print("--- data read start ---")
  train_data, train_label, test_data, test_label = input_data()
  print("--- data read finished ---")
    
  x = tf.placeholder("float", shape=[None, 1024]) # input
  y_ = tf.placeholder("float", shape=[None, 2]) # output
  sess = tf.InteractiveSession()
  
  x_image = tf.reshape(x, [-1, 32, 32, 1]) # array into image

  #layer 1
  W_conv1 = weight_variable([5, 5, 1, 32]) # 5x5patch, 32features
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  
  #layer 2
  W_conv2 = weight_variable([3, 3, 32, 48]) # 3x3patch, 48features
  b_conv2 = bias_variable([48])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  #layer 3
  W_conv3 = weight_variable([3, 3, 48, 64]) # 3x3patch, 64features
  b_conv3 = bias_variable([64])
  h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2(h_conv3)

  #fully connectted layer
  W_fc1 = weight_variable([4 * 4 * 64, 500]) # image size has been reduced to 7x7, fully connected 1024 neurons
  b_fc1 = bias_variable([500])

  h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 64]) # connect to fc layer
  h_fc1 = tf.nn.relu6(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # dropout
  W_fc2 = weight_variable([500, 2])
  b_fc2 = bias_variable([2])
 
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # softmax
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  #optimize
  #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  sess.run(tf.initialize_all_variables())

  #training
  print("--- train start ---")
  for i in range(500):
    batch = mini_batch(train_data, train_label, i)

    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print ("step {0}, training accuracy {1}".format(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  print("--- train finished ---")

  print("test accuracy {0}".format(accuracy.eval(feed_dict={
      x: test_data, y_: test_label, keep_prob: 1.0})))
  
  end_time = time.time()
  print("end time: " + str(end_time))
  print("process time: " + str(end_time - start_time))