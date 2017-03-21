# -*- coding: utf-8 -*-

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

# Encoder-Decoder model
def model(X, w_e, b_e, w_d, b_d):
    #tf.sigmoid
    encoded = tf.sigmoid(tf.matmul(X, w_e) + b_e)
    decoded = tf.sigmoid(tf.matmul(encoded, w_d) + b_d)
    
    return encoded, decoded

if __name__ == '__main__':

  start_time = time.time()
  print("start time: " + str(start_time))

  print("--- data read start ---")
  train_data, train_label, test_data, test_label = input_data()
  print("--- data read finished ---")
    
  x = tf.placeholder("float", shape=[None, 1024]) # input
  y_ = tf.placeholder("float", shape=[None, 2]) # output
  keep_prob = tf.placeholder("float")
  
  #keep_prob = tf.placeholder("float")
  w_enc = tf.Variable(tf.random_normal([1024, 625], mean=0.0, stddev=0.05))
  w_dec = tf.Variable(tf.random_normal([625, 1024], mean=0.0, stddev=0.05))
  w_dec = tf.transpose(w_enc)
  b_enc = tf.Variable(tf.zeros([625]))
  b_dec = tf.Variable(tf.zeros([1024]))

  sess = tf.InteractiveSession()

  encoded, decoded = model(x, w_enc, b_enc, w_dec, b_dec)

  #optimize
  #cross_entropy = tf.pow(x - decoded, 2)
  #cross_entropy = -tf.reduce_sum(x * tf.log(decoded))
  #cross_entropy = -1. * x * tf.log(decoded) - (1. - x) * tf.log(1. - decoded)
  print("cross_entropy:{0}".format(cross_entropy))
  loss = tf.reduce_mean(cross_entropy)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
  sess.run(tf.initialize_all_variables())

  #training
  print("--- train start ---")
  for i in range(500):
    batch = mini_batch(train_data, train_label, i)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

    if i % 100 == 0:
      train_loss = loss.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print ("step {0}, loss {1}".format(i, train_loss))
    
  print("--- train finished ---")

  print("test loss {0}".format(loss.eval(feed_dict={
      x: test_data, y_: test_label, keep_prob: 1.0})))
  
  end_time = time.time()
  print("end time: " + str(end_time))
  print("process time: " + str(end_time - start_time))