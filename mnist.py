# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time

start_time = time.time()
print("start time: " + str(start_time))

print("--- MNIST read start ---")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("--- MNIST read finished ---")

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("--- train start ---")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
print("--- train finished ---")

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("accuracy: ")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

end_time = time.time()
print("end time: " + str(end_time))
print("process time: " + str(end_time - start_time))