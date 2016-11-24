# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

__author__ = 'sean.j'

xy = np.loadtxt('./data/train_05.csv', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1., 1.))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

    print sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}) > 0.5
    print sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}) > 0.5
    print sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}) > 0.5

