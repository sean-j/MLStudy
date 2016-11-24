# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

__author__ = 'sean.j'

xy = np.loadtxt('./data/train_06.csv', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder('float', [None, 3])
Y = tf.placeholder('float', [None, 3])

W = tf.Variable(tf.zeros([3, 3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W))
strength = 0.5
l2reg = tf.reduce_sum(tf.square(W))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1)) + strength * l2reg
training = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(training, feed_dict={X: x_data, Y: y_data})
        if step % 50 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    print a, sess.run(tf.arg_max(a, 1))
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print a, sess.run(tf.arg_max(b, 1))
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print a, sess.run(tf.arg_max(c, 1))

    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print all, sess.run(tf.arg_max(all, 1))
