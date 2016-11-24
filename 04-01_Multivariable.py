# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

__author__ = 'sean.j'

'''
x_data = [[1., 1., 1., 1., 1.],
          [1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]
y_data = [1, 2, 3, 4, 5]
'''
xy = np.loadtxt('./data/train_04.csv', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

W = tf.Variable(tf.random_uniform([1,3], -5., 5.))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(train)
        if step % 100 == 0:
            print step, sess.run(cost), sess.run(W)


