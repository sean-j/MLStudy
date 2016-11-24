# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf

__author__ = 'sean.j'

x_data = [1, 3, 5, 7, 9]
y_data = [1, 2 ,3, 4, 5]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W * x_data + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(train)
        if step % 100 == 0:
            print step, sess.run(cost), sess.run(W), sess.run(b)


