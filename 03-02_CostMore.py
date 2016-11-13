# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

__author__ = 'sean.j'

x_data = [1., 2., 3.]
y_data = [3., 6., 9.]

W = tf.Variable(tf.random_uniform([1], -10., 10.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init, feed_dict={X: x_data, Y: y_data})

    for step in xrange(101):
        sess.run(update, feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

    print sess.run(hypothesis, feed_dict={X: 5})
