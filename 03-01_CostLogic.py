# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

__author__ = 'sean.j'

X = [1., 2., 3.]
Y = [1., 2., 3.]

m = n_samples = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.mul(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2))/m

init = tf.initialize_all_variables()

W_val = []
cost_val = []

with tf.Session() as sess:
    sess.run(init)

    for i in range(-30, 50):
        print i * 0, sess.run(cost, feed_dict={W: i * 0.1})
        W_val.append(i * 0.1)
        cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

    plt.plot(W_val, cost_val, 'ro')
    plt.ylabel('Cost')
    plt.xlabel('W')
    plt.show()
