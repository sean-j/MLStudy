# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf

__author__ = 'sean.j'

a = tf.constant(2)
b = tf.constant(3)

c = a + b

with tf.Session() as sess:
    print c
    print sess.run(c)