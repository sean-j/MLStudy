# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf

__author__ = 'sean.j'

hello = tf.constant('Hello, TensorFlow!')

with tf.Session() as sess:
    print hello
    print sess.run(hello)