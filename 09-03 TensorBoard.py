# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

__author__ = 'sean.j'

xy = np.loadtxt('./data/train_09.csv', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

print x_data
print y_data

X = tf.placeholder(tf.float32, name='X')
x_hist = tf.histogram_summary('x', X)
Y = tf.placeholder(tf.float32, name='Y')
y_hist = tf.histogram_summary('y', Y)

W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='Weight1')
w1_hist = tf.histogram_summary('weight1', W1)
W2 = tf.Variable(tf.random_uniform([5, 4], -1.0, 1.0), name='Weight2')
w2_hist = tf.histogram_summary('weight2', W2)
W3 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0), name='Weight3')
w3_hist = tf.histogram_summary('weight3', W3)

b1 = tf.Variable(tf.zeros([5]), name='Bias1')
b1_hist = tf.histogram_summary('bais1', b1)
b2 = tf.Variable(tf.zeros([4]), name='Bias2')
b2_hist = tf.histogram_summary('bais2', b2)
b3 = tf.Variable(tf.zeros([1]), name='Bias3')
b3_hist = tf.histogram_summary('bais3', b3)

with tf.name_scope('layer1') as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
with tf.name_scope('layer2') as scope:
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
with tf.name_scope('layer3') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

learning_rate = 0.1
#reg_strength = 0
#l2reg = tf.reduce_sum(tf.square(W1) + tf.square(W2))

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  #+ reg_strength * l2reg
    cost_sum = tf.scalar_summary('cost', cost)
with tf.name_scope('train') as scope:
    training = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/tmp/xor_logs', sess.graph)

    sess.run(init)

    for step in xrange(10001):
        sess.run(training, feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)

    print 'finished!'
    '''
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    print 'New!'
    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data})
    print 'Accuracy: ', accuracy.eval({X: x_data, Y: y_data})
    '''

