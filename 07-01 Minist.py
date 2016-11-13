# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
from lib import minist_data as md

__author__ = 'sean.j'

def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

learning_rate = 0.01
training_epochs = 5
batch_size = 100
display_step = 1

x = tf.placeholder("float", [None, 28 * 28])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x, W))

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

mnist = md.read_data_sets("MNIST_data/", one_hot=True)
checkpoint_dir = "cps/"

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print (sess.run(b))

    print ("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print 'Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    '''
    r = randint(0, mnist.test.num_examples - 1)
    print 'Label: ', sess.run(tf.argmax(mnist.test.labels[r: r+1], 1))
    print 'Prediction: ', sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r + 1]})

    plt.imshow(mnist.test.images[r:r+ 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    '''
