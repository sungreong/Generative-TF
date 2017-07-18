# -*- coding: utf-8 -*-
#
#   gan_layers2.py
#
#       date. 4/9/2017
#

import tensorflow as tf
from tensorflow.python.layers import layers
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)

# basic constants
total_epoch = 1000
batch_size = 100
n_input = 28 * 28   # eq. 784
n_noise = 128
n_class = 10
n_hidden = 256
learning_rate = 0.001


# TensorFlow placeholders
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, n_noise])


def generator(noise, labels, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # add labels to input data
        inputs = tf.concat([noise, labels], 1)

        G1 = tf.layers.dense(
                    inputs, n_hidden, activation=tf.nn.relu, name='gener1')
        G2 = tf.layers.dense(
                    G1, n_input, activation=tf.nn.relu, name='gener2')
    
    vars_g = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    return G2, vars_g


def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        # add labels to input data
        inputs = tf.concat([inputs, labels], 1)

        D1 = tf.layers.dense(
                    inputs, n_hidden, activation=tf.nn.relu, name='discr1')
        D2 = tf.layers.dense(
                    D1, n_hidden, activation=tf.nn.relu, name='discr2')
        D3 = tf.layers.dense(
                    D2, 1, activation=None, name='discr3')

        vars_d = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    return D3, vars_d


# define networks
G, vars_G = generator(Z, Y)
D_real, vars_D = discriminator(X, Y)
D_gene, _ = discriminator(G, Y, reuse=True)

# Loss function is set referring ...
# http://bamos.github.io/2016/08/09/deep-completion/

loss_D_real = tf.reduce_mean(
                    tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real), D_real))
loss_D_gene = tf.reduce_mean(
                    tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_gene), D_gene))
loss_D = loss_D_real + loss_D_gene
loss_G = tf.reduce_mean(
                    tf.losses.sigmoid_cross_entropy(tf.ones_like(D_gene), D_gene))

train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G, var_list=vars_G)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = np.random.uniform(-1., 1., size=[batch_size, n_noise])

        _, loss_val_D = sess.run([train_D, loss_D], 
                        feed_dict={X: batch_xs, Y: batch_ys, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], 
                        feed_dict={Y: batch_ys, Z: noise})

    print('Epoch: {:>4d}'.format((epoch + 1)), 
          'D loss: {:>.4f}'.format(loss_val_D),
          'G loss: {:>.4f}'.format(loss_val_G))

    if epoch % 10 == 0:
        noise = np.random.uniform(-1., 1., size=[30, n_noise])
        samples = sess.run(G, feed_dict={Y: mnist.validation.labels[:30], Z: noise})

        fig, ax = plt.subplots(6, n_class, figsize=(n_class, 6))

        for i in range(n_class):
            for j in range(6):
                ax[j][i].set_axis_off()

            for j in range(3):
                ax[0+(j*2)][i].imshow(
                    np.reshape(mnist.validation.images[i+(j*n_class)], (28, 28)))
                ax[1+(j*2)][i].imshow(
                    np.reshape(samples[i+(j*n_class)], (28, 28)))

        plt.savefig('../work/samples2/{}.png'.format(
                        str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)


print('Training is completed.')
