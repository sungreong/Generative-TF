# -*- coding: utf-8 -*-
# 
#   gan_layers.py
#
#   (ref.) golbin's TensorFlow tutorial codes
#       https://github.com/golbin/TensorFlow-Tutorials
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
total_epoch = 20
batch_size = 100
learning_rate = 0.0002
# network related constants
n_hidden = 256
n_input = 28 * 28   # eq. 784
n_noise = 128

# TensorFlow placeholders
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])


# Generator func. (G) 
def generator(noise_z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        net = tf.layers.dense(
                    noise_z, n_hidden, activation=tf.nn.relu, name='gener1')
        generator_out = tf.layers.dense(
                     net, n_input, activation=tf.sigmoid, name='gener2')

    vars_g = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    return generator_out, vars_g


# Discriminator func. (D)
def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        net = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu,
                                name='discr1')
        discriminator_out = tf.layers.dense(
                        net, 1, activation=tf.sigmoid, name='discr2')
    
    vars_d = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    return discriminator_out, vars_d


# Noise generator
def get_noise(batch_size):
    return np.random.normal(size=(batch_size, n_noise))


# Graph definition
G, G_var_list = generator(Z)
D_gene, D_var_list = discriminator(G)
D_real, _ = discriminator(X, reuse=True) 
                        # but specified shape (784, 256) and found shape (10, 256)

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

# GAN training optimizer
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

print('Training ...')

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size)

        # Train Discriminator and Generator in parallel
        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('Epoch: {:>4d}'.format((epoch + 1)), 
          'D loss: {:>.4f}'.format(loss_val_D),
          'G loss: {:>.4f}'.format(loss_val_G))

    sample_size = 10
    noise = get_noise(sample_size)

    samples = sess.run(G, feed_dict={Z: noise})

    fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

    for i in range(sample_size):
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(samples[i], (28, 28)))

    plt.savefig('../work/samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)


print('Training is completed.')
