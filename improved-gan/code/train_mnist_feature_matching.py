# -*- coding: utf-8 -*-
#
#   train_mnist_feature_matching.py
#       date. 5/8/2017
#
#   (ref.)
#   https://github.com/openai/improved-gan/tree/master/mnist_svhn_cifar10
#
'''
Feature matchingは、Discriminatorにxとx~を入力した時のそれぞれの中間層出力の二乗誤差を
小さくすることでGeneratorがより本物に近いデータを生成できるようにするテクニックです。
実装する時は出力層に一番近い中間層出力（活性化関数を通した後の値）をマッチさせれば良いと思います。
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import layers

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)


# basic constants
total_epoch = 20
batch_size = 100
learning_rate = 0.0002
# network related constants
n_hidden = 500
n_input = 28 * 28   # eq. 784
n_noise = 100

# TensorFlow placeholders
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])


# Generator func. (G) 
def generator(noise_z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        net = tf.layers.dense(noise_z,
                    n_hidden, activation=tf.nn.softplus, name='gener1')
        net = tf.layers.dense(net,
                    n_hidden, activation=tf.nn.softplus, name='gener2')
        generator_out = tf.layers.dense(net,
                    n_input, activation=tf.sigmoid, name='gener3')

    vars_g = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    return generator_out, vars_g

# Discriminator func. (D)
def gaussian_noise_layer(input_layer, sigma=0.1):
    noise = tf.random_normal(shape=tf.shape(input_layer), 
                             mean=0.0, stddev=sigma, dtype=tf.float32)
    return input_layer + noise


def discriminator(inputs, reuse=False):
    num_units = [None, 1000, 500, 250, 250, 250, 10]
    with tf.variable_scope('discriminator', reuse=reuse):
        net = gaussian_noise_layer(inputs, sigma=0.3)
        net = tf.layers.dense(net,
                    num_units[1], activation=tf.nn.relu, name='discr1')
        net = gaussian_noise_layer(net, sigma=0.5)
        net = tf.layers.dense(net,
                    num_units[2], activation=tf.nn.relu, name='discr2')
        net = gaussian_noise_layer(net, sigma=0.5)
        net = tf.layers.dense(net,
                    num_units[3], activation=tf.nn.relu, name='discr3')
        net = gaussian_noise_layer(net, sigma=0.5)
        net = tf.layers.dense(net,
                    num_units[4], activation=tf.nn.relu, name='discr4')
        net = gaussian_noise_layer(net, sigma=0.5)
        net = tf.layers.dense(net,
                    num_units[5], activation=tf.nn.relu, name='discr5')
        mom_out = net       # forwarding to feature matching
        net = gaussian_noise_layer(net, sigma=0.5)
        discriminator_out = tf.layers.dense(net, 
                    num_units[6], activation=None, name='discr6')
    
    vars_d = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    return discriminator_out, mom_out, vars_d


def inference(x_lab, x_unl, y_):
    # labelled data
    py_x_lab, feat_x_lab, _ = discriminator(x_lab)

    # unlabelled data
    py_x_unl, _, _ = discriminator(x_unl)

    # image generation
    x_generated = generator(some_noise)         # need to fix "some_noise"
    py_x_g, feat_x_g = discriminator(x_generated)
    
    return py_x_lab, feat_x_lab, py_x_unlab, py_x_g, feat_x_g


def loss(py_x_lab, py_x_unlab, py_x_g, y_, feat_actual, feat_fake):
    # supervised loss
    loss_lab = tf.losses.softmax_cross_entropy()

    # unsupervised loss
    log_zx_unl = tf.reduce_logsumexp(py_x_unl, axis=1)
    log_dx_unl = log_zx_unl - tf.nn.softplus(log_zx_unl)

    loss_unlab = tf.reduced_mean(log_dx_unl)

    # adversarial loss


    # feature matching

    return loss_tot




def fm_loss(feat_actual, feat_fake):
    '''
      calculate feature matching loss
        mom_gen = T.mean(LL.get_output(layers[-3], gen_dat), axis=0)
        mom_real = T.mean(LL.get_output(layers[-3], x_unl), axis=0)
        oss_gen = T.mean(T.square(mom_gen - mom_real))
    '''

    return tf.losses.mean_squared_error(feat_actual, feat_fake)



if __name__ == '__main__':

    # tensorflow placeholders
    x_lab = tf.placeholder(tf.float32, [None, n_input])
    x_unl = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, 10])



'''
  on-going ...
'''
