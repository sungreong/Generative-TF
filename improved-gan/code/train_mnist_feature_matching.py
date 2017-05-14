# -*- coding: utf-8 -*-
#
#   train_mnist_feature_matching.py
#       date. 5/8/2017, 5/15
#
#   (ref.)
#   https://github.com/openai/improved-gan/tree/master/mnist_svhn_cifar10
#

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import layers

from tensorflow.examples.tutorials.mnist import input_data
from mnist_prep_ssl import load_data_ssl

# MNIST dataset parameters for data loader
def dataset_params():
    params = {}
    params['n_train_lab'] = 1000
    params['n_val'] = 5000
    params['n_train_unlab'] = 60000 - 1000 - 5000

    return dataset_params

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
    py_x_g, feat_x_gen, _ = discriminator(x_generated)
    features_to_match = (feat_x_lab, feat_x_gen)
    
    return py_x_lab, py_x_unlab, py_x_g, features_to_match


def loss(py_x_lab, py_x_unlab, py_x_g, y_, feat_actual, feat_fake):
    # supervised loss
    loss_lab = tf.losses.softmax_cross_entropy(y_, py_x_lab)

    # unsupervised loss
    log_zx_unl = tf.reduce_logsumexp(py_x_unl, axis=1)
    log_dx_unl = log_zx_unl - tf.nn.softplus(log_zx_unl)
    loss_unlab = -1. * tf.reduced_mean(log_dx_unl)

    # adversarial loss
    log_zx_g = tf.reduce_logsumexp(py_x_g, axis=1)
    log_dx_g = log_zx_g - tf.nn.softplus(log_zx_g)
    loss_advarsarial = -1. * tf.reduce_mean(log_dx_g)

    # feature matching
    loss_fm = tf.losses.mean_squared_error(feat_actual, feat_fake)
    loss_advarsarial += loss_fm    

    return loss_lab, loss_unlab, loss_adversarial

def gen_fake_data():
    n_input = 784
    n_class = 10
    fake1 = np.ones([10, n_input], dtype=np.float32) * 0.1
    fake2 = np.ones([40, n_input], dtype=np.float32) * 0.1
    fake3 = np.ones([10, n_class], dtype=np.float32) * 0.1

    return fake1, fake2, fake3


if __name__ == '__main__':
    # basic constants
    total_epoch = 20
    batch_size = 100
    learning_rate = 0.0002
    # network related constants
    n_hidden = 500
    n_input = 28 * 28   # eq. 784
    n_noise = 100
    n_class = 10

    # load data
    # dataset_params = dataset_params()
    # mnist = load_data_ssl(params, '../data')
    fake1, fake2, fake3 = gen_fake_data()

    # tensorflow placeholders
    x_lab = tf.placeholder(tf.float32, [None, n_input])
    x_unl = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_class])

    # Graph definition
    py_x_lab, py_x_unlab, py_x_g, features_to_match = inference(x_lab, x_unl, y_)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    fd = {x_lab: fake1, x_unl: fake2, y_: fake3}

    py_1_np, py_2_np, py_3_np = sess.run([py_x_lab, py_x_unlab, py_x_g], feed_dict=fd)
    print('shape of py_1_np = ', py_1_np.shape)
    print('shape of py_2_np = ', py_2_np.shape)
    print('shape of py_3_np = ', py_3_np.shape)


    assert False

    '''
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
    '''

'''
  on-going ...
'''
