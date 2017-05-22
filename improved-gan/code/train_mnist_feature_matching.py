# -*- coding: utf-8 -*-
#
#   train_mnist_feature_matching.py
#       date. 5/8/2017, 5/23
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
    params['mode'] = 'random'

    return params

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
def gaussian_noise_layer(inputs, sigma=0.1):
    noise = tf.random_normal(shape=tf.shape(inputs), 
                             mean=0.0, stddev=sigma, dtype=tf.float32)
    return inputs + noise

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


# Noise generator
def get_noise(batch_size):
    return np.random.normal(size=(batch_size, n_noise))


def inference(x_lab, x_unl, y_, z):
    '''
      do infernce by network model
      args.:
        x_lab:  labelled X data
        x_unl:  unlabelled X data
        y_:     label Y data
        z:      noise for generator
    '''
    # labelled data
    py_x_lab, feat_x_lab, vars_d = discriminator(x_lab)

    # unlabelled data
    py_x_unl, _, _ = discriminator(x_unl, reuse=True)

    # image generation
    x_generated, vars_g = generator(z)
    py_x_g, feat_x_gen, _ = discriminator(x_generated, reuse=True)

    logits = (py_x_lab, py_x_unl, py_x_g)
    features_to_match = (feat_x_lab, feat_x_gen)
    vars_ = (vars_d, vars_d)
    
    return logits, features_to_match, vars_


def loss(logits, y_, features_to_match):
    # unpack logits, features
    py_x_lab, py_x_unlab, py_x_g = logits
    feat_actual, feat_fake = features_to_match
    with tf.name_scope('losses'):
        # supervised loss
        loss_lab = tf.losses.softmax_cross_entropy(y_, py_x_lab)

        # unsupervised loss
        log_zx_unl = tf.reduce_logsumexp(py_x_unlab, axis=1)
        log_dx_unl = log_zx_unl - tf.nn.softplus(log_zx_unl)
        loss_unlab = -1. * tf.reduce_mean(log_dx_unl)

        loss_discr = loss_lab, loss_unlab
        tf.summary.scalar('loss_D', loss_discr)

        # adversarial loss
        log_zx_g = tf.reduce_logsumexp(py_x_g, axis=1)
        log_dx_g = log_zx_g - tf.nn.softplus(log_zx_g)
        loss_adversarial = -1. * tf.reduce_mean(log_dx_g)
        tf.summary.scalar('loss_G', loss_adversarial)

        # feature matching
        loss_fm = tf.losses.mean_squared_error(feat_actual, feat_fake)
        loss_adversarial += loss_fm

    return loss_discr, loss_adversarial

def evaluate(y_, y_pred):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    return accuracy

def gen_fake_data():
    n_input = 784
    n_class = 10
    fake1 = np.ones([10, n_input], dtype=np.float32) * 0.1
    fake2 = np.ones([40, n_input], dtype=np.float32) * 0.1
    fake3 = np.ones([10, n_class], dtype=np.float32) * 0.1

    return fake1, fake2, fake3


if __name__ == '__main__':
    # basic constants
    total_epochs = 200
    batch_size = 100
    learning_rate = 0.0002
    # network related constants
    n_hidden = 500
    n_input = 28 * 28   # eq. 784
    n_noise = 100
    n_class = 10

    # load data
    dataset_params = dataset_params()
    mnist = load_data_ssl(dataset_params, '../data')
    # fake1, fake2, fake3 = gen_fake_data()

    # tensorflow placeholders
    x_lab = tf.placeholder(tf.float32, [None, n_input])
    x_unl = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_class])
    z = tf.placeholder(tf.float32, [None, n_noise])

    # Graph definition
    logits, features_to_match, vars_ = inference(x_lab, x_unl, y_, z)
    loss_D, loss_G = loss(logits, y_, features_to_match)
    accuracy = evaluate(y_, py_x_lab)

    opti1 = tf.train.AdamOptimizer(learning_rate)
    train_op_D = opti1.minimize(loss_D, var_list=vars_[0])
    opti2 = tf.train.AdamOptimizer(learning_rate)
    train_op_G = opti2.minimize(loss_G, var_list=vars_[1])

    merged = tf.summary.merge_all()
    summaries_dir = '/tmp/tflogs'
    init = tf.global_variables_initializer()
    n_samples = mnist.train_unlab.num_examples

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test')
        sess.run(init)
        print('Training ...')

        for epoch in range(total_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xu, _ = mnist.train_unlab.next_batch(batch_size)
                batch_xl, batch_yl = mnist.train_lab.next_batch(batch_size)
                # noise_z = get_noise(batch_size)
                train_fd_D = {x_lab: batch_xl, x_unl: batch_xu, 
                            y_: batch_yl}
                _, loss_D_np = sess.run([train_op_D, loss_D],
                                        feed_dict=train_fd_D)
                batch_xu, _ = mnist.train_unlab.next_batch(batch_size)
                batch_xl, batch_yl = mnist.train_lab.next_batch(batch_size)
                noise_z = get_noise(batch_size)
                train_fd_G = {x_lab: batch_xl, y_: batch_yl,
                              x_unl: batch_xu, z: noise_z}
                _, loss_G_np = sess.run([train_op_G, loss_G],
                                        feed_dict=train_fd_G)
            summary = sess.run(merged, feed_dict=train_fd_G)
            print('epoch ={:5d}, training loss_D = {:>10.4f}, '
                  '   loss_G = {:>10.4f}'.format(
                  epoch, loss_D_np, loss_G_np))
            train_writer.add_summary(summary, epoch)

            # validation
            batch_xv, batch_yv = mnist.validation.next_batch(batch_size)
            noise_z = get_noise(batch_size)
            val_fd = {x_lab: batch_xv, x_unl: batch_xu,
                      y_: batch_yv, z: noise_z}
            summary, loss_val_np, accu_val_np = sess.run(
                            [merged, loss_D, accuracy], feed_dict=val_fd)
            print('epoch ={:5d}, validation loss = {:>10.4f}, '
                  '  accuracy = {:>10.4f}\n'.format(
                  epoch, loss_val_np, accu_val_np))
            test_writer.add_summary(summry, epoch)
