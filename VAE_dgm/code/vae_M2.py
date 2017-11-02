# -*- coding: utf-8 -*-
#
#   vae_M2.py
#       vae M2 model for semi-supervised learning
#       date. 4/22/2017, 4/26
#

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data


def mk_config():
    '''
      set network configuration
    '''
    config = {}
    config['ndim_x'] = 28 * 28
    config['ndim_y'] = 10
    config['ndim_z'] = 50

    config['encoder_xy_z_hidden_units'] = 500
    config['encoder_x_y_hidden_units'] = 500
    config['decoder_yz_x'] = 500

    config['learning_rate'] = 3.e-4

    return config


class VAE(object):
    """ 
    Variation Autoencoder (VAE)
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. 
    """
    def __init__(self, config, name="vae"):
        self.name = name
        self.config = config
        # tensorflow placeholders
        x_lab = tf.placeholder(tf.float32, [None, config['ndim_x']])
        y_lab = tf.placeholder(tf.float32, [None, config['ndim_y']])
        x_unlab = tf.placeholder(tf.float32, [None, config['ndim_x']])
        self.x_lab = x_lab
        self.y_lab = y_lab
        self.x_unlab = x_unlab
        # prepare tensorflow graph, etc.
        self._create_loss_optimizer(config)


    def _create_loss_optimizer(self, config):
        '''
          sum up losses and set optimzer to minimize
          args:
            config  : network configuration
            alpha   : hyper parameter to change ratio of generator 
                      side / discreminator side (eq.9)
        '''
        loss_g3 = self.compute_lower_bound_loss(
                                self.x_lab, self.y_lab, self.x_unlab)
        loss_g = loss_g3[0]
        
        loss_d, accuracy = self.compute_classification_loss(
                                                self.x_lab, self.y_lab)

        loss_tot = loss_g + config['alpha'] * loss_d
        _optimizer = tf.train.AdamOptimizer(
                                    learning_rate=config['learning_rate'])

        self.train_op = _optimizer.minimize(loss_tot)
        self.loss_g = loss_g
        self.loss_d = loss_d
        self.loss_tot = loss_tot
        self.accuracy = accuracy
        self.batch_size = config['batch_size']

        return None

    def sample_x_y(self, x, argmax=False, test=False):
        ''' compute p(y) = Cat(y|pi) multinomial distribution '''
        batch_size = tf.shape(x.data)[0]
        y_dist = self._encoder_x_y(x, test=test, softmax=True)
        n_labels = tf.shape(y_dist)[1]

        if argmax:
            raise ValueError
        else:
            # Draws samples from a multinomial distribution
            label_id = tf.multinomial(tf.log(y_dist), 1)
            label_id = tf.squeeze(label_id)
            sampled_y = tf.one_hot(label_id, n_labels)
        
        return sampled_y

    def sample_x_label(self, x, test=False):
        y_dist = self._encoder_x_y(x, test=test, softmax=True)

        return tf.argmax(y_dist, axis=1)

    def bernoulli_nll_keepbatch(self, x, y):
        nll = tf.nn.softplus(y) - x * y
        return tf.reduce_sum(nll, axis=1)

    def gaussian_nll_keepbatch(self, x, mean, ln_var, clip=True):
        if clip:
            clip_min = tf.log(0.001)
            ln_var = tf.minimum(ln_var, clip_min)
            clip_max = tf.log(10.)
            ln_var = tf.maximum(ln_var, clip_max)
        x_prec = tf.exp(-ln_var)
        x_diff = x - mean
        x_power = (x_diff * x_diff) * x_prec * 0.5
        pi = tf.constant(3.1415, dtype=tf.float32)
        nll = tf.reduce_sum((tf.log(2.0 * pi) + ln_var) * 0.5 + x_power,
                            axis=1)
        return nll

    def gaussian_kl_divergence_keepbatch(self, mean, ln_var):
        var = tf.exp(ln_var)
        kld = tf.reduce_sum(mean * mean + var - ln_var - 1, axis=1) * 0.5
        return kld

    def log_px_zy(self, x, z, y, reuse=False):
        config = self.config
        x_mean, x_ln_var = self._decoder(config, y, z, reuse=reuse)
        nll = self.gaussian_nll_keepbatch(x, x_mean, x_ln_var)

        return -nll

    def log_py(self, y):
        n_types_of_label = tf.shape(y)[1]
        shape_ = (tf.shape(y)[0],)
        n_types_float = tf.to_float(n_types_of_label)
        constant = tf.log(1. / n_types_float)
        filled = tf.fill(shape_, constant)

        return filled


    def train_ae(self, sess, dataset, train_epochs=10):
        display_step = 5
        batch_size = self.batch_size

        try:
            n_samples = dataset.train_unlab._num_examples
        except:
            print('Datasets does not include (train_unlab) part.')
            raise ValueError

        # Training process
        for epoch in range(train_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = dataset.train.next_batch(batch_size)

                # Fit training using batch data
                cost = self.partial_fit(batch_xs, sess)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), 
                    "cost=", "{:.9f}".format(avg_cost))

        return None

    def train_classifier(self, sess, dataset, alpha=1.0, train_epochs=10):
        ''' This method is used for M2 classification model '''
        batch_size = self.batch_size
        display_step = 10

        try:
            n_samples = dataset.train_unlab._num_examples
        except:
            print('Datasets does not include (train_unlab) part.')
            raise ValueError

        # Training process
        for epoch in range(train_epochs):
            cost_avg = 0.
            nloops = int(n_samples / batch_size)
            
            for i in range(nloops):
                batch_xl, batch_yl = dataset.train_lab.next_batch(batch_size)
                batch_xul, _ = dataset.train_unlab.next_batch(batch_size)
                train_fd = {self.x_lab: batch_xl, 
                            self.y_lab: batch_yl, 
                            self.x_unlab: batch_xul}

                _, loss1, loss2, loss3 = sess.run([self.train_op, self.loss_tot, 
                                                   self.loss_g, self.loss_d], 
                                feed_dict=train_fd)
            # cross validation accuracy
            batch_xv, batch_yv = dataset.validation.next_batch(batch_size)
            val_fd = {self.x_lab: batch_xv, self.y_lab: batch_yv}
            accu_val = sess.run(self.accuracy, feed_dict=val_fd)
                
            if epoch % display_step == 0:
                print('epoch {:>6d}: loss_tot, (loss_g/loss_d) = {:>8.3f},\t({:>.3f}, {:>.3f})'.format(
                    epoch, loss1, loss2, loss3))
                print('              c.v. accuracy = {:>8.4f}'.format(accu_val))

        return None

    def train_jointly(self, sess, dataset, alpha=1.0):
        ''' This method is used for M1+M2 stacked model '''

        return None

    def compute_lower_bound_loss(self, labeled_x, labeled_y, unlabeled_x):
        ''' compute lower boundary of lossess '''
        def lower_bound(log_px_zy, log_py, log_pz, log_qz_xy):
            lb = log_px_zy + log_py + log_pz - log_qz_xy
            return lb

        def with_x_lab(config, labeled_x, labeled_y):
            ### Lower bound for labeled data ###
            # Compute eq.6 -L(x,y)
            z_mean_l, z_ln_var_l = self._encoder_xy_z(config, labeled_x, labeled_y)
            # Draw one sample z from Gaussian distribution
            z_data_shape = tf.shape(z_mean_l)
            eps = tf.random_normal(z_data_shape, mean=0., stddev=1., dtype=tf.float32)
            z_l = tf.add(z_mean_l, tf.multiply(tf.sqrt(tf.exp(z_ln_var_l)), eps))

            log_px_zy_l = self.log_px_zy(labeled_x, z_l, labeled_y)
            log_py_l = self.log_py(labeled_y)

            lower_bound_l = log_px_zy_l + log_py_l - self.gaussian_kl_divergence_keepbatch(
                                                z_mean_l, z_ln_var_l)

            return lower_bound_l

        def without_x_unlab():
            lower_bound_u = tf.constant(0.)

            return lower_bound_u

        def with_x_unlab(config, unlabeled_x, batchsize_u):
            ### Lower bound for unlabeled data ###
            # To marginalize y, we repeat unlabeled x, and construct a target (batchsize_u * num_types_of_label) x num_types_of_label
            # Example of n-dimensional x and target matrix for a 3 class problem and batch_size=2.
            #         unlabeled_x_ext                 y_ext
            #  [[x0[0], x0[1], ..., x0[n]]         [[1, 0, 0]
            #   [x1[0], x1[1], ..., x1[n]]          [1, 0, 0]
            #   [x0[0], x0[1], ..., x0[n]]          [0, 1, 0]
            #   [x1[0], x1[1], ..., x1[n]]          [0, 1, 0]
            #   [x0[0], x0[1], ..., x0[n]]          [0, 0, 1]
            #   [x1[0], x1[1], ..., x1[n]]]         [0, 0, 1]]
            n_label = config['ndim_y']
            tile_shape = [n_label, 1]
            unlabeled_x_ext = tf.tile(unlabeled_x, tile_shape)

            y_orig = tf.eye(n_label, dtype=tf.float32)
            y_tiled  = tf.tile(y_orig, [1, batchsize_u])
            y_ext= tf.reshape(y_tiled, [-1, n_label])

            # Compute eq.6 -L(x,y) for unlabeled data
            z_mean_u_ext, z_ln_var_u_ext = self._encoder_xy_z(
                                            config, unlabeled_x_ext, y_ext, reuse=True)
            # z_u_ext = _gaussian(z_mean_u_ext, z_mean_ln_var_u_ext)
            # Draw one sample z from Gaussian distribution
            z_data_shape = tf.shape(z_mean_u_ext)
            eps = tf.random_normal(z_data_shape, mean=0., stddev=1., dtype=tf.float32)
            z_u_ext = tf.add(z_mean_u_ext, tf.multiply(tf.sqrt(tf.exp(z_ln_var_u_ext)), eps))
            log_px_zy_u = self.log_px_zy(unlabeled_x_ext, z_u_ext, y_ext, reuse=True)
            log_py_u = self.log_py(y_ext)
   
            lower_bound_u = log_px_zy_u + log_py_u \
                - self.gaussian_kl_divergence_keepbatch(z_mean_u_ext, z_ln_var_u_ext)

            # Compute eq.7 sum_y{q(y|x){-L(x,y) + H(q(y|x))}}
            # Let LB(xn, y) be the lower bound for an input image xn and a label y (y = 0, 1, ..., 9).
            # Let bs be the batchsize.
            # 
            # lower_bound_u is a vector and it looks like...
            # [LB(x0,0), LB(x1,0), ..., LB(x_bs,0), LB(x0,1), LB(x1,1), ..., LB(x_bs,1), ..., LB(x0,9), LB(x1,9), ..., LB(x_bs,9)]
            # 
            # After reshaping. (axis 1 corresponds to label, axis 2 corresponds to batch)
            # [[LB(x0,0), LB(x1,0), ..., LB(x_bs,0)],
            #  [LB(x0,1), LB(x1,1), ..., LB(x_bs,1)],
            #                   .
            #                   .
            #                   .
            #  [LB(x0,9), LB(x1,9), ..., LB(x_bs,9)]]
            # 
            # After transposing. (axis 1 corresponds to batch)
            # [[LB(x0,0), LB(x0,1), ..., LB(x0,9)],
            #  [LB(x1,0), LB(x1,1), ..., LB(x1,9)],
            #                   .
            #                   .
            #                   .
            #  [LB(x_bs,0), LB(x_bs,1), ..., LB(x_bs,9)]]
            lower_bound_u = tf.transpose(
                        tf.reshape(lower_bound_u, (n_label, batchsize_u)))
    
            y_dist = self._encoder_x_y(config, unlabeled_x)
            lower_bound_u = y_dist * (lower_bound_u - tf.log(y_dist + 1.e-6))
            # loss_unlabeled = -tf.reduce_sum(lower_bound_u) / tf.to_float(batchsize_u)

            return lower_bound_u


        # _l: labeled
        # _u: unlabeled
        config = self.config
        batchsize_l = tf.shape(labeled_x)[0]
        batchsize_u = tf.shape(unlabeled_x)[0]
        n_label = tf.shape(labeled_y)[1]
        zero_t = tf.constant(0, dtype=tf.int32)

        # Lower bound for labeled data
        lower_bound_l = with_x_lab(config, labeled_x, labeled_y)

        # combine 2 funcs by tf.cond()
        def f1(): return with_x_unlab(config, unlabeled_x, batchsize_u)
        def f2(): return without_x_unlab()
        lower_bound_u = tf.cond(tf.greater(batchsize_u, 0), f1, f2)

        # loss_labeled = -tf.reduce_sum(lower_bound_l)
        # loss_unlabeled = -tf.reduce_sum(lower_bound_u)

        # average over batch
        loss_labeled = tf.reduce_mean(-lower_bound_l)
        loss_unlabeled = tf.reduce_mean(-lower_bound_u)
        loss = loss_labeled + loss_unlabeled

        return loss, loss_labeled, loss_unlabeled


    # Extended objective eq.9
    def compute_classification_loss(self, x_lab, y_lab):
        config = self.config
        y_pred = self._encoder_x_y(config, x_lab, reuse=True)
        # batchsize = tf.shape(labeled_x)[0]
        loss = tf.losses.softmax_cross_entropy(y_lab, y_pred)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_lab, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return loss, accuracy


class GaussianM2VAE(VAE):
    ''' Gaussian M2 VAE network models '''

    def _encoder_xy_z(self, config, input_x, input_y, reuse=False):
        ''' compute q(z|y,x) = func(theta,...) '''

        # check input data
        n_hidden = config['encoder_xy_z_hidden_units']
        n_z = config['ndim_z']

        with tf.variable_scope('encoder_xy_z', reuse=reuse):
            net_x = tf.layers.dense(input_x, n_hidden,
                                    activation=tf.nn.softplus,
                                    name='hidden1')
            net_y = tf.layers.dense(input_y, n_hidden, 
                                    activation=tf.nn.softplus,
                                    name='hidden2')
            merged = tf.concat([net_x, net_y], axis=1)
            net = tf.layers.batch_normalization(merged, name='batch_norm')

            z_mean = tf.layers.dense(net, n_z,
                                     activation=None,
                                     name='z_mean')
            z_log_sigma_2 = tf.layers.dense(net, n_z,
                                            activation=None,
                                            name='z_log_sigma_2')

        return z_mean, z_log_sigma_2
    
    
    def _encoder_x_y(self, config, inputs, reuse=False):
        ''' compute q(y|x) using softmax function '''
        n_hidden = config['encoder_x_y_hidden_units']
        n_y = config['ndim_y']

        with tf.variable_scope('encoder_x_y', reuse=reuse):
            net = tf.layers.dense(inputs, n_hidden, 
                                  activation=tf.nn.softplus, name='hidden')
            net = tf.layers.batch_normalization(net, name='batch_norm')
            readout = tf.layers.dense(net, n_y,
                                       activation=tf.nn.softmax, 
                                       name='readout')
    
        return readout

    def _decoder(self, config, input_y, input_z, reuse=False):
        '''  compute p(x|y,z) = func(x; y, z, theta) '''
        n_hidden = config['decoder_yz_x']
        n_input = config['ndim_x']
        with tf.variable_scope('decoder_yz_x', reuse=reuse):

            net_y = tf.layers.dense(input_y, n_hidden, 
                                    activation=tf.nn.softplus, name='hidden1')
            net_z = tf.layers.dense(input_z, n_hidden, 
                                    activation=tf.nn.softplus, name='hidden2')
            merged = tf.concat([net_y, net_z], axis=1)
            net = tf.layers.batch_normalization(merged, name='batch_norm')

            x_reconstr_mean = tf.layers.dense(net,
                                              n_input, activation=None)
            x_reconstr_sigma_2 = tf.layers.dense(net,
                                              n_input, activation=None)

        return x_reconstr_mean, x_reconstr_sigma_2

