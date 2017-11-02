# -*- coding: utf-8 -*-
#
#	vae_M2_train.py
#		date. 4/27/2017
#

import os, sys, time
import numpy as np
import tensorflow as tf

from mnist_prep_ssl import load_data_ssl
from vae_M2 import GaussianM2VAE

def params():	# for data loader
    params = {}
    params['n_train_lab'] = 1000
    params['n_val'] = 5000
    params['n_train_unlab'] = 60000 - 1000 - 5000
    params['percent_limit'] = (0.8, 1.2)    # accept +/-20%

    return params

# original code
# max_epoch = 1000
# num_trains_per_epoch = 2000
# batchsize_l = 100
# batchsize_u = 100

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

    config['alpha'] = 10.0
    config['learning_rate'] = 3.e-4
    config['batch_size'] = 200

    return config

if __name__ == '__main__':
	np.random.seed(seed=2017)
	params = params()
	mnist = load_data_ssl(params, '../data')

	config = mk_config()
	vae = GaussianM2VAE(config)

	loss_tot = vae.loss_tot
	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)

	# training vae net
	print('\nTraining...')
	epochs = 100

	vae.train_classifier(sess, mnist, train_epochs=epochs)

	print('finished')
