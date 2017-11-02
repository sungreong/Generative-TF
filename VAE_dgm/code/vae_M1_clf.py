#
#   vae_M1_clf.py
#       date. 4/16/2017
#

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import backend as K

from tensorflow.examples.tutorials.mnist import input_data
from vae_keras_nets import VariationalAutoencoder

N_TOT_TRAIN =60000      # total number of train samples

def load_data(mnist_dirn='../data', label_ratio=0.1):
    '''
      Returns:
        mnist:              instance of 'collections.namedtuple'
        mnist.train:        dataset for train (for unsupervised learning)
        mnist.validation    dataset with label for supervised learning
        mnist.test          dataset for test
    '''
    n_train_lab = int(N_TOT_TRAIN * label_ratio)
    mnist = input_data.read_data_sets(mnist_dirn, 
                        validation_size=n_train_lab, one_hot=True)

    return mnist

def small_nets(x):
    #  network model
    with tf.variable_scope('small_nets'):
        x = keras.layers.Dense(units=100, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(units=100, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        y_pred = keras.layers.Dense(units=10, activation='softmax')(x)
    
    return y_pred

def classify_w_encoded(x, y_, n_z=20, lr=0.05):
    y_pred = small_nets(x)
    loss = tf.losses.softmax_cross_entropy(y_, y_pred)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    train_step = tf.train.AdagradOptimizer(lr).minimize(loss)

    return loss, accuracy, train_step



if __name__ == '__main__':  
    np.random.seed(seed=2017)
    mnist = load_data(label_ratio=0.1)

    n_samples = mnist.train.num_examples     # num. samples w/o label

    net_config = \
        dict(n_hidden_recog_1=400,  # 1st layer encoder neurons
         n_hidden_recog_2=400,      # 2nd layer encoder neurons
         n_hidden_gener_1=400,      # 1st layer decoder neurons
         n_hidden_gener_2=400,      # 2nd layer decoder neurons
         n_input=784,               # MNIST data input (img shape: 28*28)
         n_z=20)                    # dimensionality of latent space
    
    vae1 = VariationalAutoencoder(
                    net_config, learning_rate=0.001, batch_size=100)
    # classification with latent features
    n_z = net_config['n_z']
    z = tf.placeholder(tf.float32, [None, n_z])
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    clf_loss, clf_accuracy, train_step = classify_w_encoded(z, y_, lr=0.01)
    
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    print('\n VAE fitting...')
    vae1.train(sess, mnist, training_epochs=100)

    # training classifier (classify_w_encoded)
    print('\nTraining...')
    print('number of train samples in this process = ', 
                                    mnist.validation.num_examples)
    batch_size_tr = 100
    epochs = 101
    n_loop_train = int(mnist.validation.num_examples / batch_size_tr)
    for e in range(epochs):
        for i in range(n_loop_train):
            batch_x, batch_y = mnist.validation.next_batch(batch_size_tr)
            batch_z = vae1.transform(sess, batch_x)

            train_fd = {z: batch_z, y_: batch_y, K.learning_phase(): 1}
            train_step.run(feed_dict=train_fd)
        
        if e % 10 == 0:
            val_fd = {z: batch_z, y_: batch_y, K.learning_phase(): 0}
            tr_loss, tr_accu = sess.run([clf_loss, clf_accuracy], val_fd)
            print('Epoch, loss, accurary = {:>3d}: {:>8.4f}, {:>8.4f}'.format(
                                            e, tr_loss, tr_accu))
    
    # test process
    batch_size_te = 100
    n_loops_test = int(mnist.test.num_examples / batch_size_te)
    test_accu = []
    for i in range(n_loops_test):
        batch_xte, batch_yte = mnist.test.next_batch(batch_size_te)
        batch_z = vae1.transform(sess, batch_xte)
        test_fd = {z: batch_z, y_: batch_yte, K.learning_phase(): 0}
        test_accu.append(clf_accuracy.eval(feed_dict=test_fd))
    test_accu = np.asarray(test_accu)
    test_accuracy = np.mean(test_accu)

    print('\nTest accuracy = ', test_accuracy)

