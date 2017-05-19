#
#   mnist_latent_space.py
#       date. 5/11/2017, 5/19
#
#   (ref.) http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html
#

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data

def load_data(mnist_dir='../data'):
    mnist = input_data.read_data_sets(mnist_dir, one_hot=False)

    return mnist

def kernel_plot(rbm):
    plt.figure(figsize=(12, 10))
    n_plot = 100

    for i, comp in enumerate(rbm.components_[:n_plot, :]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.85, 0.85, 0.05, 0.05)

    plt.savefig('../work/mnist_kernel.png')
    plt.close()

def digit_reconstruct(rbm, images):
    plt.figure()
    n_sample = images.shape[0]
    images_orig = np.copy(images)

    # perform gibbs sampling
    n_chains = 200
    for i in range(n_chains):
        images = rbm.gibbs(images)

    for i in range(n_sample):
        plt.subplot(2, 10, i + 1)
        plt.imshow(images_orig[i, :].reshape((28, 28)), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

        plt.subplot(2, 10, i + 11)
        plt.imshow(images[i, :].reshape((28, 28)), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Digits reconstruction by RBM', fontsize=16)
    plt.savefig('../work/mnist_reconst.png')
    plt.close()


if __name__ == '__main__':
    np.random.seed(seed=2017)
    mnist = load_data()

    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels
    
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.03
    rbm.n_iter = 10
    rbm.n_components = 500

    print('\nRBM Training...')
    rbm.fit(X_train)

    kernel_plot(rbm)

    sample_10 = X_test[:10, :]
    digit_reconstruct(rbm, sample_10)
