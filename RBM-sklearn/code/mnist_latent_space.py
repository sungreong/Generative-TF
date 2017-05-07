#
#   mnist_latent_space.py
#       date. 4/18/2017
#

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data


def load_data(path_from_home='Sources/Python.d/TensorFlow/MNIST_data'):
    home_dir = os.environ.get('HOME')
    mnist_dir = os.path.join(home_dir, path_from_home)
    mnist = input_data.read_data_sets(mnist_dir, one_hot=False)

    return mnist

if __name__ == '__main__':
    
    np.random.seed(seed=2017)
    mnist = load_data()

    X_train = mnist.train.images
    y_train = mnist.train.labels

    X_test = mnist.test.images
    y_test = mnist.test.labels
    
    mms = MinMaxScaler()
    X_train_s = mms.fit_transform(X_train)
    # X_validation_s = mms.transform(X_validation)
    X_test_s = mms.transform(X_test)

    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.03
    rbm.n_iter = 30
    rbm.n_components = 2

    print('\nRBM Training...')
    rbm.fit(X_train_s)
    X_train_rbmfitted = rbm.transform(X_train_s)
    # X_validation_rbmfitted = rbm.transform(X_validation_s)
    X_test_rbmfitted = rbm.transform(X_test_s)

    print('x_test_rbmfitted = \n')
    print(X_test_rbmfitted[:20,:])

    assert False

    # check data shape
    # print('Shape of original X_train = ', X_train.shape)
    # print('Shape of X_train_rbmfitted = ', X_train_rbmfitted.shape)

    # plot latent space
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_rbmfitted[:, 0], X_test_rbmfitted[:, 1], c=y_test)
    plt.colorbar()
    # plt.show()
    plt.savefig('mnist_map.png')
    plt.close()
    