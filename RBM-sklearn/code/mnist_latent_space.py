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

MNIST_RBM_MODEL = '../work/mnist_rbm.pkl'


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
    rbm.n_iter = 3
    rbm.n_components = 500

    print('\nRBM Training...')
    rbm.fit(X_train_s)

    if os.path.exists(MNIST_RBM_MODEL):
        pass
    else:
        with open(MNIST_RBM_MODEL, 'wb') as fw:
            pickled = pickle.dumps(rbm.get_params())
            fw.write(pickled)
        print('data is saved.')

    kernel_plot(rbm)
