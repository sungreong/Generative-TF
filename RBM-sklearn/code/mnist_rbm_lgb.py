#
#   mnist_rbm_lgb.py
#       date. 4/12/2017, 5/11
#

import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data

N_LABEL = 1000
N_UNLAB = 50000
N_CV = 1000

def load_data(mnist_dir='../data'):
    mnist = input_data.read_data_sets(mnist_dir, one_hot=False)

    return mnist

if __name__ == '__main__':
    
    np.random.seed(seed=2017)
    mnist = load_data()

    X_train0 = mnist.train.images
    y_train0 = mnist.train.labels
    X_train_lab = X_train0[:N_LABEL]
    y_train_lab = y_train0[:N_LABEL]
    X_train_unlab = X_train0[N_LABEL:N_UNLAB]
    X_validation = mnist.validation.images[:N_CV]
    y_validation = mnist.validation.labels[:N_CV]
    X_test = mnist.test.images
    y_test = mnist.test.labels
    
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.03
    rbm.n_iter = 10
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 500
    print('\nRBM Training...')
    rbm.fit(X_train_unlab)      # train by unlabelled data

    X_train_rbmfitted = rbm.transform(X_train_lab)
    X_validation_rbmfitted = rbm.transform(X_validation)
    X_test_rbmfitted = rbm.transform(X_test)

    gbm = lgb.LGBMClassifier(
        objective='multiclass',
        num_leaves=63,
        learning_rate=0.01,
        n_estimators=1000)

    gbm.fit(X_train_rbmfitted, y_train_lab,         # train by labbelled data
        eval_set=[(X_validation_rbmfitted, y_validation)],
        eval_metric='multi_logloss',
        early_stopping_rounds=10)
    y_pred = gbm.predict(X_test_rbmfitted, num_iteration=gbm.best_iteration)

    accu = accuracy_score(y_test, y_pred)
    print('accuracy = {:>.4f}'.format(accu))

    # result 
    # Early stopping, best iteration is:
    # [791]	valid_0's multi_logloss: 0.340837
    # accuracy = 0.9039
