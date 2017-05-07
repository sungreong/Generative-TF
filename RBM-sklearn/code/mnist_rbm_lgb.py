#
#   mnist_rbm_lgb.py
#       date. 4/12/2017
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
N_CV = 1000

def load_data(path_from_home='Sources/Python.d/TensorFlow/MNIST_data'):
    home_dir = os.environ.get('HOME')
    mnist_dir = os.path.join(home_dir, path_from_home)
    mnist = input_data.read_data_sets(mnist_dir, one_hot=False)

    return mnist

if __name__ == '__main__':
    
    np.random.seed(seed=2017)
    mnist = load_data()

    X_train0 = mnist.train.images
    y_train0 = mnist.train.labels
    X_train = X_train0[:N_LABEL]
    y_train = y_train0[:N_LABEL]

    X_train_to_rbm = X_train0[N_LABEL:]
    X_validation = mnist.validation.images[:N_CV]
    y_validation = mnist.validation.labels[:N_CV]
    X_test = mnist.test.images
    y_test = mnist.test.labels
    
    mms = MinMaxScaler()
    X_train_s = mms.fit_transform(X_train)
    X_validation_s = mms.transform(X_validation)
    X_test_s = mms.transform(X_test)

    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.03
    rbm.n_iter = 30
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 500
    print('\nRBM Training...')
    rbm.fit(X_train_to_rbm)
    X_train_rbmfitted = rbm.transform(X_train_s)
    X_validation_rbmfitted = rbm.transform(X_validation_s)
    X_test_rbmfitted = rbm.transform(X_test_s)

    # check data shape
    print('Shape of original X_train = ', X_train.shape)
    print('Shape of X_train_rbmfitted = ', X_train_rbmfitted.shape)

    gbm = lgb.LGBMClassifier(
        objective='multiclass',
        num_leaves=63,
        learning_rate=0.01,
        n_estimators=1000)

    gbm.fit(X_train_rbmfitted, y_train,
        eval_set=[(X_validation_rbmfitted, y_validation)],
        eval_metric='multi_logloss',
        early_stopping_rounds=10)
    y_pred = gbm.predict(X_test_rbmfitted, num_iteration=gbm.best_iteration)
    # y_pred_proba = gbm.predict_proba(X_test_s, num_iteration=gbm.best_iteration)

    accu = accuracy_score(y_test, y_pred)
    print('accuracy = {:>.4f}'.format(accu))

    # result 
    #    condition: rbm.learning_rate = 0.06, rbm.n_iter = 10, rbm.n_components = 1800
    # [996]	valid_0's multi_logloss:1.01218
    # [997]	valid_0's multi_logloss:1.0116
    # [998]	valid_0's multi_logloss:1.01103
    # [999]	valid_0's multi_logloss:1.01046
    # [1000]	valid_0's multi_logloss:1.00989
    # accuracy = 0.8708

    # result
    # (LightGBM parameter is changed.)
    # accuracy = 0.8993
    
    
