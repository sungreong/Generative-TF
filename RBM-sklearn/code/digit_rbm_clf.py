#
#   digit_rbm_clf.py 
#       sklearn RBM - MLP net test code
#       date. 4//17/2017
#

import numpy as np
from scipy.ndimage import convolve

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier


def load_data():
    digits = load_digits()
    y = digits.target
    X = np.asarray(digits.data, 'float32')
    X, y = nudge_dataset(X, y)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0)

    X_no_lab, X_labeled, _, y_labeled = train_test_split(
                X_train, y_train, test_size=0.2, random_state=0)

    return X_no_lab, X_labeled, X_test, y_labeled, y_test


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


if __name__ == '__main__':
    X_no_lab, X_labeled, X_test, y_labeled, y_test = load_data()

    print('number of X_no_lab  = ', X_no_lab.shape[0])
    print('number of X_labeled = ', X_labeled.shape[0])
    print('number of X_test    = ', X_test.shape[0])

    # RBM fitting process
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.03
    rbm.n_iter = 5
    rbm.n_components = 128

    print('\nRBM training...')
    rbm.fit(X_no_lab)

    # Compute the hidden layer activation probabilities, P(h=1|v=X)
    X_xformed = rbm.transform(X_labeled)
    X_test_xformed = rbm.transform(X_test)

    # MLP classification process
    print('\nMLP training...')
    clf = MLPClassifier(hidden_layer_sizes=(100, ),
                        alpha=1.e-3,        # L2 penalty 
                        max_iter=200,
                        verbose=False)
    clf.fit(X_xformed, y_labeled)

    y_pred = clf.predict(X_test_xformed)

    accu = accuracy_score(y_test, y_pred)
    print('accyracy = {:>.4f}'.format(accu))

    confmat = confusion_matrix(y_test, y_pred)
    print('\nconfusion matrix:')
    print(confmat)
