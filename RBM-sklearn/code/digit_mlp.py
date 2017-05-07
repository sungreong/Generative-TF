#
#   digit_mlp.py - simple sklearn MLP classifier example
#

import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

digits = load_digits()
y = digits.target
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

clf = MLPClassifier(hidden_layer_sizes=(100, ), verbose=True)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accu = accuracy_score(y_test, y_pred)
print('accyracy = {:>.4f}'.format(accu))

confmat = confusion_matrix(y_test, y_pred)
print('\nconfusion matrix:')
print(confmat)
