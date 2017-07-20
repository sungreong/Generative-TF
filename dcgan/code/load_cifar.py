#
#   load_cifar.py
#       date. 7/18/2017
#       load CIFAR-10 dataset
#

import os
import pickle
import collections
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

Datasets = collections.namedtuple('Datasets', 
                                  ['train', 'validation', 'test'])
NUM_CLASSES = 10

def unpickle(file):
    '''
      extract cifar-10 dataset from python pickle file
    '''
    fo = open(file, 'rb')
    d = pickle.load(fo, encoding='bytes')
    # d = cPickle.load(fo)
    fo.close()
    x = d[b'data'].reshape([-1, 3, 32, 32])
    # x = (x - 127.5) / 128.0
    x = np.asarray(x, dtype=np.float32)
    y = d[b'labels']
    y = np.asarray(y, dtype=np.uint8)    

    return {'x': x, 'y': y}


def load(data_dir, subset='train'):
    '''
      load cifar-10 dataset from specified directory
    '''
    ext = '.bin'
    if subset=='train':
        fn = 'cifar-10-batches-py/data_batch_'
        train_data = [unpickle(os.path.join(
                    data_dir, fn + str(i))) for i in range(1, 6)]
        trainx = np.concatenate([d['x'] for d in train_data], axis=0)
        trainy = np.concatenate([d['y'] for d in train_data], axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(
                                data_dir, 'cifar-10-batches-py/test_batch'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')


def onehot_label(labels):
    '''
      one-hot label encoding
    '''
    n_sample = labels.shape[0]
    oh = np.zeros([n_sample, NUM_CLASSES], dtype=np.float32)
    for i in range(n_sample):
        oh[i, labels[i]] = 1.0
    
    return oh


def random_sampling(X_train, y_train, n_validation=5000):
    '''
      sampling data by random
    '''
    n_train = X_train.shape[0]

    if n_train < 0:
        raise ValueError('n_train is not valid.')

    # use numpy.random.permutation
    train_idx = np.random.permutation(n_train)
    # split train data into 2 (labeled / unlabeld)
    X_train_subset = X_train[train_idx[n_validation:]]
    y_train_subset = y_train[train_idx[n_validation:]]
    X_val = X_train[train_idx[:n_validation]]
    y_val= y_train[train_idx[:n_validation]]

    return X_train_subset, y_train_subset,X_val, y_val


def load_data(dirn='../data'):
    '''
      load CIFAR-10 data and split into 3 blocks
    '''
    # parameter set
    n_train = 40000
    n_val = 10000

    dirn = '../data'
    X_train0, y_train0 = load(dirn, subset='train')
    X_test, y_test0 = load(dirn, subset='test')
    print('Files are loaded.')
    y_train1 = onehot_label(y_train0)
    y_test = onehot_label(y_test0)

    # split validation set
    X_train, y_train, X_validation, y_validation = \
        random_sampling(X_train0, y_train1, n_validation=n_val)
    '''
    print('X_train: ', X_train.shape, ', ', type(X_train))
    print('X_validation: ', X_validation.shape, ', ', type(X_validation))
    '''
  
    # matrix transpose (channel 1st -> channel last)
    X_train = np.transpose(X_train, (0, 2, 3, 1))
    X_validation = np.transpose(X_validation, (0, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 2, 3, 1))

    # DataSet class construction
    train_lab = DataSet(X_train, y_train, reshape=False)
    validation_set = DataSet(X_validation, y_validation, reshape=False)
    test_set = DataSet(X_test, y_test, reshape=False)

    cifar10 = Datasets(train=train_lab, 
                        validation=validation_set,
                        test=test_set)
    return cifar10
