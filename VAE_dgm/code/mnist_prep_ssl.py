#
#   mnist_prep_ssl.py
#       4/27/2017
#       prepare mnist dataset for semi-supervised leaning
#
import collections
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

Datasets4 = collections.namedtuple('Datasets4', 
                                   ['train_lab', 'train_unlab', 
                                    'validation', 'test'])
def params():
    '''
      set parameter to allocate data samples
    '''
    params = {}
    params['n_train_lab'] = 200
    params['n_val'] = 5000
    params['n_train_unlab'] = 60000 - 200 - 5000
    params['percent_limit'] = (0.7, 1.3)    # accept +/-30%

    return params


def load_data_ssl(params, dirn='../data'):
    '''
      load MNIST data and split into 4 blocks
    '''
    MAX_TRIAL = 999     # retry-max to check label balance
    percent_limit = params['percent_limit']

    n_train_lab = params['n_train_lab']
    n_train_unlab = params['n_train_unlab']
    n_val = params['n_val']
    mnist = input_data.read_data_sets(dirn, validation_size=n_val, 
                                      one_hot=True)
    n_train_total = mnist.train.num_examples
    if (n_train_lab + n_train_unlab) > n_train_total:
        raise ValueError('inconsistent parameters')

    X_train = mnist.train.images
    y_train = mnist.train.labels
    # prep. index list
    n_train = X_train.shape[0]

    num_trial = 0
    while num_trial < MAX_TRIAL:
        train_idx = np.random.permutation(n_train)
        # split train data into 2 (labeled / unlabeld)
        X_train_lab = X_train[train_idx[:n_train_lab]]
        y_train_lab = y_train[train_idx[:n_train_lab]]
        X_train_unlab = X_train[train_idx[n_train_lab:]]
        y_train_unlab = y_train[train_idx[n_train_lab:]]
        # check balance of label 
        if check_label_balance(y_train_lab, percent_limit=percent_limit):
            break
        num_trial += 1
        if num_trial == MAX_TRIAL:
            raise ValueError('percentage range looks too narrow.')

    # cancel scaling of DataSet class constructor
    X_train_lab = X_train[train_idx[:n_train_lab]] * 255.
    X_train_unlab = X_train[train_idx[n_train_lab:]] * 255.

    train_lab = DataSet(X_train_lab, y_train_lab, reshape=False)
    train_unlab = DataSet(X_train_unlab, y_train_unlab, reshape=False)

    mnist_ssl = Datasets4(train_lab=train_lab, 
                          train_unlab=train_unlab,
                          validation=mnist.validation,
                          test=mnist.test)
    return mnist_ssl


def check_label_balance(y_label_np, percent_limit=(0.85, 1.15)):
    '''
      check label balance of sampled data
      args:
        y_label_np    - label array (one-hot encoded)
        percent_limit - limit percent of each label sample
                        if it set (0.85, 1.15), acceptable percent is +/-15%
    '''
    n_sample = y_label_np.shape[0]
    counts = np.zeros([10], dtype=np.int32)
    for i in range(n_sample):
        label_i = np.argmax(y_label_np[i, :])
        counts[label_i] += 1
    
    cnt_max = np.max(counts)
    cnt_max_label = np.argmax(counts)
    cnt_min = np.min(counts)
    cnt_min_label = np.argmin(counts)
  
    upper_limit = cnt_max * 1. / (n_sample / 10.)
    lower_limit = cnt_min * 1. / (n_sample / 10.)
    judge = (percent_limit[0] < lower_limit) and (
                        upper_limit < percent_limit[1])

    return judge


def test_load_data_ssl(dirn):
    data_alloc = params()
    mnist = load_data_ssl(data_alloc, dirn=dirn)
    # read data
    print('\nSome test results:')
    print('num samples of train data without label = ',
        mnist.train_lab.num_examples)
    print('num samples of train data w/ label = ',
        mnist.train_unlab.num_examples)
    print('num samples of validation data = ',
        mnist.validation.num_examples)
    print('num samples of test data = ',
        mnist.test.num_examples)

    # next_batch
    batch_x, batch_y = mnist.train_lab.next_batch(100)
    batch_x_unlab, _ = mnist.train_unlab.next_batch(100)

    print('shape of batch_x = ', batch_x.shape)
    print('shape of batch_y = ', batch_y.shape)
    print('shape of batch_x(unlabelled) = ', batch_x_unlab.shape)


if __name__ == '__main__':
    test_load_data_ssl(dirn='../data')
