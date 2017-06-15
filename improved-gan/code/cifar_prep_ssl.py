#
#   cifar_prep_ssl.py
#       6/15/2017
#       prepare CIFAR-10 dataset for semi-supervised leaning
#

import collections
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

Datasets4 = collections.namedtuple('Datasets4', 
                                   ['train_lab', 'train_unlab', 
                                    'validation', 'test'])
def params():
    '''
      set parameter to allocate data samples
    '''
    params = {}
    params['n_train_lab'] = 100
    params['n_val'] = 10000
    params['n_train_unlab'] = 50000 - 100 - 10000   # total sample 50000
    params['mode'] = None

    return params


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


def load_data_ssl(params, dirn='../data'):
    '''
      load CIFAR-10 data and split into 4 blocks
    '''
    # parameter set
    n_train_lab = params['n_train_lab']
    n_train_unlab = params['n_train_unlab']
    n_val = params['n_val']
    mode = params['mode']

    dirn = '../data'
    X_train, y_train0 = load(dirn, subset='train')
    X_test, y_test0 = load(dirn, subset='test')
    y_train = onehot_label(y_train0)
    y_test = onehot_label(y_test0)

    # step.1 - split validation set
    X_train_tot, X_validation, y_train_tot, y_validation = \
        random_sampling(X_train, y_train, n_labeled=n_val)
    n_train_total = X_train_tot.shape[0]

    # step.2 - split train set into labeled / unlabeled
    if (n_train_lab + n_train_unlab) > n_train_total:
        raise ValueError('inconsistent parameters')

    # select bin_sampling or random_sampling (including inbalance)
    if mode == 'random':
        X_train_lab, X_train_unlab, y_train_lab, y_train_unlab = \
            random_sampling(X_train_tot, y_train_tot, n_labeled=n_train_lab)
    else:
        X_train_lab, X_train_unlab, y_train_lab, y_train_unlab = \
            bin_sampling(X_train_tot, y_train_tot, n_labeled=n_train_lab)

    # cancel scaling by DataSet class constructor
    train_lab = DataSet(X_train_lab, y_train_lab, reshape=False)
    train_unlab = DataSet(X_train_unlab, y_train_unlab, reshape=False)
    validation_set = DataSet(X_validation, y_validation, reshape=False)
    test_set = DataSet(X_test, y_test, reshape=False)

    cifar10_ssl = Datasets4(train_lab=train_lab, 
                          train_unlab=train_unlab,
                          validation=mnist.validation,
                          test=mnist.test)
    return cifar10_ssl


def bin_sampling(X_train, y_train, n_labeled=100):
    '''
      sampling to make each label same number
      args:
        X_train, y_train    : source dataset
        n_labeled           : total labeled data
    '''
    num_examples = X_train.shape[0]
    lab_each = n_labeled // 10
    cnt = np.zeros([10,], dtype=np.int)
    index_picked = [[] for lab in range(10)]

    for index, y in enumerate(y_train):
        y_i = np.argmax(y)
        if cnt[y_i] < lab_each:
            cnt[y_i] += 1
            index_picked[y_i].append(index)

        if np.min(cnt) == lab_each:
            break

    index_picked = np.asarray(index_picked)
    index_picked = index_picked.ravel()
    
    index_not_picked = set(np.arange(num_examples)) - set(index_picked)
    index_not_picked = np.asarray(list(index_not_picked))

    X_lab = X_train[index_picked]
    y_lab = y_train[index_picked]
    X_unlab = X_train[index_not_picked]
    y_unlab = y_train[index_not_picked]

    return X_lab, X_unlab, y_lab, y_unlab


def check_label_balance(y_label_np, percent_limit=(0.8, 1.2)):
    '''
      check label balance of sampled data
      args:
        y_label_np    - label array (one-hot encoded)
        percent_limit - limit percent of each label sample
                        if set (0.85, 1.15), acceptable percent is +/-15%
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


def random_sampling(X_train, y_train, n_labeled=100):
    '''
      sampling data by random
    '''
    MAX_TRIAL = 999     # retry-max to check label balance
    n_train = X_train.shape[0]

    num_trial = 0
    while num_trial < MAX_TRIAL:
        # use numpy.random.permutation
        train_idx = np.random.permutation(n_train)
        # split train data into 2 (labeled / unlabeld)
        X_lab = X_train[train_idx[:n_labeled]]
        y_lab = y_train[train_idx[:n_labeled]]
        X_unlab = X_train[train_idx[n_labeled:]]
        y_unlab = y_train[train_idx[n_labeled:]]
        # check balance of label 
        if check_label_balance(y_lab):
            break
        num_trial += 1
        if num_trial == MAX_TRIAL:
            raise ValueError('percentage range looks too narrow.')    

    return X_lab, X_unlab, y_lab, y_unlab


def test_load_data_ssl(dirn):
    data_alloc = params()
    cifar = load_data_ssl(data_alloc, dirn=dirn)
    # read data
    print('\nSome test results:')
    print('num samples of train data without label = ',
        cifar.train_unlab.num_examples)
    print('num samples of train data w/ label = ',
        cifar.train_lab.num_examples)
    print('num samples of validation data = ',
        cifar.validation.num_examples)
    print('num samples of test data = ',
        cifar.test.num_examples)

    # next_batch
    batch_x, batch_y = cifar.train_lab.next_batch(100)
    batch_x_unlab, _ = cifar.train_unlab.next_batch(100)
    batch_xv, batch_yv = cifar.validation.next_batch(100)

    print('shape of batch_x = ', batch_x.shape)
    print('shape of batch_y = ', batch_y.shape)
    print('shape of batch_x(unlabelled) = ', batch_x_unlab.shape)
    print('shape of batch_xv = ', batch_xv.shape)
    print('shape of batch_yb = ', batch_yv.shape)


if __name__ == '__main__':
    # test_load_data_ssl(dirn='../data')
    dirn = '../data'
    test_load_data_ssl(dirn=dirn)
