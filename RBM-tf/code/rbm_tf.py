#
#   rbm_tf.py
#       date. 4/14/2017
#

import os
import timeit
import numpy as np
import tensorflow as tf
import PIL.Image as Image
from utils import tile_raster_images
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.examples.tutorials.mnist import input_data

class RBM(object):
    '''
      Restricted Boltzmann Machine (RBM)  
    '''
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None):
        '''
          RBM constructor:
          args.:
            input:      input tensor, if TensorFlow, this is tf.placeholder
            n_visible:  number of visible units
            n_hidden:   number of hidden units
            W:          RBM parameter, shape is [n_visible, n_hidden]
            hbias:      RBM parameter, shape is [n_hidden]
            vbias:      RBM parameter, shape is [n_visible]
        '''
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        with tf.variable_scope('RBM'):
            if W is None:
                # using default initializer i.e. glorot_uniform_initializer
                W = tf.get_variable('W', [n_visible, n_hidden], 
                                    dtype=tf.float32)
            if hbias is None:
                init_hb = tf.zeros_initializer([n_hidden])
                hbias = tf.get_variable('hbias', [n_hidden],
                                    dtype=tf.float32, initializer=init_hb)
            if vbias is None:
                init_vb = tf.zeros_initializer([n_visible])
                vbias = tf.get_variable('vbias', [n_visible],
                                    dtype=tf.float32, initializer=init_vb)

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.params = [self.W, self.hbias, self.vbias]


    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''

        wx_b = tf.matmul(v_sample, self.W) + self.hbias
        vbias_term = tf.matmul(v_sample, tf.reshape(self.vbias, [-1, 1]))
        vbias_term = tf.squeeze(vbias_term)

        hidden_term = tf.reduce_sum(tf.log(1. + tf.exp(wx_b)), axis=1)
        tf.assert_equal(tf.shape(hidden_term), tf.shape(vbias_term))

        return -hidden_term - vbias_term
    

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units
        '''

        pre_sigmoid_activation = tf.matmul(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, tf.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given visible samples
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        dist = Bernoulli(probs=h1_mean, dtype=tf.float32)
        h1_sample = dist.sample()

        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units
        '''
        pre_sigmoid_activation = tf.matmul(hid, tf.transpose(self.W)) + self.vbias
        return [pre_sigmoid_activation, tf.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        dist = Bernoulli(probs=v1_mean, dtype=tf.float32)
        v1_sample = dist.sample()

        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state '''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state '''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]
    

    def grads_cd(self, v0_samples, h0_samples, nv_means, nh_samples):
        '''
          calculate parameter updates according to Contrastve Divergence
          method
        '''
        n_vis = self.n_visible
        n_hid = self.n_hidden

        dvbias = tf.reduce_mean(v0_samples - nv_means, axis=0)
        dhbias = tf.reduce_mean(h0_samples - nh_samples, axis=0)

        v0_samples = tf.reshape(v0_samples, [-1, n_vis, 1])
        h0_samples = tf.reshape(h0_samples, [-1, 1, n_hid])
        nv_means = tf.reshape(nv_means, [-1, n_vis, 1])
        nh_samples = tf.reshape(nh_samples, [-1, 1, n_hid])

        dW = tf.reduce_mean(tf.matmul(v0_samples, h0_samples)
                - tf.matmul(nv_means, nh_samples), axis=0)

        return [dW, dhbias, dvbias]


    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        '''
          This functions implements one step of CD-k (PCD-k)
        
          args.:
            lr      : learning rate used to train the RBM
            persistent: None for CD. For PCD, shared variable
                containing old state of Gibbs chain. This must be a shared
                variable of size (batch size, number of hidden units).
            k       : number of Gibbs steps to do in CD-k/PCD-k
        '''
        n_vis = self.n_visible
        n_hid = self.n_hidden
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        # compute negative phase
        tf.assert_rank(ph_sample, 2)
        _, pv_mean, _ = self.sample_v_given_h(ph_sample)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            # chain_start = persistent
            print('presistent is not implemented')
            raise ValueError

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        if k == 1:
            [pre_sigmoid_nvs, nv_means, nv_samples,
            pre_sigmoid_nhs, nh_means, nh_samples] = self.gibbs_hvh(chain_start)

        else:   # k > 1 case
            # prepare variables to execute while_loop process
            bat_size = tf.shape(self.input)[0]            
            v_zeros1 = tf.zeros([bat_size, n_vis], dtype=tf.float32)
            v_zeros2 = tf.zeros([bat_size, n_vis], dtype=tf.float32)
            v_zeros3 = tf.zeros([bat_size, n_vis], dtype=tf.float32)
            h_zeros1 = tf.zeros([bat_size, n_hid], dtype=tf.float32)
            h_zeros2 = tf.zeros([bat_size, n_hid], dtype=tf.float32)

            state = [v_zeros1, v_zeros2, v_zeros3, h_zeros1, h_zeros2, chain_start]
            i = tf.constant(0)      # loop counter
            i_1 = tf.constant(1)

            def body(i, state):     # output has to be same struct. of input
                i = tf.add(i, i_1)
                h_samples = state[-1]
                state = self.gibbs_hvh(h_samples)

                return i, state

            cond = lambda i, _: tf.less(i, k)

            final_i, final_state = tf.while_loop(cond, body, [i, state])
            [pre_sigmoid_nvs, nv_means, nv_samples,
            pre_sigmoid_nhs, nh_means, nh_samples] = final_state

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples
        cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(
                                                    self.free_energy(chain_end))

        gradients = self.grads_cd(self.input, ph_mean, nv_means, nh_means)

        W_ = self.W + lr * gradients[0]
        hbias_ = self.hbias + lr * gradients[1]
        vbias_ = self.vbias + lr * gradients[2]
        train_op = tf.group(
            (self.W).assign(W_),
            (self.hbias).assign(hbias_),
            (self.vbias).assign(vbias_))

        if persistent:
            print('presistent is not implemented')
            raise ValueError
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(pre_sigmoid_nvs)

        # return monitoring_cost
        return cost, monitoring_cost, train_op
        

    def get_reconstruction_cost(self, pre_sigmoid_nv):
        '''
          Approximation to the reconstruction error
        
          original code:
            cross_entropy = tf.reduce_mean(
                tf.reduce_sum(
                self.input * tf.log(tf.sigmoid(pre_sigmoid_nv)) +
                (1. - self.input) * tf.log(1. - tf.sigmoid(pre_sigmoid_nv)),
                axis=1))
        '''
        net_cost = tf.losses.sigmoid_cross_entropy(self.input, pre_sigmoid_nv)
        reconstruction_cost = -1. * net_cost * self.n_visible

        return reconstruction_cost


def plot_filters(sess, rbm, epoch, dirn='../work/rbm_imgs'):
    w_T = np.transpose(sess.run(rbm.W))
    image = Image.fromarray(
            tile_raster_images(
                X=w_T,
                # X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
    fname = 'filters_at_epoch_{}.png'.format(epoch)
    path = os.path.join(dirn, fname)
    image.save(path)


def test_rbm(learning_rate=0.1, training_epochs=10,
             batch_size=20, n_samples=10, n_hidden=500):
    '''
      Demonstrate how to train and afterwards sample from it on MNIST.
      args.:
        learning_rate:      learning rate used for training
        training_epochs:    number of epochs used for training
        batch_size:         size of a batch used to train the RBM
        n_samples:          number of samples to plot for each chain
    '''

    datasets = input_data.read_data_sets('../data', one_hot=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = datasets.train.labels.shape[0] // batch_size
    
    # allocate symbolic variables for the data
    # index = T.lscalar()    # index to a [mini]batch
    x = tf.placeholder(tf.float32, [None, 784])  

    # construct the RBM class
    rbm = RBM(input=x, n_visible=784, n_hidden=n_hidden)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, monitoring_cost, train_op = rbm.get_cost_updates(
                                    lr=learning_rate, persistent=None, k=15)
    # Training the RBM
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print('\nTraining...')
        # go through training epochs
        for epoch in range(training_epochs):
            e_start_time = timeit.default_timer()
            # go through the training set
            mean_mcost = []
                        
            for i in range(n_train_batches):
                batch_x, batch_y = datasets.train.next_batch(batch_size)
                train_fd = {x: batch_x}
                sess.run(train_op, feed_dict=train_fd)
                mon_cost_i = sess.run(monitoring_cost, feed_dict=train_fd)
                mean_mcost += [mon_cost_i]

            mean_mcost = np.asarray(mean_mcost)
            e_end_time = timeit.default_timer()
            epoch_time = e_end_time - e_start_time

            print('epoch{:>3d}, cost = {:>8.3f} (time = {:>8.2f} s)'.format(
                                epoch, np.mean(mean_mcost), epoch_time))
            plot_filters(sess, rbm, epoch)            
#

if __name__ == '__main__':
    test_rbm()
