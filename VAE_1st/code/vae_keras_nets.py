#
#   vae_keras_nets.py
#       date. 3/21/2017
#       date. 4/4/2017 - switch library "Keras" to "tf.contrib.keras"
#   

import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Dropout, Activation
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

class VariationalAutoencoder(object):
    """ 
    Variation Autoencoder (VAE) with an sklearn-like interface 
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    """
    def __init__(self, network_architecture, 
                    learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, 
                        [None, network_architecture['n_input']])        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        

    def _create_network(self):
        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent space
        self.z_mean, self.z_log_sigma_sq = self._encoder(
                                            **self.network_architecture)
        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture['n_z']
        eps = tf.random_normal((self.batch_size, n_z), mean=0., stddev=1., 
                                                        dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                    tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self._decoder(**self.network_architecture)


    def _encoder(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1, n_hidden_gener_2, 
                            n_input, n_z):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        net = Dense(units=n_hidden_recog_1, activation='softplus')(self.x)
        net = Dense(units=n_hidden_recog_2, activation='softplus')(net)
        z_mean = Dense(units=n_z, activation='linear')(net)
        z_log_sigma_sq = Dense(units=n_z, activation='linear')(net)
        
        return z_mean, z_log_sigma_sq

    def _decoder(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1, n_hidden_gener_2, 
                            n_input, n_z):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        net = Dense(units=n_hidden_gener_1, activation='softplus')(self.z)
        net = Dense(units=n_hidden_gener_2, activation='softplus')(net)
        x_reconstr_mean = Dense(units=n_input, activation='linear')(net)

        return x_reconstr_mean
        
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        reconstr_loss = tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=self.x, logits=self.x_reconstr_mean), 1)
        
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + kl_loss)   # average over batch
        
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X, sess):
        '''
          Train model based on mini-batch of input data.
          Return cost of mini-batch. This function is invoked from train()
        '''
        opt, cost = sess.run((self.optimizer, self.cost), 
                                                    feed_dict={self.x: X})
        return cost

    def transform(self, sess, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution

        return sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, sess, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        activated = tf.sigmoid(self.x_reconstr_mean)

        return sess.run(activated, feed_dict={self.z: z_mu})
    
    def reconstruct(self, sess, X):
        """ Use VAE to reconstruct given data. """
        activated = tf.sigmoid(self.x_reconstr_mean)
        return sess.run(activated, feed_dict={self.x: X})

    def train(self, sess, dataset, training_epochs=10, display_step=5):
        batch_size = self.batch_size
        n_samples = dataset.train._num_examples
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = dataset.train.next_batch(batch_size)

                # Fit training using batch data
                cost = self.partial_fit(batch_xs, sess)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), 
                    "cost=", "{:.9f}".format(avg_cost))


if __name__ == '__main__':
    # Load MNIST dataset
    mnist = input_data.read_data_sets('../data', one_hot=True)
    n_samples = mnist.train.num_examples

    network_architecture = \
        dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
         n_hidden_recog_2=500,      # 2nd layer encoder neurons
         n_hidden_gener_1=500,      # 1st layer decoder neurons
         n_hidden_gener_2=500,      # 2nd layer decoder neurons
         n_input=784,               # MNIST data input (img shape: 28*28)
         n_z=20)                    # dimensionality of latent space
    
    with tf.variable_scope('vae_1'):
        vae1 = VariationalAutoencoder(network_architecture, 
                        learning_rate=0.001, batch_size=100)
    # Initializing the tensor flow variables
    init = tf.global_variables_initializer()

    # Launch the session
    sess = tf.InteractiveSession()
    # self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(init)
    print('Train model 1 ...')
    vae1.train(sess, mnist, training_epochs=75)

    # --- snippet 7

    x_sample = mnist.test.next_batch(100)[0]
    x_reconstruct = vae1.reconstruct(sess, x_sample)

    fig1 = plt.figure(figsize=(8, 12))
    for i in range(5):

        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('../work/vae_ae.png')

    plt.close()

    # --- snippet 8

    network_architecture = \
        dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
         n_hidden_recog_2=500,      # 2nd layer encoder neurons
         n_hidden_gener_1=500,      # 1st layer decoder neurons
         n_hidden_gener_2=500,      # 2nd layer decoder neurons
         n_input=784,       # MNIST data input (img shape: 28*28)
         n_z=2)             # dimensionality of latent space

    with tf.variable_scope('vae_2'):
        vae2 = VariationalAutoencoder(network_architecture, 
                        learning_rate=0.001, batch_size=100)
    
    vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='vae_2')
    init2 = tf.variables_initializer(vars)
    sess.run(init2)
    print('Train model 2 ...')
    vae2.train(sess, mnist, training_epochs=75)

    # --- snippet 9

    x_sample, y_sample = mnist.test.next_batch(5000)
    z_mu = vae2.transform(sess, x_sample)
    fig2 = plt.figure(figsize=(8, 6)) 
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.grid()
    plt.savefig('../work/class_plot.png')
    plt.close()

    # --- snippet 10

    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]]*vae1.batch_size)
            x_mean = vae2.generate(sess, z_mu)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

    fig3 = plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()

    plt.savefig('../work/canvas.png')
    plt.close()
    

