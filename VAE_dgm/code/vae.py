#
#   vae.py
#       date. 2/25/2017, 3/15/2017
#       https://github.com/jmetzen/jmetzen.github.com/blob/master/notebooks/vae.ipynb
#

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(0)
tf.set_random_seed(0)

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:

from tensorflow.python.layers import layers
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data', one_hot=True)
n_samples = mnist.train.num_examples


def xavier_init(fan_in, fan_out, constant=1): 
    '''
      Xavier initialization of network weights
    '''
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

# Full-connected Layer   
class VAE_fc(object):
    def __init__(self, input, n_in, n_out, activation='relu', vs='var_scope'):
        self.input = input

        with tf.variable_scope(vs):    
            w_init = xavier_init(n_in, n_out)
            w_h = tf.get_variable('w', initializer=w_init)

            b_init = tf.zeros_initializer([n_out])
            b_h = tf.get_variable('b', [n_out], initializer=b_init)
        self.w = w_h
        self.b = b_h
        self.activation = activation
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        if self.activation in ['relu', 'softplus', 'sigmoid']:
            if self.activation == 'relu':
                self.output = tf.nn.relu(linarg)
            if self.activation == 'softplus':
                self.output = tf.nn.softplus(linarg)
            if self.activation == 'sigmoid':
                self.output = tf.nn.sigmoid(linarg)
        else:
            self.output = linarg

        return self.output



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
        self.z_mean, self.z_log_sigma_sq = self._recognition_network(
                                            **self.network_architecture)

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture['n_z']
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self._generator_network(
                                        **self.network_architecture)


    def _recognition_network(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1, n_hidden_gener_2, 
                            n_input, n_z):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        net = tf.layers.dense(self.x, n_hidden_recog_1, 
                                activation=tf.nn.softplus, name='recog/layer1')
        net = tf.layers.dense(net, n_hidden_recog_2,
                                activation=tf.nn.softplus, name='recog/layer2')
        z_mean = tf.layers.dense(net, n_z,
                                activation=None, name='recog/zmean')
        z_log_sigma_sq = tf.layers.dense(net, n_z,
                                activation=None, name='recog/zlog_sig_sq')
        
        return z_mean, z_log_sigma_sq



    def _generator_network(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1, n_hidden_gener_2, 
                            n_input, n_z):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        net = tf.layers.dense(self.z, n_hidden_gener_1,
                                activation=tf.nn.softplus, name='gener/layer1')
        net = tf.layers.dense(net, n_hidden_gener_2,
                                activation=tf.nn.softplus, name='gener/layer2')
        x_reconstr_mean = tf.layers.dense(net, n_input,
                                activation=None, name='gener/reconstr_mean')

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
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})

        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X, sess):
        """ Use VAE to reconstruct given data. """
        return sess.run(tf.sigmoid(self.x_reconstr_mean), 
                             feed_dict={self.x: X})


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Initializing the tensor flow variables
    init = tf.global_variables_initializer()

    # Launch the session
    sess = tf.InteractiveSession()
    # self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs, sess)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(avg_cost))
    return vae, sess


if __name__ == '__main__':
    network_architecture = \
        dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

    vae, sess = train(network_architecture, training_epochs=75)

    # --- snippet 7

    x_sample = mnist.test.next_batch(100)[0]
    x_reconstruct = vae.reconstruct(x_sample, sess)

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
    plt.savefig('vae_ae.png')

    plt.close()

    assert False
    # --- snippet 8

    network_architecture = \
        dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
         n_hidden_recog_2=500,      # 2nd layer encoder neurons
         n_hidden_gener_1=500,      # 1st layer decoder neurons
         n_hidden_gener_2=500,      # 2nd layer decoder neurons
         n_input=784,       # MNIST data input (img shape: 28*28)
         n_z=2)             # dimensionality of latent space

    vae_2d = train(network_architecture, training_epochs=75)        # this might make error.
                                                                    # due to scope reuse
    # --- snippet 9

    x_sample, y_sample = mnist.test.next_batch(5000)
    z_mu = vae_2d.transform(x_sample)
    fig2 = plt.figure(figsize=(8, 6)) 
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.grid()
    plt.savefig('class_plot.png')
    plt.close()

    # --- snippet 10

    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]]*vae.batch_size)
            x_mean = vae_2d.generate(z_mu)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

    fig3 = plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()

    plt.savefig('canvas.png')
    plt.close()
    

