#
#   model.py
#       model definition using "tf.layers" API - network is adapted to CIFAR-10
#       date. 7/20/2017
#

import numpy as np
import tensorflow as tf

def conv_cond_concat(x, y, batch_size):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    
    return tf.concat([
        x, y*tf.ones([batch_size, x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


class Generator:
    def __init__(self, depths=[256, 128, 64], s_size=4, batch_size=128, 
                 y_dim=None, vs='generator'):
        '''
          args.:
            depths: list of image depth(channel)
            s_size: starting image size (i.e. starting [4, 4, 1024])
        '''
        output_depth = 3    # for cifar-10 data
        self.vs = vs        # variable scope
        self.depths = depths + [output_depth]
        self.s_size = s_size
        self.batch_size = batch_size
        self.y_dim = y_dim
        if y_dim != None:
            self.with_label = True
        else:
            self.with_label = False
        self.gfc_dim = 1024
        self.reuse = False

    def __call__(self, z, label=None, training=False):
        z = tf.convert_to_tensor(z)
        y_dim = self.y_dim
        batch_size = self.batch_size
        gf_dim = 64     # Dimension of gen filters in first conv layer. [64]
        if not self.with_label:
            with tf.variable_scope(self.vs, reuse=self.reuse):
                # reshape from inputs
                with tf.variable_scope('reshape'):
                    net = tf.layers.dense(z, self.depths[0] * self.s_size * self.s_size)
                    net = tf.reshape(net, [-1, self.s_size, self.s_size, self.depths[0]])   # [-1, 4, 4, 256]
                    net = tf.nn.relu(tf.layers.batch_normalization(net, training=training))
                # deconvolution (transpose of convolution) x 4
                with tf.variable_scope('deconv1'):
                    net = tf.layers.conv2d_transpose(net, self.depths[1], (5, 5), 
                                                     strides=(2, 2), padding='SAME')        # [-1, 8, 8, 128]
                    net = tf.nn.relu(tf.layers.batch_normalization(net, training=training))
                with tf.variable_scope('deconv2'):
                    net = tf.layers.conv2d_transpose(net, self.depths[2], (5, 5),
                                                     strides=(2, 2), padding='SAME')        # [-1, 16, 16, 64]
                    net = tf.layers.batch_normalization(net, training=training)
                with tf.variable_scope('deconv3'):
                    net = tf.layers.conv2d_transpose(net, self.depths[3], (5, 5),
                                                     strides=(2, 2), padding='SAME')        # [-1, 32, 32, 3]
                    net = tf.layers.batch_normalization(net, training=training)
                output = net

        else:   # generator with Labels
            with tf.variable_scope('g_cond', reuse=self.reuse):
                # image size
                s_output = 32
                s_output2 = int(s_output/2) # 16
                s_output4 = int(s_output/4) #  8

                # reshape from inputs
                with tf.variable_scope('linear1'):
                    yb = tf.reshape(label, [self.batch_size, 1, 1, y_dim])
                    net = tf.concat([z, label], axis=1)
                    net = tf.layers.dense(z, self.gfc_dim, activation=None)
                    net = tf.nn.relu(tf.layers.batch_normalization(net, training=training))
                    net = tf.concat([net, label], axis=1)   # h0

                with tf.variable_scope('linear2'):
                    net = tf.layers.dense(net, gf_dim*2*s_output4*s_output4, activation=None)
                    net = tf.nn.relu(tf.layers.batch_normalization(net, training=training))
                    net = tf.reshape(net, [self.batch_size, s_output4, s_output4, self.depths[1]])  # [-1, 8, 8, 128]
                    net = conv_cond_concat(net, yb, batch_size)

                # deconvolution (transpose of convolution)
                with tf.variable_scope('deconv1'):
                    net = tf.layers.conv2d_transpose(net, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                    net = tf.nn.relu(tf.layers.batch_normalization(net, training=training))     # [-1, 16, 16, 64]
                    net = conv_cond_concat(net, yb, batch_size)

                # output images
                with tf.variable_scope('deconv3'):
                    net = tf.layers.conv2d_transpose(net, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                    output = net
        
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        
        return output


class Discriminator:
    def __init__(self, depths=[64, 128, 256], batch_size=128, 
                 y_dim=None, vs='discriminator'):
        input_depth = 3     # for cifar-10 data
        self.depths = [input_depth] + depths
        self.reuse = False
        self.batch_size = batch_size
        self.y_dim = y_dim
        if y_dim != None:
            self.with_label = True
        else:
            self.with_label = False
        self.vs = vs
        self.dfc_dim = 1024

    def __call__(self, inputs, label=None, training=False):
        def leaky_relu(x, leak=0.2, name='lrelu'):
            return tf.maximum(x, x * leak, name=name)

        batch_size = self.batch_size
        if not self.with_label:
            with tf.variable_scope(self.vs, reuse=self.reuse):
                # convolution x 4
                with tf.variable_scope('conv1'):
                    net = tf.layers.conv2d(inputs, self.depths[1], (5, 5), 
                                           strides=(2, 2), padding='same')  # [-1, 16, 16, 64]
                    net = leaky_relu(tf.layers.batch_normalization(net, training=training))
                with tf.variable_scope('conv2'):
                    net = tf.layers.conv2d(net, self.depths[2], (5, 5),
                                           strides=(2, 2), padding='same')  # [-1, 8, 8, 128]
                    net = leaky_relu(tf.layers.batch_normalization(net, training=training))
                with tf.variable_scope('conv3'):
                    net = tf.layers.conv2d(net, self.depths[3], (5, 5),
                                           strides=(2, 2), padding='same')  # [-1, 4, 4, 256]
                    net = leaky_relu(tf.layers.batch_normalization(net, training=training))

                with tf.variable_scope('classify'):
                    # batch_size = outputs.get_shape()[0].value
                    reshaped = tf.reshape(net, [batch_size, 4*4*256])
                    output = tf.layers.dense(reshaped, 1, activation=None, name='logits')
            
        else:   # discriminator with Labels
            with tf.variable_scope('d_cond', reuse=self.reuse):
                yb = tf.reshape(label, [batch_size, 1, 1, self.y_dim])
                net = conv_cond_concat(inputs, yb, batch_size)

                with tf.variable_scope('conv1'):
                    net = tf.layers.conv2d(net, self.depths[1], [5, 5],
                                           strides=(2, 2), padding='same')
                    net = leaky_relu(net)
                    net = conv_cond_concat(net, yb, batch_size)

                with tf.variable_scope('conv2'):
                    net = tf.layers.conv2d(net, self.depths[2], [5, 5],
                                           strides=(2, 2), padding='same')
                    net = leaky_relu(tf.layers.batch_normalization(net, training=training))
                    net = tf.reshape(net, [batch_size, -1])
                    net = tf.concat([net, label], 1)

                with tf.variable_scope('dense1'):              
                    net = tf.layers.dense(net, self.dfc_dim, activation=None)
                    net = leaky_relu(tf.layers.batch_normalization(net, training=training))
                    net = tf.concat([net, label], 1)

                output = tf.layers.dense(net, 1, activation=None, name='d_before_sigmoid')
                # outputs = tf.nn.sigmoid(net)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')

        return output


class DCGAN:
    def __init__(self,
                 batch_size=128, s_size=4, z_dim=100, y_dim=None,
                 g_depths=[256, 128, 64],
                 d_depths=[64, 128, 256]):
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.g = Generator(depths=g_depths, s_size=self.s_size, y_dim=y_dim)
        self.d = Discriminator(depths=d_depths, batch_size=batch_size, y_dim=y_dim)

    # Noise generator
    def get_noise(self):
        return tf.random_uniform([self.batch_size, self.z_dim], 
                                 minval=-1.0, maxval=1.0)

    # inference by feedforward
    def inference(self, data_x, data_y):
        '''
          args:
            traindata: 4-D Tensor of shape `[batch, height, width, channels]`.
        '''
        generated = self.g(self.get_noise(), label=data_y, training=True)
        d_on_generated = self.d(generated, label=data_y, training=True)
        d_on_givendata = self.d(data_x, label=data_y, training=True)

        logits_list = [generated, d_on_generated, d_on_givendata]

        return logits_list

    def debug_output_shapes(self, traindata):

        generated = self.g(self.get_noise(), training=True)
        d_on_generated = self.d(generated, training=True)

        shape1 = tf.shape(generated)        # [128, 784]
        shape2 = tf.shape(d_on_generated)
        shape3 = tf.shape(traindata)

        return shape1, shape2, shape3


    def loss(self, logits_list):
        '''
          build models, calculate losses.
        '''
        _, d_on_generated, d_on_givendata = logits_list
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=d_on_generated, 
                        labels=tf.ones_like(d_on_generated)))        # ONES_LIKE
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=d_on_generated,
                        labels=tf.zeros_like(d_on_generated)))  # ZEORS_LIKE
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=d_on_givendata,
                        labels = tf.ones_like(d_on_givendata))) # ONES_LIKE
        d_loss = d_loss_fake + d_loss_real

        return g_loss, d_loss


    def train(self, g_loss, d_loss, learning_rate=0.0002, beta1=0.5):
        '''
          args:
            losses list.

          returns:
            train op.
        '''
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(g_loss, var_list=self.g.variables)
        d_opt_op = d_opt.minimize(d_loss, var_list=self.d.variables)

        # with tf.control_dependencies([g_opt_op, d_opt_op]):
        #     return tf.no_op(name='train')

        return [g_opt_op, d_opt_op]


    def sample_images(self, row=8, col=8, inputs=None, label=None):
        if inputs is None:
            inputs = self.get_noise()
        images = tf.tanh(self.g(inputs, label=label, training=True))
        images = tf.image.convert_image_dtype(((images + 1.) / 2.), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))
