#
#   main_cifar.py
#       date. 7/20/2017
#

import numpy as np
import tensorflow as tf

from model import DCGAN
from load_cifar import load_data

def main():
    # load data
    cifar = load_data('../data/')
    # fake_x = np.ones([128, 28, 28, 1], dtype=np.float32) * 0.1

    # condition
    k = 1             # # of discrim updates for each gen update
    l2 = 2.5e-5       # l2 weight decay
    b1 = 0.5          # momentum term of adam
    nc = 3            # # of channels in image
    ny = 10           # # of classes
    batch_size = 128  # # of examples in batch
    npx = 32          # # of pixels width/height of images
    nz = 100          # # of dim for Z
    ngfc = 1024       # # of gen units for fully connected layers
    ndfc = 1024       # # of discrim units for fully connected layers
    ngf = 64          # # of gen filters in first conv layer
    ndf = 64          # # of discrim filters in first conv layer
    nx = npx*npx*nc   # # of dimensions in X
    niter = 100       # # of iter at starting learning rate
    niter_decay = 100 # # of iter to linearly decay learning rate to zero
    lr = 0.0002       # initial learning rate for adam

    # tensorflow placeholder
    x = tf.placeholder(tf.float32, [None, npx, npx, nc])
    y = tf.placeholder(tf.float32, [None, ny])          # for training w/ label
    y_target = tf.placeholder(tf.float32, [None, ny])   # for image generation

    # graphs
    dcgan = DCGAN(batch_size=batch_size, s_size=4, z_dim=nz, y_dim=ny)
    logits_list = dcgan.inference(x, y)     # (x, y)
    g_loss, d_loss = dcgan.loss(logits_list)
    train_op = dcgan.train(g_loss, d_loss, learning_rate=lr)

    # images
    images = dcgan.sample_images(label=y_target)
    # images = dcgan.sample_images()

    init = tf.global_variables_initializer()

    # Training
    n_epochs = 300
    with tf.Session() as sess:
        sess.run(init)

        # loop control
        n_sample = cifar.train.num_examples
        n_loop = n_sample // batch_size
        if n_sample % batch_size != 0:
            n_loop += 1

        for e in range(1, n_epochs+1):
            for i in range(n_loop):
                batch_x, batch_y = cifar.train.next_batch(batch_size)
                batch_img = batch_x.reshape([-1, 32, 32, 3])
                fd_train = {x: batch_img, y: batch_y}
                # fd_train = {x: batch_img}
                sess.run(train_op, feed_dict=fd_train)
                g_loss_np, d_loss_np = sess.run([g_loss, d_loss], 
                                                feed_dict=fd_train)

            print('ecpoch {:>5d}: g_loss={:>11.4f}, d_loss={:>11.4f}'.format(
                                            e, g_loss_np, d_loss_np))

            # Generate sample images after training
            if e in [10, 20, 50, 100, 200, 300]:
                _, batch_yv = cifar.validation.next_batch(batch_size)
                fn_sample = '../work/samples/cifar_' + str(e) + '.jpg'
                generated = sess.run(images, feed_dict={y_target: batch_yv})
                # generated = sess.run(images)
                with open(fn_sample, 'wb') as fp:
                    fp.write(generated)
            

if __name__ == '__main__':
    main()


