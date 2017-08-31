#
#   main_lsun.py
#       date. 8/31/2017
#

import numpy as np
import tensorflow as tf

from model import DCGAN
from load_lsun import LSUNdataset

def main():
    # load data
    bedroom = LSUNdataset(dirn='../data', category='bedroom')
    # fake_x = np.ones([128, 28, 28, 1], dtype=np.float32) * 0.1

    # condition
    k = 1             # # of discrim updates for each gen update
    l2 = 2.5e-5       # l2 weight decay
    b1 = 0.5          # momentum term of adam
    nc = 3            # # of channels in image
    ny = 10           # # of classes
    batch_size = 128  # # of examples in batch
    npx = 32         # # of pixels width/height of images
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
    # y = tf.placeholder(tf.float32, [None, ny])          # for training w/ label
    y = None    # without label data
    # y_target = tf.placeholder(tf.float32, [None, ny])   # for image generation

    # graphs
    dcgan = DCGAN(batch_size=batch_size, s_size=4, z_dim=nz, y_dim=None)
    logits_list = dcgan.inference(x, y)     # (x, y)
    g_loss, d_loss = dcgan.loss(logits_list)
    train_op = dcgan.train(g_loss, d_loss, learning_rate=lr)

    # images
    images = dcgan.sample_images(label=None)
    # images = dcgan.sample_images()

    init = tf.global_variables_initializer()

    # Training
    n_epochs = 10
    with tf.Session() as sess:
        sess.run(init)

        # loop control
        n_sample = bedroom.num_examples
        n_loop = n_sample // batch_size     # for LSUN bedroom dataset, n_loop = 23696
        if n_sample % batch_size != 0:
            n_loop += 1
        
        for e in range(1, n_epochs+1):
            for i in range(n_loop):
                batch_x = bedroom.next_batch(batch_size, img_size=32)
                batch_img = batch_x.reshape([-1, 32, 32, 3])
                fd_train = {x: batch_img}
                # fd_train = {x: batch_img}
                sess.run(train_op, feed_dict=fd_train)
                g_loss_np, d_loss_np = sess.run([g_loss, d_loss], 
                                                feed_dict=fd_train)
                # print status
                if i % 10 == 0:
                    print((' ecpoch {:>5d}: ({:>8d} /{:>8d}) :'
                          'g_loss={:>10.4f}, d_loss={:>10.4f}').format(
                          e, i, n_loop, g_loss_np, d_loss_np))

                if i == 100:
                    break

            # Generate sample images after training
            if e in [1, 2, 5, 10]:
                fn_sample = '../work/samples/bedroom_' + str(e) + '.jpg'
                generated = sess.run(images)
                # generated = sess.run(images)
                with open(fn_sample, 'wb') as fp:
                    fp.write(generated)

if __name__ == '__main__':
    main()
