#
#   load_lsun.py
#       date. 8/30/2017
#
#       tested w/:
#           python3 (3.5.2)
#           python-lmdb 0.9.2
#           lmdb  0.9.21
#

import io
import os
import numpy as np
import lmdb
from PIL import Image


class LSUNdataset(object):
    def __init__(self, dirn='.', category='church_outdoor'):
        self.category=category
        self.db_path = os.path.join(dirn, category + '_train_lmdb')
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.keys = np.array([])

        if os.path.exists(self.db_path) and os.path.isdir(self.db_path):
            try:
                env = lmdb.open(self.db_path, max_readers=1, 
                                readonly=True, lock=False,
                                readahead=False, meminit=False)
                self.env = env
            except NameError:
                print('lmdb open error. check files in directory.')
        else:
            raise OSError('check path (to data directory) name.')

    def keys_fetch(self):
        """
        pre-fetching keys of LSUN image db
          args:
            env - lmdb.Environment object
        """
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            tot = txn.stat()['entries']
            i = 0
            keys = []
            for key, _ in cursor:
                i += 1
                if i % 100 == 0 or i == tot:
                    print('Fetching {:>8d} /{:>8d} keys'.format(i, tot),
                          end='\r')
                keys.append(key)
            print('\nDone.')
            self._num_examples = txn.stat()['entries']

        self.keys = np.asarray(keys)

    def get_image_by_keys(self, keys, img_size=64):
        txn = self.env.begin(write=False)
        images = []
        for key in keys:
            val = txn.get(key)
            byteImgIO = io.BytesIO(val)
            img = Image.open(byteImgIO)
            resized = self.img_preprocess(img, output_size=img_size)
            images.append(resized)
        images = np.asarray(images)

        return images

    def img_preprocess(self, img, output_size=64):
        """
          get same resized (BOX) image
        """
        w, h = img.size
        if w > h:
            box_param = (int(w * 0.5 - h * 0.5), 0, int(w * 0.5 + h * 0.5), h)
            cropped = img.crop(box_param)
        else:   # w < h
            box_param = (0, int(h * 0.5 - w * 0.5), w, int(h * 0.5 + w * 0.5))
            cropped = img.crop(box_param)

        resized = cropped.resize((output_size, output_size))
        resized = np.asarray(resized)

        return resized

    def next_batch(self, batch_size, img_size=64, shuffle=True):
        """
          provide batch image samples
        """
        if self.keys.shape[0] == 0:
            self.keys_fetch()

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.random.permutation(self._num_examples)
            self.keys = self.keys[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            keys_rest_part = self.keys[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.random.permutation(self._num_examples)
                self.keys = self.keys[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            keys_new_part = self.keys[start:end]
            images = self.get_image_by_keys(
                np.concatenate((keys_rest_part, keys_new_part), axis=0),
                img_size=img_size)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            images = self.get_image_by_keys(self.keys[start:end], 
                                            img_size=img_size)
        return images

    @property
    def num_examples(self):
        with self.env.begin(write=False) as txn:
            self._num_examples = txn.stat()['entries']
        return self._num_examples


if __name__ == '__main__':
    church = LSUNdataset(dirn='../LSUNdataset', category='church_outdoor')
    
    church.keys_fetch()     # takes several minutes
    print('size of one key = ', len(church.keys[0]))
    im = church.next_batch(10)
    print('shape of batch image = ', im.shape)

    im = church.next_batch(100, img_size=64)
    imgsiz = im.shape[1]
    bigsiz = imgsiz * 8
    big_image = np.zeros([bigsiz, bigsiz, 3], dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            idx = row * 10 + col
            big_image[row*imgsiz:(row+1)*imgsiz, 
                      col*imgsiz:(col+1)*imgsiz, :] = im[idx]

    # show data sample images
    im = Image.fromarray(big_image)
    im.save('sample.png')
