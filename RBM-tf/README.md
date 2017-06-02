# README.md

## RBM(Restricted Boltzmann Machine) TensorFlow implementation

Before you run rbm_tf.py, make directories as below.

```text
+---+ code  - python program directory
    | 
    + data  - directory to store MNIST datasets (...ubyte.gz 4 files)
    |
    + work  - directory to save plot by "rbm_tf.py"
```

If you don't have MNIST datasets, tensorflow MNIST utility will download dataset automatically.

### Package requirement (my test environment)

- Numpy (1.12.1)
- Pillow (3.2.0)
- TensorFlow (1.1.0)



**Running condition...**

```text
$ python rbm_tf.py 
Extracting ../data/train-images-idx3-ubyte.gz
Extracting ../data/train-labels-idx1-ubyte.gz
Extracting ../data/t10k-images-idx3-ubyte.gz
Extracting ../data/t10k-labels-idx1-ubyte.gz

Training...
epoch  0, cost = -190.706 (time =    21.09 s)
epoch  1, cost = -164.263 (time =    20.99 s)
epoch  2, cost = -160.005 (time =    21.07 s)
epoch  3, cost = -156.966 (time =    21.11 s)
epoch  4, cost = -154.970 (time =    21.08 s)
epoch  5, cost = -153.724 (time =    21.10 s)
epoch  6, cost = -153.094 (time =    21.14 s)
epoch  7, cost = -152.028 (time =    21.21 s)
epoch  8, cost = -150.945 (time =    21.18 s)
epoch  9, cost = -149.921 (time =    21.19 s)

```
