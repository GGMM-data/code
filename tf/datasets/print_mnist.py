from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    mnist = input_data.read_data_sets("/home/mxxmhh/MNIST_data", one_hot=True)
    batch_size = 128

    # 1. number
    print("1.1 mnist.train.num_examples: ")
    print(mnist.train.num_examples)
    train_xs, train_ys = mnist.train.next_batch(batch_size)
    print("1.2 size of batch train images")
    print(train_xs.shape)
    print("1.3 size of batch train labels")
    print(train_ys.shape)
    print(train_ys[:5])
    print(np.argmax(train_ys[:5], 1))
    print(tf.argmax(train_ys[:5], 1))
    
    
    print("1.4 mnist.test.num_examples: ")
    print(mnist.test.num_examples)
    test_xs, test_ys = mnist.test.next_batch(batch_size)
    print("1.5 size of batch test images")
    print(test_xs.shape)
    print("1.6 size of batch test labels")
    print(test_ys.shape)

    # 2. shape 
    train_data = mnist.train.images
    train_labels = mnist.train.labels
    test_data = mnist.test.images
    test_labels = mnist.test.labels
    print("2.1 mnist train data shape:")
    print(train_data.shape)
    print("2.2 mnist train labels shape:")
    print(train_labels.shape)
    print("2.3 mnist test data shape:")
    print(test_data.shape)
    print("2.4 mnist test labels shape:")
    print(test_labels.shape)


