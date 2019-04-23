import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)
mnist = input_data.read_data_sets("./data/MNIST", one_hot=True)

