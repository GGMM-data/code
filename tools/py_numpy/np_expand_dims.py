import numpy as np


a = np.ones([2, 3, 4])
print(a.shape)

a1 = np.expand_dims(a, 0)
print(a1.shape)

a2 = np.expand_dims(a, 1)
print(a2.shape)

a3 = np.expand_dims(a, 2)
print(a3.shape)

a4 = np.expand_dims(a, 3)
print(a4.shape)
