import numpy as np


a1 = np.random.permutation(9)
print(a1)

a2 = np.random.permutation([1, 2, 4, 9, 8])
print(a2)

a3 = np.random.permutation(np.arange(9).reshape(3, 3))
print(a3)
