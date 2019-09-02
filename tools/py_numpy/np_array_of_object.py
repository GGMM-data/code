import numpy as np

a0 = (3, np.ones((3,3)))
a1 = (3, np.ones((3,3)))

b = [a0, a1]
b1 = np.array(b)
print(b1)

b2 = b1[:, 1]
print(type(b2))
print(b2)
print(b2.shape)

