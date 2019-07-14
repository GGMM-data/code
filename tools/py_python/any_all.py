import numpy as np


a = np.array([0.0, 0.0, 0.0])
print(a.any())

a = np.array([1.0, 0.0, 0.0])
print(a.any())

a = np.array([1.0, 0.0, 0.0])
print(a.all())

a = np.array([1.0, 1.0, 1.0])
print(a.all())

print(any([0.0, 1.0]))
