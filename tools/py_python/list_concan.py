import numpy as np


a1 = np.random.randint(1, 10, (3, 10))
a2 = np.random.randint(1, 10, (3, 10))

a3 = np.random.randint(1, 10, (3, 5))
a4 = np.random.randint(1, 10, (3, 5))
b1 = a1.tolist()
b2 = a2.tolist()
b3 = a3.tolist()
b4 = a4.tolist()

c = b1+b2+b3+b4
print(c)
d = np.concatenate(c, 1)
print(d)
