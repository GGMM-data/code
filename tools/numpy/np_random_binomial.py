import numpy as np

for i in range(10):
    rand = np.random.binomial(2, 0.9)
    print(rand)

rand = np.random.binomial(3, 0.9, 5)
print(rand)

