import numpy as np
import time

length = 3
a = [np.ones((1, 400, 4)), np.ones((1, 400, 4)), np.ones((1, 400, 4))]
t = time.time()
for i in range(length):
    np.concatenate(a, 0)

print(time.time() -t)
