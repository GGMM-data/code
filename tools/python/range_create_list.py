import time
import numpy as np

number = 10000

t1 = time.time()
l = []
for i in range(number):
    l.append(number)
t2 = time.time()
print(t2-t1)

t1 = time.time()
l = list(range(number))
t2 = time.time()
print(t2-t1)
    
t1 = time.time()
l = np.arange(number)
print(type(l))
t2 = time.time()
print(t2-t1)
