import multiprocessing
import time
import random

def func(x):
    res = x*x
    t = random.random()
    print(x, t)
    time.sleep(t)
    return res


p = multiprocessing.Pool()
results = []
for i in range(4):
    results.append(p.apply_async(func, args=(i, )))

for res in results:
    print(res.get())

# https://stackoverflow.com/questions/42843203/how-to-get-the-result-of-multiprocessing-pool-apply-async
