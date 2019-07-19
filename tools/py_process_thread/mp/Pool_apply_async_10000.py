from multiprocessing import Pool
import multiprocessing as mp
import os
import time
import random


def func(i):
    print(i)
    #with open("./files/"+str(i), "w") as f:
    #    f.write(str(i))

print(mp.cpu_count())
pool = Pool(mp.cpu_count())
#pool = Pool(1)


if __name__ == "__main__":
    number = 100
    t = time.time()
    for i in range(number):
        pool.apply_async(func, args=(i,))
    pool.close()
    pool.join()
    print("Done", time.time() - t)

