from multiprocessing import Pool
import time
import os
from itertools import repeat


def f(string, x):
    print(string)
    return x*x

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        number = 10
        s = "hello"
        print(pool.starmap(f, zip(repeat(s), range(number))))


