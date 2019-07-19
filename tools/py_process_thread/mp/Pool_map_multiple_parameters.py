from multiprocessing import Pool
import time
import os


def f(string, x):
    print(string)
    return x*x

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        number = 10
        s = "hello"
        print(pool.map(f, zip(s*number, range(number))))


