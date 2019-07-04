from multiprocessing import Pool
import time
import os


def f(x):
    return x*x

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        number = 10
        # 1.map有序
        print(pool.map(f, range(number)))

        # print(pool.imap_unordered(f, range(number)))
        # 返回一个迭代器
        # 2.imap_unodered无序
        for i in pool.imap_unordered(f, range(number)):
            print(i)

        # 3.并行
        # evaluate "os.getpid()" asynchronously
        results = []
        for i in range(number):
            results.append(pool.apply_async(f, args=(i,)))
        print([res.get() for res in results])


