from multiprocessing import Pool
import random
import os
import time


def p1():
    print("A")
    # print("Run task %s (%s)" % (name, os.getpid()))
    start = time.time()
    print("start time", start)
    time.sleep(0.1)
    #time.sleep(random.random()*3)
    end = time.time()
    print("task A runs %0.2f seconds." % (end-start))
    return "return A"

def p2():
    print("B")
    # print("Run task %s (%s)" % (name, os.getpid()))
    start = time.time()
    print("start time", start)
    #time.sleep(random.random()*3)
    time.sleep(0.3)
    end = time.time()
    print("task B runs %0.2f seconds." % (end-start))
    return "return B"

def p3():
    print("C")
    # print("Run task %s (%s)" % (name, os.getpid()))
    start = time.time()
    print("start time", start)
    #time.sleep(random.random()*3)
    time.sleep(0.2)
    end = time.time()
    print("task C runs %0.2f seconds." % (end-start))
    return "return C"


if __name__ == "__main__":
    # 1. Pool.apply_async多进程
    print("1. 多进程")
    begin = time.time()
    pool = Pool(2) 
    p = [p1, p2, p3]
    results = []
    for h in p:
        results.append(pool.apply_async(h, args=()))
    pool.close()
    pool.join()
    for res in results:
        print(res.get())
    print("Done.")
    print("Total time", time.time() - begin)

    # 2. Pool.apply单进程
    print("2. 单进程")
    begin = time.time()
    pool = Pool(2) 
    p = [p1, p2, p3]
    results = []
    for h in p:
        results.append(pool.apply(h, args=()))
    pool.close()
    pool.join()
    for res in results:
        print(res)
    print("Total time", time.time() - begin)

    # 3.示例
    print("3. 示例")
    pool = Pool(4) 
    multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    pool.close()
    pool.join()
    print([res.get(timeout=1) for res in multiple_results])


