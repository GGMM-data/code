import multiprocessing as mp
import tensorflow as tf
import os
import time
import random

n = 0
def add(index):
    sess = tf.Session()
    y = tf.ones([1,2])
    print(sess.run(y))
    global n
    while True:
        if n > 10:
            print(index, " done")
            #coord.request_stop()
            break
        print("before A, thread ", index, n)
        n += 1
        print("after A, thread ", index, n)
        time.sleep(random.random())
        print("before B, thread ", index, n)
        n += 1
        print("after B, thread ", index, n)


if __name__ == "__main__":
    pool = mp.Pool()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    jobs = []
    for i in range(1):
        jobs.append(mp.Process(target=add, args=(i,)))
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    print("Done")


