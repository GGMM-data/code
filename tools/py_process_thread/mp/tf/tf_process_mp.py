import multiprocessing as mp
import tensorflow as tf
import os
import time
import random

def add(index):
    sess = tf.Session()
    y = tf.ones([1,2])
    print(sess.run(y))
    n = 0
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
    # 这个的话是jobs的数量要小于等于cpu数量
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    length = 500

    begin_time = time.time()
    # 1.mp.Process, mutlri thread
    jobs = []
    for i in range(length):
        jobs.append(mp.Process(target=add, args=(i,)))
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    # ==========================================

    end_time = time.time()
    time1 = end_time - begin_time
    begin_time = time.time()

    # 2.single process
    for i in range(length):
        pass
        #add(i)
    # ==========================================

    print("Multi processes total time: ", time1)
    print("Single process total time: ", time.time() - begin_time)
    print("Done")

