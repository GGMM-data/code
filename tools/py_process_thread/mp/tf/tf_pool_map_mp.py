import multiprocessing as mp
import os
import time
import random
import tensorflow as tf

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    length = 500
    
    begin_time = time.time()

    # 1.pool.map, multi processes
    pool = mp.Pool(mp.cpu_count())
    pool.map(add, range(2))
    pool.map(add, range(length))
    # ===================================

    end_time = time.time()
    time1 = end_time - begin_time
    begin_time = time.time()

    # 2.single process
    for i in range(length):
        add(i)
        #pass
    # ===================================

    print("total time: ", time1)
    print("total time: ", time.time() - begin_time)
    print("Done")
