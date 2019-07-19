import tensorflow as tf
import time
import random
import multiprocessing as mp
import threading
import os


class Hello:
    def __init__(self, sess):
        self.sess = sess

    def add(self, index):
        print("1243")
        y = tf.ones([1,2])
        print(self.sess.run(y))
        global n
        # while not coord.should_stop():
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

def add(index, sess):
        print("1243")
        y = tf.ones([1,2])
        print(sess.run(y))
        n = 0
        # while not coord.should_stop():
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
    with tf.Session() as sess:
        length = 2
        hhh = []
        for i in range(length):
            hhh.append(Hello(sess))
        coord = tf.train.Coordinator()

        jobs = []
        pool = mp.Pool()
        for i,h in enumerate(hhh):
            #jobs.append(threading.Thread(target=h.add, args=(i, sess)))
            jobs.append(mp.Process(target=h.add, args=(i, )))
            # pool.apply_async(h.add, args=(i,sess))
        pool.close()
        pool.join()
        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        #coord.join(jobs)
        print("Hello World!")
        
