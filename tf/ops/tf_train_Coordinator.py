import tensorflow as tf
import time
import random
import threading

n = 0

class Hello:

    def add(self, index, y, sess):
        print(sess.run(y))
        global n
        while not coord.should_stop():
        #while True:
            if n > 10:
                print(index, " done")
                coord.request_stop()
                #break
            print("before A, thread ", index, n)
            n += 1
            print("after A, thread ", index, n)
            time.sleep(random.random())
            print("before B, thread ", index, n)
            n += 1
            print("after B, thread ", index, n)
    
if __name__ == "__main__":
    with tf.Session() as sess:
        length = 2
        hhh = []
        for i in range(length):
            hhh.append(Hello())
        coord = tf.train.Coordinator()
        y = tf.ones([1,2])

        jobs = []
        for i,h in enumerate(hhh):
            jobs.append(threading.Thread(target=h.add, args=(i, y, sess)))

        for j in jobs:
            j.start()

        coord.join(jobs)
        print("Hello World!")
        
        
