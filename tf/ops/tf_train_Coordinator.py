import tensorflow as tf
import time
import random
import threading

n = 0

def add(index):
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
        coord = tf.train.Coordinator()

        jobs = []
        for i in range(2):
            jobs.append(threading.Thread(target=add, args=(i,)))

        for j in jobs:
            j.start()

        coord.join(jobs)
        print("Hello World!")
        
        
