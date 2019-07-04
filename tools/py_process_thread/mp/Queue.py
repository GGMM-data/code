from multiprocessing import Process,Queue

import os, time, random

def write(q):
    print("process to write: %s" % os.getpid())
    for value in ['A','B','C']:
        print("Put %s to queue" % value)
        q.put(value)
        time.sleep(random.random())

def write2(q):
    print("process to write: %s" % os.getpid())
    for value in ['D','E','F']:
        print("Put %s to queue" % value)
        q.put(value)
        time.sleep(random.random())

def read(q):
    print("Process to read: %s" % os.getpid())
    while True:
        value = q.get(True)
        print("get %s from queue" % value)

if __name__ == "__main__":
    q = Queue()
    pw = Process(target=write,args=(q,))
    pw2 = Process(target=write2,args=(q,))
    pr = Process(target=read,args=(q,))
    pw.start()
    pw2.start()
    pr.start()
    pw.join()
    pw2.join()
    pr.terminate()
