from multiprocessing import Process,Pool
import os,random,time


def run_proc(name):
    time.sleep(2)
    print("Run child process %s (%s)" % (name,os.getpid()))


if __name__ == '__main__':
    print("==========1. join==========")
    print("Parent process %s" % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print("child process will start.")
    p.start()
    p.join()
    print("child process end.")

 
    print("==========2. no join==========")
    print("Parent process %s" % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print("child process will start.")
    p.start()
    # p.join()
    print("child process end.")

