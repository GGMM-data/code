from multiprocessing import Process,Pool
import os,random,time


def run_proc(name):
    print("Run child process %s (%s)" % (name,os.getpid()))


def long_time_task(name):
    print("Run task %s (%s)" % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random()*3)
    end = time.time()
    print("task %s runs %0.2f seconds." % (name,end-start))


if __name__ == '__main__':
    print("Parent process %s" % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print("child process will start.")
    p.start()
    p.join()
    print("child process end.")

    print("Paranet Process %s" % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task,args=(i,))
    print("Waitting for done.")
    p.close()
    p.join()
    print("All subprocesses done")

