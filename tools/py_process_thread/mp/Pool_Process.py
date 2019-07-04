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
    # 1.Process
    print("Parent process %s" % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print("child process will start.")
    p.start()
    p.join()
    print("child process end.")

    # 2.Pool
    print("Paranet Process %s" % os.getpid())
    pool = Pool(4)
    for i in range(5):
        pool.apply_async(long_time_task,args=(i,))
    print("Waitting for done.")
    pool.close()    # 回收Pool
    pool.join()
    print("All subprocesses done")
