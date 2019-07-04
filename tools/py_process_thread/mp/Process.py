from multiprocessing import Process, Queue
import os,random,time


def proc1(name):
    print("Run child process %s (%s)" % (name,os.getpid()))
    print(time.time())
    time.sleep(random.random())
    print("%s end" % (name))
    return "return A"

def proc2(name):
    print("Run child process %s (%s)" % (name,os.getpid()))
    print(time.time())
    time.sleep(random.random())
    print("%s end" % (name))
    return "return B"

def proc(length, output):
    result = "Hello! " + str(length) + "!"
    time.sleep(random.random())
    output.put(result)

if __name__ == '__main__':
    # 1. mp.Process
    print("Parent process %s" % os.getpid())
    p1 = Process(target=proc1, args=('p1',))
    p2 = Process(target=proc2, args=('p2',))
    print("child processes will start.")
    p1.start()
    p2.start()
    # 上面两行代码意思是p1.start()有返回值时，开始执行p2.start()。p1.start()有返回值并不是说p1执行完了
    p1.join()
    p2.join()
    # 上面两行代码中，p1.join()执行完之后才会执行p2.join()。所以只有p1执行完之后，p2才能尝试结束。。
    #  The interpreter will, however, wait until P1 finishes before attempting to wait for P2 to finish.
    print("child processes end.")


    # 2.获得mp.Process的返回值
    print("# 2.获得mp.Process的返回值")
    output = Queue()
    processes = [Process(target=proc, args=(x, output)) for x in range(4)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = [output.get() for p in processes]
    print(results)

# https://stackoverflow.com/questions/31711378/python-multiprocessing-how-to-know-to-use-pool-or-process
