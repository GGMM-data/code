import os

print("Process %s start..." % os.getpid())
pid = os.fork()

if pid == 0:
    print("child process %s and my parent process is %s" % (os.getpid(),os.getppid()))
else:
    print("I (%s) just crated a child process (%s)" % (os.getpid(), pid))
