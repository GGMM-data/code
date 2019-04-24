import torch.multiprocessing as mp
import torch
import time
import os


def proc(sec, x):
   print(os.getpid(),"  ", x)
   time.sleep(sec)
   print(os.getpid(), "  ", x)
   x += sec
   print(str(os.getpid()) + "  over.  ", x)


if __name__ == '__main__':
   num_processes = 3
   processes = []
   x = torch.ones([3,])
   x.share_memory_()
   for rank in range(num_processes):
     p = mp.Process(target=proc, args=(rank, x))
     p.start() 
     processes.append(p)
   for p in processes:
     p.join()
   print(x)
