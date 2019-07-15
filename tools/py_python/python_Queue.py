import queue
import numpy as np


q = queue.Queue(4)

q.put(np.ones((2,2)))
q.put(np.ones((2,2)))
q.put(np.ones((2,2)))
q.put(np.ones((2,2)))
print(q.queue)

a = q.queue
a = np.array(a)
print(a.shape)
print(q.queue)
