import numpy as np
import time

info = np.ones([3086,], dtype=np.float32)

begin = time.time()
pos = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
vec = [1,1]
for i in range(6):
#     print(type([pos]))
#     print(type([vec]))
#     print(type(info.tolist()))
#     print(pos)
#     print([vec])
#     print([info.tolist()])
     a = np.concatenate(([vec]+pos, [info.tolist()]))
#     print(a.shape)
#     print(a)
print(time.time()-begin)
