import numpy as np
import time

# 2.example2
print("===========example 2: ==========")
length = 6
task = 3
a = [[] for _ in range(length)]

for i in range(length):
    print(a[i])
    for j in range(task):
        a[i].append(np.ones((1, 10, 4)))

temp = np.concatenate(a[0], 0)
print(type(temp))
print(temp.shape)

action = []
for i in range(length):
    action.append(np.ones((task, 5)))
print(len(action))
for a in action:
    print(a.shape)
print(np.array(action).shape)

# 1.example1
print("===========example 1: ==========")
info = np.ones([3086,], dtype=np.float32)
begin = time.time()
pos = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
vec = [1,1]
for i in range(len(pos)):
#     print(type([pos]))
#     print(type([vec]))
#     print(type(info.tolist()))
#     print(pos)
#     print([vec])
#     print([info.tolist()])
#     a = np.concatenate(([vec]+pos, [info.tolist()]))
#     print(a.shape)
#     print(a)
    print(time.time()-begin)


