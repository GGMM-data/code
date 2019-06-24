import numpy as np
import time

size = 32
poi = []
pos = [[1,1],[2,2],[3,3], [4,4], [5,5],[6,6]]
for i in range(32):
    for j in range(32):
        poi.append([i, j])

begin = time.time()
for i in range(32):
    for j in range(32):
        for k in range(6):
            pos_x = pos[k][0] - poi[i*size+j][0]
            pos_y = pos[k][1] - poi[i*size+j][1]
            distances = pos_x**2 + pos_y**2
 
end = time.time()
print(end-begin)

begin = time.time()
flag = np.zeros((size*size), dtype=bool)
for k in range(6):
    temp=np.square(np.asarray(poi) - np.asarray(pos[k]))
    dis = temp[:,0] + temp[:, 1]
    flag[dis<4] = True
flag = np.reshape(flag, (size,size))
end = time.time()
print(end-begin)
