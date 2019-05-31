import numpy as np
import time

# 目的，计算不断更新的两个列表的乘积的和
numbers = 100000
a = []
b = []
c1 = []

begin_time = time.time() 
for i in range(numbers):
    a.append(i-1)
    b.append(i+1)
    c1.append(np.sum(np.multiply(a, b)))
# print(c1)
end_time = time.time()
print("Total time: ", end_time - begin_time)

# 我用的是上面的代码，然后，，，，太多重复运算，效率惨不忍睹

a = []
b = []
c2 = []

begin_time = time.time() 
for i in range(numbers):
    a.append(i-1)
    b.append(i+1)
    c2.append(a[i]*b[i])
results = np.add.accumulate(c2)
# print(results)
end_time = time.time()
print("Total time: ", end_time - begin_time)

