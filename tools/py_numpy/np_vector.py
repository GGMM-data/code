import numpy as np 

a = np.ones((2,2))
print(a.shape)

b = np.ones((2,2))
print(b.shape)

t1 = np.sum(a*b)
t2 = (np.sqrt(np.sum(np.square(a)))*(np.sqrt(np.sum(np.square(b)))))

print(t1)
print(t2)

a1 = np.reshape(a, (-1,))
b1 = np.reshape(b, (-1,))
z1 = a1*b1
z2 = (np.sqrt(np.sum(np.square(a1)))*(np.sqrt(np.sum(np.square(b1)))))
print(z1)
print(z2)
