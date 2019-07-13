import numpy as np

# np.newaxis相当于None。
print(np.newaxis is None)

a = np.ones((2, 3, 4))
print(a.shape)

a1 = a[np.newaxis]
print(a1.shape)

a2 = a[:, np.newaxis]
print(a2.shape)

a3 = a[:, :, np.newaxis]
print(a3.shape)

a4 = a[:, :, :, np.newaxis]
print(a4.shape)

a5 = a[:, :, np.newaxis, np.newaxis, :]
print(a5.shape)


b = np.array([0, 10, 20, 30])
print(b.shape)
c = np.array([1, 2, 3])
print(c.shape)
print(c)
b1 = b[:, np.newaxis]
print(b1.shape)
print(b1)
d = b1 + c
print(d)
print(d.shape)
