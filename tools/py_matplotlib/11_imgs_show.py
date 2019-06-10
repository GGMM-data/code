import numpy as np
import matplotlib.pyplot as plt

plt.figure()
#img = np.random.random((32, 32, 1)) error
# np.random.random返回区间[0.0, 1.0)之间的随机浮点数
img = np.random.rand(32, 32)
# img = np.random.randn(32, 32)
plt.imshow(img)
plt.show()

plt.figure()
img1 = np.random.randint(low=0, high=255, size=(3, 32, 32))
img1 = np.random.randint(low=0, high=255, size=(32, 32, 3))
img1 = np.array(img1, dtype=np.float32)
plt.imshow(img1)
plt.show()

plt.figure()
img2 = np.ones((32, 32, 3), dtype=np.float32)
img2 = img2/2
plt.imshow(img2)
plt.show()


