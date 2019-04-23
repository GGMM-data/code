import numpy as np
import matplotlib.pyplot as plt

plt.figure()
#img = np.random.random((32, 32, 1)) error
img = np.random.random((32, 32))
plt.imshow(img)
plt.show()

plt.figure()
img1 = np.random.randint(low=0, high=255, size=(3,32,32))
img1 = np.random.randint(low=0, high=255, size=(32,32,3))
img1 = np.array(img1, dtype=np.float32)
plt.imshow(img1)

plt.figure()
img2 = np.ones((32, 32, 3), dtype=np.float32)
img2 = img2/2
plt.imshow(img2)
plt.show()


