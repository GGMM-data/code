import numpy as np
import matplotlib.pyplot as plt

plt.figure()
img = np.random.randint(low=0, high=255, size=(32,32,3))
img = np.array(img, dtype=np.int32)
plt.imshow(img)

plt.figure()
img2 = np.ones((32, 32, 3), dtype=np.float32)
img2 = img2/2
plt.imshow(img2)
plt.show()
