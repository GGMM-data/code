import matplotlib.pyplot as plt
import numpy as np

img = np.random.randint(0, 255, [32, 32])
print(img.shape)

plt.imshow(img)
plt.show()

