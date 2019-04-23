import matplotlib.pyplot as plt
import numpy as np

a = np.random.randint(0,255,[32,32])
print(a.shape)

plt.imshow(a)
plt.show()
