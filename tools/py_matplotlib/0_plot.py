import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y1 = np.arange(10)
y2 = y1 + 1
print(y1)
print(y2)

for y in [y1, y2]:
    plt.plot(x, y)
plt.show()
