import matplotlib.pyplot as plt
import numpy as np

color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

for i in range(len(color)):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    plt.plot(x, y+i, color=color[i])

plt.show()


plt.plot(range(10), range(10), color='w')
plt.show()

