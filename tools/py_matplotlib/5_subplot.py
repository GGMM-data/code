import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 1)
y1 = 2 * x
y2 = 3 * x
y3 = 4 * x
y4 = 5 * x

plt.figure()

plt.subplot(2, 2, 1)
plt.plot(x, y1, marker='s', lw=4)

plt.subplot(2, 2, 2)
plt.plot(x, y2, ls='-.')

plt.subplot(2, 2, 3)
plt.plot(x, y3, color='r')

plt.subplot(2, 2, 4)
plt.plot(x, y4, ms=10, marker='o')

plt.show()

print(type(l))
