import matplotlib.pyplot as plt
import numpy as np

x = np.arange(500)
y1 = 2 * x
y2 = 3 * x
y3 = 4 * x
y4 = 5 * x
y5 = 5 * x
y6 = 5 * x
y7 = 5 * x
y8 = 5 * x

plt.figure(figsize=(40, 20))

plt.subplot(4, 2, 1)
plt.plot(x, y1, marker='s', lw=4)

plt.subplot(4, 2, 2)
plt.plot(x, y2, ls='-.')

plt.subplot(4, 2, 3)
plt.plot(x, y3, color='r')

plt.subplot(4, 2, 4)
plt.plot(x, y4, ms=10, marker='1')

plt.subplot(4, 2, 5)
plt.plot(x, y5, ms=10, marker='2')

plt.subplot(4, 2, 6)
plt.plot(x, y6, ms=10, marker='3')

plt.subplot(4, 2, 7)
plt.plot(x, y7, ms=10, marker='4')
plt.savefig("5.pdf")

plt.show()

