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

plt.figure(figsize=(50, 30))
# plt.subplots_adjust(top=1,bottom=0.1,left=0,right=1,hspace=0,wspace=0)

plt.subplot(3, 3, 1)
plt.xlabel("No. of step")
plt.ylabel("Energy")
plt.plot(x, y1, marker='s', lw=4)

plt.subplot(3, 3, 2)
plt.plot(x, y2, ls='-.')

plt.subplot(3, 3, 3)
plt.plot(x, y3, color='r')

plt.subplot(3, 3, 4)
plt.plot(x, y4, ms=10, marker='1')

plt.subplot(3, 3, 5)
plt.plot(x, y5, ms=10, marker='2')

plt.subplot(3, 3, 6)
plt.plot(x, y6, ms=10, marker='3')

plt.subplot(3, 3, 7)
plt.plot(x, y7, ms=10, marker='4')
plt.subplots_adjust(hspace=0.4)
plt.savefig("5.png")

# plt.show()

