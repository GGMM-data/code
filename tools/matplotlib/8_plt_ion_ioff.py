import matplotlib.pyplot as plt
import numpy as np

count = 1
flag = True

plt.figure()
ax = plt.gca()
x = np.arange(20)
plt.figure()
ax2 = plt.gca()

while flag:
    plt.ion()
    y = pow(x[:count], 2)
    temp = x[:count]
    ax.plot(temp, y, linewidth=1)
    plt.pause(1)
    plt.ioff()

    ax2.plot(x, x+count)
    count += 1
    if count > 20:
        break

plt.show() 

