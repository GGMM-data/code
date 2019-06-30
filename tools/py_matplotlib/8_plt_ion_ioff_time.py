import matplotlib.pyplot as plt
import time
import numpy as np

count = 1
flag = True

fig = plt.figure()
ax = fig.gca()

x = np.arange(4000)

while True:
        begin = time.time()
        plt.ion()
        # y = pow(x[:count], 2)
        y = x[:count]
        temp = x[:count]
        ax.plot(temp, y, linewidth=1)
        plt.ioff()
        plt.savefig("8.png")
        print(count, time.time() - begin)
        count += 1

