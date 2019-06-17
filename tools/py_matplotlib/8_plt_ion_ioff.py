import matplotlib.pyplot as plt
import numpy as np

count = 1
flag = True

figures = [plt.figure(), plt.figure(), plt.figure()]

axes = []
for fig in figures:
    axes.append(fig.gca())
x = np.arange(20)

while flag:
    for i in range(3):
        plt.ion()
        y = pow(x[:count], 2)
        temp = x[:count]
        axes[i].plot(temp, y, linewidth=1)
        plt.pause(0.01) # 
        plt.savefig("8_" +str(i) + ".png")
        plt.ioff()
    count += 1
    if count > 20:
        break

# plt.show() 
