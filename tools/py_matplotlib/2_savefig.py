import matplotlib.pyplot as plt
import numpy as np
    
x = np.arange(0, 10, 1)

y1 = x**2
y2 = 2*x +5

plt.plot(x,y1)
plt.savefig("2.png") # 保存图像，名字为2.png
plt.show()
