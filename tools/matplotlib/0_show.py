import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10,1)

y1 = x**2
y2 = 2*x +5

######
# imshow

plt.plot(x,y1)
plt.savefig("0_1.png")
#plt.show()

plt.plot(x,y2)
plt.show()
