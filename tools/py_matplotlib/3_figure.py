import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10,1)

y1 = x**2
y2 = 2*x +5

# figure
plt.figure()
plt.plot(x,y1)

plt.figure(num=6,figsize=(10,10))
plt.plot(x,y2)
plt.show()

