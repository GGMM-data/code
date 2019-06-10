import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,11,1)
y = np.arange(1,20,2)
print(x)
print(y)

plt.plot(x,y,'r',lw=5) # x, y, color, lw is linewidth
plt.bar(x,y,width=0.1) # x,height,width,color,alpha,lw or linewidth,tick_table,linestyle....

for i in range(6):
    x = np.zeros([2])
    x[0] = i
    x[1] = i
    y = np.zeros([2])
    y[0] = 10
    y[1] = 20
    plt.plot(x,y,'y',lw=5)

plt.show()

x = np.arange(1,10,1)
y1 = x**2
print(x,y1)
plt.plot(x,y1)
plt.show()

y2 = 2*x + 1
plt.plot(x,y2)
plt.show()
