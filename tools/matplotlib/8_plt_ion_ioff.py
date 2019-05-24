import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.ion()
x = np.arange(20)
for i in range(20):
  # x[i] = x[i] + i 
  y = pow(x[:i], 2)
  temp = x[:i]*100
  print(temp)
  plt.plot(temp, y, linewidth=1)
  plt.pause(0.1)

plt.ioff()
plt.show() 

