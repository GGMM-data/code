import matplotlib.pyplot as plt
import numpy as np


figure,axes = plt.subplots(2, 3, figsize=[40,20])
axes = axes.flatten()

x = np.arange(0, 20) 
y1 = pow(x, 2)
axes[0].plot(x, y1) 
axes[0].set_xlabel("test")

y5 = pow(x, 3)
axes[5].plot(x, y5) 

plt.show()

