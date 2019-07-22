import  seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x = np.random.randn(10, 10)
f, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)
sns.heatmap(x, annot=True, ax=ax1)
sns.heatmap(x, annot=True, ax=ax2)
ax1.set_xlabel("test")
plt.savefig("test.png")
plt.show()
plt.close()
