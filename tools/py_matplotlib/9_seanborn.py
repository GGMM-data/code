import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

values = np.zeros((21,21), dtype=np.int)
fig, axes = plt.subplots(2, 3, figsize=(40,20))
plt.subplots_adjust(wspace=0.1, hspace=0.2)
axes = axes.flatten()

# cmap is the paramter to specify color type, ax is the parameter to specify where to show the picture
# np.flipud(matrix), flip the column in the up/down direction, rows are preserved
figure = sns.heatmap(np.flipud(values), cmap="YlGnBu", ax=axes[0])
figure.set_xlabel("cars at second location", fontsize=30)
figure.set_title("policy", fontsize=30)
figure.set_ylabel("cars at first location", fontsize=30)
figure.set_yticks(list(reversed(range(21))))

figure = sns.heatmap(np.flipud(values), ax=axes[1])
figure.set_ylabel("cars at first location", fontsize=30)
figure.set_yticks(list(reversed(range(21))))
figure.set_title("policy", fontsize=30)
figure.set_xlabel("cars at second location", fontsize=30)

plt.savefig("./pics/hello.pdf")
plt.show()
plt.close()
