import matplotlib.pyplot as plt

"""
marker
'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'
"""

marker_list = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

for i in range(len(marker_list)):
    plt.scatter(i, i, marker=marker_list[i])

plt.show()
plt.close()
