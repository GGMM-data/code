import matplotlib.pyplot as plt

"""
marker
'.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', 'P', 'X', '|', '_'
"""

marker_list = [',', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', 'P', 'X', '|', '_']

for i in range(len(marker_list)):
    plt.scatter(i, i, marker=marker_list[i])

plt.show()
plt.close()
